# nl2scene3d/scene_state.py
"""
Loading and construction of SceneState from an open Blender scene.

Responsibilities:
    1. Extract objects and transforms from the current bpy scene.
    2. Classify each object (category + is_movable).
    3. Compute RoomBounds from structural geometry.
    4. Detect parent-child (grouping) relationships and annotate them
       directly on each SceneObject (parent / children fields).
       Grouping is computed exactly once here, not at every pipeline step.
    5. Serialize / deserialize SceneState to and from JSON.

This module contains no randomization, LLM, or rendering logic.
Must be executed inside Blender's embedded Python environment.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

from nl2scene3d.config import PipelineConfig
from nl2scene3d.models import RoomBounds, SceneObject, SceneState, Transform

try:
    import bpy  # type: ignore
except ImportError:
    bpy = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Object classification
# ---------------------------------------------------------------------------

def _has_kw(keywords: list[str] | tuple[str, ...], text: str) -> bool:
    """Return True if any keyword from the list is found in text."""
    return any(k in text for k in keywords)


def classify_object(
    name: str,
    object_type: str,
    dimensions: list[float],
    config: PipelineConfig,
) -> tuple[str, bool]:
    """
    Determine the (category, is_movable) tuple for an object.

    Args:
        name:        Blender object name.
        object_type: Blender object type string.
        dimensions:  Object dimensions [x, y, z] in meters.
        config:      Pipeline configuration.

    Returns:
        (category, is_movable)
    """
    name_lower = name.lower()

    # Non-mesh types are always static.
    if object_type in config.non_mesh_types:
        return "technical", False

    # Very small objects are fixed decorations (knobs, screws, etc.).
    max_dim = max(dimensions) if dimensions else 0.0
    if max_dim < config.min_object_dimension:
        return "decoration_small", False

    # Lights.
    if _has_kw(("lamp", "lampada", "light"), name_lower):
        if _has_kw(config.ceiling_light_patterns, name_lower):
            return "light_ceiling", False
        return "light_floor", True

    # Knobs and handles are always fixed.
    if _has_kw(("knob", "pomello", "handle", "maniglia"), name_lower):
        return "technical", False

    # Decorations and desk electronics.
    if _has_kw(
        (
            "decor", "decoration", "ornament", "book", "bottle",
            "monitor", "pc", "computer", "keyboard", "mouse", "trashbin",
        ),
        name_lower,
    ):
        return "decoration", True

    # Structural elements are always static.
    # Checked after decorations to ensure correct precedence.
    if _has_kw(config.structural_patterns, name_lower):
        return "structural", False

    # Main furniture categories.
    if _has_kw(("sofa", "couch", "divano"), name_lower):
        return "seating_large", True
    if _has_kw(("chair", "sedia", "stool", "sgabello"), name_lower):
        return "seating_small", True
    if _has_kw(("table", "tavolo", "desk", "scrivania"), name_lower):
        return "table", True
    if _has_kw(("shelf", "scaffale", "bookcase", "libreria"), name_lower):
        return "storage", True
    if _has_kw(("bed", "letto", "mattress", "materasso"), name_lower):
        return "bed", True
    if _has_kw(("wardrobe", "armadio", "cabinet", "dresser"), name_lower):
        return "storage", True
    if _has_kw(("rug", "tappeto", "carpet"), name_lower):
        return "rug", True
    if _has_kw(("plant", "pianta", "vase", "vaso"), name_lower):
        return "decoration", True

    # Nightstands are always parent-eligible (never a child), so that an
    # overlying lamp remains attached as a child and moves with them.
    if _has_kw(("nightstand", "comodino", "bedside", "bedside_table"), name_lower):
        return "furniture", True

    return "furniture", True


# ---------------------------------------------------------------------------
# Room bounds computation
# ---------------------------------------------------------------------------

def compute_room_bounds(objects: list[SceneObject]) -> RoomBounds:
    """
    Compute room bounds from the structural objects in the scene.

    Strategy:
        1. If a single dominant room mesh exists (volume > 50% of total structural
           volume), use its dimensions as the room boundary.
        2. Otherwise combine the AABBs of all structural objects.
        3. z_ceiling is derived from objects whose name contains 'ceiling', 'room',
           or 'roof'. If none are found, the maximum structural Z is used, with
           2.5 m as a fallback minimum.
    """
    structural = [o for o in objects if o.category == "structural"]
    if not structural:
        logger.warning("No structural objects found. Using the full object set.")
        structural = objects
    if not structural:
        logger.warning("Empty scene. Using default bounds of +/- 5 m.")
        return RoomBounds(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0)

    # Determine ceiling height.
    ceiling_kws  = ("ceiling", "room", "roof", "soffitto")
    ceiling_objs = [o for o in structural if _has_kw(ceiling_kws, o.name.lower())]

    if ceiling_objs:
        z_ceiling = max(o.transform.z_range()[1] for o in ceiling_objs)
    else:
        max_z = max(
            (o.transform.z_range()[1] for o in structural),
            default=2.5,
        )
        z_ceiling = max_z if max_z > 1.0 else 2.5

    # Strategy 1: single dominant room mesh.
    vols = [
        (o, o.transform.dimensions[0] * o.transform.dimensions[1] * o.transform.dimensions[2])
        for o in structural
    ]
    largest_obj, max_vol = max(vols, key=lambda x: x[1])
    total_vol = sum(v for _, v in vols)

    if total_vol > 0 and max_vol > 0.5 * total_vol and max_vol > 1.0:
        x_min, x_max, y_min, y_max = largest_obj.transform.aabb_xy(margin=0.0)
        logger.info(
            "Room identified from single object '%s' (AABB: X[%.2f, %.2f] Y[%.2f, %.2f]).",
            largest_obj.name, x_min, x_max, y_min, y_max,
        )
        return RoomBounds(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_floor=0.0,
            z_ceiling=z_ceiling,
        )

    # Strategy 2: union AABB of all structural objects.
    aabbs = [o.transform.aabb_xy(margin=0.0) for o in structural]
    return RoomBounds(
        x_min=min(a[0] for a in aabbs),
        x_max=max(a[1] for a in aabbs),
        y_min=min(a[2] for a in aabbs),
        y_max=max(a[3] for a in aabbs),
        z_floor=0.0,
        z_ceiling=z_ceiling,
    )


# ---------------------------------------------------------------------------
# Grouping (parent-child detection) --- computed once
# ---------------------------------------------------------------------------

def _volume(o: SceneObject) -> float:
    """Return the volume of an object's bounding box."""
    d = o.transform.dimensions
    return d[0] * d[1] * d[2]


def _sat_overlap(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
) -> bool:
    """Separating Axis Theorem for two convex 2D polygons."""

    def axes(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
        result = []
        n = len(poly)
        for i in range(n):
            p1, p2 = poly[i], poly[(i + 1) % n]
            ex, ey = p2[0] - p1[0], p2[1] - p1[1]
            mag = math.hypot(ex, ey)
            if mag > 1e-6:
                result.append((-ey / mag, ex / mag))
        return result

    def project(
        poly: list[tuple[float, float]],
        axis: tuple[float, float],
    ) -> tuple[float, float]:
        dots = [p[0] * axis[0] + p[1] * axis[1] for p in poly]
        return min(dots), max(dots)

    for axis in axes(poly_a) + axes(poly_b):
        mn_a, mx_a = project(poly_a, axis)
        mn_b, mx_b = project(poly_b, axis)
        if mx_a < mn_b or mx_b < mn_a:
            return False  # Separating axis found: no collision.

    return True  # No separating axis found: collision.


def compute_grouping(objects: list[SceneObject]) -> None:
    """
    Detect parent-child relationships and annotate them in-place on each SceneObject.

    An object B is considered a child of A if ONE of the following is true:
        1. Surface:   B rests on top of A (B's z_bottom ~= A's z_top, +/- 15 cm)
                      AND their XY footprints overlap.
        2. Content:   B is contained within A's Z range (e.g. book on a shelf)
                      AND their XY footprints overlap.
        3. Proximity: A and B have significant Z overlap (>= 30% of B's height)
                      AND their XY footprints are within 60 cm
                      (e.g. chair under a desk, PC tower beside a desk).

    In all cases A must have a volume >= 1.5x that of B.

    Each object receives:
        obj.parent:   name of its direct parent (or None).
        obj.children: list of its direct children.

    The grouping is direct (parent-child only), not transitive.
    """
    # Reset existing grouping data.
    for obj in objects:
        obj.parent   = None
        obj.children = []

    movable = [o for o in objects if o.is_movable and o.category != "structural"]
    by_name = {o.name: o for o in objects}

    # Attempt to load an existing grouping from Blender custom properties.
    # This keeps grouping stable across multiple calls within the same session.
    has_custom_props = False
    if bpy:
        for obj in objects:
            if obj.name in bpy.data.objects:  # type: ignore[operator]
                b_obj  = bpy.data.objects[obj.name]  # type: ignore[index]
                p_name = b_obj.get("nl2_parent")
                if p_name and p_name in by_name:
                    obj.parent = p_name
                    by_name[p_name].children.append(obj.name)
                    has_custom_props = True

    if has_custom_props:
        logger.info("Persistent grouping loaded from Blender custom properties.")
        return

    # Compute grouping from scratch.
    # Main furniture pieces (beds, tables, wardrobes) must always be independent roots.
    ALLOWED_CHILD_CATEGORIES  = {"decoration", "decoration_small", "seating_small", "light_floor"}
    ALLOWED_PARENT_CATEGORIES = {"table", "desk", "storage", "seating_large", "bed", "furniture"}

    for child in movable:
        # Main furniture pieces cannot be children of anything.
        if child.category not in ALLOWED_CHILD_CATEGORIES:
            continue

        child_z_min, child_z_max = child.transform.z_range()
        child_vol    = _volume(child)
        child_height = child_z_max - child_z_min
        child_poly   = child.transform.obb_corners_xy(margin=0.0)

        best_parent: Optional[str] = None
        best_score: float = float("inf")  # Lower score = better parent.

        for candidate in movable:
            if candidate.name == child.name:
                continue

            # Only allowed parent categories can have children.
            if candidate.category not in ALLOWED_PARENT_CATEGORIES:
                continue

            cand_vol = _volume(candidate)

            # XY footprint comparison is more reliable than volume for stacked objects.
            # A tall lamp has a large volume but a small footprint; using volume alone
            # would prevent a small nightstand from being recognized as a parent.
            child_area = child.transform.dimensions[0] * child.transform.dimensions[1]
            cand_area  = candidate.transform.dimensions[0] * candidate.transform.dimensions[1]

            parent_bigger_footprint = cand_area  >= child_area  * 1.05  # for "resting on top"
            parent_bigger_volume    = cand_vol   >= child_vol   * 1.2   # for "contained" / "proximity"

            par_z_min, par_z_max = candidate.transform.z_range()

            # Criterion 1: child rests on top of the parent (window slightly widened).
            z_diff_top = child_z_min - par_z_max
            is_on_top  = -0.08 <= z_diff_top <= 0.20

            # Criterion 2: child is contained within the parent's Z range.
            is_inside = (
                child_z_min >= par_z_min - 0.05
                and child_z_max <= par_z_max + 0.05
            )

            # Criterion 3: Z overlap + XY proximity (chair under desk, PC beside desk).
            z_overlap     = max(0.0, min(child_z_max, par_z_max) - max(child_z_min, par_z_min))
            has_z_overlap = (child_height > 0) and (z_overlap / child_height >= 0.30)

            matched = False
            score   = 0.0

            if (is_on_top and parent_bigger_footprint) or (is_inside and parent_bigger_volume):
                par_poly = candidate.transform.obb_corners_xy(margin=0.0)
                if _sat_overlap(child_poly, par_poly):
                    matched = True
                    score = abs(z_diff_top) if is_on_top else 0.0

            if not matched and has_z_overlap:
                # Proximity check: only for minor categories.
                # Nightstands and dressers remain independent roots.
                allowed_proximity_children = {
                    "seating_small", "decoration", "decoration_small", "light_floor"
                }
                allowed_proximity_parents = {
                    "table", "desk", "storage", "seating_large", "bed", "furniture"
                }

                if (
                    child.category     in allowed_proximity_children
                    and candidate.category in allowed_proximity_parents
                    and parent_bigger_volume
                ):
                    par_poly_expanded = candidate.transform.obb_corners_xy(margin=0.15)
                    if _sat_overlap(child_poly, par_poly_expanded):
                        matched = True
                        # Score = XY distance from parent center.
                        cx, cy = candidate.transform.geometric_center_xy()
                        bx, by = child.transform.geometric_center_xy()
                        score  = 10.0 + math.hypot(bx - cx, by - cy)

            if not matched:
                continue

            # Choose the parent with the lowest score (closest / most directly on top).
            if score < best_score:
                best_score  = score
                best_parent = candidate.name

        if best_parent is not None:
            child.parent = best_parent
            by_name[best_parent].children.append(child.name)
            logger.debug(
                "Grouping: '%s' -> parent '%s' (score=%.2f).",
                child.name, best_parent, best_score,
            )

    # Persist the computed grouping to Blender custom properties for future calls.
    if bpy:
        for obj in objects:
            if obj.name in bpy.data.objects:  # type: ignore[operator]
                b_obj = bpy.data.objects[obj.name]  # type: ignore[index]
                b_obj["nl2_parent"] = obj.parent if obj.parent else ""

    n_groups   = sum(1 for o in objects if o.children)
    n_children = sum(1 for o in objects if o.parent is not None)
    logger.info("Grouping complete: %d groups, %d child objects.", n_groups, n_children)


# ---------------------------------------------------------------------------
# Static placement rules for wall- and ceiling-mounted objects
# ---------------------------------------------------------------------------

def apply_static_placement_rules(objects, room_bounds, config) -> None:
    """
    Freeze (is_movable=False) root objects that are mounted high on walls
    (shelves, wall lamps, pictures) or attached to the ceiling, propagating
    the static state to all their children.

    Must be called AFTER compute_grouping:
        - Uses obj.parent / obj.children to propagate static state.
        - Acts only on root objects (parent is None). An object resting on a
          piece of furniture (e.g. a table lamp) is a child and moves rigidly
          with its parent, so it is NOT frozen here.

    Criteria:
        - z_min >= static_height_threshold -> high wall-mounted object.
        - z_max close to the ceiling       -> ceiling-mounted object.
    """
    by_name = {o.name: o for o in objects}

    def freeze(name: str) -> None:
        """Recursively freeze an object and all its children."""
        o = by_name.get(name)
        if not o:
            return
        o.is_movable = False
        for c in o.children:
            freeze(c)

    threshold = config.static_height_threshold
    ceiling   = room_bounds.z_ceiling
    frozen    = 0

    for o in objects:
        if not o.is_movable or o.parent is not None:
            continue  # Only movable root objects are considered.

        z_min, z_max = o.transform.z_range()
        on_wall_high = z_min >= threshold
        on_ceiling   = config.freeze_ceiling_objects and z_max >= ceiling - 0.15

        if on_wall_high or on_ceiling:
            freeze(o.name)
            frozen += 1
            logger.info(
                "Object frozen: '%s' (z_min=%.2f, z_max=%.2f, ceiling=%.2f).",
                o.name, z_min, z_max, ceiling,
            )

    if frozen:
        logger.info("Static placement rules: %d groups frozen.", frozen)


# ---------------------------------------------------------------------------
# SceneLoader
# ---------------------------------------------------------------------------

class SceneLoader:
    """
    Loads and inspects a Blender scene, producing a complete SceneState.

    Must be used inside Blender's embedded Python environment (bpy available).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        logger.info(
            "SceneLoader initialized. Max movable objects: %d.", config.max_movable_objects
        )

    def load_blend_file(self, blend_path: Path) -> None:
        """Open a .blend file, replacing the current scene."""
        if not blend_path.exists():
            raise FileNotFoundError(f".blend file not found: {blend_path}")

        try:
            import bpy  # type: ignore  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("bpy requires the Blender environment.") from exc

        logger.info("Opening '%s'...", blend_path)
        bpy.ops.wm.open_mainfile(filepath=str(blend_path))
        logger.info(".blend file opened.")

    def extract_scene_state(self, scene_name: Optional[str] = None) -> SceneState:
        """
        Extract the complete state of the current Blender scene.

        Steps:
            1. Iterate all objects -> classify -> build SceneObject instances.
            2. Compute RoomBounds.
            3. Compute parent-child grouping and annotate all objects.

        Returns:
            Complete SceneState with pipeline_step='original'.
        """
        try:
            import bpy       # type: ignore  # noqa: PLC0415
            import mathutils # type: ignore  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("bpy/mathutils require the Blender environment.") from exc

        blender_scene  = bpy.context.scene  # type: ignore[union-attr]
        effective_name = scene_name or blender_scene.name  # type: ignore[union-attr]

        logger.info(
            "Extracting scene '%s' (%d Blender objects).",
            effective_name, len(blender_scene.objects),  # type: ignore[union-attr]
        )

        objects: list[SceneObject] = []
        movable_count = 0

        for blender_obj in blender_scene.objects:  # type: ignore[union-attr]
            name       = blender_obj.name
            obj_type   = blender_obj.type
            dimensions = [
                blender_obj.dimensions.x,
                blender_obj.dimensions.y,
                blender_obj.dimensions.z,
            ]

            category, is_movable = classify_object(name, obj_type, dimensions, self.config)

            # Enforce the movable object limit.
            if is_movable and movable_count >= self.config.max_movable_objects:
                logger.debug(
                    "'%s' demoted: movable object limit of %d reached.",
                    name, self.config.max_movable_objects,
                )
                is_movable = False

            # Force Euler rotation mode for compatibility.
            if blender_obj.rotation_mode not in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"):
                blender_obj.rotation_mode = "XYZ"

            # Origin offset relative to the geometric center (meters, already scaled).
            local_center = (
                sum(
                    (mathutils.Vector(p) for p in blender_obj.bound_box),
                    mathutils.Vector(),
                ) / 8
            )
            origin_offset = [
                local_center.x * blender_obj.scale.x,
                local_center.y * blender_obj.scale.y,
                local_center.z * blender_obj.scale.z,
            ]

            # Always use world coordinates (matrix_world) for consistency,
            # even when the object has a native Blender parent.
            world_mat = blender_obj.matrix_world
            world_loc = world_mat.translation
            world_rot = world_mat.to_euler("XYZ")

            transform = Transform(
                location=[world_loc.x, world_loc.y, world_loc.z],
                rotation_euler=[world_rot.x, world_rot.y, world_rot.z],
                dimensions=dimensions,
                origin_offset=origin_offset,
            )

            scene_obj = SceneObject(
                name=name,
                object_type=obj_type,
                transform=transform,
                category=category,
                is_movable=is_movable,
            )
            objects.append(scene_obj)
            if is_movable:
                movable_count += 1

        logger.info("Extracted %d objects (%d movable).", len(objects), movable_count)

        # Compute room bounds.
        room_bounds = compute_room_bounds(objects)
        logger.info(
            "Bounds: X[%.2f, %.2f] Y[%.2f, %.2f] Z[%.2f, %.2f].",
            room_bounds.x_min, room_bounds.x_max,
            room_bounds.y_min, room_bounds.y_max,
            room_bounds.z_floor, room_bounds.z_ceiling,
        )

        # Compute grouping and annotate parent/children on each object.
        compute_grouping(objects)

        # Freeze shelves, wall lamps, and ceiling objects (and their children).
        apply_static_placement_rules(objects, room_bounds, self.config)

        return SceneState(
            scene_name=effective_name,
            objects=objects,
            room_bounds=room_bounds,
            pipeline_step="original",
        )

    # ---------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------

    def save_state(self, state: SceneState, output_path: Path) -> None:
        """Serialize a SceneState to a JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(state.to_dict(), fh, indent=2, ensure_ascii=False)
        logger.info(
            "State '%s' (step: %s) saved to: %s",
            state.scene_name, state.pipeline_step, output_path,
        )

    @staticmethod
    def load_state(json_path: Path) -> SceneState:
        """Deserialize a SceneState from a JSON file."""
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        state = SceneState.from_dict(data)
        logger.info(
            "State '%s' (step: %s) loaded from: %s",
            state.scene_name, state.pipeline_step, json_path,
        )
        return state