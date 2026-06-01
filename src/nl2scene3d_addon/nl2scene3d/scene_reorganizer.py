# nl2scene3d/scene_reorganizer.py
"""
LLM-based scene reorganization (text-only, via Gemini).

Responsibilities:
  - Build a flat JSON with ONLY the movable root objects and their dimensions,
    keeping token usage low.
  - Build a JSON with fixed structural elements (walls, windows, doors) to give
    the LLM full spatial context of the room.
  - Send the prompt and receive new (X, Y, Rz) coordinates.
  - Validate and sanitize the output: group bounds clamp, Z lock, 90-degree
    rotation snap.
  - Move children with a rigid transform relative to their parent, always
    preserving the group structure.
  - Resolve post-LLM collisions with a Minimum Translation Vector (MTV)
    applied rigidly to the entire group.

Fundamental rules:
  - Z is NEVER modified: it is always taken from the original scene.
  - The LLM receives ONLY movable root objects (is_child=false). Children
    are managed internally via rigid transforms.
  - Parent/child grouping is pre-computed by SceneLoader.extract_scene_state()
    and is not recalculated here.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path

from nl2scene3d.gemini_client   import GeminiClient, GeminiParsingError
from nl2scene3d.models          import SceneObject, SceneState, Transform, RoomBounds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Wall margin for the reorganizer (must match RandomizerConfig.wall_margin and
# wall_margin_meters in settings.toml). Room bounds include wall thickness, so
# a margin that is too small leaves objects clipping through the inner wall face.
REORDER_WALL_MARGIN = 0.20


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _load_prompt_template(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    with open(prompt_path, encoding="utf-8") as fh:
        return fh.read()


def _build_flat_json_for_llm(state: SceneState) -> str:
    """
    Builds a flat JSON array with all movable ROOT objects.

    Each entry contains:
      - name, category
      - Center position (x, y) and rotation (rz_deg) in meters / degrees
      - Dimensions (w, d, h) in meters
      - Exact rotated 2D AABB (x_min, x_max, y_min, y_max)
      - Required clearances: wall_margin and collision_margin
    """
    entries = []

    for obj in state.objects:
        if not obj.is_movable or obj.parent is not None:
            continue

        rz_rad = obj.transform.rotation_euler[2]
        rz_deg = round(math.degrees(rz_rad) % 360.0, 1)
        x_min, x_max, y_min, y_max = obj.transform.aabb_xy(margin=0.0)

        entry: dict = {
            "name":     obj.name,
            "category": obj.category,
            "x":        round(obj.transform.location[0], 3),
            "y":        round(obj.transform.location[1], 3),
            "rz_deg":   rz_deg,
            "w":        round(obj.transform.dimensions[0], 3),
            "d":        round(obj.transform.dimensions[1], 3),
            "h":        round(obj.transform.dimensions[2], 3),
            "aabb_2d": {
                "x_min": round(x_min, 3),
                "x_max": round(x_max, 3),
                "y_min": round(y_min, 3),
                "y_max": round(y_max, 3),
            },
            "wall_margin_required":      REORDER_WALL_MARGIN,
            "collision_margin_required": 0.05,
        }

        if obj.children:
            entry["children"] = obj.children

        entries.append(entry)

    return json.dumps(entries, indent=2, ensure_ascii=False)


def _build_structural_json_for_llm(state: SceneState) -> str:
    """
    Builds a JSON list of structural elements (walls, doors, windows).

    These elements are NOT reorganized, but are provided as spatial
    reference for the LLM. Doors and windows include a clearance_zone
    that must remain free of furniture.
    """
    entries = []

    for obj in state.objects:
        if obj.is_movable or obj.category != "structural":
            continue

        rz_rad = obj.transform.rotation_euler[2]
        rz_deg = round(math.degrees(rz_rad) % 360.0, 1)
        x_min, x_max, y_min, y_max = obj.transform.aabb_xy(margin=0.0)

        obj_type      = "wall"
        clearance_zone = None
        name_lower    = obj.name.lower()

        if "door" in name_lower or "porta" in name_lower:
            obj_type           = "door"
            clearance_distance = 0.90   # clearance in front of the door, in meters
            if abs(rz_deg % 180) < 45 or abs(rz_deg % 180 - 180) < 45:
                # Door aligned with the X axis (0 deg or 180 deg).
                clearance_zone = {
                    "x_min": round(x_min - 0.05, 3),
                    "x_max": round(x_max + 0.05, 3),
                    "y_min": round(y_min - clearance_distance, 3),
                    "y_max": round(y_max + clearance_distance, 3),
                }
            else:
                # Door aligned with the Y axis (90 deg or 270 deg).
                clearance_zone = {
                    "x_min": round(x_min - clearance_distance, 3),
                    "x_max": round(x_max + clearance_distance, 3),
                    "y_min": round(y_min - 0.05, 3),
                    "y_max": round(y_max + 0.05, 3),
                }

        elif "window" in name_lower or "finestra" in name_lower:
            obj_type       = "window"
            clearance_zone = {
                "x_min": round(x_min - 0.10, 3),
                "x_max": round(x_max + 0.10, 3),
                "y_min": round(y_min - 0.10, 3),
                "y_max": round(y_max + 0.10, 3),
            }

        entry: dict = {
            "name":   obj.name,
            "type":   obj_type,
            "x":      round(obj.transform.location[0], 3),
            "y":      round(obj.transform.location[1], 3),
            "rz_deg": rz_deg,
            "w":      round(obj.transform.dimensions[0], 3),
            "d":      round(obj.transform.dimensions[1], 3),
            "h":      round(obj.transform.dimensions[2], 3),
            "aabb_2d": {
                "x_min": round(x_min, 3),
                "x_max": round(x_max, 3),
                "y_min": round(y_min, 3),
                "y_max": round(y_max, 3),
            },
        }

        if clearance_zone is not None:
            entry["clearance_zone"] = clearance_zone
            entry["clearance_note"] = (
                "MUST be clear of furniture. No objects allowed in this zone."
                if obj_type == "door"
                else "Should be kept clear. Avoid tall furniture."
            )

        entries.append(entry)

    return json.dumps(entries, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Geometry utilities (imported from nl2scene3d.utils.geometry)
# ---------------------------------------------------------------------------

from nl2scene3d.utils.geometry import (
    group_aabb_xy,
    has_collision,
    is_finite_float,
    penetration_vector,
    snap_rotation_90,
)


# ---------------------------------------------------------------------------
# Child transform helper
# ---------------------------------------------------------------------------

def _apply_rigid_child_transform(
    child:             SceneObject,
    old_parent_loc:    list[float],
    old_parent_rz:     float,
    new_parent_loc:    list[float],
    new_parent_rz:     float,
    original_child_z:  float | None = None,
) -> SceneObject:
    """
    Returns a COPY of the child, rigidly moved in the XY plane relative to its parent.

    Z is never changed: it is set to `original_child_z` if provided, otherwise
    the child's current Z is preserved unchanged.
    """
    new_child = child.copy()

    rel_x        = child.transform.location[0] - old_parent_loc[0]
    rel_y        = child.transform.location[1] - old_parent_loc[1]
    d_rz         = new_parent_rz - old_parent_rz
    cos_a, sin_a = math.cos(d_rz), math.sin(d_rz)

    new_child.transform.location[0] = new_parent_loc[0] + rel_x * cos_a - rel_y * sin_a
    new_child.transform.location[1] = new_parent_loc[1] + rel_x * sin_a + rel_y * cos_a
    new_child.transform.location[2] = (
        original_child_z if original_child_z is not None else child.transform.location[2]
    )
    new_child.transform.rotation_euler[2] = (
        child.transform.rotation_euler[2] + d_rz
    ) % (2 * math.pi)

    return new_child


# ---------------------------------------------------------------------------
# Group geometry helpers
# ---------------------------------------------------------------------------

def _clamp_parent_group_location(
    orig_parent:   SceneObject,
    proposed_loc:  list[float],
    proposed_rz:   float,
    orig_children: list[SceneObject],
    room_bounds:   RoomBounds,
) -> list[float]:
    """
    Clamps the parent position using the group's real AABB so that the entire
    group (parent + children) stays inside the room bounds with the wall margin.
    """
    g_x_min, g_x_max, g_y_min, g_y_max = group_aabb_xy(
        orig_parent, proposed_loc, proposed_rz, orig_children, margin=0.0
    )

    px, py = proposed_loc[0], proposed_loc[1]
    wm     = REORDER_WALL_MARGIN

    overflow_left  = max(0.0, (room_bounds.x_min + wm) - g_x_min)
    overflow_right = max(0.0, g_x_max - (room_bounds.x_max - wm))
    overflow_front = max(0.0, (room_bounds.y_min + wm) - g_y_min)
    overflow_back  = max(0.0, g_y_max - (room_bounds.y_max - wm))

    dx = overflow_left if overflow_left > overflow_right else -overflow_right
    dy = overflow_front if overflow_front > overflow_back else -overflow_back

    return [px + dx, py + dy, proposed_loc[2]]


def _group_has_collision(
    root_obj:        SceneObject,
    current_children: list[SceneObject],
    collidable:      list[SceneObject],
    wall_margin:     float,
    furniture_margin: float,
    room_bounds=None,
) -> bool:
    """
    Returns True if the root or any of its children collides with `collidable`
    or exits the room bounds.
    """
    if has_collision(
        root_obj, collidable,
        wall_margin=wall_margin,
        furniture_margin=furniture_margin,
        room_bounds=room_bounds,
    ):  # type: ignore[call-arg]
        return True
    for child in current_children:
        if has_collision(
            child, collidable,
            wall_margin=wall_margin,
            furniture_margin=furniture_margin,
            room_bounds=room_bounds,
        ):  # type: ignore[call-arg]
            return True
    return False


def _group_penetration_vector(
    root_obj:         SceneObject,
    current_children: list[SceneObject],
    other:            SceneObject,
    margin:           float = 0.05,
) -> tuple[float, float]:
    """
    Computes the MTV between the entire group (root + children) and one other object.

    Returns the displacement vector of the member with the deepest penetration.
    """
    best_dx, best_dy = 0.0, 0.0
    best_pen         = 0.0

    for member in [root_obj] + current_children:
        dx, dy = penetration_vector(member, other, margin=margin)
        pen    = abs(dx) + abs(dy)
        if pen > best_pen:
            best_pen         = pen
            best_dx, best_dy = dx, dy

    return best_dx, best_dy


def _find_free_group_position(
    orig_obj:     SceneObject,
    obj:          SceneObject,
    current_children: list[SceneObject],
    collidable:   list[SceneObject],
    room_bounds:  RoomBounds,
    by_name_orig: dict[str, SceneObject],
    corrected:    dict[str, SceneObject],
) -> bool:
    """
    Searches a grid of positions for a collision-free placement of the group.

    Used as a fallback when MTV iterations fail to resolve a collision. Scans
    the room with decreasing step size and stops at the first free position.
    Updates `obj` and child entries in `corrected` in-place.

    Returns True if a free position was found.
    """
    rz     = obj.transform.rotation_euler[2]
    z_lock = obj.transform.location[2]

    gx_min, gx_max, gy_min, gy_max = group_aabb_xy(
        obj, obj.transform.location, rz, current_children, margin=0.0
    )
    half_w = (gx_max - gx_min) / 2.0
    half_d = (gy_max - gy_min) / 2.0
    cx_now = obj.transform.location[0]
    cy_now = obj.transform.location[1]
    wm     = REORDER_WALL_MARGIN

    x_lo = room_bounds.x_min + wm + half_w
    x_hi = room_bounds.x_max - wm - half_w
    y_lo = room_bounds.y_min + wm + half_d
    y_hi = room_bounds.y_max - wm - half_d

    if x_hi <= x_lo or y_hi <= y_lo:
        return False

    def _try(cx: float, cy: float) -> bool:
        test = obj.copy()
        test.transform.location = [cx, cy, z_lock]
        test_children = [
            _apply_rigid_child_transform(
                by_name_orig[c],
                old_parent_loc=orig_obj.transform.location,
                old_parent_rz=orig_obj.transform.rotation_euler[2],
                new_parent_loc=[cx, cy, z_lock],
                new_parent_rz=rz,
            )
            for c in orig_obj.children
            if c in by_name_orig
        ]
        if _group_has_collision(
            test, test_children, collidable,
            wall_margin=wm, furniture_margin=0.02, room_bounds=room_bounds,
        ):
            return False
        # Position is free: commit the change.
        obj.transform.location = [cx, cy, z_lock]
        for c in orig_obj.children:
            if c in by_name_orig:
                corrected[c] = _apply_rigid_child_transform(
                    by_name_orig[c],
                    old_parent_loc=orig_obj.transform.location,
                    old_parent_rz=orig_obj.transform.rotation_euler[2],
                    new_parent_loc=[cx, cy, z_lock],
                    new_parent_rz=rz,
                )
        return True

    # Try with increasing grid resolution, prioritizing positions close to the
    # current location to minimize unnecessary displacement.
    for steps in (5, 9, 15):
        xs = [x_lo + (x_hi - x_lo) * k / (steps - 1) for k in range(steps)]
        ys = [y_lo + (y_hi - y_lo) * k / (steps - 1) for k in range(steps)]
        candidates = sorted(
            ((x, y) for x in xs for y in ys),
            key=lambda p: (p[0] - cx_now) ** 2 + (p[1] - cy_now) ** 2,
        )
        for cx, cy in candidates:
            if _try(cx, cy):
                return True

    return False


# ---------------------------------------------------------------------------
# LLM output validation and sanitization
# ---------------------------------------------------------------------------

def _validate_and_sanitize_llm_output(
    llm_output:     dict | list,
    original_state: SceneState,
) -> SceneState:
    """
    Validates and sanitizes the raw JSON output from the LLM.

    Expected LLM output format — a list of root objects with fields:
      "name": exact Blender object name
      "x", "y": new coordinates (float, meters)
      "rz_deg": new Z rotation (0-360 degrees)

    Sanitization rules applied in order:
      1. Lock Z to the original value (never changed).
      2. Snap Z rotation to multiples of 90 degrees.
      3. Preserve original X and Y rotations.
      4. Move children with a rigid transform relative to their parent.
      5. Clamp every group to the room bounds.
      6. Resolve post-LLM collisions with MTV; fall back to grid search if needed.
    """
    room_bounds = original_state.room_bounds

    logger.info("Validating LLM output. Type: %s.", type(llm_output).__name__)

    # Normalize: accept both a flat list or a dict with an "objects" key.
    if isinstance(llm_output, list):
        llm_objects_list: list = llm_output
    elif isinstance(llm_output, dict):
        llm_objects_list = llm_output.get("objects", [])
    else:
        logger.error("Invalid LLM output type (%s). Returning original state.", type(llm_output))
        return original_state

    logger.info("Processing %d LLM objects.", len(llm_objects_list))

    # Build name -> LLM entry map.
    llm_by_name: dict[str, dict] = {}
    for i, item in enumerate(llm_objects_list):
        if not isinstance(item, dict):
            logger.warning("Item %d is not a dict (type=%s). Skipped.", i, type(item).__name__)
            continue
        if "name" not in item:
            logger.warning("Item %d has no 'name' field (keys=%s). Skipped.", i, list(item.keys()))
            continue
        name = item["name"]
        if not isinstance(name, str):
            logger.warning("Item %d 'name' is not a string (type=%s). Skipped.", i, type(name).__name__)
            continue
        llm_by_name[name] = item

    logger.info("LLM name map: %d entries.", len(llm_by_name))

    by_name_orig  = {obj.name: obj for obj in original_state.objects}
    corrected:     dict[str, SceneObject] = {}
    clamped_count = 0
    missing_count = 0

    for orig_obj in original_state.objects:
        # Structural objects are always left unchanged.
        if not orig_obj.is_movable:
            corrected[orig_obj.name] = orig_obj.copy()
            continue

        # Children are processed as part of their parent — skip for now.
        if orig_obj.parent is not None:
            if orig_obj.name not in corrected:
                corrected[orig_obj.name] = orig_obj.copy()
            continue

        # Root movable object: read the LLM-proposed position.
        llm_data = llm_by_name.get(orig_obj.name)
        if llm_data is None:
            logger.warning(
                "Root object '%s' absent from LLM output. Keeping original position.",
                orig_obj.name,
            )
            missing_count += 1
            corrected[orig_obj.name] = orig_obj.copy()
            continue

        try:
            if "x" in llm_data and "y" in llm_data:
                new_x   = float(llm_data["x"])
                new_y   = float(llm_data["y"])
                rz_deg  = float(
                    llm_data.get("rz_deg", math.degrees(orig_obj.transform.rotation_euler[2]))
                )
                rz_rad  = rz_deg * (math.pi / 180.0)
                new_loc = [new_x, new_y, orig_obj.transform.location[2]]
                new_rot = [
                    orig_obj.transform.rotation_euler[0],
                    orig_obj.transform.rotation_euler[1],
                    rz_rad,
                ]
            else:
                # Fall back to the verbose format {location, rotation_euler}.
                new_loc = list(llm_data.get("location", orig_obj.transform.location))
                new_rot = list(llm_data.get("rotation_euler", orig_obj.transform.rotation_euler))
                if len(new_loc) != 3 or len(new_rot) != 3:
                    raise ValueError("location or rotation_euler must have exactly 3 components.")
                new_loc = [float(v) for v in new_loc]
                new_rot = [float(v) for v in new_rot]

            # Lock Z and X/Y rotations to original values.
            new_loc[2] = orig_obj.transform.location[2]
            new_rot[0] = orig_obj.transform.rotation_euler[0]
            new_rot[1] = orig_obj.transform.rotation_euler[1]

            if not all(is_finite_float(v) for v in new_loc + new_rot):
                raise ValueError("Non-finite values detected.")

            new_rot[2] = snap_rotation_90(new_rot[2])

        except (TypeError, ValueError, KeyError) as exc:
            logger.warning(
                "Invalid coordinates for '%s': %s. Keeping original position.",
                orig_obj.name, exc,
            )
            corrected[orig_obj.name] = orig_obj.copy()
            continue

        # Clamp the entire group to the room bounds.
        if room_bounds is not None:
            child_objs = [by_name_orig[c] for c in orig_obj.children if c in by_name_orig]
            clamped    = _clamp_parent_group_location(orig_obj, new_loc, new_rot[2], child_objs, room_bounds)
            if clamped != new_loc:
                logger.debug("'%s' (group): clamped %s -> %s.", orig_obj.name, new_loc, clamped)
                new_loc       = clamped
                clamped_count += 1

        # Absolute Z lock: always use the original value.
        new_loc[2] = orig_obj.transform.location[2]

        new_obj = orig_obj.copy()
        new_obj.transform = Transform(
            location=new_loc,
            rotation_euler=new_rot,
            dimensions=list(orig_obj.transform.dimensions),
            origin_offset=list(orig_obj.transform.origin_offset),
        )
        corrected[orig_obj.name] = new_obj

        # Move children rigidly (Z of each child is restored to its original value).
        for child_name in orig_obj.children:
            orig_child = by_name_orig.get(child_name)
            if orig_child is None:
                continue
            corrected[child_name] = _apply_rigid_child_transform(
                orig_child,
                old_parent_loc=orig_obj.transform.location,
                old_parent_rz=orig_obj.transform.rotation_euler[2],
                new_parent_loc=new_loc,
                new_parent_rz=new_rot[2],
                original_child_z=orig_child.transform.location[2],
            )

    logger.info(
        "LLM output validation: %d roots processed, %d clamped, %d missing.",
        sum(1 for o in original_state.objects if o.is_movable and o.parent is None),
        clamped_count, missing_count,
    )

    # --- Post-LLM collision resolution (MTV + grid fallback) ---

    main_cats  = {"bed", "table", "storage", "seating_large", "seating_small", "furniture"}
    final_list: list[SceneObject] = []

    # Non-movable objects are placed first as fixed obstacles.
    for obj in original_state.objects:
        if not obj.is_movable:
            final_list.append(corrected[obj.name])

    jitter_resolved = 0
    jitter_failed   = 0

    def _group_volume(o: SceneObject) -> float:
        d   = o.transform.dimensions
        vol = d[0] * d[1] * d[2]
        for c in o.children:
            cc = by_name_orig.get(c)
            if cc:
                dd = cc.transform.dimensions
                vol += dd[0] * dd[1] * dd[2]
        return vol

    # Resolve collisions starting from the largest groups (same order as the randomizer)
    # so that a small object placed early never blocks a large piece of furniture.
    movable_roots = [
        o for o in original_state.objects if o.is_movable and o.parent is None
    ]
    movable_roots.sort(key=_group_volume, reverse=True)

    for orig_obj in movable_roots:
        obj = corrected[orig_obj.name]

        if obj.category in main_cats:
            collidable = [
                o for o in final_list
                if o.category in main_cats or not o.is_movable
            ]
            current_children = [corrected[c] for c in orig_obj.children if c in corrected]

            for i in range(60):
                if not _group_has_collision(
                    obj, current_children, collidable,
                    wall_margin=REORDER_WALL_MARGIN, furniture_margin=0.02,
                    room_bounds=room_bounds,
                ):
                    break

                moved = False
                for other in collidable:
                    dx, dy = _group_penetration_vector(obj, current_children, other, margin=0.02)
                    if dx != 0.0 or dy != 0.0:
                        obj.transform.location[0] += dx
                        obj.transform.location[1] += dy

                        # Clamp the group after each MTV step.
                        if room_bounds is not None:
                            clamped = _clamp_parent_group_location(
                                obj,
                                obj.transform.location,
                                obj.transform.rotation_euler[2],
                                current_children,
                                room_bounds,
                            )
                            obj.transform.location = clamped

                        # Update children rigidly.
                        for child_name in orig_obj.children:
                            orig_child = by_name_orig.get(child_name)
                            if orig_child:
                                corrected[child_name] = _apply_rigid_child_transform(
                                    orig_child,
                                    old_parent_loc=orig_obj.transform.location,
                                    old_parent_rz=orig_obj.transform.rotation_euler[2],
                                    new_parent_loc=obj.transform.location,
                                    new_parent_rz=obj.transform.rotation_euler[2],
                                )
                        current_children = [corrected[c] for c in orig_obj.children if c in corrected]
                        moved = True
                        break

                if not moved:
                    break
            else:
                # MTV exhausted: try a grid-based free-position search.
                relocated = False
                if room_bounds is not None:
                    relocated = _find_free_group_position(
                        orig_obj, obj, current_children, collidable,
                        room_bounds, by_name_orig, corrected,
                    )
                if relocated:
                    jitter_resolved += 1
                    current_children = [corrected[c] for c in orig_obj.children if c in corrected]
                    logger.info("Group '%s' relocated to a free position (MTV failed).", obj.name)
                else:
                    jitter_failed += 1
                    logger.warning(
                        "Unresolvable collision for '%s' after 60 iterations (no free position).",
                        obj.name,
                    )

            if not _group_has_collision(
                obj, current_children, collidable,
                wall_margin=REORDER_WALL_MARGIN, furniture_margin=0.02,
                room_bounds=room_bounds,
            ):
                if i > 0:
                    jitter_resolved += 1

        # Final containment guarantee: clamp every group (including decorations/lights)
        # using the full group AABB so that no child can protrude outside the walls.
        if room_bounds is not None:
            current_children = [corrected[c] for c in orig_obj.children if c in corrected]
            clamped = _clamp_parent_group_location(
                obj,
                obj.transform.location,
                obj.transform.rotation_euler[2],
                current_children,
                room_bounds,
            )
            if clamped != obj.transform.location:
                obj.transform.location = clamped
                for child_name in orig_obj.children:
                    oc = by_name_orig.get(child_name)
                    if oc:
                        corrected[child_name] = _apply_rigid_child_transform(
                            oc,
                            old_parent_loc=orig_obj.transform.location,
                            old_parent_rz=orig_obj.transform.rotation_euler[2],
                            new_parent_loc=obj.transform.location,
                            new_parent_rz=obj.transform.rotation_euler[2],
                        )

        final_list.append(obj)
        for child_name in orig_obj.children:
            if child_name in corrected:
                final_list.append(corrected[child_name])

    if jitter_resolved or jitter_failed:
        logger.info(
            "Post-LLM collisions: %d resolved, %d unresolvable.",
            jitter_resolved, jitter_failed,
        )

    return SceneState(
        scene_name=original_state.scene_name,
        objects=final_list,
        room_bounds=original_state.room_bounds,
        pipeline_step="reordered",
        metadata={
            "clamped_count":    clamped_count,
            "missing_count":    missing_count,
            "jitter_resolved":  jitter_resolved,
            "jitter_failed":    jitter_failed,
        },
    )


# ---------------------------------------------------------------------------
# SceneReorganizer
# ---------------------------------------------------------------------------

class SceneReorganizer:
    """Orchestrates the LLM text call for scene reorganization."""

    def __init__(self, client: GeminiClient, prompts_dir: Path) -> None:
        self.client      = client
        self.prompts_dir = prompts_dir
        logger.info("SceneReorganizer initialized.")

    def _build_user_prompt(self, state: SceneState) -> str:
        """
        Builds the user prompt from the template, injecting scene data.

        JSON blocks are substituted via dedicated placeholders (###FLAT_JSON###
        and ###STRUCTURAL_JSON###) rather than .format() to avoid conflicts
        with curly braces inside the JSON strings.
        """
        template    = _load_prompt_template(self.prompts_dir / "reorder_user.txt")
        room_bounds = state.room_bounds
        flat_json   = _build_flat_json_for_llm(state)
        struct_json = _build_structural_json_for_llm(state)
        n_roots     = len(state.root_movable_objects)

        if room_bounds is not None:
            wm        = REORDER_WALL_MARGIN
            safe_xmin = room_bounds.x_min + wm
            safe_xmax = room_bounds.x_max - wm
            safe_ymin = room_bounds.y_min + wm
            safe_ymax = room_bounds.y_max - wm

            prompt = template.format(
                scene_name=state.scene_name,
                x_min=safe_xmin,
                x_max=safe_xmax,
                y_min=safe_ymin,
                y_max=safe_ymax,
                room_width=safe_xmax - safe_xmin,
                room_depth=safe_ymax - safe_ymin,
                room_height=room_bounds.height,
                n_roots=n_roots,
            )
        else:
            prompt = (
                "You are an interior designer. Reorganize this 3D scene following "
                "professional layout principles. Return ONLY the updated JSON array.\n\n"
                f"{flat_json}"
            )
            return prompt

        prompt = prompt.replace("###FLAT_JSON###",       flat_json)
        prompt = prompt.replace("###STRUCTURAL_JSON###", struct_json)
        return prompt

    def reorganize(self, disordered_state: SceneState) -> SceneState:
        """
        Reorganizes the scene via a text-only LLM call.

        Sends a flat JSON of root movable objects to Gemini and applies the
        returned positions after validation and collision resolution.

        Args:
            disordered_state: SceneState after randomization.

        Returns:
            A new SceneState with pipeline_step='reordered', or
            'reordered_failed' if the LLM response could not be parsed.
        """
        logger.info(
            "Starting LLM reorganization for '%s'. Root objects: %d.",
            disordered_state.scene_name,
            len(disordered_state.root_movable_objects),
        )

        system_prompt = _load_prompt_template(self.prompts_dir / "reorder_system.txt")
        user_prompt   = self._build_user_prompt(disordered_state)

        logger.debug(
            "Prompt lengths — system: %d chars, user: %d chars.",
            len(system_prompt), len(user_prompt),
        )

        try:
            llm_output = self.client.call_text(system_prompt, user_prompt)
        except GeminiParsingError as exc:
            logger.error("LLM parsing failed: %s. Returning disordered state.", exc)
            return SceneState(
                scene_name=disordered_state.scene_name,
                objects=copy.deepcopy(disordered_state.objects),
                room_bounds=disordered_state.room_bounds,
                pipeline_step="reordered_failed",
                metadata={"error": str(exc)},
            )

        reordered_state = _validate_and_sanitize_llm_output(llm_output, disordered_state)
        logger.info("LLM reorganization complete for '%s'.", reordered_state.scene_name)
        return reordered_state

    def reorganize_multimodal(
            self,
            disordered_state: SceneState,
            image_paths: list[Path],
    ) -> SceneState:
        """
        Reorganizes the scene via a MULTIMODAL LLM call (reference images + JSON).

        Sends rendered views of the current (disordered) scene together with the
        flat JSON, so the model can see the room shape and the real size of each
        object. Output is validated and sanitized exactly like reorganize().
        """
        logger.info(
            "Starting MULTIMODAL reorganization for '%s'. Root objects: %d, images: %d.",
            disordered_state.scene_name,
            len(disordered_state.root_movable_objects),
            len(image_paths),
        )

        system_prompt = _load_prompt_template(self.prompts_dir / "reorder_system.txt")
        user_prompt = self._build_user_prompt(disordered_state)

        # call_vision_multi has no separate system parameter: merge everything into
        # a single prompt and describe the attached views.
        visual_note = (
            "\n\n--- REFERENCE IMAGES ---\n"
            "You are also given rendered views of the CURRENT (disordered) scene "
            "(a top-down view and an isometric view). Use them to understand the "
            "room shape, the real size of each object, and where things are now. "
            "Then produce the reorganized layout. Return ONLY the updated JSON array, "
            "using the EXACT object names from the JSON above."
        )
        combined_prompt = f"{system_prompt}\n\n{user_prompt}{visual_note}"

        try:
            llm_output = self.client.call_vision_multi(image_paths, combined_prompt)
        except GeminiParsingError as exc:
            logger.error("LLM parsing failed: %s. Returning disordered state.", exc)
            return SceneState(
                scene_name=disordered_state.scene_name,
                objects=copy.deepcopy(disordered_state.objects),
                room_bounds=disordered_state.room_bounds,
                pipeline_step="reordered_failed",
                metadata={"error": str(exc)},
            )

        reordered_state = _validate_and_sanitize_llm_output(llm_output, disordered_state)
        logger.info("Multimodal reorganization complete for '%s'.", reordered_state.scene_name)
        return reordered_state