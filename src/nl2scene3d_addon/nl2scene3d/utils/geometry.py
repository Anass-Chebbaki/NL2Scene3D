# nl2scene3d/utils/geometry.py
"""
Collision detection for the NL2Scene3D pipeline.

Architecture:
  - Basic geometric computations (AABB, OBB corners, z_range) live on Transform.
  - This module is responsible only for:
      1. has_collision():         binary check between a candidate and a list of objects.
      2. wall_collision():        dedicated check for walls (AABB + Z overlap).
      3. furniture_collision():   SAT check between OBBs for furniture.

Collision margin:
  Each object is expanded by `margin` meters on all sides before the check.
  This guarantees that after placement there is always at least 2 * margin
  of clearance between adjacent objects, preventing visual interpenetration
  even when the LLM or randomizer places them very close together.

  Recommended values:
    - Randomizer:       margin = config.collision_margin  (default 0.05 m = 5 cm)
    - Post-LLM check:   margin = 0.02 m  (more tolerant; the LLM reasons about large furniture)
    - Wall check:       margin = config.wall_margin       (default 0.20 m)
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nl2scene3d.models import SceneObject, RoomBounds

logger = logging.getLogger(__name__)

# Blender object types that do not participate in collision detection.
_IGNORED_TYPES = frozenset({"CAMERA", "LIGHT", "EMPTY", "SPEAKER", "ARMATURE", "CURVE"})


# ---------------------------------------------------------------------------
# Basic geometry utilities
# ---------------------------------------------------------------------------

def is_finite_float(val: Any) -> bool:
    """Return True if the value is a finite float (not NaN or inf)."""
    try:
        f = float(val)
        return math.isfinite(f)
    except (ValueError, TypeError):
        return False


def snap_rotation_90(rz: float) -> float:
    """Snap a Z rotation to the nearest multiple of 90 degrees (0, 90, 180, 270)."""
    multiples = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
    return min(multiples, key=lambda m: abs(m - (rz % (2 * math.pi))))


# ---------------------------------------------------------------------------
# Floor validation
# ---------------------------------------------------------------------------

def object_above_floor(
    obj: "SceneObject",
    z_floor: float = 0.0,
    tolerance: float = 0.05,
) -> bool:
    """
    Return True if the object is correctly positioned at or above the floor.

    Returns False if the object descends more than `tolerance` meters below
    the floor level.
    """
    z_min, _ = obj.transform.z_range()
    return z_min >= (z_floor - tolerance)


# ---------------------------------------------------------------------------
# Wall collision (SAT OBB vs OBB)
# ---------------------------------------------------------------------------

def wall_collision(
    candidate: "SceneObject",
    wall_objects: list["SceneObject"],
    wall_margin: float = 0.20,
) -> bool:
    """
    Check whether the candidate (expanded by margin) penetrates a physical wall.

    Uses SAT OBB (not just AABB) so that furniture rotated at 45 degrees does
    not produce false negatives against thin walls. The wall_margin is added
    around the candidate to keep furniture away from walls.

    Doors, windows, and room meshes are excluded from this check.
    """
    cand_poly = candidate.transform.obb_corners_xy(margin=wall_margin)
    c_z_min, c_z_max = candidate.transform.z_range()

    for wall in wall_objects:
        name_lower = wall.name.lower()
        if any(k in name_lower for k in ("door", "window", "room", "porta", "finestra")):
            continue

        # Quick Z-overlap check before running SAT.
        w_z_min, w_z_max = wall.transform.z_range()
        z_overlap = max(0.0, min(c_z_max, w_z_max) - max(c_z_min, w_z_min))
        if z_overlap <= 0.01:
            continue

        wall_poly = wall.transform.obb_corners_xy(margin=0.0)
        if _sat_overlap(cand_poly, wall_poly):
            logger.debug(
                "Wall collision (SAT): '%s' vs '%s' (Z overlap: %.3f).",
                candidate.name, wall.name, z_overlap,
            )
            return True

    return False


# ---------------------------------------------------------------------------
# Furniture collision (SAT OBB vs OBB)
# ---------------------------------------------------------------------------

def _sat_overlap(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
) -> bool:
    """Separating Axis Theorem for two convex 2D polygons."""

    def get_axes(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
        axes = []
        n = len(poly)
        for i in range(n):
            p1, p2 = poly[i], poly[(i + 1) % n]
            ex, ey = p2[0] - p1[0], p2[1] - p1[1]
            mag = math.hypot(ex, ey)
            if mag > 1e-6:
                axes.append((-ey / mag, ex / mag))
        return axes

    def project(
        poly: list[tuple[float, float]],
        axis: tuple[float, float],
    ) -> tuple[float, float]:
        dots = [p[0] * axis[0] + p[1] * axis[1] for p in poly]
        return min(dots), max(dots)

    for axis in get_axes(poly_a) + get_axes(poly_b):
        mn_a, mx_a = project(poly_a, axis)
        mn_b, mx_b = project(poly_b, axis)
        if mx_a < mn_b or mx_b < mn_a:
            return False  # Separating axis found: no collision.

    return True  # No separating axis found: collision.


def furniture_collision(
    candidate: "SceneObject",
    furniture_objects: list["SceneObject"],
    margin: float = 0.05,
) -> bool:
    """
    Check whether the candidate overlaps another piece of furniture.

    Uses SAT on 2D OBBs to correctly handle rotated objects. The margin
    expands each OBB by `margin` meters to guarantee physical clearance
    between objects after placement.

    The Z threshold is reduced to 0.01 m to also catch nearly coplanar
    objects (e.g. rug vs chair, objects on a table vs table edge).
    """
    cand_poly = candidate.transform.obb_corners_xy(margin=margin)
    cand_z_min, cand_z_max = candidate.transform.z_range()

    for other in furniture_objects:
        if other.name == candidate.name:
            continue

        o_z_min, o_z_max = other.transform.z_range()
        z_overlap = max(0.0, min(cand_z_max, o_z_max) - max(cand_z_min, o_z_min))
        if z_overlap < 0.01:  # 1 cm threshold instead of 2 cm.
            continue

        other_poly = other.transform.obb_corners_xy(margin=margin)
        if _sat_overlap(cand_poly, other_poly):
            logger.debug(
                "SAT collision: '%s' vs '%s'.", candidate.name, other.name
            )
            return True

    return False


def check_openings_clearance(
    candidate: "SceneObject",
    structural_objects: list["SceneObject"],
) -> bool:
    """
    Check whether the candidate invades the clearance zone in front of a door or window.

    Returns True if there is an invasion (collision with the clearance zone).
    """
    for obj in structural_objects:
        name_lower = obj.name.lower()
        is_door   = any(k in name_lower for k in ("door", "porta"))
        is_window = any(k in name_lower for k in ("window", "finestra"))

        if not (is_door or is_window):
            continue

        # Clearance depth on each side:
        #   Door:   0.90 m for passage and swing radius.
        #   Window: 0.50 m for light and access.
        clearance_depth = 0.90 if is_door else 0.50

        cx, cy = obj.transform.geometric_center_xy()
        rz = obj.transform.rotation_euler[2]
        cos_z, sin_z = math.cos(rz), math.sin(rz)
        dim = obj.transform.dimensions

        # Extend local Y (depth) to include clearance on both sides of the panel.
        w = dim[0] / 2.0                    # structural half-width
        h = dim[1] / 2.0 + clearance_depth  # extended half-depth

        local_corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
        clearance_poly = [
            (cx + lx * cos_z - ly * sin_z, cy + lx * sin_z + ly * cos_z)
            for lx, ly in local_corners
        ]

        # Candidate OBB with a minimum 2 cm tolerance margin.
        cand_poly = candidate.transform.obb_corners_xy(margin=0.02)

        if _sat_overlap(clearance_poly, cand_poly):
            c_z_min, c_z_max = candidate.transform.z_range()
            o_z_min, o_z_max = obj.transform.z_range()
            z_overlap = max(0.0, min(c_z_max, o_z_max) - max(c_z_min, o_z_min))

            if is_door:
                # Block if there is real Z overlap between the object and the door.
                # Doors typically span floor to ceiling, so this fires almost always.
                if z_overlap > 0.05:
                    logger.debug(
                        "Door collision: '%s' blocks the passage of '%s'.",
                        candidate.name, obj.name,
                    )
                    return True
            elif is_window:
                # Block if the object rises above the window sill.
                if c_z_max > o_z_min + 0.10 and z_overlap > 0.05:
                    logger.debug(
                        "Window collision: '%s' covers the light of '%s'.",
                        candidate.name, obj.name,
                    )
                    return True

    return False


# ---------------------------------------------------------------------------
# Main collision entry point
# ---------------------------------------------------------------------------

def has_collision(
    candidate: "SceneObject",
    placed_objects: list["SceneObject"],
    wall_margin: float = 0.20,
    furniture_margin: float = 0.05,
    check_walls: bool = True,
    room_bounds: "RoomBounds | None" = None,
) -> bool:
    """
    Check whether the candidate collides with any already-placed object.

    Args:
        candidate:        Object to test.
        placed_objects:   Already placed objects (structural and furniture).
        wall_margin:      Minimum clearance from walls in meters.
        furniture_margin: OBB expansion margin for furniture in meters.
                          With 0.05 m every pair of objects will have >= 10 cm clearance.
        check_walls:      If False, skip wall checks (useful for wall-mounted decorations).
        room_bounds:      Optional RoomBounds. When provided, also checks that the
                          candidate AABB is fully contained within the room bounds.
                          This is the most reliable wall check as it does not depend
                          on the presence of physical wall meshes.

    Returns:
        True if at least one collision is detected.
    """
    # Room bounds containment check (most reliable).
    if check_walls and room_bounds is not None:
        c_aabb = candidate.transform.aabb_xy(margin=0.0)
        if not room_bounds.contains_aabb(c_aabb, margin=wall_margin):
            logger.debug(
                "Out of bounds: '%s' AABB %s not contained within bounds (margin=%.2f).",
                candidate.name, c_aabb, wall_margin,
            )
            return True

    walls: list["SceneObject"]     = []
    furniture: list["SceneObject"] = []

    for obj in placed_objects:
        if obj.name == candidate.name:
            continue
        if obj.object_type in _IGNORED_TYPES:
            continue
        if obj.category == "structural":
            walls.append(obj)
        else:
            furniture.append(obj)

    if check_walls and walls:
        if wall_collision(candidate, walls, wall_margin):
            return True
        if check_openings_clearance(candidate, walls):
            return True

    if furniture:
        if furniture_collision(candidate, furniture, furniture_margin):
            return True

    return False


# ---------------------------------------------------------------------------
# Scoring and solver utilities
# ---------------------------------------------------------------------------

def collision_score(
    candidate: "SceneObject",
    placed_objects: list["SceneObject"],
    wall_margin: float = 0.20,
    furniture_margin: float = 0.05,
    check_walls: bool = True,
    room_bounds: "RoomBounds | None" = None,
) -> float:
    """
    Return a "badness" score for the candidate's current position.

    0.0   = no collision (perfect placement).
    > 0.0 = overlap present; higher values indicate worse placement.

    Useful in the randomizer to choose the least problematic position when
    a completely collision-free placement cannot be found within max_attempts.

    If room_bounds is provided, a penalty proportional to the out-of-bounds
    overflow is added.
    """
    # Return 0.0 immediately if the exact SAT algorithm detects no collision,
    # preventing false positives from AABB approximations on rotated objects.
    if not has_collision(
        candidate, placed_objects, wall_margin, furniture_margin, check_walls, room_bounds
    ):
        return 0.0

    total = 0.0

    c_aabb_base = candidate.transform.aabb_xy(margin=0.0)
    c_aabb_wall = candidate.transform.aabb_xy(margin=wall_margin)
    c_aabb_furn = candidate.transform.aabb_xy(margin=furniture_margin)
    c_z_min, c_z_max = candidate.transform.z_range()

    # Penalty for objects outside the room bounds.
    if check_walls and room_bounds is not None:
        if not room_bounds.contains_aabb(c_aabb_base, margin=wall_margin):
            overflow = (
                max(0.0, room_bounds.x_min + wall_margin - c_aabb_base[0])
                + max(0.0, c_aabb_base[1] - (room_bounds.x_max - wall_margin))
                + max(0.0, room_bounds.y_min + wall_margin - c_aabb_base[2])
                + max(0.0, c_aabb_base[3] - (room_bounds.y_max - wall_margin))
            )
            total += 100.0 + overflow * 10.0  # Proportional blocking penalty.

    for obj in placed_objects:
        if obj.name == candidate.name:
            continue
        if obj.object_type in _IGNORED_TYPES:
            continue

        o_aabb = obj.transform.aabb_xy(margin=0.0)
        o_z_min, o_z_max = obj.transform.z_range()

        z_overlap = max(0.0, min(c_z_max, o_z_max) - max(c_z_min, o_z_min))
        if z_overlap < 0.02:
            continue

        if obj.category == "structural" and check_walls:
            name_lower = obj.name.lower()
            if any(k in name_lower for k in ("door", "window", "room", "porta", "finestra")):
                is_door   = any(k in name_lower for k in ("door", "porta"))
                is_window = any(k in name_lower for k in ("window", "finestra"))
                clearance_depth = 0.90 if is_door else 0.50

                cx, cy = obj.transform.geometric_center_xy()
                rz = obj.transform.rotation_euler[2]
                cos_z, sin_z = math.cos(rz), math.sin(rz)
                dim = obj.transform.dimensions
                w = dim[0] / 2.0
                h = dim[1] / 2.0 + clearance_depth

                local_corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
                clearance_poly = [
                    (cx + lx * cos_z - ly * sin_z, cy + lx * sin_z + ly * cos_z)
                    for lx, ly in local_corners
                ]
                cand_poly = candidate.transform.obb_corners_xy(margin=0.02)

                if _sat_overlap(clearance_poly, cand_poly):
                    real_z_overlap = max(
                        0.0, min(c_z_max, o_z_max) - max(c_z_min, o_z_min)
                    )
                    if is_door and real_z_overlap > 0.05:
                        total += 50.0  # Heavy blocking penalty for doors.
                    elif is_window and c_z_max > o_z_min + 0.10 and real_z_overlap > 0.05:
                        total += 25.0  # Lighter penalty for windows.
                continue

            ratio = aabb_overlap_ratio(c_aabb_wall, o_aabb)
            total += ratio * 2.0  # Walls carry double weight.
        else:
            ratio = aabb_overlap_ratio(c_aabb_furn, o_aabb)
            total += ratio

    return total


def penetration_vector(
    candidate: "SceneObject",
    other: "SceneObject",
    margin: float = 0.05,
) -> tuple[float, float]:
    """
    Compute the Minimum Translation Vector (MTV) between two objects in the XY plane.

    Returns (dx, dy) to apply to the candidate to resolve the overlap.
    Returns (0.0, 0.0) if there is no overlap.

    Useful in the post-LLM solver to displace overlapping objects intelligently
    instead of using random jitter.
    """
    c_cx, c_cy = candidate.transform.geometric_center_xy()
    o_cx, o_cy = other.transform.geometric_center_xy()

    c_aabb = candidate.transform.aabb_xy(margin=margin)
    o_aabb = other.transform.aabb_xy(margin=0.0)

    x_overlap = min(c_aabb[1], o_aabb[1]) - max(c_aabb[0], o_aabb[0])
    y_overlap = min(c_aabb[3], o_aabb[3]) - max(c_aabb[2], o_aabb[2])

    if x_overlap <= 0 or y_overlap <= 0:
        return 0.0, 0.0  # No overlap.

    # Push along the axis with the smaller penetration (standard MTV).
    if x_overlap < y_overlap:
        dx = x_overlap + 0.01  # +1 cm buffer.
        return (dx if c_cx > o_cx else -dx), 0.0
    else:
        dy = y_overlap + 0.01
        return 0.0, (dy if c_cy > o_cy else -dy)


def aabb_overlap_ratio(
    aabb_a: tuple[float, float, float, float],
    aabb_b: tuple[float, float, float, float],
) -> float:
    """
    Compute the overlap ratio between two 2D AABBs.

    Returns:
        Value in [0.0, 1.0]. 0.0 means no overlap.
        Uses the smaller area as the denominator to protect small objects.
    """
    x_overlap = max(0.0, min(aabb_a[1], aabb_b[1]) - max(aabb_a[0], aabb_b[0]))
    y_overlap = max(0.0, min(aabb_a[3], aabb_b[3]) - max(aabb_a[2], aabb_b[2]))
    intersection = x_overlap * y_overlap

    if intersection < 1e-6:
        return 0.0

    area_a = (aabb_a[1] - aabb_a[0]) * (aabb_a[3] - aabb_a[2])
    area_b = (aabb_b[1] - aabb_b[0]) * (aabb_b[3] - aabb_b[2])
    min_area = min(area_a, area_b)

    if min_area < 1e-6:
        return 0.0

    return min(intersection / min_area, 1.0)


# ---------------------------------------------------------------------------
# Shared group geometry helper
# ---------------------------------------------------------------------------

def group_aabb_xy(
    orig_parent:   "SceneObject",
    proposed_loc:  list[float],
    proposed_rz:   float,
    orig_children: list["SceneObject"],
    margin:        float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Computes the combined XY AABB for a parent+children group at a proposed position.

    Uses each member's actual AABB (including rotation and origin offset) via
    the Transform class, so the result is always geometrically accurate.

    Previously duplicated in randomizer.py and scene_reorganizer.py.
    """
    from nl2scene3d.models import Transform  # local import to avoid circular dependency

    old_parent_loc = orig_parent.transform.location
    old_parent_rz  = orig_parent.transform.rotation_euler[2]
    d_rz           = proposed_rz - old_parent_rz
    cos_a, sin_a   = math.cos(d_rz), math.sin(d_rz)

    temp_parent_tf = Transform(
        location=[proposed_loc[0], proposed_loc[1], orig_parent.transform.location[2]],
        rotation_euler=[
            orig_parent.transform.rotation_euler[0],
            orig_parent.transform.rotation_euler[1],
            proposed_rz,
        ],
        dimensions=orig_parent.transform.dimensions,
        origin_offset=orig_parent.transform.origin_offset,
    )
    x_min, x_max, y_min, y_max = temp_parent_tf.aabb_xy(margin=margin)

    for orig_child in orig_children:
        rel_x  = orig_child.transform.location[0] - old_parent_loc[0]
        rel_y  = orig_child.transform.location[1] - old_parent_loc[1]
        new_cx = proposed_loc[0] + rel_x * cos_a - rel_y * sin_a
        new_cy = proposed_loc[1] + rel_x * sin_a + rel_y * cos_a
        c_rz   = (orig_child.transform.rotation_euler[2] + d_rz) % (2 * math.pi)

        temp_child_tf = Transform(
            location=[new_cx, new_cy, orig_child.transform.location[2]],
            rotation_euler=[
                orig_child.transform.rotation_euler[0],
                orig_child.transform.rotation_euler[1],
                c_rz,
            ],
            dimensions=orig_child.transform.dimensions,
            origin_offset=orig_child.transform.origin_offset,
        )
        cx_min, cx_max, cy_min, cy_max = temp_child_tf.aabb_xy(margin=margin)
        x_min = min(x_min, cx_min)
        x_max = max(x_max, cx_max)
        y_min = min(y_min, cy_min)
        y_max = max(y_max, cy_max)

    return x_min, x_max, y_min, y_max