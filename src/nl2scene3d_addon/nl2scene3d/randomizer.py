# nl2scene3d/randomizer.py
"""
Controlled randomization of a 3D scene layout.

Goal: disorganize the scene in a plausible way — objects out of place but
not overlapping, not outside the room walls, and with original Z values intact.

Core design rules:
  - The Z coordinate is NEVER modified. An object originally at 0.80 m
    (e.g. on a desk) stays at 0.80 m after randomization. Physics-based
    drop-to-floor is intentionally excluded: it would place shelves and
    monitors on the floor.
  - Children move with their parent via a rigid XY transform, preserving
    their relative positions.
  - Each object (or parent+children group) is treated as a blob with an
    AABB expanded by collision_margin, so visually objects never overlap
    even when a collision is accepted.
  - Objects are sorted by volume (descending): large pieces (beds, wardrobes)
    are placed first; smaller ones fill the remaining space.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Optional

from nl2scene3d.config import RandomizerConfig
from nl2scene3d.models import RoomBounds, SceneObject, SceneState, Transform
from nl2scene3d.utils.geometry import collision_score, group_aabb_xy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rigid group transform
# ---------------------------------------------------------------------------

def apply_rigid_transform(
    child:          SceneObject,
    old_parent_loc: list[float],
    old_parent_rz:  float,
    new_parent_loc: list[float],
    new_parent_rz:  float,
    original_z:     float | None = None,
) -> None:
    """
    Moves and rotates a child object rigidly relative to its parent.

    Only the XY plane is transformed. The Z coordinate is never modified:
    it is set to `original_z` if provided, otherwise left unchanged.
    Modifies `child.transform` in-place.
    """
    rel_x = child.transform.location[0] - old_parent_loc[0]
    rel_y = child.transform.location[1] - old_parent_loc[1]

    d_rz         = new_parent_rz - old_parent_rz
    cos_a, sin_a = math.cos(d_rz), math.sin(d_rz)

    child.transform.location[0] = new_parent_loc[0] + rel_x * cos_a - rel_y * sin_a
    child.transform.location[1] = new_parent_loc[1] + rel_x * sin_a + rel_y * cos_a

    if original_z is not None:
        child.transform.location[2] = original_z

    child.transform.rotation_euler[2] = (
        child.transform.rotation_euler[2] + d_rz
    ) % (2 * math.pi)



def _clamp_parent_group_location_rand(
    orig_parent:   SceneObject,
    proposed_loc:  list[float],
    proposed_rz:   float,
    orig_children: list[SceneObject],
    room_bounds:   RoomBounds,
    wall_margin:   float,
) -> list[float]:
    """
    Clamps the parent position so that both the parent and all children remain
    inside the room bounds with the given wall margin.

    Uses the combined group AABB to avoid errors from double-rotation.
    """
    g_x_min, g_x_max, g_y_min, g_y_max = group_aabb_xy(
        orig_parent, proposed_loc, proposed_rz, orig_children, margin=0.0
    )

    px, py = proposed_loc[0], proposed_loc[1]

    overflow_left  = max(0.0, (room_bounds.x_min + wall_margin) - g_x_min)
    overflow_right = max(0.0, g_x_max - (room_bounds.x_max - wall_margin))
    overflow_front = max(0.0, (room_bounds.y_min + wall_margin) - g_y_min)
    overflow_back  = max(0.0, g_y_max - (room_bounds.y_max - wall_margin))

    # Apply only the dominant overflow direction per axis.
    dx = overflow_left if overflow_left > overflow_right else -overflow_right
    dy = overflow_front if overflow_front > overflow_back else -overflow_back

    return [px + dx, py + dy, proposed_loc[2]]


# ---------------------------------------------------------------------------
# Random position / rotation generators
# ---------------------------------------------------------------------------

def _random_location(
    original_location: list[float],
    dimensions:        list[float],
    origin_offset:     list[float],
    rotation_z:        float,
    room_bounds:       RoomBounds,
    jitter_ratio:      float,
    wall_margin:       float,
    rng:               random.Random,
) -> list[float]:
    """
    Generates a valid XY position inside the room bounds, accounting for
    the rotated AABB and origin offset. Z is always kept at its original value.
    """
    # Compute the AABB of the object placed at the origin to get the exact
    # safe coordinate range after rotation.
    temp_tf = Transform(
        location=[0.0, 0.0, original_location[2]],
        rotation_euler=[0.0, 0.0, rotation_z],
        dimensions=dimensions,
        origin_offset=origin_offset,
    )
    t_xmin, t_xmax, t_ymin, t_ymax = temp_tf.aabb_xy(margin=0.0)

    safe_x_min = room_bounds.x_min + wall_margin - t_xmin
    safe_x_max = room_bounds.x_max - wall_margin - t_xmax
    safe_y_min = room_bounds.y_min + wall_margin - t_ymin
    safe_y_max = room_bounds.y_max - wall_margin - t_ymax

    if safe_x_max <= safe_x_min or safe_y_max <= safe_y_min:
        # Object is too large for the room — leave it in place.
        return list(original_location)

    # Restrict the valid range to a jitter window around the original position.
    jitter_x = room_bounds.width * jitter_ratio
    jitter_y = room_bounds.depth * jitter_ratio
    cx_orig  = original_location[0]
    cy_orig  = original_location[1]

    range_x_min = max(safe_x_min, cx_orig - jitter_x)
    range_x_max = min(safe_x_max, cx_orig + jitter_x)
    range_y_min = max(safe_y_min, cy_orig - jitter_y)
    range_y_max = min(safe_y_max, cy_orig + jitter_y)

    # Fall back to the full safe range if jitter collapsed it.
    if range_x_max < range_x_min:
        range_x_min, range_x_max = safe_x_min, safe_x_max
    if range_y_max < range_y_min:
        range_y_min, range_y_max = safe_y_min, safe_y_max

    new_x = rng.uniform(range_x_min, range_x_max)
    new_y = rng.uniform(range_y_min, range_y_max)
    return [new_x, new_y, original_location[2]]


def _random_rotation(original_rz: float, rng: random.Random) -> float:
    """Returns the original Z rotation incremented by a random multiple of 90 degrees."""
    delta = rng.choice([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
    return (original_rz + delta) % (2 * math.pi)


# ---------------------------------------------------------------------------
# SceneRandomizer
# ---------------------------------------------------------------------------

class SceneRandomizer:
    """
    Artificially disorganizes a 3D scene layout.

    Algorithm:
      1. Collect all root movable objects (those without a parent).
      2. Sort them by volume descending (large pieces are placed first).
      3. For each root, attempt up to max_placement_attempts random positions:
           a. Generate a random Z rotation (multiple of 90 deg).
           b. Generate a valid position consistent with that rotation and the
              room bounds.
           c. Check for wall and furniture collisions using expanded AABBs.
           d. If no collision-free position is found, use the one with the
              lowest collision score as a fallback.
      4. Move children with a rigid transform relative to their parent.
    """

    def __init__(self, config: Optional[RandomizerConfig] = None) -> None:
        self.config = config or RandomizerConfig()
        seed        = self.config.seed if self.config.seed != 0 else None
        self._rng   = random.Random(seed)
        logger.info(
            "SceneRandomizer initialized. seed=%s, jitter=%.2f, "
            "wall_margin=%.2f, collision_margin=%.2f.",
            seed,
            self.config.jitter_ratio,
            self.config.wall_margin,
            self.config.collision_margin,
        )

    def randomize(self, state: SceneState) -> SceneState:
        """
        Applies randomization to a deep copy of the scene.

        The original state is never modified.

        Args:
            state: SceneState with pipeline_step='original' and pre-computed grouping.

        Returns:
            A new SceneState with pipeline_step='randomized'.
        """
        if state.room_bounds is None:
            raise ValueError(
                "SceneState has no room_bounds. Extract the scene with SceneLoader first."
            )

        bounds = state.room_bounds
        logger.info(
            "Randomizing '%s': %d movable objects (%d roots).",
            state.scene_name,
            len(state.movable_objects),
            len(state.root_movable_objects),
        )

        new_objects: list[SceneObject] = [obj.copy() for obj in state.objects]
        by_name = {obj.name: obj for obj in new_objects}

        # Static objects act as fixed obstacles from the start.
        placed: list[SceneObject] = [obj for obj in new_objects if not obj.is_movable]

        # Sort movable roots by volume descending so large objects are placed first.
        roots = sorted(
            [obj for obj in new_objects if obj.is_movable and obj.is_root],
            key=lambda o: (
                o.transform.dimensions[0]
                * o.transform.dimensions[1]
                * o.transform.dimensions[2]
            ),
            reverse=True,
        )

        placed_count   = 0
        fallback_count = 0

        for root in roots:
            old_loc = list(root.transform.location)
            old_rz  = root.transform.rotation_euler[2]

            # Start with the original position as a safe baseline.
            best_obj   = root.copy()
            best_score = self._group_collision_score(
                root_candidate=best_obj,
                original_root=root,
                all_objects_by_name=by_name,
                placed_objects=placed,
                bounds=bounds,
            )

            for _ in range(self.config.max_placement_attempts):
                candidate = root.copy()

                new_rz = _random_rotation(old_rz, self._rng)
                candidate.transform.rotation_euler[2] = new_rz

                new_loc = _random_location(
                    old_loc,
                    candidate.transform.dimensions,
                    candidate.transform.origin_offset,
                    new_rz,
                    bounds,
                    self.config.jitter_ratio,
                    self.config.wall_margin,
                    self._rng,
                )
                candidate.transform.location = new_loc

                score = self._group_collision_score(
                    root_candidate=candidate,
                    original_root=root,
                    all_objects_by_name=by_name,
                    placed_objects=placed,
                    bounds=bounds,
                )

                if score == 0.0:
                    best_obj   = candidate
                    best_score = 0.0
                    break

                if score < best_score:
                    best_score = score
                    best_obj   = candidate

            assert best_obj is not None

            if best_score > 0.0:
                fallback_count += 1
                logger.debug(
                    "'%s': no collision-free position found after %d attempts. "
                    "Using best fallback (score=%.3f).",
                    root.name,
                    self.config.max_placement_attempts,
                    best_score,
                )

            final_root = by_name[root.name]
            new_loc    = list(best_obj.transform.location)
            new_rz     = best_obj.transform.rotation_euler[2]

            # Clamp the entire group to the room bounds.
            orig_children: list[SceneObject] = []
            orig_by_name  = {o.name: o for o in state.objects}

            def gather_children(p_name: str):
                p_obj = orig_by_name.get(p_name)
                if p_obj:
                    for c_name in p_obj.children:
                        c_obj = orig_by_name.get(c_name)
                        if c_obj:
                            orig_children.append(c_obj)
                            gather_children(c_name)

            gather_children(root.name)

            clamped_loc = _clamp_parent_group_location_rand(
                root, new_loc, new_rz, orig_children, bounds, self.config.wall_margin
            )
            if clamped_loc != new_loc:
                logger.debug(
                    "Randomizer group clamp for '%s': %s -> %s.",
                    root.name, new_loc, clamped_loc,
                )
                new_loc = clamped_loc

            # Absolute Z lock: always use the original value from the scene.
            new_loc[2] = root.transform.location[2]

            final_root.transform.location           = new_loc
            final_root.transform.rotation_euler[2]  = new_rz
            placed.append(final_root)
            placed_count += 1

            self._move_children(
                root_name=root.name,
                old_loc=old_loc,
                old_rz=old_rz,
                new_loc=new_loc,
                new_rz=new_rz,
                by_name=by_name,
                placed=placed,
                bounds=bounds,
                orig_by_name=orig_by_name,
            )

        logger.info(
            "Randomization complete: %d roots placed, %d with fallback position.",
            placed_count, fallback_count,
        )

        return SceneState(
            scene_name=state.scene_name,
            objects=new_objects,
            room_bounds=bounds,
            pipeline_step="randomized",
            metadata={
                "randomizer_seed": self.config.seed,
                "placed_count":    placed_count,
                "fallback_count":  fallback_count,
            },
        )

    def _group_collision_score(
        self,
        root_candidate:       SceneObject,
        original_root:        SceneObject,
        all_objects_by_name:  dict[str, SceneObject],
        placed_objects:       list[SceneObject],
        bounds:               RoomBounds,
    ) -> float:
        """
        Computes the cumulative collision score for the entire group
        (parent + all descendants) at the candidate position.

        Only spatial collisions (SAT/AABB vs walls, furniture, room bounds) are
        evaluated. No floor check is performed: Z is locked by apply_rigid_transform,
        and calling object_above_floor here would penalise objects whose pivot sits
        slightly below z=0 (e.g. a shelf at z=0.8 whose AABB base dips to -0.01 due
        to origin offset), making score==0 structurally impossible and freezing every
        object in place after the first randomization.
        """
        total_score = collision_score(
            root_candidate,
            placed_objects,
            wall_margin=self.config.wall_margin,
            furniture_margin=self.config.collision_margin,
            room_bounds=bounds,
        )

        def _collect_and_score_children(
            parent_name:        str,
            current_parent_loc: list[float],
            current_parent_rz:  float,
        ):
            nonlocal total_score
            orig_parent = all_objects_by_name[parent_name]
            for child_name in orig_parent.children:
                orig_child  = all_objects_by_name[child_name]
                moved_child = orig_child.copy()
                apply_rigid_transform(
                    moved_child,
                    old_parent_loc=orig_parent.transform.location,
                    old_parent_rz=orig_parent.transform.rotation_euler[2],
                    new_parent_loc=current_parent_loc,
                    new_parent_rz=current_parent_rz,
                    original_z=orig_child.transform.location[2],
                )
                total_score += collision_score(
                    moved_child,
                    placed_objects,
                    wall_margin=self.config.wall_margin,
                    furniture_margin=self.config.collision_margin,
                    room_bounds=bounds,
                )
                if bounds is not None and not bounds.contains_aabb(
                    moved_child.transform.aabb_xy(margin=0.0),
                    margin=self.config.wall_margin,
                ):
                    total_score += 100.0

                # Note: no floor check here; Z is already locked at the original value.

                _collect_and_score_children(
                    child_name,
                    moved_child.transform.location,
                    moved_child.transform.rotation_euler[2],
                )

        _collect_and_score_children(
            original_root.name,
            root_candidate.transform.location,
            root_candidate.transform.rotation_euler[2],
        )
        return total_score

    def _move_children(
        self,
        root_name:    str,
        old_loc:      list[float],
        old_rz:       float,
        new_loc:      list[float],
        new_rz:       float,
        by_name:      dict[str, SceneObject],
        placed:       list[SceneObject],
        bounds:       RoomBounds,
        orig_by_name: dict[str, SceneObject] | None = None,
    ) -> None:
        """
        Recursively moves all descendants of the root with a rigid XY transform.

        Each child's Z is restored to its original value from the pre-randomization
        scene to guarantee that no child ever changes elevation.
        """
        root_obj = by_name.get(root_name)
        if root_obj is None:
            return

        for child_name in root_obj.children:
            child = by_name.get(child_name)
            if child is None:
                continue

            original_child_z = None
            orig_child_loc   = old_loc
            orig_child_rz    = old_rz

            if orig_by_name:
                orig_child = orig_by_name.get(child_name)
                if orig_child:
                    original_child_z = orig_child.transform.location[2]
                    orig_child_loc   = list(orig_child.transform.location)
                    orig_child_rz    = orig_child.transform.rotation_euler[2]

            apply_rigid_transform(
                child,
                old_loc, old_rz,
                new_loc, new_rz,
                original_z=original_child_z,
            )
            placed.append(child)

            # Recurse using the child's original position as the old reference,
            # not the just-computed one, to avoid compounding errors.
            self._move_children(
                child_name,
                orig_child_loc,
                orig_child_rz,
                list(child.transform.location),
                child.transform.rotation_euler[2],
                by_name, placed, bounds, orig_by_name,
            )