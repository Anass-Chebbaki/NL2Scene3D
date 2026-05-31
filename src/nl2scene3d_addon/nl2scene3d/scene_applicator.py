# nl2scene3d/scene_applicator.py
"""
Applies a SceneState to the currently open Blender scene.

Core principle:
  Only location and rotation_euler are written — nothing else.
  The Z coordinate is NEVER modified via raycast or surface snapping.
  Z values are the sole responsibility of the original .blend file:
  if an object floats in the original, it will keep that elevation
  throughout the entire pipeline.

Must be executed inside the Blender Python environment.
"""

from __future__ import annotations

import logging
from pathlib import Path

from nl2scene3d.models import SceneState

logger = logging.getLogger(__name__)


class SceneApplicator:
    """
    Applies a SceneState to the scene currently open in Blender.

    Updates positions and rotations of Blender objects to match the values
    in the SceneState, without adding, removing, or dropping any objects.

    Attributes:
        tolerance: Minimum difference required to actually write a change.
                   Changes below this threshold are skipped to avoid noise.
    """

    def __init__(self, tolerance: float = 0.001) -> None:
        self.tolerance = tolerance
        logger.info("SceneApplicator initialized. Tolerance: %.4f.", tolerance)

    def apply_state(self, state: SceneState) -> dict[str, int]:
        """
        Applies the SceneState to the current Blender scene.

        For each object in the SceneState, finds the matching Blender object
        by name and updates its location and rotation_euler.

        Non-movable objects and objects whose delta is below the tolerance
        threshold are skipped. Z is never independently modified: it is
        always taken directly from the SceneState (which inherits the
        original value from the .blend file).

        Args:
            state: SceneState containing the new transforms.

        Returns:
            Counters dict: {'updated': N, 'not_found': N, 'skipped': N}.

        Raises:
            ImportError: If bpy is not available (not running inside Blender).
        """
        try:
            import bpy          # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("The 'bpy' module requires the Blender environment.") from exc

        counters: dict[str, int] = {"updated": 0, "not_found": 0, "skipped": 0}
        blender_scene = bpy.context.scene

        logger.info(
            "Applying state '%s' (step: %s, objects: %d).",
            state.scene_name, state.pipeline_step, len(state.objects),
        )

        # Split objects into roots and children to process parent matrices
        # before their children, avoiding world-space desync.
        roots_to_process:    list[tuple] = []
        children_to_process: list[tuple] = []

        for scene_obj in state.objects:
            blender_obj = blender_scene.objects.get(scene_obj.name)  # type: ignore[union-attr]
            if blender_obj is None:
                logger.warning("Object '%s' not found in Blender. Skipped.", scene_obj.name)
                counters["not_found"] += 1
                continue

            if not scene_obj.is_movable or blender_obj.type in ("CAMERA", "LIGHT"):
                counters["skipped"] += 1
                continue

            if blender_obj.parent is None:
                roots_to_process.append((scene_obj, blender_obj))
            else:
                children_to_process.append((scene_obj, blender_obj))

        def process_object(scene_obj, blender_obj) -> bool:
            """
            Writes location and rotation from the SceneState to the Blender object.

            For objects with a native Blender parent, coordinates are converted
            from world space to local space. Returns True if any value was written.
            """
            t       = scene_obj.transform
            updated = False

            # --- Location ---
            cur_loc = [
                blender_obj.matrix_world.translation.x,
                blender_obj.matrix_world.translation.y,
                blender_obj.matrix_world.translation.z,
            ]
            if any(abs(t.location[i] - cur_loc[i]) > self.tolerance for i in range(3)):
                if blender_obj.parent is not None:
                    try:
                        import mathutils                                  # noqa: PLC0415
                        world_vec = mathutils.Vector(t.location)
                        local_vec = blender_obj.parent.matrix_world.inverted() @ world_vec
                        blender_obj.location.x = local_vec.x
                        blender_obj.location.y = local_vec.y
                        blender_obj.location.z = local_vec.z
                    except Exception:                                     # noqa: BLE001
                        blender_obj.location.x = t.location[0]
                        blender_obj.location.y = t.location[1]
                        blender_obj.location.z = t.location[2]
                else:
                    blender_obj.location.x = t.location[0]
                    blender_obj.location.y = t.location[1]
                    blender_obj.location.z = t.location[2]
                updated = True
                logger.debug("'%s': location %s -> %s.", scene_obj.name, cur_loc, t.location)

            # --- Rotation Euler ---
            cur_rot = [
                blender_obj.matrix_world.to_euler("XYZ").x,
                blender_obj.matrix_world.to_euler("XYZ").y,
                blender_obj.matrix_world.to_euler("XYZ").z,
            ]
            if any(abs(t.rotation_euler[i] - cur_rot[i]) > self.tolerance for i in range(3)):
                blender_obj.rotation_mode = "XYZ"
                if blender_obj.parent is not None:
                    try:
                        import mathutils                                  # noqa: PLC0415
                        world_rot_euler = mathutils.Euler(t.rotation_euler, "XYZ")
                        local_mat       = (
                            blender_obj.parent.matrix_world.to_3x3().inverted()
                            @ world_rot_euler.to_matrix()
                        )
                        local_rot = local_mat.to_euler("XYZ")
                        blender_obj.rotation_euler.x = local_rot.x
                        blender_obj.rotation_euler.y = local_rot.y
                        blender_obj.rotation_euler.z = local_rot.z
                    except Exception:                                     # noqa: BLE001
                        blender_obj.rotation_euler.x = t.rotation_euler[0]
                        blender_obj.rotation_euler.y = t.rotation_euler[1]
                        blender_obj.rotation_euler.z = t.rotation_euler[2]
                else:
                    blender_obj.rotation_euler.x = t.rotation_euler[0]
                    blender_obj.rotation_euler.y = t.rotation_euler[1]
                    blender_obj.rotation_euler.z = t.rotation_euler[2]
                updated = True
                logger.debug("'%s': rotation %s -> %s.", scene_obj.name, cur_rot, t.rotation_euler)

            return updated

        # Pass 1: process all Blender root objects.
        for scene_obj, blender_obj in roots_to_process:
            updated = process_object(scene_obj, blender_obj)
            counters["updated" if updated else "skipped"] += 1

        # Sync world matrices of parents before processing their children.
        try:
            bpy.context.view_layer.update()     # type: ignore[union-attr]
        except Exception as exc:                # noqa: BLE001
            logger.debug("view_layer.update() not available after Pass 1: %s.", exc)

        # Pass 2: process children, updating matrices after each write so that
        # deeper hierarchies (grandchildren) always see the correct parent matrix.
        for scene_obj, blender_obj in children_to_process:
            updated = process_object(scene_obj, blender_obj)
            counters["updated" if updated else "skipped"] += 1
            if updated:
                try:
                    bpy.context.view_layer.update()     # type: ignore[union-attr]
                except Exception:
                    pass

        # Final sync.
        try:
            bpy.context.view_layer.update()     # type: ignore[union-attr]
        except Exception as exc:                # noqa: BLE001
            logger.debug("Final view_layer.update() not available: %s.", exc)

        logger.info(
            "Apply complete: %d updated, %d not found, %d unchanged.",
            counters["updated"], counters["not_found"], counters["skipped"],
        )
        return counters

    def save_blend_file(self, output_path: Path) -> None:
        """Saves the current Blender scene to a .blend file."""
        try:
            import bpy  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("The 'bpy' module requires the Blender environment.") from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
        logger.info("Blender scene saved: %s.", output_path)