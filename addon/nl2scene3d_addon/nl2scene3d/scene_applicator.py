# nl2scene3d/scene_applicator.py
"""
Applicazione delle trasformazioni a una scena Blender.

Principio fondamentale:
    Questo modulo applica SOLO location e rotation_euler — nient'altro.
    La coordinata Z NON viene mai modificata tramite raycast o surface snapping.
    La Z è responsabilità esclusiva del file .blend originale: se un oggetto
    fluttua nell'originale, rimane a quella quota in tutta la pipeline.

Deve essere eseguito all'interno dell'ambiente Python di Blender.
"""
from __future__ import annotations

import logging
from pathlib import Path

from nl2scene3d.models import SceneState

logger = logging.getLogger(__name__)


class SceneApplicator:
    """
    Applica uno SceneState alla scena attualmente aperta in Blender.

    Aggiorna posizioni e rotazioni degli oggetti Blender in base ai valori
    contenuti nello SceneState, senza aggiungere, rimuovere o far cadere oggetti.

    Attributes:
        tolerance: Soglia di differenza minima per applicare una modifica.
    """

    def __init__(self, tolerance: float = 0.001) -> None:
        self.tolerance = tolerance
        logger.info("SceneApplicator inizializzato. Tolerance: %.4f.", tolerance)

    def apply_state(self, state: SceneState) -> dict[str, int]:
        """
        Applica lo SceneState alla scena Blender corrente.

        Per ogni oggetto nello SceneState, cerca l'oggetto corrispondente
        in Blender per nome e ne aggiorna location e rotation_euler.
        Gli oggetti non movibili e quelli sotto la tolerance vengono saltati.

        La Z non viene mai modificata: rimane sempre quella dello SceneState
        (che a sua volta eredita la Z originale del file .blend).

        Args:
            state: SceneState con le nuove trasformazioni.

        Returns:
            {'updated': N, 'not_found': N, 'skipped': N}

        Raises:
            ImportError: Se bpy non e' disponibile.
        """
        try:
            import bpy  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Il modulo 'bpy' richiede l'ambiente Blender.") from exc

        counters: dict[str, int] = {"updated": 0, "not_found": 0, "skipped": 0}
        blender_scene = bpy.context.scene

        logger.info(
            "Applicazione stato '%s' (step: %s, oggetti: %d).",
            state.scene_name, state.pipeline_step, len(state.objects),
        )

        # Raggruppa gli oggetti in base alla presenza di un parent nativo in Blender
        # per evitare sfasamenti temporali durante l'aggiornamento delle matrici
        roots_to_process = []
        children_to_process = []

        for scene_obj in state.objects:
            blender_obj = blender_scene.objects.get(scene_obj.name)  # type: ignore[union-attr]

            if blender_obj is None:
                logger.warning("Oggetto '%s' non trovato in Blender. Ignorato.", scene_obj.name)
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
            t = scene_obj.transform
            updated = False

            # --- Location ---
            cur_loc = [
                blender_obj.matrix_world.translation.x,
                blender_obj.matrix_world.translation.y,
                blender_obj.matrix_world.translation.z,
            ]
            if any(abs(t.location[i] - cur_loc[i]) > self.tolerance for i in range(3)):
                if blender_obj.parent is not None:
                    # Oggetto con parent nativo: converti world → local
                    try:
                        import mathutils  # noqa: PLC0415
                        world_vec = mathutils.Vector(t.location)
                        local_vec = blender_obj.parent.matrix_world.inverted() @ world_vec
                        blender_obj.location.x = local_vec.x
                        blender_obj.location.y = local_vec.y
                        blender_obj.location.z = local_vec.z
                    except Exception:  # noqa: BLE001
                        blender_obj.location.x = t.location[0]
                        blender_obj.location.y = t.location[1]
                        blender_obj.location.z = t.location[2]
                else:
                    blender_obj.location.x = t.location[0]
                    blender_obj.location.y = t.location[1]
                    blender_obj.location.z = t.location[2]
                updated = True
                logger.debug("'%s': location %s → %s.", scene_obj.name, cur_loc, t.location)

            # --- Rotation Euler ---
            cur_rot = [
                blender_obj.matrix_world.to_euler('XYZ').x,
                blender_obj.matrix_world.to_euler('XYZ').y,
                blender_obj.matrix_world.to_euler('XYZ').z,
            ]
            if any(abs(t.rotation_euler[i] - cur_rot[i]) > self.tolerance for i in range(3)):
                blender_obj.rotation_mode = "XYZ"
                if blender_obj.parent is not None:
                    # Oggetto con parent nativo: converti world_rot → local_rot
                    try:
                        import mathutils  # noqa: PLC0415
                        world_rot_euler = mathutils.Euler(t.rotation_euler, 'XYZ')
                        parent_matrix = blender_obj.parent.matrix_world
                        local_mat = parent_matrix.to_3x3().inverted() @ world_rot_euler.to_matrix()
                        local_rot_euler = local_mat.to_euler('XYZ')
                        blender_obj.rotation_euler.x = local_rot_euler.x
                        blender_obj.rotation_euler.y = local_rot_euler.y
                        blender_obj.rotation_euler.z = local_rot_euler.z
                    except Exception:  # noqa: BLE001
                        blender_obj.rotation_euler.x = t.rotation_euler[0]
                        blender_obj.rotation_euler.y = t.rotation_euler[1]
                        blender_obj.rotation_euler.z = t.rotation_euler[2]
                else:
                    blender_obj.rotation_euler.x = t.rotation_euler[0]
                    blender_obj.rotation_euler.y = t.rotation_euler[1]
                    blender_obj.rotation_euler.z = t.rotation_euler[2]
                updated = True
                logger.debug("'%s': rotation %s → %s.", scene_obj.name, cur_rot, t.rotation_euler)

            return updated

        # Pass 1: Processa tutti i root nativi di Blender
        for scene_obj, blender_obj in roots_to_process:
            updated = process_object(scene_obj, blender_obj)
            counters["updated" if updated else "skipped"] += 1

        # Aggiorna la view layer per sincronizzare le matrici del mondo dei parent prima dei figli
        try:
            bpy.context.view_layer.update()  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.debug("view_layer.update() non applicabile in Pass 1: %s", exc)

        # Pass 2: Processa tutti i figli nativi di Blender (che ora possono calcolare correttamente la posizione/rotazione locale)
        for scene_obj, blender_obj in children_to_process:
            updated = process_object(scene_obj, blender_obj)
            counters["updated" if updated else "skipped"] += 1

        # Aggiornamento finale
        try:
            bpy.context.view_layer.update()  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.debug("view_layer.update() finale non applicabile: %s", exc)

        logger.info(
            "Applicazione completata: %d aggiornati, %d non trovati, %d invariati.",
            counters["updated"], counters["not_found"], counters["skipped"],
        )
        return counters

    def save_blend_file(self, output_path: Path) -> None:
        """Salva la scena Blender corrente in un file .blend."""
        try:
            import bpy  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Il modulo 'bpy' richiede l'ambiente Blender.") from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
        logger.info("Scena Blender salvata: %s", output_path)