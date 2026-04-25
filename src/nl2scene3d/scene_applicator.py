"""
Applicazione delle trasformazioni suggerite dall'LLM alla scena Blender.

Questo modulo:
- Legge il JSON dello stato riordinato/raffinato
- Aggiorna le proprieta' location e rotation_euler degli oggetti in Blender
- Garantisce che le modifiche vengano applicate correttamente prima del rendering

Deve essere eseguito all'interno dell'ambiente Python di Blender.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from nl2scene3d.models import SceneState

logger = logging.getLogger(__name__)

class SceneApplicator:
    """
    Applica uno SceneState alla scena attualmente aperta in Blender.

    Aggiorna posizioni e rotazioni degli oggetti Blender in base ai valori
    contenuti nello SceneState, senza aggiungere o rimuovere oggetti.

    Attributes:
        tolerance: Soglia di differenza minima per applicare una modifica.
                   Evita aggiornamenti inutili per valori quasi identici.
    """

    def __init__(self, tolerance: float = 0.001) -> None:
        """
        Inizializza l'applicatore.

        Args:
            tolerance: Soglia minima di variazione per aggiornare una proprieta'.
        """
        self.tolerance = tolerance
        logger.info(
            "SceneApplicator inizializzato. Tolerance: %.4f", tolerance
        )

    def apply_state(self, state: SceneState) -> dict[str, int]:
        """
        Applica lo SceneState alla scena Blender corrente.

        Per ogni oggetto nello SceneState, cerca l'oggetto corrispondente
        in Blender per nome e ne aggiorna location e rotation_euler.

        Args:
            state: SceneState con le nuove trasformazioni da applicare.

        Returns:
            Dizionario con contatori: 'updated', 'not_found', 'skipped'.

        Raises:
            ImportError: Se bpy non e' disponibile.
        """
        try:
            import bpy  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Il modulo 'bpy' richiede l'ambiente Blender."
            ) from exc

        counters = {"updated": 0, "not_found": 0, "skipped": 0}
        blender_scene = bpy.context.scene

        logger.info(
            "Applicazione stato '%s' alla scena Blender. ",
            state.scene_name,
            state.pipeline_step,
            len(state.objects),
        )

        for scene_obj in state.objects:
            # Cerca l'oggetto Blender per nome
            blender_obj = blender_scene.objects.get(scene_obj.name)

            if blender_obj is None:
                logger.warning(
                    "Oggetto '%s' non trovato nella scena Blender. Ignorato.",
                    scene_obj.name,
                )
                counters["not_found"] += 1
                continue

            # Gli oggetti non movibili non vengono modificati
            if not scene_obj.is_movable:
                counters["skipped"] += 1
                continue

            transform = scene_obj.transform
            updated = False

            # Aggiorna la posizione se la differenza supera la tolerance
            current_loc = [
                blender_obj.location.x,
                blender_obj.location.y,
                blender_obj.location.z,
            ]
            new_loc = transform.location

            if any(
                abs(new_loc[i] - current_loc[i]) > self.tolerance
                for i in range(3)
            ):
                blender_obj.location.x = new_loc[0]
                blender_obj.location.y = new_loc[1]
                blender_obj.location.z = new_loc[2]
                updated = True
                logger.debug(
                    "Oggetto '%s': location aggiornata da %s a %s",
                    scene_obj.name,
                    current_loc,
                    new_loc,
                )

            # Aggiorna la rotazione se la differenza supera la tolerance
            current_rot = [
                blender_obj.rotation_euler.x,
                blender_obj.rotation_euler.y,
                blender_obj.rotation_euler.z,
            ]
            new_rot = transform.rotation_euler

            if any(
                abs(new_rot[i] - current_rot[i]) > self.tolerance
                for i in range(3)
            ):
                blender_obj.rotation_euler.x = new_rot[0]
                blender_obj.rotation_euler.y = new_rot[1]
                blender_obj.rotation_euler.z = new_rot[2]
                updated = True
                logger.debug(
                    "Oggetto '%s': rotation_euler aggiornata da %s a %s",
                    scene_obj.name,
                    current_rot,
                    new_rot,
                )

            if updated:
                counters["updated"] += 1
            else:
                counters["skipped"] += 1

        # Aggiorna la viewport di Blender
        try:
            bpy.context.view_layer.update()
        except Exception as exc:  # noqa: BLE001
            logger.debug("view_layer.update() non applicabile: %s", exc)

        logger.info(
            "Applicazione completata: %d aggiornati, %d non trovati, %d invariati.",
            counters["updated"],
            counters["not_found"],
            counters["skipped"],
        )
        return counters

    def save_blend_file(self, output_path: Path) -> None:
        """
        Salva la scena Blender corrente in un nuovo file .blend.

        Args:
            output_path: Percorso del file .blend di destinazione.

        Raises:
            ImportError: Se bpy non e' disponibile.
        """
        try:
            import bpy  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Il modulo 'bpy' richiede l'ambiente Blender."
            ) from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
        logger.info("Scena Blender salvata: %s", output_path)
