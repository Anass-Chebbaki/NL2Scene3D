# src/nl2scene3d/visual_critic.py
"""
Critica visiva del layout riordinato tramite chiamata LLM Vision.

Questo modulo analizza il render isometrico della scena riordinata
e produce una lista di correzioni suggerite per migliorare il layout.
Include logica di confronto qualitativo: se il modello assegna uno score
elevato o non suggerisce correzioni, lo stato viene restituito invariato.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Optional

from nl2scene3d.gemini_client import GeminiClient, GeminiParsingError
from nl2scene3d.models import LLMCorrection, ObjectTransform, RoomBounds, SceneState
from nl2scene3d.config import PipelineConfig

logger = logging.getLogger(__name__)

# Score minimo default al di sopra del quale vengono comunque applicate
# le correzioni per ottimizzare ulteriormente il layout.
_DEFAULT_MIN_QUALITY_SCORE: int = 7

# Numero massimo default di correzioni da applicare per singola iterazione.
_DEFAULT_MAX_CORRECTIONS: int = 5


def _parse_corrections_from_llm(
    llm_output: dict | list,
) -> tuple[int, str, list[LLMCorrection]]:
    """
    Parsa l'output dell'LLM Vision in una lista di correzioni strutturate.

    Args:
        llm_output: Output JSON grezzo del modello vision.

    Returns:
        Tupla (score, quality_assessment, lista di LLMCorrection).
    """
    if isinstance(llm_output, list):
        logger.warning(
            "Output vision e' una lista anziche' un dizionario. "
            "Interpretato come lista diretta di correzioni."
        )
        corrections = [
            LLMCorrection.from_dict(item)
            for item in llm_output
            if isinstance(item, dict)
        ]
        return 5, "Parsed from list format", corrections

    if not isinstance(llm_output, dict):
        logger.error(
            "Output vision non e' un dizionario valido: %s. Nessuna correzione applicata.",
            type(llm_output),
        )
        return 5, "Invalid format", []

    score = int(llm_output.get("score", 5))
    quality_assessment: str = llm_output.get("quality_assessment", "")
    raw_corrections: list = llm_output.get("corrections", [])

    corrections: list[LLMCorrection] = []
    for item in raw_corrections:
        if not isinstance(item, dict):
            continue
        try:
            correction = LLMCorrection.from_dict(item)
            corrections.append(correction)
        except (KeyError, TypeError) as exc:
            logger.warning(
                "Correzione non valida ignorata: %s. Errore: %s",
                item,
                exc,
            )

    logger.info(
        "Critica visiva: score=%d/10, assessment='%s', correzioni suggerite=%d.",
        score,
        quality_assessment,
        len(corrections),
    )
    return score, quality_assessment, corrections


def _apply_corrections_to_state(
    state: SceneState,
    corrections: list[LLMCorrection],
    max_corrections: int,
    room_bounds: Optional[RoomBounds] = None,
) -> SceneState:
    """
    Applica le correzioni suggerite dall'LLM Vision allo stato della scena.

    Args:
        state: Stato corrente da aggiornare.
        corrections: Lista di correzioni da applicare.
        max_corrections: Numero massimo di correzioni da applicare.
        room_bounds: Bounds per il clamp delle coordinate.

    Returns:
        Nuovo SceneState con le correzioni applicate.
    """
    objects_by_name: dict[str, "SceneObject"] = {  # type: ignore[name-defined]
        obj.name: obj.copy() for obj in state.objects
    }

    applied_count = 0
    skipped_count = 0

    corrections_to_apply = corrections[:max_corrections]
    if len(corrections) > max_corrections:
        logger.info(
            "Correzioni limitate da %d a %d per non alterare eccessivamente il layout.",
            len(corrections),
            max_corrections,
        )

    for correction in corrections_to_apply:
        target_obj = objects_by_name.get(correction.object_name)

        if target_obj is None:
            logger.warning(
                "Correzione ignorata: oggetto '%s' non trovato nella scena.",
                correction.object_name,
            )
            skipped_count += 1
            continue

        if not target_obj.is_movable:
            logger.warning(
                "Correzione ignorata: oggetto '%s' non e' movibile.",
                correction.object_name,
            )
            skipped_count += 1
            continue

        new_location = list(target_obj.transform.location)
        new_rotation = list(target_obj.transform.rotation_euler)

        if correction.action in ("move", "move_and_rotate"):
            if correction.new_location and len(correction.new_location) == 3:
                candidate_location = [float(v) for v in correction.new_location]
                # Preserva la quota Z originale.
                candidate_location[2] = target_obj.transform.location[2]
                if room_bounds is not None:
                    candidate_location = room_bounds.clamp_location(candidate_location)
                new_location = candidate_location

        if correction.action in ("rotate", "move_and_rotate"):
            if (
                correction.new_rotation_euler
                and len(correction.new_rotation_euler) == 3
            ):
                new_rotation = [float(v) for v in correction.new_rotation_euler]

        target_obj.transform = ObjectTransform(
            location=new_location,
            rotation_euler=new_rotation,
            dimensions=target_obj.transform.dimensions,
        )

        logger.debug(
            "Correzione applicata a '%s': azione=%s, location=%s, rotation=%s.",
            correction.object_name,
            correction.action,
            new_location,
            new_rotation,
        )
        applied_count += 1

    logger.info(
        "Correzioni applicate: %d su %d richieste (%d saltate).",
        applied_count,
        len(corrections_to_apply),
        skipped_count,
    )

    return SceneState(
        scene_name=state.scene_name,
        objects=list(objects_by_name.values()),
        room_bounds=state.room_bounds,
        pipeline_step="refined",
        metadata={
            "applied_corrections": applied_count,
            "skipped_corrections": skipped_count,
        },
    )


class VisualCritic:
    """
    Esegue la critica visiva del layout riordinato tramite LLM Vision.

    Attributes:
        client: Client Gemini per le chiamate API.
        prompts_dir: Directory contenente i template dei prompt.
        config: Configurazione della pipeline (opzionale).
        min_quality_score: Score minimo al di sopra del quale si applicano correzioni.
        max_corrections: Numero massimo di correzioni per iterazione.
    """

    def __init__(
        self,
        client: GeminiClient,
        prompts_dir: Path,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        """
        Inizializza il visual critic.

        Args:
            client: Client Gemini configurato.
            prompts_dir: Directory dei template dei prompt.
            config: Configurazione della pipeline per soglie e limiti.
                    Se None, vengono usati i valori default.
        """
        self.client = client
        self.prompts_dir = prompts_dir
        self.config = config

        if config is not None:
            self.min_quality_score: int = config.min_quality_score
            self.max_corrections: int = config.max_corrections
        else:
            self.min_quality_score = _DEFAULT_MIN_QUALITY_SCORE
            self.max_corrections = _DEFAULT_MAX_CORRECTIONS

        logger.info(
            "VisualCritic inizializzato. "
            "min_quality_score=%d, max_corrections=%d.",
            self.min_quality_score,
            self.max_corrections,
        )

    def _build_critic_prompt(self, state: SceneState) -> str:
        """
        Costruisce il prompt per la critica visiva caricandolo dal template.

        Args:
            state: Stato della scena riordinata.

        Returns:
            Prompt formattato con le informazioni della stanza.

        Raises:
            FileNotFoundError: Se il template del prompt non esiste.
        """
        template_path = self.prompts_dir / "critic_user.txt"
        if not template_path.exists():
            raise FileNotFoundError(
                f"Template del prompt per la critica visiva non trovato: {template_path}"
            )

        with open(template_path, encoding="utf-8") as fh:
            template = fh.read()

        room_bounds = state.room_bounds
        if room_bounds is not None:
            return template.format(
                x_min=room_bounds.x_min,
                x_max=room_bounds.x_max,
                y_min=room_bounds.y_min,
                y_max=room_bounds.y_max,
            )

        return template.format(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0)

    def critique_and_refine(
        self,
        reordered_state: SceneState,
        render_iso_path: Path,
    ) -> SceneState:
        """
        Analizza il render isometrico e applica le correzioni suggerite.

        Se il modello non suggerisce correzioni, lo stato viene restituito
        invariato con pipeline_step="refined". Se la chiamata LLM fallisce,
        viene restituito lo stato invariato con i metadati dell'errore.

        Args:
            reordered_state: Stato della scena dopo il riordino LLM.
            render_iso_path: Percorso al render isometrico da analizzare.

        Returns:
            SceneState raffinato, oppure invariato se le correzioni non sono
            applicabili o la chiamata LLM fallisce.
        """
        logger.info(
            "Avvio critica visiva della scena '%s'. Render: %s",
            reordered_state.scene_name,
            render_iso_path,
        )

        user_prompt = self._build_critic_prompt(reordered_state)

        try:
            llm_output = self.client.call_vision(render_iso_path, user_prompt)
        except GeminiParsingError as exc:
            logger.error(
                "Parsing della risposta vision fallito: %s. "
                "Restituisce lo stato riordinato invariato.",
                exc,
            )
            refined = reordered_state.copy()
            refined.pipeline_step = "refined"
            refined.metadata = {"error": str(exc), "applied_corrections": 0}
            return refined

        score, quality_assessment, corrections = _parse_corrections_from_llm(
            llm_output
        )

        logger.info(
            "Punteggio qualita' visiva: %d/10. Assessment: '%s'.",
            score,
            quality_assessment,
        )

        if not corrections:
            logger.info("Nessuna correzione suggerita. Layout considerato ottimale.")
            refined = reordered_state.copy()
            refined.pipeline_step = "refined"
            refined.metadata = {
                "quality_score": score,
                "quality_assessment": quality_assessment,
                "applied_corrections": 0,
            }
            return refined

        if score >= self.min_quality_score:
            logger.info(
                "Score %d >= soglia %d. Correzioni applicate per ottimizzare il layout.",
                score,
                self.min_quality_score,
            )

            refined_state = _apply_corrections_to_state(
                reordered_state,
                corrections,
                self.max_corrections,
                reordered_state.room_bounds,
            )
            refined_state.metadata["quality_score"] = score
            refined_state.metadata["quality_assessment"] = quality_assessment
            return refined_state

        logger.info(
            "Score %d < soglia %d. Correzioni applicate per recuperare qualita' insufficiente.",
            score,
            self.min_quality_score,
        )
        refined_state = _apply_corrections_to_state(
            reordered_state,
            corrections,
            self.max_corrections,
            reordered_state.room_bounds,
        )
        refined_state.metadata["quality_score"] = score
        refined_state.metadata["quality_assessment"] = quality_assessment
        
        return refined_state