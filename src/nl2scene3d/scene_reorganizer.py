"""
Riordino della scena tramite chiamata testuale a Gemini (Step 7).

Questo modulo costruisce il prompt per l'LLM, invia lo stato disordinato
della scena e riceve il JSON con le nuove coordinate degli oggetti.

Include validazione e sanitizzazione dell'output dell'LLM per garantire
che le coordinate siano coerenti e nei bounds della stanza.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Optional

from nl2scene3d.gemini_client import GeminiClient, GeminiParsingError
from nl2scene3d.models import ObjectTransform, RoomBounds, SceneObject, SceneState

logger = logging.getLogger(__name__)

# Soglia di movimento minimo in metri: correzioni sotto questa soglia vengono ignorate
MIN_MOVEMENT_THRESHOLD: float = 0.01

def _load_prompt_template(prompt_path: Path) -> str:
    """
    Carica un template di prompt da file.

    Args:
        prompt_path: Percorso al file del template.

    Returns:
        Contenuto del template come stringa.

    Raises:
        FileNotFoundError: Se il file non esiste.
    """
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Template di prompt non trovato: {prompt_path}"
        )
    with open(prompt_path, encoding="utf-8") as fh:
        return fh.read()

def _build_scene_json_for_llm(state: SceneState) -> str:
    """
    Costruisce la rappresentazione JSON della scena ottimizzata per l'LLM.

    Include solo i campi necessari per il ragionamento spaziale,
    escludendo informazioni ridondanti che aumenterebbero la lunghezza del prompt.

    Args:
        state: Stato corrente della scena.

    Returns:
        Stringa JSON della scena.
    """
    scene_dict = {
        "scene_name": state.scene_name,
        "objects": [obj.to_dict() for obj in state.objects],
    }
    return json.dumps(scene_dict, indent=2, ensure_ascii=False)

def _validate_and_sanitize_llm_output(
    llm_output: dict | list,
    original_state: SceneState,
) -> SceneState:
    """
    Valida e sanitizza l'output JSON dell'LLM.

    Verifica che:
    - Tutti gli oggetti originali siano presenti nella risposta
    - Le coordinate siano numericamente valide
    - Le coordinate siano nei bounds della stanza

    In caso di coordinate fuori bounds, applica il clamp.
    In caso di oggetti mancanti, mantiene la posizione originale.

    Args:
        llm_output: Output JSON grezzo dell'LLM.
        original_state: Stato originale della scena (usato come fallback).

    Returns:
        SceneState validato e sanitizzato.
    """
    room_bounds = original_state.room_bounds

    # Estrae la lista degli oggetti dall'output LLM
    # Il formato atteso e' {"objects": [...]} oppure direttamente [...]
    if isinstance(llm_output, list):
        llm_objects_list = llm_output
    elif isinstance(llm_output, dict):
        llm_objects_list = llm_output.get("objects", [])
    else:
        logger.error("Output LLM non e' ne' dict ne' list: %s", type(llm_output))
        return original_state

    # Costruisce un dizionario nome -> dati per lookup rapido
    llm_objects_by_name: dict[str, dict] = {}
    for llm_obj_data in llm_objects_list:
        if isinstance(llm_obj_data, dict) and "name" in llm_obj_data:
            llm_objects_by_name[llm_obj_data["name"]] = llm_obj_data

    corrected_objects: list[SceneObject] = []
    clamped_count = 0
    missing_count = 0

    for original_obj in original_state.objects:
        llm_obj_data = llm_objects_by_name.get(original_obj.name)

        if llm_obj_data is None:
            # Oggetto non presente nella risposta: mantieni posizione originale
            logger.warning(
                "Oggetto '%s' assente dalla risposta LLM. "
                "Mantenuta posizione originale.",
                original_obj.name,
            )
            corrected_objects.append(copy.deepcopy(original_obj))
            missing_count += 1
            continue

        # Oggetti non movibili: ignora eventuali modifiche dell'LLM
        if not original_obj.is_movable:
            corrected_objects.append(copy.deepcopy(original_obj))
            continue

        # Estrae le nuove coordinate dalla risposta LLM
        try:
            new_location = list(llm_obj_data.get("location", original_obj.transform.location))
            new_rotation = list(
                llm_obj_data.get("rotation_euler", original_obj.transform.rotation_euler)
            )

            # Verifica che siano liste di 3 numeri
            if len(new_location) != 3 or len(new_rotation) != 3:
                raise ValueError("location o rotation_euler non hanno 3 componenti")

            # Converte in float e verifica valori NaN/Inf
            new_location = [float(v) for v in new_location]
            new_rotation = [float(v) for v in new_rotation]

            for v in new_location + new_rotation:
                if not (v == v) or abs(v) == float("inf"):  # NaN check
                    raise ValueError(f"Valore non valido: {v}")

        except (TypeError, ValueError, KeyError) as exc:
            logger.warning(
                "Coordinate non valide per oggetto '%s': %s. "
                "Mantenuta posizione originale.",
                original_obj.name,
                exc,
            )
            corrected_objects.append(copy.deepcopy(original_obj))
            continue

        # Preserva la Z originale (gli oggetti restano sul pavimento)
        new_location[2] = original_obj.transform.location[2]

        # Clamp delle coordinate nei bounds della stanza
        if room_bounds is not None:
            clamped_location = room_bounds.clamp_location(new_location)
            if clamped_location != new_location:
                logger.debug(
                    "Oggetto '%s': coordinate clampate da %s a %s",
                    original_obj.name,
                    new_location,
                    clamped_location,
                )
                new_location = clamped_location
                clamped_count += 1

        new_obj = copy.deepcopy(original_obj)
        new_obj.transform = ObjectTransform(
            location=new_location,
            rotation_euler=new_rotation,
            dimensions=original_obj.transform.dimensions,
        )
        corrected_objects.append(new_obj)

    logger.info(
        "Validazione output LLM: %d clamped, %d mancanti su %d totali.",
        clamped_count,
        missing_count,
        len(original_state.objects),
    )

    return SceneState(
        scene_name=original_state.scene_name,
        objects=corrected_objects,
        room_bounds=original_state.room_bounds,
        pipeline_step="reordered",
        metadata={
            "clamped_count": clamped_count,
            "missing_count": missing_count,
        },
    )

class SceneReorganizer:
    """
    Coordina la prima chiamata LLM per il riordino testuale della scena.

    Corrisponde allo Step 7 della pipeline.

    Attributes:
        client: Client Gemini per le chiamate API.
        prompts_dir: Directory contenente i template dei prompt.
    """

    def __init__(
        self,
        client: GeminiClient,
        prompts_dir: Path,
    ) -> None:
        """
        Inizializza il reorganizer.

        Args:
            client: Client Gemini configurato.
            prompts_dir: Directory dei template dei prompt.
        """
        self.client = client
        self.prompts_dir = prompts_dir
        logger.info("SceneReorganizer inizializzato.")

    def _build_user_prompt(self, state: SceneState) -> str:
        """
        Costruisce il prompt utente con i dati della scena disordinata.

        Args:
            state: Stato disordinato della scena.

        Returns:
            Prompt utente formattato.
        """
        template_path = self.prompts_dir / "reorder_user.txt"
        template = _load_prompt_template(template_path)

        room_bounds = state.room_bounds
        scene_json = _build_scene_json_for_llm(state)

        if room_bounds:
            return template.format(
                scene_name=state.scene_name,
                x_min=room_bounds.x_min,
                x_max=room_bounds.x_max,
                y_min=room_bounds.y_min,
                y_max=room_bounds.y_max,
                room_width=room_bounds.width,
                room_depth=room_bounds.depth,
                room_height=room_bounds.height,
                scene_json=scene_json,
            )
        else:
            return f"Please reorganize the following scene:\n\n{scene_json}"

    def reorganize(self, disordered_state: SceneState) -> SceneState:
        """
        Esegue il riordino della scena tramite chiamata LLM.

        Args:
            disordered_state: Stato disordinato della scena (output Step 4-6).

        Returns:
            Nuovo SceneState con le posizioni suggerite dall'LLM, validate
            e sanitizzate.

        Raises:
            GeminiClientError: In caso di errori API non recuperabili.
        """
        logger.info(
            "Avvio riordino LLM della scena '%s'. "
            "Oggetti movibili: %d",
            disordered_state.scene_name,
            len(disordered_state.movable_objects),
        )

        # Carica il system prompt
        system_prompt_path = self.prompts_dir / "reorder_system.txt"
        system_prompt = _load_prompt_template(system_prompt_path)

        # Costruisce il user prompt
        user_prompt = self._build_user_prompt(disordered_state)

        logger.debug(
            "Lunghezza system_prompt: %d caratteri, user_prompt: %d caratteri",
            len(system_prompt),
            len(user_prompt),
        )

        # Prima chiamata LLM
        try:
            llm_output = self.client.call_text(system_prompt, user_prompt)
        except GeminiParsingError as exc:
            logger.error(
                "Parsing della risposta LLM fallito: %s. "
                "Restituisce lo stato disordinato invariato.",
                exc,
            )
            return SceneState(
                scene_name=disordered_state.scene_name,
                objects=copy.deepcopy(disordered_state.objects),
                room_bounds=disordered_state.room_bounds,
                pipeline_step="reordered_failed",
                metadata={"error": str(exc)},
            )

        # Validazione e sanitizzazione dell'output
        reordered_state = _validate_and_sanitize_llm_output(
            llm_output, disordered_state
        )

        logger.info(
            "Riordino LLM completato per scena '%s'.",
            reordered_state.scene_name,
        )
        return reordered_state
