# src/nl2scene3d/scene_reorganizer.py
"""
Riordino della scena tramite chiamata testuale a Gemini.

Questo modulo costruisce il prompt per l'LLM, invia lo stato disordinato
della scena e riceve il JSON con le nuove coordinate degli oggetti.

Include validazione e sanitizzazione dell'output dell'LLM per garantire
che le coordinate siano coerenti e nei bounds della stanza.
"""
from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path
from typing import Optional

from nl2scene3d.gemini_client import GeminiClient, GeminiParsingError
from nl2scene3d.models import ObjectTransform, RoomBounds, SceneObject, SceneState

logger = logging.getLogger(__name__)

# (Costante MIN_MOVEMENT_THRESHOLD rimossa poiche' non utilizzata)


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

    Esclude gli elementi strutturali (muri, pavimenti, soffitti) gia'
    rappresentati dai room_bounds, per risparmiare token nel prompt.

    Args:
        state: Stato corrente della scena.

    Returns:
        Stringa JSON della scena filtrata.
    """
    relevant_objects = [
        obj for obj in state.objects 
        if obj.is_movable or "door" in obj.name.lower() or "window" in obj.name.lower()
    ]
    scene_dict = {
        "scene_name": state.scene_name,
        "objects": [obj.to_dict() for obj in relevant_objects],
    }
    return json.dumps(scene_dict, indent=2, ensure_ascii=False)


def _is_valid_float(value: object) -> bool:
    """
    Verifica che un valore sia un float finito e non NaN.

    Args:
        value: Valore da verificare.

    Returns:
        True se il valore e' un float finito, False altrimenti.
    """
    try:
        f = float(value)  # type: ignore[arg-type]
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


def _validate_and_sanitize_llm_output(
    llm_output: dict | list,
    original_state: SceneState,
) -> SceneState:
    """
    Valida e sanitizza l'output JSON dell'LLM.

    Verifica che:
    - Tutti gli oggetti originali siano presenti nella risposta
    - Le coordinate siano numericamente valide (no NaN, no Inf)
    - Le coordinate siano nei bounds della stanza (applica clamp se necessario)

    La coordinata Z di ciascun oggetto movibile viene sempre preservata
    da original_state, che deve pertanto rappresentare uno stato in cui
    la quota Z degli oggetti e' identica a quella originale (il randomizer
    non altera la Z per costruzione).

    In caso di oggetti mancanti o coordinate non valide, mantiene
    la posizione originale come fallback.

    Args:
        llm_output: Output JSON grezzo dell'LLM.
        original_state: Stato di riferimento per fallback e preservazione della Z.

    Returns:
        SceneState validato e sanitizzato.
    """
    room_bounds = original_state.room_bounds

    if isinstance(llm_output, list):
        llm_objects_list: list = llm_output
    elif isinstance(llm_output, dict):
        llm_objects_list = llm_output.get("objects", [])
    else:
        logger.error(
            "Output LLM non e' ne' dict ne' list: %s. Restituisce stato originale.",
            type(llm_output),
        )
        return original_state

    llm_objects_by_name: dict[str, dict] = {}
    for llm_obj_data in llm_objects_list:
        if isinstance(llm_obj_data, dict) and "name" in llm_obj_data:
            llm_objects_by_name[llm_obj_data["name"]] = llm_obj_data
        elif isinstance(llm_obj_data, list) and len(llm_obj_data) >= 4:
            # Formato compatto: [name, x, y, rz_degrees]
            name = str(llm_obj_data[0])
            rz_rad = float(llm_obj_data[3]) * (math.pi / 180.0)
            llm_objects_by_name[name] = {
                "name": name,
                "location": [float(llm_obj_data[1]), float(llm_obj_data[2]), 0.0],
                "rotation_euler": [0.0, 0.0, rz_rad],
            }

    corrected_objects: list[SceneObject] = []
    clamped_count = 0
    missing_count = 0

    for original_obj in original_state.objects:
        llm_obj_data = llm_objects_by_name.get(original_obj.name)

        if llm_obj_data is None:
            if original_obj.category == "structural":
                logger.debug(
                    "Oggetto strutturale '%s' assente dalla risposta LLM (comportamento atteso).",
                    original_obj.name,
                )
            else:
                logger.warning(
                    "Oggetto movibile '%s' assente dalla risposta LLM. Mantenuta posizione originale.",
                    original_obj.name,
                )
                missing_count += 1
            corrected_objects.append(copy.deepcopy(original_obj))
            continue

        if not original_obj.is_movable:
            corrected_objects.append(copy.deepcopy(original_obj))
            continue

        try:
            new_location = list(
                llm_obj_data.get("location", original_obj.transform.location)
            )
            new_rotation = list(
                llm_obj_data.get(
                    "rotation_euler", original_obj.transform.rotation_euler
                )
            )

            if len(new_location) != 3 or len(new_rotation) != 3:
                raise ValueError(
                    "location o rotation_euler non hanno esattamente 3 componenti."
                )

            if not all(_is_valid_float(v) for v in new_location + new_rotation):
                raise ValueError(
                    "Uno o piu' valori non sono float finiti validi."
                )

            new_location = [float(v) for v in new_location]
            
            raw_rotation = [float(v) for v in new_rotation]
            # CLAMP X e Y alla rotazione originale per evitare inclinazioni fantasiose dell'LLM
            new_rotation = [
                original_obj.transform.rotation_euler[0],
                original_obj.transform.rotation_euler[1],
                raw_rotation[2]
            ]
            
            # SNAP Z ai multipli di 90 gradi per posizionamenti ortogonali perfetti
            multiples = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
            best_z = min(multiples, key=lambda m: abs(m - (new_rotation[2] % (2 * math.pi))))
            new_rotation[2] = best_z

        except (TypeError, ValueError, KeyError) as exc:
            logger.warning(
                "Coordinate non valide per oggetto '%s': %s. "
                "Mantenuta posizione originale.",
                original_obj.name,
                exc,
            )
            corrected_objects.append(copy.deepcopy(original_obj))
            continue

        # Preserviamo la Z originale. Se l'oggetto fluttua, l'applicatore lo farà cadere.
        # Se l'oggetto finisce dentro un altro, has_collision lo rileverà.
        new_location[2] = original_obj.transform.location[2]

        if room_bounds is not None:
            clamped_location = room_bounds.clamp_location(new_location, original_obj.transform.dimensions)
            if clamped_location != new_location:
                logger.debug(
                    "Oggetto '%s': coordinate clampate da %s a %s.",
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

    # Passaggio finale: Risoluzione collisioni con separazione intelligente
    from nl2scene3d.utils.geometry import has_collision, compute_aabb_2d
    import random as _rng
    
    final_objects: list[SceneObject] = []
    # Prima aggiungiamo tutti gli oggetti non movibili (strutturali)
    for obj in corrected_objects:
        if not obj.is_movable:
            final_objects.append(obj)
            
    # Poi aggiungiamo i movibili uno ad uno, risolvendo eventuali collisioni
    # con separazione basata sul vettore di penetrazione anziché jitter casuale.
    main_furniture_categories = {"bed", "table", "storage", "seating_large", "seating_small", "furniture"}
    jitter_resolved = 0
    jitter_failed = 0
    
    for obj in corrected_objects:
        if not obj.is_movable:
            continue
            
        if obj.category in main_furniture_categories:
            original_loc = list(obj.transform.location)
            # Include sia i mobili che i muri per il check collisioni
            collidable_objects = [
                o for o in final_objects 
                if o.category in main_furniture_categories or o.category == "structural"
            ]
            
            # Tenta di risolvere la collisione con step incrementali
            # di dimensione crescente (0.1m, 0.2m, 0.3m)
            max_attempts = 30
            jitter_sizes = [0.15, 0.25, 0.35]  # Incrementi crescenti
            attempt = 0
            resolved = False
            
            while has_collision(obj, collidable_objects, check_walls=True) and attempt < max_attempts:
                jitter_mag = jitter_sizes[min(attempt // 10, len(jitter_sizes) - 1)]
                angle = _rng.uniform(0, 2 * math.pi)
                dx = jitter_mag * math.cos(angle)
                dy = jitter_mag * math.sin(angle)
                
                new_x = original_loc[0] + dx * (1 + attempt * 0.1)
                new_y = original_loc[1] + dy * (1 + attempt * 0.1)
                
                # Clamp ai room bounds
                if room_bounds is not None:
                    half_dim_x = obj.transform.dimensions[0] / 2.0
                    half_dim_y = obj.transform.dimensions[1] / 2.0
                    new_x = max(room_bounds.x_min + half_dim_x + 0.05,
                                min(room_bounds.x_max - half_dim_x - 0.05, new_x))
                    new_y = max(room_bounds.y_min + half_dim_y + 0.05,
                                min(room_bounds.y_max - half_dim_y - 0.05, new_y))
                
                obj.transform.location[0] = new_x
                obj.transform.location[1] = new_y
                attempt += 1
            
            if attempt > 0:
                if not has_collision(obj, collidable_objects, check_walls=True):
                    jitter_resolved += 1
                    logger.debug(
                        "Collisione per '%s' risolta dopo %d tentativi (spostamento: %.2fm).",
                        obj.name, attempt,
                        math.sqrt((obj.transform.location[0] - original_loc[0])**2 +
                                  (obj.transform.location[1] - original_loc[1])**2),
                    )
                else:
                    jitter_failed += 1
                    logger.warning(
                        "Impossibile risolvere collisione per '%s' dopo %d tentativi. "
                        "Mantenuta posizione LLM.",
                        obj.name, max_attempts,
                    )
                    obj.transform.location = original_loc
            
        final_objects.append(obj)

    if jitter_resolved > 0 or jitter_failed > 0:
        logger.info(
            "Risoluzione collisioni post-LLM: %d risolte, %d irrisolvibili.",
            jitter_resolved, jitter_failed,
        )

    return SceneState(
        scene_name=original_state.scene_name,
        objects=final_objects,
        room_bounds=original_state.room_bounds,
        pipeline_step="reordered",
        metadata={
            "clamped_count": clamped_count,
            "missing_count": missing_count,
            "jitter_resolved": jitter_resolved,
            "jitter_failed": jitter_failed,
        },
    )


class SceneReorganizer:
    """
    Coordina la prima chiamata LLM per il riordino testuale della scena.

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

    def _build_footprint_table(self, state: SceneState) -> str:
        """
        Costruisce una tabella dei footprint AABB per ogni oggetto movibile.
        Questo aiuta l'LLM a ragionare sullo spazio fisico occupato.
        """
        from nl2scene3d.utils.geometry import compute_aabb_2d
        
        lines = ["\nOBJECT FOOTPRINT TABLE (pre-computed AABB on XY plane):"]
        lines.append("| Object Name | Width(X) | Depth(Y) | Height(Z) | Floor Area |")
        lines.append("|-------------|----------|----------|-----------|------------|")
        
        for obj in state.movable_objects:
            dim = obj.transform.dimensions
            rz = obj.transform.rotation_euler[2]
            cos_z = abs(math.cos(rz))
            sin_z = abs(math.sin(rz))
            eff_x = dim[0] * cos_z + dim[1] * sin_z
            eff_y = dim[0] * sin_z + dim[1] * cos_z
            area = eff_x * eff_y
            lines.append(
                f"| {obj.name} | {eff_x:.2f}m | {eff_y:.2f}m | {dim[2]:.2f}m | {area:.2f}m² |"
            )
        
        lines.append("")
        lines.append(
            "USE THIS TABLE to avoid placing objects where their footprints would overlap. "
            "For each object, its center must be at least (Width/2 + OtherWidth/2) apart "
            "in X or (Depth/2 + OtherDepth/2) apart in Y from every other object."
        )
        return "\n".join(lines)

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
        footprint_table = self._build_footprint_table(state)

        if room_bounds is not None:
            prompt = template.format(
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
            compact_instr = (
                "\n\nOUTPUT FORMAT (COMPACT JSON):"
                "\nReturn a JSON list of lists, where each inner list is: [\"object_name\", x, y, rotation_z_degrees]"
                "\nExample: [[\"furniture_bed\", 1.2, -0.5, 90], [\"furniture_desk\", -2.1, 1.0, 0]]"
                "\n\nONLY include movable objects. DO NOT add any text, descriptions, or other fields."
            )
            return prompt + footprint_table + compact_instr

        return (
            "You are an interior designer. Please reorganize the following 3D scene JSON "
            "to be professionally arranged, functional, and aesthetically pleasing. "
            "Return ONLY the updated JSON.\n\n"
            f"{scene_json}"
        )

    def reorganize(self, disordered_state: SceneState) -> SceneState:
        """
        Esegue il riordino della scena tramite chiamata LLM.

        Args:
            disordered_state: Stato disordinato della scena.

        Returns:
            Nuovo SceneState con le posizioni suggerite dall'LLM,
            validate e sanitizzate.

        Raises:
            GeminiClientError: In caso di errori API non recuperabili.
        """
        logger.info(
            "Avvio riordino LLM della scena '%s'. Oggetti movibili: %d.",
            disordered_state.scene_name,
            len(disordered_state.movable_objects),
        )

        system_prompt_path = self.prompts_dir / "reorder_system.txt"
        system_prompt = _load_prompt_template(system_prompt_path)
        user_prompt = self._build_user_prompt(disordered_state)

        logger.debug(
            "Lunghezza system_prompt: %d caratteri, user_prompt: %d caratteri.",
            len(system_prompt),
            len(user_prompt),
        )

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

        reordered_state = _validate_and_sanitize_llm_output(
            llm_output, disordered_state
        )

        logger.info(
            "Riordino LLM completato per scena '%s'.",
            reordered_state.scene_name,
        )

        return reordered_state

    def reorganize_with_image(
            self,
            disordered_state: SceneState,
            image_path: Path,
    ) -> SceneState:
        """
        Esegue il riordino della scena con prompt multimodale (testo + immagine).

        A differenza di 'reorganize', questo metodo manda al modello sia
        i dati JSON della scena sia uno screenshot del viewport corrente.
        Permette al LLM di "vedere" effettivamente la stanza e di prendere
        decisioni piu' contestuali (orientamento muri, posizione finestre, ecc.).

        Args:
            disordered_state: Stato disordinato della scena.
            image_path: Percorso allo screenshot della viewport corrente.

        Returns:
            Nuovo SceneState con le posizioni suggerite dall'LLM,
            validate e sanitizzate.
        """
        logger.info(
            "Avvio riordino LLM multimodale per scena '%s'. "
            "Oggetti movibili: %d. Immagine: %s",
            disordered_state.scene_name,
            len(disordered_state.movable_objects),
            image_path,
        )

        # Caricamento del system prompt classico
        system_prompt_path = self.prompts_dir / "reorder_system.txt"
        system_prompt = _load_prompt_template(system_prompt_path)

        # Costruzione del user prompt classico (bounds, JSON, footprint)
        user_prompt_body = self._build_user_prompt(disordered_state)

        # Aggiunta istruzioni specifiche per la modalita' visiva
        vision_instructions = (
            "\n\n=== VISUAL CONTEXT ===\n"
            "Along with this textual data, you are receiving an image showing "
            "the CURRENT state of the room from a representative viewpoint.\n"
            "Use the image to understand information that coordinates alone cannot convey:\n"
            "- The actual orientation and layout of walls\n"
            "- The position of doors, windows, and other openings\n"
            "- The real proportions and shape of the room (may not be a perfect rectangle)\n"
            "- Natural focal points (windows for desks, walls for beds, etc.)\n"
            "\nBase your reorganization on BOTH the numerical data AND the visual context.\n"
            "For example: place the bed with headboard against an empty wall visible in "
            "the image, position the desk near a window if present, leave space in front "
            "of doors."
        )

        # call_vision accetta un singolo prompt (non system + user separati).
        # Concateniamo system + user + istruzioni visive in un unico prompt.
        combined_prompt = (
            f"{system_prompt}\n\n"
            f"{user_prompt_body}\n"
            f"{vision_instructions}"
        )

        logger.debug(
            "Lunghezza combined_prompt: %d caratteri.",
            len(combined_prompt),
        )

        try:
            llm_output = self.client.call_vision(
                image_path=image_path,
                user_prompt=combined_prompt,
            )
        except GeminiParsingError as exc:
            logger.error(
                "Parsing della risposta LLM (multimodale) fallito: %s. "
                "Restituisce lo stato disordinato invariato.",
                exc,
            )
            return SceneState(
                scene_name=disordered_state.scene_name,
                objects=copy.deepcopy(disordered_state.objects),
                room_bounds=disordered_state.room_bounds,
                pipeline_step="reordered_failed",
                metadata={"error": str(exc), "mode": "multimodal"},
            )

        reordered_state = _validate_and_sanitize_llm_output(
            llm_output, disordered_state
        )

        logger.info(
            "Riordino LLM multimodale completato per scena '%s'.",
            reordered_state.scene_name,
        )

        return reordered_state