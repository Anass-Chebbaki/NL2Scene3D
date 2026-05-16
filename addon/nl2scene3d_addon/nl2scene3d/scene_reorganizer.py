# nl2scene3d/scene_reorganizer.py
"""
Riordino della scena tramite LLM (Gemini).

Responsabilità:
    - Costruire il prompt per l'LLM con JSON completo della scena e relazioni
      parent-child già calcolate da SceneState.
    - Inviare il prompt e ricevere le nuove coordinate.
    - Validare/sanitizzare l'output (bounds clamp, Z lock, snap a 90°).
    - Spostare i figli con trasformazione rigida rispetto al parent.
    - Risolvere collisioni post-LLM con vettore MTV invece di jitter casuale.

Principi fondamentali:
    - La Z non viene MAI modificata: si prende sempre dalla scena originale.
    - Il grouping (parent/children) è già calcolato su ogni SceneObject da
      SceneLoader.extract_scene_state() — non viene ricalcolato qui.
    - Nessuna dipendenza da utils.grouping (deprecato).
"""
from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path
from nl2scene3d.gemini_client import GeminiClient, GeminiParsingError
from nl2scene3d.models import SceneObject, SceneState, Transform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _load_prompt_template(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Template di prompt non trovato: {prompt_path}")
    with open(prompt_path, encoding="utf-8") as fh:
        return fh.read()


def _build_scene_json_for_llm(state: SceneState) -> str:
    """
    Costruisce il JSON della scena ottimizzato per l'LLM.

    Include SOLO gli oggetti root movibili e i loro figli (con is_child=True
    per segnalare all'LLM che seguono il genitore).
    Esclude gli strutturali (già rappresentati dai room_bounds).
    """
    objects_data = []
    by_name = {obj.name: obj for obj in state.objects}

    for obj in state.objects:
        if not obj.is_movable:
            continue

        entry = obj.to_dict()

        # Arricchisce con info utili all'LLM
        if obj.parent is not None:
            entry["is_child"] = True
            entry["follows_parent"] = obj.parent
        else:
            entry["is_child"] = False
            if obj.children:
                entry["children_that_follow"] = obj.children

        objects_data.append(entry)

    scene_dict = {
        "scene_name": state.scene_name,
        "note": (
            "Objects with is_child=true follow their parent rigidly. "
            "Only specify new positions for root objects (is_child=false). "
            "Children will be repositioned automatically."
        ),
        "objects": objects_data,
    }
    return json.dumps(scene_dict, indent=2, ensure_ascii=False)


def _build_footprint_table(state: SceneState) -> str:
    """Tabella AABB per ogni oggetto root movibile — aiuta l'LLM a ragionare sugli spazi."""
    lines = ["\nOBJECT FOOTPRINT TABLE (pre-computed AABB, XY plane):"]
    lines.append("| Object Name | Width(X) | Depth(Y) | Height(Z) | Floor Area | is_child |")
    lines.append("|-------------|----------|----------|-----------|------------|----------|")

    for obj in state.objects:
        if not obj.is_movable:
            continue
        dim = obj.transform.dimensions
        rz = obj.transform.rotation_euler[2]
        cos_z = abs(math.cos(rz))
        sin_z = abs(math.sin(rz))
        eff_x = dim[0] * cos_z + dim[1] * sin_z
        eff_y = dim[0] * sin_z + dim[1] * cos_z
        area = eff_x * eff_y
        is_child = "yes" if obj.parent is not None else "no"
        lines.append(
            f"| {obj.name} | {eff_x:.2f}m | {eff_y:.2f}m | {dim[2]:.2f}m | {area:.2f}m² | {is_child} |"
        )

    lines.append("")
    lines.append(
        "Only move ROOT objects (is_child=no). Their children will follow automatically. "
        "Keep at least (Width/2 + OtherWidth/2) distance in X and "
        "(Depth/2 + OtherDepth/2) distance in Y between each pair of objects."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validazione e sanitizzazione output LLM
# ---------------------------------------------------------------------------

from nl2scene3d.utils.geometry import (
    has_collision,
    is_finite_float,
    penetration_vector,
    snap_rotation_90,
)


def _apply_rigid_child_transform(
    child: SceneObject,
    old_parent_loc: list[float],
    old_parent_rz: float,
    new_parent_loc: list[float],
    new_parent_rz: float,
) -> SceneObject:
    """
    Restituisce una COPIA del figlio spostato rigidamente rispetto al parent.
    Non modifica il figlio in-place.
    """
    new_child = child.copy()

    rel_x = child.transform.location[0] - old_parent_loc[0]
    rel_y = child.transform.location[1] - old_parent_loc[1]
    rel_z = child.transform.location[2] - old_parent_loc[2]

    d_rz = new_parent_rz - old_parent_rz
    cos_a, sin_a = math.cos(d_rz), math.sin(d_rz)

    new_child.transform.location[0] = new_parent_loc[0] + rel_x * cos_a - rel_y * sin_a
    new_child.transform.location[1] = new_parent_loc[1] + rel_x * sin_a + rel_y * cos_a
    new_child.transform.location[2] = new_parent_loc[2] + rel_z  # Z: solo traslazione
    new_child.transform.rotation_euler[2] = (child.transform.rotation_euler[2] + d_rz) % (2 * math.pi)

    return new_child


def _validate_and_sanitize_llm_output(
    llm_output: dict | list,
    original_state: SceneState,
) -> SceneState:
    """
    Valida e sanitizza l'output JSON dell'LLM.

    - Legge solo le posizioni per i root movibili (is_child=False).
    - Blocca la Z sull'originale.
    - Snappa la rotazione Z a multipli di 90°.
    - Preserva le rotazioni X/Y originali (l'LLM non le tocca mai).
    - Sposta i figli con trasformazione rigida rispetto al parent mosso.
    - Risolve collisioni post-LLM con vettore di penetrazione (MTV).
    """
    # --- Local helper already imported from geometry ---

    room_bounds = original_state.room_bounds

    # --- Parse output ---
    if isinstance(llm_output, list):
        llm_objects_list: list = llm_output
    elif isinstance(llm_output, dict):
        llm_objects_list = llm_output.get("objects", [])
    else:
        logger.error("Output LLM non valido (%s). Stato originale restituito.", type(llm_output))
        return original_state

    # Indicizza per nome
    llm_by_name: dict[str, dict] = {}
    for item in llm_objects_list:
        if isinstance(item, dict) and "name" in item:
            llm_by_name[item["name"]] = item
        elif isinstance(item, list) and len(item) >= 4:
            # Formato compatto: [name, x, y, rz_degrees]
            name = str(item[0])
            rz_rad = float(item[3]) * (math.pi / 180.0)
            llm_by_name[name] = {
                "name": name,
                "location": [float(item[1]), float(item[2]), 0.0],
                "rotation_euler": [0.0, 0.0, rz_rad],
            }

    # --- Applica le posizioni LLM ai root movibili ---
    by_name_orig = {obj.name: obj for obj in original_state.objects}
    corrected: dict[str, SceneObject] = {}  # name → nuovo obj

    clamped_count = 0
    missing_count = 0

    for orig_obj in original_state.objects:
        # Strutturali: sempre invariati
        if not orig_obj.is_movable:
            corrected[orig_obj.name] = orig_obj.copy()
            continue

        # Figli: verranno aggiornati dopo il parent
        if orig_obj.parent is not None:
            corrected[orig_obj.name] = orig_obj.copy()  # placeholder, sovrascritto sotto
            continue

        # Root movibile: leggi posizione LLM
        llm_data = llm_by_name.get(orig_obj.name)
        if llm_data is None:
            logger.warning("Root movibile '%s' assente dall'output LLM. Posizione originale mantenuta.", orig_obj.name)
            missing_count += 1
            corrected[orig_obj.name] = orig_obj.copy()
            continue

        try:
            new_loc = list(llm_data.get("location", orig_obj.transform.location))
            new_rot = list(llm_data.get("rotation_euler", orig_obj.transform.rotation_euler))

            if len(new_loc) != 3 or len(new_rot) != 3:
                raise ValueError("location o rotation_euler non hanno 3 componenti.")
            if not all(is_finite_float(v) for v in new_loc + new_rot):
                raise ValueError("Valori non finiti rilevati.")

            new_loc = [float(v) for v in new_loc]
            new_rot = [float(v) for v in new_rot]

            # Blocca Z sull'originale
            new_loc[2] = orig_obj.transform.location[2]
            # Preserva RX e RY originali, snap RZ a 90°
            new_rot[0] = orig_obj.transform.rotation_euler[0]
            new_rot[1] = orig_obj.transform.rotation_euler[1]
            new_rot[2] = snap_rotation_90(new_rot[2])

        except (TypeError, ValueError, KeyError) as exc:
            logger.warning("Coordinate non valide per '%s': %s. Posizione originale.", orig_obj.name, exc)
            corrected[orig_obj.name] = orig_obj.copy()
            continue

        # Clamp ai bounds
        if room_bounds is not None:
            clamped = room_bounds.clamp_location(new_loc, orig_obj.transform.dimensions)
            if clamped != new_loc:
                logger.debug("'%s': clamp %s → %s.", orig_obj.name, new_loc, clamped)
                new_loc = clamped
                clamped_count += 1

        new_obj = orig_obj.copy()
        new_obj.transform = Transform(
            location=new_loc,
            rotation_euler=new_rot,
            dimensions=list(orig_obj.transform.dimensions),
            origin_offset=list(orig_obj.transform.origin_offset),
        )
        corrected[orig_obj.name] = new_obj

        # Sposta i figli con trasformazione rigida
        for child_name in orig_obj.children:
            orig_child = by_name_orig.get(child_name)
            if orig_child is None:
                continue
            new_child = _apply_rigid_child_transform(
                orig_child,
                old_parent_loc=orig_obj.transform.location,
                old_parent_rz=orig_obj.transform.rotation_euler[2],
                new_parent_loc=new_loc,
                new_parent_rz=new_rot[2],
            )
            # Verifica bounds per il figlio
            if room_bounds is not None:
                child_aabb = new_child.transform.aabb_xy(margin=0.0)
                if not room_bounds.contains_aabb(child_aabb, margin=0.0):
                    logger.debug(
                        "Figlio '%s' fuori bounds dopo trasformazione rigida — posizione originale mantenuta.",
                        child_name,
                    )
                    corrected[child_name] = orig_child.copy()
                    continue
            corrected[child_name] = new_child

    logger.info(
        "Validazione output LLM: %d root processati, %d clamped, %d mancanti.",
        len([o for o in original_state.objects if o.is_movable and o.parent is None]),
        clamped_count,
        missing_count,
    )

    # --- Risoluzione collisioni post-LLM ---
    # Usa il vettore MTV (Minimum Translation Vector) invece di jitter casuale.
    main_cats = {"bed", "table", "storage", "seating_large", "seating_small", "furniture"}

    final_list: list[SceneObject] = []
    # Prima i non movibili
    for obj in original_state.objects:
        if not obj.is_movable:
            final_list.append(corrected[obj.name])

    jitter_resolved = 0
    jitter_failed = 0

    for orig_obj in original_state.objects:
        if not orig_obj.is_movable or orig_obj.parent is not None:
            continue  # Strutturali già aggiunti; figli vengono dopo il parent

        obj = corrected[orig_obj.name]

        if obj.category in main_cats:
            collidable = [
                o for o in final_list
                if o.category in main_cats or o.category == "structural"
            ]
            max_iter = 20
            for i in range(max_iter):
                if not has_collision(obj, collidable, wall_margin=0.05, furniture_margin=0.02):  # type: ignore[call-arg]
                    break
                # Cerca l'oggetto con cui si sovrappone di più e applica MTV
                moved = False
                for other in collidable:
                    dx, dy = penetration_vector(obj, other, margin=0.02)
                    if dx != 0.0 or dy != 0.0:
                        obj.transform.location[0] += dx
                        obj.transform.location[1] += dy
                        if room_bounds is not None:
                            clamped = room_bounds.clamp_location(
                                obj.transform.location, obj.transform.dimensions, margin=0.02  # type: ignore[call-arg]
                            )
                            obj.transform.location = clamped
                        moved = True
                        break
                if not moved:
                    break
            else:
                jitter_failed += 1
                logger.warning("Collisione irrisolvibile per '%s' dopo %d iterazioni.", obj.name, max_iter)

            if not has_collision(obj, collidable, wall_margin=0.05, furniture_margin=0.02):  # type: ignore[call-arg]
                if i > 0:
                    jitter_resolved += 1

        final_list.append(obj)

        # Aggiungi i figli (già calcolati con trasformazione rigida)
        for child_name in orig_obj.children:
            if child_name in corrected:
                final_list.append(corrected[child_name])

    if jitter_resolved or jitter_failed:
        logger.info("Collisioni post-LLM: %d risolte, %d irrisolvibili.", jitter_resolved, jitter_failed)

    return SceneState(
        scene_name=original_state.scene_name,
        objects=final_list,
        room_bounds=original_state.room_bounds,
        pipeline_step="reordered",
        metadata={
            "clamped_count": clamped_count,
            "missing_count": missing_count,
            "jitter_resolved": jitter_resolved,
            "jitter_failed": jitter_failed,
        },
    )


# ---------------------------------------------------------------------------
# SceneReorganizer
# ---------------------------------------------------------------------------

class SceneReorganizer:
    """
    Coordina la chiamata LLM per il riordino testuale della scena.

    Attributes:
        client:     Client Gemini per le chiamate API.
        prompts_dir: Directory contenente i template dei prompt.
    """

    def __init__(self, client: GeminiClient, prompts_dir: Path) -> None:
        self.client = client
        self.prompts_dir = prompts_dir
        logger.info("SceneReorganizer inizializzato.")

    def _build_user_prompt(self, state: SceneState) -> str:
        template_path = self.prompts_dir / "reorder_user.txt"
        template = _load_prompt_template(template_path)

        room_bounds = state.room_bounds
        scene_json = _build_scene_json_for_llm(state)
        footprint_table = _build_footprint_table(state)

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
            json_instr = (
                "\n\nOUTPUT FORMAT:"
                "\nReturn the requested JSON dictionary format exactly as specified in the system instructions."
                "\nOnly include ROOT movable objects (is_child=false). DO NOT include children — they are repositioned automatically."
                "\nDO NOT add markdown, comments or extra fields."
            )
            return prompt + footprint_table + json_instr

        return (
            "You are an interior designer. Reorganize this 3D scene JSON professionally. "
            "Return ONLY the updated JSON.\n\n"
            f"{scene_json}"
        )

    def reorganize(self, disordered_state: SceneState) -> SceneState:
        """
        Riordina la scena tramite LLM (solo testo).

        Args:
            disordered_state: Stato disordinato della scena.

        Returns:
            SceneState riordinato, validato e sanitizzato.
        """
        logger.info(
            "Avvio riordino LLM per '%s'. Root movibili: %d.",
            disordered_state.scene_name,
            len(disordered_state.root_movable_objects),
        )

        system_prompt = _load_prompt_template(self.prompts_dir / "reorder_system.txt")
        user_prompt = self._build_user_prompt(disordered_state)

        logger.debug(
            "Prompt lunghezze — system: %d, user: %d.",
            len(system_prompt), len(user_prompt),
        )

        try:
            llm_output = self.client.call_text(system_prompt, user_prompt)
        except GeminiParsingError as exc:
            logger.error("Parsing LLM fallito: %s. Stato disordinato restituito.", exc)
            return SceneState(
                scene_name=disordered_state.scene_name,
                objects=copy.deepcopy(disordered_state.objects),
                room_bounds=disordered_state.room_bounds,
                pipeline_step="reordered_failed",
                metadata={"error": str(exc)},
            )

        reordered_state = _validate_and_sanitize_llm_output(llm_output, disordered_state)
        logger.info("Riordino LLM completato per '%s'.", reordered_state.scene_name)
        return reordered_state

    def reorganize_with_image(
        self,
        disordered_state: SceneState,
        image_path: Path,
    ) -> SceneState:
        """
        Riordina la scena con prompt multimodale (testo + immagine viewport).

        Args:
            disordered_state: Stato disordinato della scena.
            image_path:       Screenshot della viewport corrente.

        Returns:
            SceneState riordinato, validato e sanitizzato.
        """
        logger.info(
            "Avvio riordino LLM multimodale per '%s'. Immagine: %s",
            disordered_state.scene_name, image_path,
        )

        system_prompt = _load_prompt_template(self.prompts_dir / "reorder_system.txt")
        user_prompt_body = self._build_user_prompt(disordered_state)

        vision_instructions = (
            "\n\n=== VISUAL CONTEXT ===\n"
            "Along with this textual data, you are receiving an image showing "
            "the CURRENT state of the room from a representative viewpoint.\n"
            "Use the image to understand:\n"
            "- The actual orientation and layout of walls\n"
            "- The position of doors, windows, and other openings\n"
            "- The real proportions of the room\n"
            "- Natural focal points (windows for desks, walls for beds, etc.)\n"
            "\nBase your reorganization on BOTH the numerical data AND the visual context."
        )

        combined_prompt = f"{system_prompt}\n\n{user_prompt_body}\n{vision_instructions}"

        try:
            llm_output = self.client.call_vision(image_path=image_path, user_prompt=combined_prompt)
        except GeminiParsingError as exc:
            logger.error("Parsing LLM multimodale fallito: %s. Stato disordinato restituito.", exc)
            return SceneState(
                scene_name=disordered_state.scene_name,
                objects=copy.deepcopy(disordered_state.objects),
                room_bounds=disordered_state.room_bounds,
                pipeline_step="reordered_failed",
                metadata={"error": str(exc), "mode": "multimodal"},
            )

        reordered_state = _validate_and_sanitize_llm_output(llm_output, disordered_state)
        logger.info("Riordino LLM multimodale completato per '%s'.", reordered_state.scene_name)
        return reordered_state