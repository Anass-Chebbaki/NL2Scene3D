# nl2scene3d/scene_reorganizer.py
"""
Riordino della scena tramite LLM (Gemini) — modalità text-only.

Responsabilità:
    - Costruire un JSON piatto con SOLO gli oggetti padre movibili e le loro
      dimensioni/relazioni, abbattendo i token necessari all'LLM.
    - Costruire un JSON con gli elementi strutturali fissi (muri, finestre, porte)
      per dare all'LLM pieno contesto della stanza.
    - Inviare il prompt e ricevere le nuove coordinate (X, Y, Rz).
    - Validare/sanitizzare l'output (bounds clamp di gruppo, Z lock, snap a 90°).
    - Spostare i figli con trasformazione rigida rispetto al parent e preservarli
      senza MAI staccarli (clamping e collisioni applicate al gruppo parent+figli).
    - Risolvere collisioni post-LLM con vettore MTV applicato rigidamente al gruppo.

Principi fondamentali:
    - La Z non viene MAI modificata: si prende sempre dalla scena originale.
    - L'LLM riceve SOLO i root movibili (is_child=false): i figli sono gestiti
      internamente tramite trasformazione rigida e non devono essere menzionati.
    - Il grouping (parent/children) è già calcolato su ogni SceneObject da
      SceneLoader.extract_scene_state() — non viene ricalcolato qui.
"""
from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path
from nl2scene3d.gemini_client import GeminiClient, GeminiParsingError
from nl2scene3d.models import SceneObject, SceneState, Transform, RoomBounds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Margine dai muri per il reorder.
# DEVE essere uguale al randomizer (RandomizerConfig.wall_margin = 0.20):
# i room_bounds arrivano alla faccia ESTERNA dei muri (includono lo spessore),
# quindi un margine troppo piccolo (es. 0.10) lascia gli oggetti a filo o dentro
# la faccia interna del muro -> la scrivania "buca" la parete.
# Tienilo allineato a wall_margin_meters in settings.toml.
# ---------------------------------------------------------------------------
REORDER_WALL_MARGIN = 0.20


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _load_prompt_template(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Template di prompt non trovato: {prompt_path}")
    with open(prompt_path, encoding="utf-8") as fh:
        return fh.read()


def _build_flat_json_for_llm(state: SceneState) -> str:
    """
    Costruisce un JSON piatto con gli oggetti ROOT movibili.
    
    Ogni oggetto include:
    - name, category, posizione centro (x, y), rotazione (rz_deg)
    - Dimensioni (w, d, h)
    - AABB 2D dopo rotazione (x_min, x_max, y_min, y_max)
    - Margini richiesti: wall_margin (0.10 m da muri), collision_margin (0.05 m da altri oggetti)
    """
    entries = []

    for obj in state.objects:
        if not obj.is_movable:
            continue
        if obj.parent is not None:
            continue  # figlio: il LLM non lo vede

        rz_rad = obj.transform.rotation_euler[2]
        rz_deg = round(math.degrees(rz_rad) % 360.0, 1)

        # Calcola AABB 2D dell'oggetto con la sua rotazione corrente
        x_min, x_max, y_min, y_max = obj.transform.aabb_xy(margin=0.0)

        entry: dict = {
            "name": obj.name,
            "category": obj.category,
            # Posizione del centro (in metri)
            "x": round(obj.transform.location[0], 3),
            "y": round(obj.transform.location[1], 3),
            "rz_deg": rz_deg,
            # Dimensioni base (in metri)
            "w": round(obj.transform.dimensions[0], 3),
            "d": round(obj.transform.dimensions[1], 3),
            "h": round(obj.transform.dimensions[2], 3),
            # AABB 2D esatto dopo rotazione (in metri) — uso per calcoli precisi
            "aabb_2d": {
                "x_min": round(x_min, 3),
                "x_max": round(x_max, 3),
                "y_min": round(y_min, 3),
                "y_max": round(y_max, 3),
            },
            # Margini richiesti (in metri)
            "wall_margin_required": REORDER_WALL_MARGIN,  # minima distanza dai muri della stanza
            "collision_margin_required": 0.05,  # minima distanza fra due oggetti
        }
        if obj.children:
            entry["children"] = obj.children  # solo i nomi, per info

        entries.append(entry)

    return json.dumps(entries, indent=2, ensure_ascii=False)


def _build_structural_json_for_llm(state: SceneState) -> str:
    """
    Costruisce una lista JSON degli elementi strutturali (muri, porte, finestre).
    
    NON includere questi nel riordino, ma usarli come riferimento spaziale.
    Per PORTE e FINESTRE, aggiunge una zona di esclusione (clearance zone) attorno
    a ciascuna, che deve rimanere LIBERA (no mobili in quella area).
    """
    entries = []
    structural_types = {}  # per contare le porte/finestre e marcare le zone vietate
    
    for obj in state.objects:
        if obj.is_movable or obj.category != "structural":
            continue
            
        rz_rad = obj.transform.rotation_euler[2]
        rz_deg = round(math.degrees(rz_rad) % 360.0, 1)
        
        # Calcola AABB 2D esatto
        x_min, x_max, y_min, y_max = obj.transform.aabb_xy(margin=0.0)
        
        obj_type = "wall"  # default
        clearance_zone = None  # zona di esclusione attorno a porte/finestre
        
        # Distingui tipo di elemento strutturale
        name_lower = obj.name.lower()
        if "door" in name_lower or "porta" in name_lower:
            obj_type = "door"
            # Zone di esclusione: 0.90 m davanti alla porta
            door_width = obj.transform.dimensions[0]
            door_depth = obj.transform.dimensions[1]
            # La zona di esclusione si estende 0.90 m verso l'interno della stanza dalla porta
            clearance_distance = 0.90
            
            # La porta ha un'orientazione data da rz_deg
            # Calcola la zona di esclusione dipendentemente dall'orientamento
            if abs(rz_deg % 180) < 45 or abs(rz_deg % 180 - 180) < 45:
                # Porta allineata con asse X (orientamento 0° o 180°)
                clearance_zone = {
                    "x_min": round(x_min - 0.05, 3),
                    "x_max": round(x_max + 0.05, 3),
                    "y_min": round(y_min - clearance_distance, 3),
                    "y_max": round(y_max + clearance_distance, 3),
                }
            else:
                # Porta allineata con asse Y (orientamento 90° o 270°)
                clearance_zone = {
                    "x_min": round(x_min - clearance_distance, 3),
                    "x_max": round(x_max + clearance_distance, 3),
                    "y_min": round(y_min - 0.05, 3),
                    "y_max": round(y_max + 0.05, 3),
                }
        elif "window" in name_lower or "finestra" in name_lower:
            obj_type = "window"
            # Zone di esclusione: nessun mobile alto davanti
            # Lascio info sulla zona, l'LLM eviterà di mettere cose alte lì
            clearance_zone = {
                "x_min": round(x_min - 0.10, 3),
                "x_max": round(x_max + 0.10, 3),
                "y_min": round(y_min - 0.10, 3),
                "y_max": round(y_max + 0.10, 3),
            }
        
        entry = {
            "name": obj.name,
            "type": obj_type,  # "wall", "door", or "window"
            "x": round(obj.transform.location[0], 3),
            "y": round(obj.transform.location[1], 3),
            "rz_deg": rz_deg,
            "w": round(obj.transform.dimensions[0], 3),
            "d": round(obj.transform.dimensions[1], 3),
            "h": round(obj.transform.dimensions[2], 3),
            "aabb_2d": {
                "x_min": round(x_min, 3),
                "x_max": round(x_max, 3),
                "y_min": round(y_min, 3),
                "y_max": round(y_max, 3),
            }
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
    original_child_z: float | None = None,
) -> SceneObject:
    """
    Restituisce una COPIA del figlio spostato rigidamente nel piano XY rispetto al parent.
    La Z del figlio NON cambia mai: viene ripristinata al valore `original_child_z` se fornito,
    altrimenti rimane quella corrente del child.
    """
    new_child = child.copy()

    rel_x = child.transform.location[0] - old_parent_loc[0]
    rel_y = child.transform.location[1] - old_parent_loc[1]

    d_rz = new_parent_rz - old_parent_rz
    cos_a, sin_a = math.cos(d_rz), math.sin(d_rz)

    new_child.transform.location[0] = new_parent_loc[0] + rel_x * cos_a - rel_y * sin_a
    new_child.transform.location[1] = new_parent_loc[1] + rel_x * sin_a + rel_y * cos_a
    # Z ASSOLUTAMENTE INVARIATA
    if original_child_z is not None:
        new_child.transform.location[2] = original_child_z
    else:
        new_child.transform.location[2] = child.transform.location[2]
    new_child.transform.rotation_euler[2] = (child.transform.rotation_euler[2] + d_rz) % (2 * math.pi)

    return new_child


def _group_aabb_xy(
    orig_parent: SceneObject,
    proposed_loc: list[float],
    proposed_rz: float,
    orig_children: list[SceneObject],
    margin: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Calcola l'AABB 2D combinata (XY) di un gruppo parent+figli nella posizione/rotazione proposta.
    Usa l'AABB reale di ogni membro (con la rotazione corretta) tramite la classe Transform,
    garantendo che venga sempre tenuto conto di origin_offset e rotazione Z.
    """
    old_parent_loc = orig_parent.transform.location
    old_parent_rz = orig_parent.transform.rotation_euler[2]
    d_rz = proposed_rz - old_parent_rz
    cos_a, sin_a = math.cos(d_rz), math.sin(d_rz)

    # Transform temporanea del parent
    temp_parent_tf = Transform(
        location=[proposed_loc[0], proposed_loc[1], orig_parent.transform.location[2]],
        rotation_euler=[
            orig_parent.transform.rotation_euler[0],
            orig_parent.transform.rotation_euler[1],
            proposed_rz
        ],
        dimensions=orig_parent.transform.dimensions,
        origin_offset=orig_parent.transform.origin_offset
    )
    x_min, x_max, y_min, y_max = temp_parent_tf.aabb_xy(margin=margin)

    # Estendi l'AABB con ogni figlio nella sua nuova posizione rigida
    for orig_child in orig_children:
        rel_x = orig_child.transform.location[0] - old_parent_loc[0]
        rel_y = orig_child.transform.location[1] - old_parent_loc[1]
        
        # Ruota il vettore relativo
        new_cx = proposed_loc[0] + rel_x * cos_a - rel_y * sin_a
        new_cy = proposed_loc[1] + rel_x * sin_a + rel_y * cos_a
        c_rz = (orig_child.transform.rotation_euler[2] + d_rz) % (2 * math.pi)

        temp_child_tf = Transform(
            location=[new_cx, new_cy, orig_child.transform.location[2]],
            rotation_euler=[
                orig_child.transform.rotation_euler[0],
                orig_child.transform.rotation_euler[1],
                c_rz
            ],
            dimensions=orig_child.transform.dimensions,
            origin_offset=orig_child.transform.origin_offset
        )
        cx_min, cx_max, cy_min, cy_max = temp_child_tf.aabb_xy(margin=margin)
        x_min = min(x_min, cx_min)
        x_max = max(x_max, cx_max)
        y_min = min(y_min, cy_min)
        y_max = max(y_max, cy_max)

    return x_min, x_max, y_min, y_max


def _clamp_parent_group_location(
    orig_parent: SceneObject,
    proposed_loc: list[float],
    proposed_rz: float,
    orig_children: list[SceneObject],
    room_bounds: RoomBounds,
) -> list[float]:
    """
    Clampa la posizione del parent usando l'AABB reale del gruppo (parent+figli)
    per garantire che tutto il gruppo rimanga dentro i bounds con margine di 15 cm.
    """
    wall_margin = REORDER_WALL_MARGIN
    g_x_min, g_x_max, g_y_min, g_y_max = _group_aabb_xy(
        orig_parent, proposed_loc, proposed_rz, orig_children, margin=0.0
    )
    px, py = proposed_loc[0], proposed_loc[1]

    overflow_left  = max(0.0, (room_bounds.x_min + wall_margin) - g_x_min)
    overflow_right = max(0.0, g_x_max - (room_bounds.x_max - wall_margin))
    overflow_front = max(0.0, (room_bounds.y_min + wall_margin) - g_y_min)
    overflow_back  = max(0.0, g_y_max - (room_bounds.y_max - wall_margin))

    dx = overflow_left if overflow_left > overflow_right else -overflow_right
    dy = overflow_front if overflow_front > overflow_back else -overflow_back

    return [px + dx, py + dy, proposed_loc[2]]


def _group_has_collision(
    root_obj: SceneObject,
    current_children: list[SceneObject],
    collidable: list[SceneObject],
    wall_margin: float,
    furniture_margin: float,
    room_bounds=None,
) -> bool:
    """True se il root o uno qualsiasi dei suoi figli collide con collidable o esce dai bounds."""
    if has_collision(root_obj, collidable, wall_margin=wall_margin, furniture_margin=furniture_margin, room_bounds=room_bounds):  # type: ignore[call-arg]
        return True
    for child in current_children:
        if has_collision(child, collidable, wall_margin=wall_margin, furniture_margin=furniture_margin, room_bounds=room_bounds):  # type: ignore[call-arg]
            return True
    return False


def _group_penetration_vector(
    root_obj: SceneObject,
    current_children: list[SceneObject],
    other: SceneObject,
    margin: float = 0.05,
) -> tuple[float, float]:
    """
    MTV tra l'intero gruppo candidato (root+figli) e un singolo oggetto placed.
    Restituisce lo spostamento maggiore tra tutte le coppie del gruppo vs other.
    """
    best_dx, best_dy = 0.0, 0.0
    best_pen = 0.0

    members = [root_obj] + current_children
    for member in members:
        dx, dy = penetration_vector(member, other, margin=margin)
        pen = abs(dx) + abs(dy)
        if pen > best_pen:
            best_pen = pen
            best_dx, best_dy = dx, dy

    return best_dx, best_dy


def _validate_and_sanitize_llm_output(
    llm_output: dict | list,
    original_state: SceneState,
) -> SceneState:
    """
    Valida e sanitizza l'output JSON dell'LLM.

    L'LLM restituisce una lista di oggetti ROOT con campi:
        - "name": nome esatto dell'oggetto
        - "x", "y": nuove coordinate (float)
        - "rz_deg": nuova rotazione Z in gradi (0–360)

    Regole di sanitizzazione:
    - Blocca la Z sull'originale (non viene mai modificata).
    - Snappa la rotazione Z a multipli di 90°.
    - Preserva le rotazioni X/Y originali.
    - Sposta i figli con trasformazione rigida rispetto al parent mosso.
    - Risolve collisioni post-LLM con vettore di penetrazione (MTV) applicato al gruppo.
    """
    room_bounds = original_state.room_bounds

    logger.info(f"Validating LLM output. Type: {type(llm_output).__name__}")
    if isinstance(llm_output, list):
        logger.info(f"LLM output is list with {len(llm_output)} entries")
        if len(llm_output) > 0:
            logger.debug(f"First entry: {llm_output[0]}")
    elif isinstance(llm_output, dict):
        logger.info(f"LLM output is dict with keys: {llm_output.keys()}")

    # --- Parse output (lista flat o dict con "objects") ---
    if isinstance(llm_output, list):
        llm_objects_list: list = llm_output
    elif isinstance(llm_output, dict):
        llm_objects_list = llm_output.get("objects", [])
    else:
        logger.error("Output LLM non valido (%s). Stato originale restituito.", type(llm_output))
        return original_state

    logger.info(f"Processing {len(llm_objects_list)} LLM objects")

    # Costruisce mappa: name → entry LLM
    llm_by_name: dict[str, dict] = {}
    for i, item in enumerate(llm_objects_list):
        logger.debug(f"Item {i}: type={type(item).__name__}, value={item}")
        
        if not isinstance(item, dict):
            logger.warning(
                "Item %d nell'output LLM non è un dict: %s (tipo: %s). Skipped.",
                i, item, type(item).__name__,
            )
            continue
        
        if "name" not in item:
            logger.warning(
                "Item %d nell'output LLM manca il campo 'name'. Keys: %s. Item: %s. Skipped.",
                i, list(item.keys()), item,
            )
            continue
        
        name = item["name"]
        logger.debug(f"Item {i}: name={name} (type: {type(name).__name__})")
        
        if not isinstance(name, str):
            logger.warning(
                "Item %d: campo 'name' non è una stringa: %s (tipo: %s). Skipped.",
                i, name, type(name).__name__,
            )
            continue
        
        llm_by_name[name] = item
    
    logger.info(f"Built llm_by_name map with {len(llm_by_name)} entries: {list(llm_by_name.keys())}")

    # --- Applica le posizioni LLM ai root movibili ---
    by_name_orig = {obj.name: obj for obj in original_state.objects}
    corrected: dict[str, SceneObject] = {}

    clamped_count = 0
    missing_count = 0

    for orig_obj in original_state.objects:
        # Strutturali: sempre invariati
        if not orig_obj.is_movable:
            corrected[orig_obj.name] = orig_obj.copy()
            continue

        # Figli: placeholder, sovrascritto dopo il parent
        if orig_obj.parent is not None:
            if orig_obj.name not in corrected:
                corrected[orig_obj.name] = orig_obj.copy()
            continue

        # Root movibile: leggi posizione LLM
        llm_data = llm_by_name.get(orig_obj.name)
        if llm_data is None:
            logger.warning(
                "Root movibile '%s' assente dall'output LLM. Posizione originale mantenuta.",
                orig_obj.name,
            )
            missing_count += 1
            corrected[orig_obj.name] = orig_obj.copy()
            continue

        try:
            # Accetta sia formato flat {"x","y","rz_deg"} sia {"location","rotation_euler"}
            if "x" in llm_data and "y" in llm_data:
                new_x = float(llm_data["x"])
                new_y = float(llm_data["y"])
                rz_deg = float(llm_data.get("rz_deg", math.degrees(orig_obj.transform.rotation_euler[2])))
                rz_rad = rz_deg * (math.pi / 180.0)
                new_loc = [new_x, new_y, orig_obj.transform.location[2]]
                new_rot = [
                    orig_obj.transform.rotation_euler[0],
                    orig_obj.transform.rotation_euler[1],
                    rz_rad,
                ]
            else:
                # Fallback al formato verboso {location, rotation_euler}
                new_loc = list(llm_data.get("location", orig_obj.transform.location))
                new_rot = list(llm_data.get("rotation_euler", orig_obj.transform.rotation_euler))
                if len(new_loc) != 3 or len(new_rot) != 3:
                    raise ValueError("location o rotation_euler non hanno 3 componenti.")
                new_loc = [float(v) for v in new_loc]
                new_rot = [float(v) for v in new_rot]
                # Blocca Z
                new_loc[2] = orig_obj.transform.location[2]
                new_rot[0] = orig_obj.transform.rotation_euler[0]
                new_rot[1] = orig_obj.transform.rotation_euler[1]

            if not all(is_finite_float(v) for v in new_loc + new_rot):
                raise ValueError("Valori non finiti rilevati.")

            # Snap Rz a 90°
            new_rot[2] = snap_rotation_90(new_rot[2])

        except (TypeError, ValueError, KeyError) as exc:
            logger.warning(
                "Coordinate non valide per '%s': %s. Posizione originale.",
                orig_obj.name, exc,
            )
            corrected[orig_obj.name] = orig_obj.copy()
            continue

        # Clamp ai bounds a livello di gruppo (parent + figli)
        if room_bounds is not None:
            child_objs = [by_name_orig[c] for c in orig_obj.children if c in by_name_orig]
            clamped = _clamp_parent_group_location(orig_obj, new_loc, new_rot[2], child_objs, room_bounds)
            if clamped != new_loc:
                logger.debug("'%s' (Gruppo): clamp %s → %s.", orig_obj.name, new_loc, clamped)
                new_loc = clamped
                clamped_count += 1

        # Z ASSOLUTAMENTE INVARIATA: ripristina dal valore originale (ignora qualsiasi cosa l'LLM abbia proposto)
        new_loc[2] = orig_obj.transform.location[2]

        new_obj = orig_obj.copy()
        new_obj.transform = Transform(
            location=new_loc,
            rotation_euler=new_rot,
            dimensions=list(orig_obj.transform.dimensions),
            origin_offset=list(orig_obj.transform.origin_offset),
        )
        corrected[orig_obj.name] = new_obj

        # Sposta i figli con trasformazione rigida (Z di ogni figlio ripristinata all'originale)
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
                original_child_z=orig_child.transform.location[2],  # Z INVARIATA
            )
            corrected[child_name] = new_child

    logger.info(
        "Validazione output LLM: %d root processati, %d clamped, %d mancanti.",
        len([o for o in original_state.objects if o.is_movable and o.parent is None]),
        clamped_count,
        missing_count,
    )

    # --- Risoluzione collisioni post-LLM (MTV) con spostamento rigido dei figli ---
    main_cats = {"bed", "table", "storage", "seating_large", "seating_small", "furniture"}

    final_list: list[SceneObject] = []
    # Prima i non movibili
    for obj in original_state.objects:
        if not obj.is_movable:
            final_list.append(corrected[obj.name])

    jitter_resolved = 0
    jitter_failed = 0

    def _group_volume(o: SceneObject) -> float:
        """Volume del gruppo (root + figli): le ancore grandi vanno risolte prima."""
        d = o.transform.dimensions
        vol = d[0] * d[1] * d[2]
        for c in o.children:
            cc = by_name_orig.get(c)
            if cc:
                dd = cc.transform.dimensions
                vol += dd[0] * dd[1] * dd[2]
        return vol

    # Risolviamo le collisioni partendo dai gruppi piu' grandi (come il randomizer):
    # cosi' un oggetto piccolo piazzato prima non blocca un mobile grande.
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
            max_iter = 20
            # Figli correnti del candidato (già spostati rigidamente dopo il LLM)
            current_children = [corrected[c] for c in orig_obj.children if c in corrected]

            for i in range(max_iter):
                # Controlla l'intero gruppo (root + figli) per collisioni
                if not _group_has_collision(obj, current_children, collidable, wall_margin=REORDER_WALL_MARGIN, furniture_margin=0.02, room_bounds=room_bounds):
                    break
                moved = False
                for other in collidable:
                    # MTV calcolato sul membro del gruppo con la penetrazione maggiore
                    dx, dy = _group_penetration_vector(obj, current_children, other, margin=0.02)
                    if dx != 0.0 or dy != 0.0:
                        # Sposta il parent
                        obj.transform.location[0] += dx
                        obj.transform.location[1] += dy

                        # Clampa l'intero gruppo ai bounds della stanza
                        # Usa obj (parent corrente) e current_children (già aggiornati post-MTV)
                        if room_bounds is not None:
                            clamped = _clamp_parent_group_location(
                                obj,
                                obj.transform.location,
                                obj.transform.rotation_euler[2],
                                current_children,
                                room_bounds,
                            )
                            obj.transform.location = clamped

                        # Aggiorna rigidamente tutti i figli rispetto alla nuova posizione finale del parent
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
                        # Aggiorna la lista figli correnti per il prossimo controllo
                        current_children = [corrected[c] for c in orig_obj.children if c in corrected]
                        moved = True
                        break
                if not moved:
                    break
            else:
                jitter_failed += 1
                logger.warning(
                    "Collisione irrisolvibile per '%s' dopo %d iterazioni.", obj.name, max_iter
                )

            if not _group_has_collision(obj, current_children, collidable, wall_margin=REORDER_WALL_MARGIN, furniture_margin=0.02, room_bounds=room_bounds):
                if i > 0:
                    jitter_resolved += 1

        # GARANZIA DI CONTENIMENTO: ogni gruppo (anche decorazioni/luci non in
        # main_cats) viene riportato dentro i bounds usando l'AABB del gruppo
        # (parent + figli). Evita che un figlio sporga dai muri rispetto al parent.
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

        # Aggiungi i figli preservando l'attaccamento rigido e le nuove posizioni finali
        for child_name in orig_obj.children:
            if child_name in corrected:
                final_list.append(corrected[child_name])

    if jitter_resolved or jitter_failed:
        logger.info(
            "Collisioni post-LLM: %d risolte, %d irrisolvibili.", jitter_resolved, jitter_failed
        )

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
    """

    def __init__(self, client: GeminiClient, prompts_dir: Path) -> None:
        self.client = client
        self.prompts_dir = prompts_dir
        logger.info("SceneReorganizer inizializzato.")

    def _build_user_prompt(self, state: SceneState) -> str:
        template_path = self.prompts_dir / "reorder_user.txt"
        template = _load_prompt_template(template_path)

        room_bounds = state.room_bounds
        flat_json = _build_flat_json_for_llm(state)
        structural_json = _build_structural_json_for_llm(state)

        root_objects = state.root_movable_objects
        n_roots = len(root_objects)

        if room_bounds is not None:
            # Sottrai un margine di sicurezza da ogni lato per evitare incastri coi muri
            wall_safety_margin = REORDER_WALL_MARGIN
            safe_x_min = room_bounds.x_min + wall_safety_margin
            safe_x_max = room_bounds.x_max - wall_safety_margin
            safe_y_min = room_bounds.y_min + wall_safety_margin
            safe_y_max = room_bounds.y_max - wall_safety_margin
            
            # IMPORTANTE: Ordine corretto per evitare conflitti con parentesi graffe nel JSON:
            # 1. Fai .format() con i placeholder numerici (non contengono i JSON)
            # 2. Poi sostituisci i placeholder temporanei con i JSON
            prompt = template.format(
                scene_name=state.scene_name,
                x_min=safe_x_min,
                x_max=safe_x_max,
                y_min=safe_y_min,
                y_max=safe_y_max,
                room_width=safe_x_max - safe_x_min,
                room_depth=safe_y_max - safe_y_min,
                room_height=room_bounds.height,
                n_roots=n_roots,
            )
            
            # Ora sostituisci i placeholder temporanei "sicuri" con i JSON veri
            prompt = prompt.replace("###FLAT_JSON###", flat_json)
            prompt = prompt.replace("###STRUCTURAL_JSON###", structural_json)
        else:
            prompt = (
                "You are an interior designer. Reorganize this 3D scene following "
                "professional layout principles. Return ONLY the updated JSON array.\n\n"
                f"{flat_json}"
            )

        return prompt

    def reorganize(self, disordered_state: SceneState) -> SceneState:
        """
        Riordina la scena tramite LLM (solo testo, JSON piatto con root objects).
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