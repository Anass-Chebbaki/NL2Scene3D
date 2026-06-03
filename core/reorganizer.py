# nl2scene3d/core/reorganizer.py
"""
Riorganizzazione assistita dall'LLM - parte PURA (niente bpy, niente rete).

Tre responsabilita', tutte testabili da riga di comando:
  1. build_request / build_prompt: dai un SceneState, produce i dati (JSON) e il
     testo del prompt da inviare al modello. Vengono elencati SOLO gli oggetti
     mobili root (i figli seguono il padre) piu' i confini stanza e gli ostacoli
     fissi come contesto.
  2. extract_json: estrae in modo robusto l'oggetto JSON dalla risposta del
     modello, che spesso e' "sporca" (testo prima/dopo, recinti ```).
  3. sanitize_response: valida e mette in sicurezza le posizioni proposte
     dall'LLM -> dentro i muri, niente sovrapposizioni (risolte con MTV), Z mai
     modificata, figli spostati rigidamente col padre. Ritorna un nuovo SceneState.

La chiamata vera al modello (Ollama) e l'I/O vivono altrove (Step 4): qui dentro
non si tocca ne' Blender ne' la rete, cosi' la logica resta verificabile.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Optional

from .geometry import is_finite_float, penetration_vector, snap_rotation_90
from .models import RoomBounds, SceneObject, SceneState
from .randomizer import _clamp_parent_group_location, apply_rigid_transform
from .settings import CONST, Constants

logger = logging.getLogger(__name__)

_SKIP_TYPES = ("CAMERA", "LIGHT")


# ---------------------------------------------------------------------------
# 1) Costruzione della richiesta e del prompt
# ---------------------------------------------------------------------------

def build_request(state: SceneState) -> dict:
    """
    Costruisce il payload JSON per il modello: confini stanza, ostacoli fissi e
    oggetti mobili ROOT (i figli non si elencano, seguono il padre).

    Coordinate = centro geometrico in metri; w/d = impronta XY attuale (tiene
    conto della rotazione corrente).
    """
    rb = state.room_bounds
    fixed: list[dict] = []
    movable: list[dict] = []

    for o in state.objects:
        if o.object_type in _SKIP_TYPES:
            continue
        cx, cy = o.transform.geometric_center_xy()
        x_min, x_max, y_min, y_max = o.transform.aabb_xy(margin=0.0)
        item = {
            "name": o.name,
            "x": round(cx, 3), "y": round(cy, 3),
            "w": round(x_max - x_min, 3), "d": round(y_max - y_min, 3),
        }
        if o.is_movable and o.is_root:
            item["rotation_deg"] = int(round(math.degrees(o.transform.rotation_euler[2]))) % 360
            movable.append(item)
        elif (not o.is_movable) and o.category != "structural":
            fixed.append(item)  # ostacolo fisso (la stanza/i muri sono esclusi)

    return {
        "room": {
            "x_min": round(rb.x_min, 3), "x_max": round(rb.x_max, 3),
            "y_min": round(rb.y_min, 3), "y_max": round(rb.y_max, 3),
        },
        "fixed_objects": fixed,
        "movable_objects": movable,
    }


DEFAULT_INSTRUCTION = (
    "Sei un interior designer esperto. Le posizioni attuali degli oggetti sono "
    "DISORDINATE (sono state mescolate apposta): il tuo compito e' produrre una "
    "disposizione COMPLETAMENTE NUOVA, ordinata, realistica e funzionale. "
    "Allinea i mobili ai muri quando ha senso, lascia spazio di passaggio libero "
    "al centro, raggruppa gli oggetti correlati e crea una stanza in cui si vive "
    "bene. NON limitarti a piccoli aggiustamenti: ripensa il layout."
)

# Regole tecniche SEMPRE appese (anche se l'utente personalizza l'istruzione):
# garantiscono il contratto col resto della pipeline.
TECHNICAL_RULES = (
    "Regole vincolanti:\n"
    "- Ogni oggetto deve restare dentro i confini indicati in \"room\".\n"
    "- Niente sovrapposizioni: ne' tra oggetti mobili, ne' con gli \"fixed_objects\".\n"
    "- Puoi ruotare un oggetto solo di multipli di 90 gradi (rotation_deg in 0, 90, 180, 270).\n"
    "- NON modificare l'altezza: la z non esiste in questi dati e non va inventata.\n"
    "- Le coordinate (x, y) sono il CENTRO dell'oggetto, in metri.\n\n"
    "Rispondi ESCLUSIVAMENTE con un oggetto JSON, senza testo prima o dopo, "
    "in questo formato esatto:\n"
    "{\"placements\":[{\"name\":\"<nome>\",\"x\":<float>,\"y\":<float>,"
    "\"rotation_deg\":<int>}]}\n"
    "Includi una voce per ogni oggetto presente in \"movable_objects\"."
)


def build_prompt(state: SceneState, instruction: Optional[str] = None) -> str:
    """
    Prompt completo = istruzione (personalizzabile dall'utente) + regole tecniche
    fisse + dati della scena in JSON.

    Se `instruction` e' None o vuota, usa DEFAULT_INSTRUCTION. Le regole tecniche
    e il formato JSON vengono SEMPRE aggiunti, cosi' l'utente non puo' rompere il
    contratto modificando il testo.
    """
    instr = (instruction or "").strip() or DEFAULT_INSTRUCTION
    payload = json.dumps(build_request(state), ensure_ascii=False, indent=2)
    return f"{instr}\n\n{TECHNICAL_RULES}\n\nDati della scena:\n{payload}"


# ---------------------------------------------------------------------------
# 2) Estrazione robusta del JSON dalla risposta del modello
# ---------------------------------------------------------------------------

def extract_json(text) -> Optional[dict]:
    """
    Estrae il primo oggetto JSON ben bilanciato da un testo potenzialmente
    sporco (recinti ```json, testo prima/dopo). Ritorna un dict o None.
    """
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None

    s = text.strip()
    # Rimuove i recinti di codice ```json ... ``` se presenti.
    if "```" in s:
        s = s.replace("```json", "```")
        parts = s.split("```")
        # Prende il pezzo che contiene una graffa.
        for part in parts:
            if "{" in part:
                s = part
                break

    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = s[start:i + 1]
                try:
                    return json.loads(blob)
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


# ---------------------------------------------------------------------------
# 3) Sanificazione: dentro i muri, niente collisioni (MTV), Z intatta
# ---------------------------------------------------------------------------

def _clamp_object(obj: SceneObject, rb: RoomBounds, wall_margin: float) -> None:
    """Riporta l'AABB (ruotato) dell'oggetto dentro i muri. Solo XY."""
    x_min, x_max, y_min, y_max = obj.transform.aabb_xy(margin=0.0)
    dx = 0.0
    if x_min < rb.x_min + wall_margin:
        dx = (rb.x_min + wall_margin) - x_min
    elif x_max > rb.x_max - wall_margin:
        dx = (rb.x_max - wall_margin) - x_max
    dy = 0.0
    if y_min < rb.y_min + wall_margin:
        dy = (rb.y_min + wall_margin) - y_min
    elif y_max > rb.y_max - wall_margin:
        dy = (rb.y_max - wall_margin) - y_max
    obj.transform.location[0] += dx
    obj.transform.location[1] += dy


def _shift(obj: SceneObject, dx: float, dy: float, rb: RoomBounds, wall_margin: float) -> None:
    obj.transform.location[0] += dx
    obj.transform.location[1] += dy
    _clamp_object(obj, rb, wall_margin)


def _resolve_collisions(
    movable_roots: list[SceneObject],
    fixed_obstacles: list[SceneObject],
    rb: RoomBounds,
    const: Constants,
    max_iter: int = 80,
) -> int:
    """
    Risolve le sovrapposizioni a livello di oggetti root con il Minimum
    Translation Vector: gli ostacoli fissi spingono via i mobili; due mobili che
    si sovrappongono si scostano a meta' a testa. Itera finche' non ci sono piu'
    collisioni (o si esauriscono i tentativi). Ritorna il numero di iterazioni.
    """
    margin = const.collision_margin
    it = 0
    for it in range(1, max_iter + 1):
        moved = False

        # Mobili vs fissi: spinge interamente il mobile.
        for m in movable_roots:
            for f in fixed_obstacles:
                dx, dy = penetration_vector(m, f, margin)
                if dx or dy:
                    _shift(m, dx, dy, rb, const.wall_margin)
                    moved = True

        # Mobili vs mobili: spinta divisa.
        for i in range(len(movable_roots)):
            for j in range(i + 1, len(movable_roots)):
                a, b = movable_roots[i], movable_roots[j]
                dx, dy = penetration_vector(a, b, margin)
                if dx or dy:
                    _shift(a,  dx / 2.0,  dy / 2.0, rb, const.wall_margin)
                    _shift(b, -dx / 2.0, -dy / 2.0, rb, const.wall_margin)
                    moved = True

        if not moved:
            break

    return it


def sanitize_response(
    state: SceneState,
    llm_output,
    const: Constants = CONST,
) -> SceneState:
    """
    Valida e mette in sicurezza la risposta dell'LLM, producendo un nuovo
    SceneState 'reorganized'.

    llm_output puo' essere il testo grezzo del modello o un dict gia' parsato,
    nella forma {"placements": [{"name","x","y","rotation_deg"}, ...]}.

    Garanzie:
      - Z mai modificata (root e figli).
      - ogni gruppo resta dentro i muri (clamp di gruppo + clamp per-oggetto).
      - nessuna sovrapposizione residua tra root mobili / con gli ostacoli fissi.
      - i figli seguono rigidamente il padre.
      - nomi sconosciuti ignorati; oggetti senza proposta restano dov'erano.
    """
    data = extract_json(llm_output) or {}
    placements: dict[str, tuple] = {}
    raw = data.get("placements")
    if isinstance(raw, list):
        for p in raw:
            if not isinstance(p, dict):
                continue
            name = p.get("name")
            x, y = p.get("x"), p.get("y")
            rot = p.get("rotation_deg")
            if isinstance(name, str) and is_finite_float(x) and is_finite_float(y):
                placements[name] = (float(x), float(y), rot)

    new_objs = [o.copy() for o in state.objects]
    by_name = {o.name: o for o in new_objs}
    orig_by_name = {o.name: o for o in state.objects}
    rb = state.room_bounds

    movable_roots = [
        o for o in new_objs
        if o.is_movable and o.is_root and o.object_type not in _SKIP_TYPES
    ]
    fixed_obstacles = [
        o for o in new_objs
        if (not o.is_movable) and o.category != "structural" and o.object_type == "MESH"
    ]

    applied = 0

    # --- 1) Posa proposta (o originale) + rotazione + clamp di gruppo ---
    for root in movable_roots:
        orig = orig_by_name[root.name]
        orig_children = [orig_by_name[c] for c in orig.children if c in orig_by_name]
        z = orig.transform.location[2]

        if root.name in placements:
            px, py, rot = placements[root.name]
            proposed_rz = root.transform.rotation_euler[2]
            if rot is not None and is_finite_float(rot):
                proposed_rz = snap_rotation_90(math.radians(float(rot)))
            proposed_loc = [float(px), float(py), z]
            applied += 1
        else:
            proposed_loc = [orig.transform.location[0], orig.transform.location[1], z]
            proposed_rz  = orig.transform.rotation_euler[2]

        # Imposta la rotazione PRIMA del clamp, cosi' l'AABB e' quello giusto.
        root.transform.rotation_euler[2] = proposed_rz
        clamped = _clamp_parent_group_location(
            orig, proposed_loc, proposed_rz, orig_children, rb, const.wall_margin
        )
        root.transform.location[0] = clamped[0]
        root.transform.location[1] = clamped[1]
        root.transform.location[2] = z  # Z intatta.

    # --- 2) Collisioni (MTV) ---
    iters = _resolve_collisions(movable_roots, fixed_obstacles, rb, const)

    # --- 3) I figli seguono il padre (rigido, Z originale) ---
    for root in movable_roots:
        orig    = orig_by_name[root.name]
        old_loc = orig.transform.location
        old_rz  = orig.transform.rotation_euler[2]
        new_loc = root.transform.location
        new_rz  = root.transform.rotation_euler[2]
        for cname in orig.children:
            child  = by_name.get(cname)
            ochild = orig_by_name.get(cname)
            if child is None or ochild is None:
                continue
            child.transform.location       = list(ochild.transform.location)
            child.transform.rotation_euler = list(ochild.transform.rotation_euler)
            apply_rigid_transform(
                child, old_loc, old_rz, new_loc, new_rz,
                original_z=ochild.transform.location[2],
            )

    logger.info(
        "Sanificazione: %d/%d proposte applicate, collisioni risolte in %d iterazioni.",
        applied, len(movable_roots), iters,
    )

    return SceneState(
        scene_name=state.scene_name,
        objects=new_objs,
        room_bounds=rb,
        pipeline_step="reorganized",
    )
