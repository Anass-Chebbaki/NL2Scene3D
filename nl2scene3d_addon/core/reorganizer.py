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

La chiamata vera al modello e l'I/O vivono altrove (negli operatori): qui dentro
non si tocca ne' Blender ne' la rete, cosi' la logica resta verificabile.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Optional

from .geometry import group_aabb_xy, is_finite_float, penetration_vector, snap_rotation_90
from .models import RoomBounds, SceneObject, SceneState
from .randomizer import _clamp_parent_group_location, apply_rigid_transform
from .settings import CONST, Constants

logger = logging.getLogger(__name__)

_SKIP_TYPES = ("CAMERA", "LIGHT")


# ---------------------------------------------------------------------------
# 1) Costruzione della richiesta e del prompt
# ---------------------------------------------------------------------------

def _descendants(root: SceneObject, by_name: dict[str, SceneObject]) -> list[SceneObject]:
    """Tutti i discendenti (figli, nipoti, ...) di un root, in ordine di visita."""
    out: list[SceneObject] = []

    def rec(name: str) -> None:
        node = by_name.get(name)
        if node is None:
            return
        for cname in node.children:
            child = by_name.get(cname)
            if child is not None:
                out.append(child)
                rec(cname)

    rec(root.name)
    return out


def _footprint_item(o: SceneObject) -> dict:
    """Voce {name,x,y,w,d} con centro geometrico e impronta XY (rotazione inclusa)."""
    cx, cy = o.transform.geometric_center_xy()
    x_min, x_max, y_min, y_max = o.transform.aabb_xy(margin=0.0)
    return {
        "name": o.name,
        "x": round(cx, 3), "y": round(cy, 3),
        "w": round(x_max - x_min, 3), "d": round(y_max - y_min, 3),
    }


def build_request(state: SceneState) -> dict:
    """
    Costruisce il payload JSON per il modello.

    - movable_objects: SOLO i root mobili (quelli che il modello deve riposizionare).
      x/y = centro del root; w/d = impronta XY dell'INTERO GRUPPO (root + figli),
      cosi' il modello lascia spazio anche per cio' che e' attaccato. Ogni root puo'
      avere un campo "contains" con i figli (read-only): il modello sa che ci sono e
      a chi appartengono, ma NON deve produrre una posizione per loro (seguono il padre).
    - fixed_objects: ostacoli fissi NON strutturali (la stanza/i muri sono esclusi),
      inclusi gli oggetti che l'utente ha marcato "fisso" a mano.
    """
    rb = state.room_bounds
    by_name = {o.name: o for o in state.objects}
    fixed: list[dict] = []
    movable: list[dict] = []

    for o in state.objects:
        if o.object_type in _SKIP_TYPES:
            continue

        if o.is_movable and o.is_root:
            cx, cy = o.transform.geometric_center_xy()
            descendants = _descendants(o, by_name)
            # Impronta dell'intero gruppo alla posa CORRENTE (transform identita').
            gx_min, gx_max, gy_min, gy_max = group_aabb_xy(
                o, list(o.transform.location), o.transform.rotation_euler[2],
                descendants, margin=0.0,
            )
            item = {
                "name": o.name,
                "x": round(cx, 3), "y": round(cy, 3),
                "w": round(gx_max - gx_min, 3), "d": round(gy_max - gy_min, 3),
                "rotation_deg": int(round(math.degrees(o.transform.rotation_euler[2]))) % 360,
            }
            if descendants:
                # Figli come contesto read-only (il modello non li piazza).
                item["contains"] = [_footprint_item(c) for c in descendants]
            movable.append(item)

        elif (not o.is_movable) and o.category != "structural":
            # Ostacolo fisso (auto o marcato a mano dall'utente). La stanza/muri esclusi.
            fixed.append(_footprint_item(o))

    return {
        "room": {
            "x_min": round(rb.x_min, 3), "x_max": round(rb.x_max, 3),
            "y_min": round(rb.y_min, 3), "y_max": round(rb.y_max, 3),
        },
        "fixed_objects": fixed,
        "movable_objects": movable,
    }


PROMPT_TEMPLATE = """# Interior Design Layout Optimization Task

You are a professional interior designer working on indoor scenes of ANY type:
bedrooms, living rooms, kitchens, bathrooms, offices, dining rooms, retail or
commercial spaces, and any other indoor environment made of furniture and props.
Do not assume a specific room type in advance: infer it from the data and images.

You are provided with:
- A JSON description of the scene (schema below).
- A perspective image of the room.
- A top-down (floor plan) image of the room.

The current object positions have been intentionally randomized and MUST NOT be
considered a valid layout. Your task is to design a COMPLETELY NEW arrangement of
the objects from scratch. Do not make small adjustments to the current layout:
rethink the whole organization and produce the most realistic, functional layout
possible, as a human interior designer would.

## How to read the scene JSON

Interpret every field exactly as defined here.

- `room`: the rectangular floor boundary, in meters (`x_min, x_max, y_min, y_max`).
  Every object must stay fully inside it.
- `fixed_objects`: obstacles you must NOT move and must NOT overlap. They never
  appear in your output. Treat them as hard, immovable constraints.
- `movable_objects`: the ONLY objects you reposition. For each one you output a
  new `x`, `y`, and `rotation_deg`.
  - `x`, `y`: the object CENTER, in meters (current/randomized value, to replace).
  - `w`, `d`: the footprint of the WHOLE GROUP, i.e. the object PLUS everything in
    its `contains`. Treat each group as a single rigid block of size `w` x `d`.
  - `rotation_deg`: current rotation; your new value must be 0, 90, 180 or 270.
  - `contains` (optional): child objects rigidly attached to this object (for
    example a desk's chair, monitor and keyboard, or a bed's nightstand and lamp).
    They move together with their parent. DO NOT output placements for them:
    placing the parent already places them.


What you MAY change: only `x`, `y` and `rotation_deg` of each `movable_objects`
entry. What you MUST preserve: all object names, all `w`/`d` dimensions, any
height, and the parent-child grouping. Do not add, remove, rename, resize, merge
or split objects.

## Goal

Produce a layout that is realistic, functional, organized, visually balanced and
plausible for a real-world environment.

## Design Principles

First infer the room's function from object names, dimensions and the images,
then apply the conventions appropriate to THAT function.

### Functional grouping
Identify the likely purpose of each object and organize the space into coherent
functional areas. Objects that belong together should sit near one another.

### Furniture placement
Place furniture where the room's function dictates. Many rooms anchor large
pieces along the walls and keep the center clear; but some layouts have
intentionally central elements (a dining or conference table, a kitchen island,
a retail display). Decide based on the inferred room type, not by default.

### Accessibility
Keep natural circulation paths. People should move comfortably between areas
without obstacles blocking doorways or passages.

### Spatial coherence
Position objects so their relationships make sense. Related objects should feel
intentionally associated, not scattered.

### Visual order
Prefer clean alignments and structured arrangements. Avoid arbitrary orientations
or placements that create visual clutter.

### Space usage
Use the available floor area efficiently. Avoid both overcrowding and large,
purposeless empty zones.

## Geometric Constraints (Mandatory)

- Every object must remain completely inside the `room` boundaries.
- No movable object may overlap another movable object.
- No movable object may overlap any fixed object.
- Respect the `w`, `d` group footprint; do not modify dimensions.
- `x`, `y` are object centers, in meters.
- Do not invent height (`z`) values; height is out of scope.

## Rotation Constraints

`rotation_deg` may only be 0, 90, 180 or 270. Any other value is invalid.

## Optimization Strategy

Before answering: (1) infer the room type from JSON and images; (2) identify the
primary furniture and the focal areas; (3) consider several plausible layouts;
(4) compare them on functionality, accessibility, realism and space efficiency;
(5) pick the best; (6) verify every geometric and rotation constraint holds.

## Output Requirements

Return ONLY a raw JSON object. No explanations, no comments, no markdown, no code
fences, no extra text. Use exactly this structure:

```json
{
  "placements": [
    { "name": "<object_name>", "x": <float>, "y": <float>, "rotation_deg": <int> }
  ]
}
```

Output exactly one placement for every entry in `movable_objects`, and never for
objects listed under `contains` or in `fixed_objects`."""


def build_prompt(state: SceneState) -> str:
    """
    Prompt completo = template generico fisso (PROMPT_TEMPLATE) + dati della scena
    in JSON, appesi sotto. Il JSON viene rigenerato a ogni chiamata da build_request,
    quindi rispecchia sempre lo stato corrente (root + contains + fissi).
    """
    payload = json.dumps(build_request(state), ensure_ascii=False, indent=2)
    return f"{PROMPT_TEMPLATE}\n\n## JSON Scene data:\n```json\n{payload}\n```\n"


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
