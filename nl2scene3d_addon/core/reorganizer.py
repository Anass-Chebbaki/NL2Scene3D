# nl2scene3d/core/reorganizer.py
"""
Riorganizzazione assistita dall'LLM: modulo puro (nessuna dipendenza da bpy
o da connessioni di rete).

Il modulo e' suddiviso in tre responsabilita' distinte, tutte testabili
indipendentemente dalla riga di comando:

    1. build_request / build_prompt
       Dato un SceneState, produce il payload JSON e il testo del prompt da
       inviare al modello linguistico. Vengono elencati solo gli oggetti root
       mobili (i figli seguono il padre in modo rigido) piu' i confini della
       stanza e gli ostacoli fissi come contesto di vincolo.

    2. extract_json
       Estrae in modo robusto il primo oggetto JSON ben bilanciato dalla
       risposta del modello, che spesso contiene testo libero prima e dopo il
       JSON e/o recinti con tripli backtick.

    3. sanitize_response
       Valida e mette in sicurezza le posizioni proposte dall'LLM:
       - tiene gli oggetti dentro i muri (clamp),
       - risolve le sovrapposizioni con il Minimum Translation Vector (MTV),
       - non tocca mai la coordinata Z,
       - aggiorna i figli con una trasformazione rigida rispetto al padre.
       Restituisce un nuovo SceneState con pipeline_step="reorganized".

La chiamata effettiva al modello e l'I/O di rete vivono negli operatori, non
qui: in questo modo la logica rimane completamente verificabile.
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

# Tipi di oggetti Blender ignorati nella costruzione del payload.
_SKIP_TYPES = ("CAMERA", "LIGHT")


# ---------------------------------------------------------------------------
# 1) Costruzione della richiesta e del prompt
# ---------------------------------------------------------------------------

def _descendants(root: SceneObject, by_name: dict[str, SceneObject]) -> list[SceneObject]:
    """
    Restituisce tutti i discendenti (figli, nipoti, ...) di un oggetto root,
    in ordine di visita BFS/DFS (prima i figli diretti, poi i loro figli).

    Args:
        root:    L'oggetto radice da cui iniziare la visita.
        by_name: Dizionario {nome: SceneObject} per la risoluzione dei figli.

    Returns:
        Lista ordinata di tutti i discendenti, escluso il root stesso.
    """
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
    """
    Costruisce il dizionario descrittivo di un oggetto per il payload JSON,
    con il centro geometrico e le dimensioni dell'impronta XY (rotazione inclusa).

    Args:
        o: L'oggetto da descrivere.

    Returns:
        Dict con chiavi 'name', 'x', 'y', 'w', 'd'.
    """
    cx, cy = o.transform.geometric_center_xy()
    x_min, x_max, y_min, y_max = o.transform.aabb_xy(margin=0.0)
    return {
        "name": o.name,
        "x": round(cx, 3),
        "y": round(cy, 3),
        "w": round(x_max - x_min, 3),
        "d": round(y_max - y_min, 3),
    }


def build_request(state: SceneState) -> dict:
    """
    Costruisce il payload JSON completo da inviare al modello linguistico.

    Il payload contiene:
    - "room": i confini rettangolari della stanza in metri.
    - "fixed_objects": ostacoli fissi non strutturali che il modello non deve
      spostare e che non devono apparire nell'output. Gli elementi strutturali
      (muri, soffitto, ecc.) sono esclusi perche' gia' impliciti nei confini.
    - "movable_objects": gli unici oggetti che il modello deve riposizionare.
      Per ciascuno vengono forniti il centro corrente (x, y), l'impronta XY
      dell'INTERO GRUPPO (root + figli), la rotazione corrente e,
      opzionalmente, la lista dei figli come contesto read-only. Il modello
      non deve produrre posizioni per i figli: seguono il padre rigidamente.

    Args:
        state: Lo stato corrente della scena.

    Returns:
        Dizionario strutturato pronto per json.dumps.
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

            # Impronta dell'intero gruppo nella posa corrente.
            gx_min, gx_max, gy_min, gy_max = group_aabb_xy(
                o,
                list(o.transform.location),
                o.transform.rotation_euler[2],
                descendants,
                margin=0.0,
            )

            item = {
                "name": o.name,
                "x": round(cx, 3),
                "y": round(cy, 3),
                "w": round(gx_max - gx_min, 3),
                "d": round(gy_max - gy_min, 3),
                "rotation_deg": int(round(math.degrees(o.transform.rotation_euler[2]))) % 360,
            }

            # I figli sono contesto read-only: il modello sa che esistono ma
            # non deve produrre una posizione per loro.
            if descendants:
                item["contains"] = [_footprint_item(c) for c in descendants]

            movable.append(item)

        elif (not o.is_movable) and o.category != "structural":
            # Ostacolo fisso non strutturale (automatico o marcato dall'utente).
            fixed.append(_footprint_item(o))

    return {
        "room": {
            "x_min": round(rb.x_min, 3),
            "x_max": round(rb.x_max, 3),
            "y_min": round(rb.y_min, 3),
            "y_max": round(rb.y_max, 3),
        },
        "fixed_objects": fixed,
        "movable_objects": movable,
    }


# Template del prompt inviato al modello linguistico.
# Il testo e' in inglese per massimizzare la compatibilita' con i modelli.
PROMPT_TEMPLATE = """# Interior Design Layout Optimization Task

You are a professional interior designer working on indoor scenes of ANY type:
bedrooms, living rooms, kitchens, bathrooms, offices, dining rooms, retail or
commercial spaces, and any other indoor environment made of furniture and props.
Do not assume a specific room type in advance: infer it from the data and images.

You are provided with:
- A JSON description of the scene (schema below).
- One or more rendered images of the room: a top-down (floor plan) view and one
  or more angled views (perspective and/or isometric).

Every image has each object's name printed on top of it. Those names are exactly
the `name` values used in the JSON below: use the labels to match what you see in
the images to the objects in the data.
Each ortho image (top-down and isometric) also shows a scale bar (a labeled
segment, e.g. "0.5 m" or "1 m") indicating real-world size, and a small X/Y
axes compass (X in red, Y in green) showing world orientation. Use the scale
bar to judge real distances and the compass to read directions; the perspective
view shows only the compass, not the scale bar.

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
  appear in your output. Treat them as hard, immovable constraints. They may
  include fixtures that mark an entrance (for example door hardware): keep the
  area in front of such a fixture clear so the doorway stays usable. If
  `fixed_objects` is empty, no fixed obstacle is defined and you only need to
  respect the `room` boundary.
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
    Assembla il prompt completo da inviare al modello linguistico.

    Il prompt e' composto dal template generico (PROMPT_TEMPLATE) seguito dai
    dati JSON della scena corrente, generati da build_request. Il JSON viene
    rigenerato ad ogni chiamata, quindi rispecchia sempre lo stato attuale
    (root mobili, figli come contesto, ostacoli fissi).

    Args:
        state: Lo stato corrente della scena.

    Returns:
        Stringa contenente il prompt completo.
    """
    payload = json.dumps(build_request(state), ensure_ascii=False, indent=2)
    return f"{PROMPT_TEMPLATE}\n\n## JSON Scene data:\n```json\n{payload}\n```\n"


# ---------------------------------------------------------------------------
# 2) Estrazione robusta del JSON dalla risposta del modello
# ---------------------------------------------------------------------------

def _iter_balanced_objects(s: str):
    """
    Generatore che restituisce, nell'ordine in cui compaiono, tutte le
    sottostringhe '{...}' con le graffe bilanciate trovate in `s`, ignorando
    le graffe che si trovano dentro stringhe JSON.

    Tollera testo libero e recinti ```` ```json ```` perche' i backtick non
    sono graffe e quindi non interferiscono con lo scanner.
    """
    depth  = 0
    in_str = False
    escape = False
    start  = -1

    for i, ch in enumerate(s):
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
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield s[start:i + 1]
                    start = -1


def extract_json(text) -> Optional[dict]:
    """
    Estrae l'oggetto JSON utile da un testo potenzialmente "sporco", che puo'
    contenere testo libero prima e/o dopo il JSON e recinti con tripli backtick.

    A differenza di un semplice split sui recinti, lo scanner cerca TUTTI gli
    oggetti '{...}' ben bilanciati e poi sceglie quello rilevante: se piu' di
    uno e' valido (es. una graffa nel preambolo come "il {layout} richiesto")
    viene preferito il primo che contiene la chiave "placements", altrimenti il
    primo che si parsa correttamente.

    Args:
        text: Stringa grezza dalla risposta del modello, oppure dict gia' parsato.

    Returns:
        L'oggetto JSON come dict Python, oppure None se non e' stato possibile
        estrarne uno valido.
    """
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None

    first_valid: Optional[dict] = None
    for blob in _iter_balanced_objects(text):
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        if "placements" in obj:
            return obj  # match esatto sul payload atteso
        if first_valid is None:
            first_valid = obj

    return first_valid


# ---------------------------------------------------------------------------
# 3) Sanificazione: dentro i muri, niente collisioni (MTV), Z intatta
# ---------------------------------------------------------------------------

def _clamp_object(obj: SceneObject, rb: RoomBounds, wall_margin: float) -> None:
    """
    Riporta l'oggetto dentro i confini della stanza agendo sulla sua
    location XY, calcolando l'AABB ruotato per tenere conto dell'orientamento.
    La coordinata Z non viene mai modificata.

    Args:
        obj:         L'oggetto da correggere.
        rb:          I confini della stanza.
        wall_margin: Margine minimo da mantenere rispetto ai muri (in metri).
    """
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
    """
    Sposta un oggetto di (dx, dy) e poi lo riclampa dentro i muri.

    Args:
        obj:         L'oggetto da spostare.
        dx:          Spostamento sull'asse X in metri.
        dy:          Spostamento sull'asse Y in metri.
        rb:          I confini della stanza.
        wall_margin: Margine minimo dai muri.
    """
    obj.transform.location[0] += dx
    obj.transform.location[1] += dy
    _clamp_object(obj, rb, wall_margin)


def _footprint_area(o: SceneObject) -> float:
    """
    Calcola l'area dell'impronta XY dell'oggetto (larghezza x profondita').
    Usata per decidere quale oggetto e' "piu' grande" in caso di collisione.

    Args:
        o: L'oggetto di cui calcolare l'area.

    Returns:
        Area in metri quadrati.
    """
    d = o.transform.dimensions
    return float(d[0]) * float(d[1])


def _resolve_collisions(
    movable_roots: list[SceneObject],
    fixed_obstacles: list[SceneObject],
    rb: RoomBounds,
    const: Constants,
    max_iter: int = 80,
) -> int:
    """
    Risolve le sovrapposizioni tra oggetti root mobili e tra mobili e
    ostacoli fissi, usando il Minimum Translation Vector (MTV).

    Strategia:
    - Gli ostacoli fissi spingono interamente il mobile sovrapposto.
    - Due mobili sovrapposti si scostano di meta' ciascuno.
    Il processo itera fino alla convergenza o al raggiungimento di max_iter.

    Args:
        movable_roots:    Lista degli oggetti root mobili.
        fixed_obstacles:  Lista degli ostacoli fissi.
        rb:               I confini della stanza.
        const:            Costanti di configurazione (collision_margin, wall_margin).
        max_iter:         Numero massimo di iterazioni.

    Returns:
        Il numero di iterazioni effettivamente eseguite.
    """
    margin = const.collision_margin
    it = 0

    while it < max_iter:
        it += 1
        moved = False

        # Mobili vs fissi: il mobile viene spostato interamente.
        for m in movable_roots:
            for f in fixed_obstacles:
                dx, dy = penetration_vector(m, f, margin)
                if dx or dy:
                    _shift(m, dx, dy, rb, const.wall_margin)
                    moved = True

        # Mobili vs mobili: la spinta e' divisa a meta'.
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


def _propagate_to_descendants(
    parent_name:    str,
    old_parent_loc: list[float],
    old_parent_rz:  float,
    new_parent_loc: list[float],
    new_parent_rz:  float,
    by_name:        dict[str, SceneObject],
    orig_by_name:   dict[str, SceneObject],
) -> None:
    """
    Propaga ricorsivamente la trasformazione rigida del padre a TUTTI i suoi
    discendenti (figli, nipoti, ...), non solo ai figli diretti.

    Per ogni discendente:
      - ripristina la posa originale (cosi' la trasformazione e' sempre
        relativa al delta padre, mai cumulativa tra applicazioni successive);
      - applica la trasformazione rigida XY rispetto al padre;
      - ricorre usando la posa ORIGINALE del discendente come "old" e quella
        appena calcolata come "new" per i suoi stessi figli.

    La coordinata Z di ogni discendente resta sempre quella originale.
    """
    oparent = orig_by_name.get(parent_name)
    if oparent is None:
        return

    for cname in oparent.children:
        child  = by_name.get(cname)
        ochild = orig_by_name.get(cname)
        if child is None or ochild is None:
            continue

        # Ripristina la posa originale del figlio prima della trasformazione.
        child.transform.location       = list(ochild.transform.location)
        child.transform.rotation_euler = list(ochild.transform.rotation_euler)

        apply_rigid_transform(
            child,
            old_parent_loc, old_parent_rz,
            new_parent_loc, new_parent_rz,
            original_z=ochild.transform.location[2],
        )

        # Ricorsione: la posa originale del figlio diventa il riferimento "old"
        # per i nipoti, la sua nuova posa il riferimento "new".
        _propagate_to_descendants(
            cname,
            list(ochild.transform.location),
            ochild.transform.rotation_euler[2],
            list(child.transform.location),
            child.transform.rotation_euler[2],
            by_name,
            orig_by_name,
        )


def sanitize_response(
    state: SceneState,
    llm_output,
    const: Constants = CONST,
) -> SceneState:
    """
    Valida e mette in sicurezza la risposta dell'LLM, producendo un nuovo
    SceneState con pipeline_step="reorganized".

    Il parametro llm_output puo' essere la stringa grezza della risposta del
    modello o un dict gia' parsato, nella forma:
        {"placements": [{"name": ..., "x": ..., "y": ..., "rotation_deg": ...}]}

    Garanzie offerte da questa funzione:
    - La coordinata Z non viene mai modificata, ne' per i root ne' per i figli.
    - Ogni gruppo rimane dentro i muri (clamp di gruppo + clamp per oggetto).
    - Non rimangono sovrapposizioni tra root mobili o con gli ostacoli fissi.
    - I figli seguono il padre con una trasformazione rigida.
    - I nomi sconosciuti vengono ignorati silenziosamente.
    - Gli oggetti senza proposta mantengono la posizione originale.

    Args:
        state:      Lo stato corrente della scena (prima della riorganizzazione).
        llm_output: Risposta del modello (stringa grezza o dict).
        const:      Costanti di configurazione.

    Returns:
        Un nuovo SceneState con le posizioni sanitizzate.
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

    if rb is None:
        raise ValueError(
            "SceneState senza room_bounds: impossibile sanificare. Estrai prima la scena."
        )

    movable_roots = [
        o for o in new_objs
        if o.is_movable and o.is_root and o.object_type not in _SKIP_TYPES
    ]
    fixed_obstacles = [
        o for o in new_objs
        if (not o.is_movable) and o.category != "structural" and o.object_type == "MESH"
    ]

    applied = 0

    # --- Fase 1: applica la posa proposta (o quella originale) e clamp di gruppo ---
    for root in movable_roots:
        orig = orig_by_name[root.name]
        # Il clamp di gruppo deve tenere conto dell'INTERA gerarchia (figli,
        # nipoti, ...), non solo dei figli diretti.
        orig_descendants = _descendants(orig, orig_by_name)
        z = orig.transform.location[2]  # La Z originale viene sempre preservata.

        if root.name in placements:
            px, py, rot = placements[root.name]
            proposed_rz = root.transform.rotation_euler[2]

            if rot is not None and is_finite_float(rot):
                proposed_rz = snap_rotation_90(math.radians(float(rot)))

            proposed_loc = [float(px), float(py), z]
            applied += 1
        else:
            # Nessuna proposta: mantieni la posizione originale.
            proposed_loc = [orig.transform.location[0], orig.transform.location[1], z]
            proposed_rz = orig.transform.rotation_euler[2]

        # La rotazione deve essere impostata PRIMA del clamp, affinche'
        # l'AABB calcolato per il clamp rifletta l'orientamento corretto.
        root.transform.rotation_euler[2] = proposed_rz

        clamped = _clamp_parent_group_location(
            orig, proposed_loc, proposed_rz, orig_descendants, rb, const.wall_margin
        )
        root.transform.location[0] = clamped[0]
        root.transform.location[1] = clamped[1]
        root.transform.location[2] = z  # Z intatta.

    # --- Fase 2: risoluzione delle collisioni (MTV) ---
    iters = _resolve_collisions(movable_roots, fixed_obstacles, rb, const)

    # --- Fase 3: l'intera gerarchia segue il padre con trasformazione rigida ---
    # (figli, nipoti, ... non solo i figli diretti).
    for root in movable_roots:
        orig = orig_by_name[root.name]
        _propagate_to_descendants(
            root.name,
            orig.transform.location,
            orig.transform.rotation_euler[2],
            root.transform.location,
            root.transform.rotation_euler[2],
            by_name,
            orig_by_name,
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
