# nl2scene3d/core/classify.py
"""
Classificazione minima, confini stanza e grouping di NL2Scene3D.

FILOSOFIA (dopo il refactor):
  - NIENTE piu' categorie semantiche per i mobili (bed/table/chair/...): erano
    liste di parole fragili che non generalizzavano tra scene diverse.
  - L'utente decide cosa e' fisso/mobile e chi e' figlio di chi, dal pannello.
  - L'automatico fa solo il minimo che serve e che e' robusto:
      * camera/luci e oggetti dal nome STRUTTURALE (muro/pavimento/stanza/...)
        sono FISSI di default (l'utente puo' sempre cambiare);
      * i confini della stanza si ricavano dalla geometria;
      * il grouping automatico e' solo un SUGGERIMENTO geometrico opzionale
        (bottone "Suggerisci gruppi"), non viene piu' imposto.

Tutto PURO Python: nessun bpy, quindi testabile da riga di comando.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from .geometry import sat_overlap
from .models import RoomBounds, SceneObject
from .settings import CONST, Constants, NON_MESH_TYPES, STRUCTURAL_PATTERNS

logger = logging.getLogger(__name__)


def _has_kw(keywords, text: str) -> bool:
    """True se almeno una keyword e' contenuta in text."""
    return any(k in text for k in keywords)


# ---------------------------------------------------------------------------
# Classificazione minima (solo strutturale vs resto)
# ---------------------------------------------------------------------------

def default_classification(
    name: str,
    object_type: str,
    dimensions: list[float],
    const: Constants = CONST,
) -> tuple[str, bool]:
    """
    Classificazione di DEFAULT, senza categorie di mobili.

    Ritorna (category, is_movable) dove category e' solo una delle tre:
        "technical"  -> camera/luci/non-mesh    (fisso)
        "structural" -> muro/pavimento/stanza... (fisso)
        "object"     -> tutto il resto           (mobile)

    'category' serve unicamente alle collisioni (strutturale vs resto) e a
    mostrare lo stato nell'Inspect. Non esistono piu' categorie semantiche.
    L'utente puo' comunque ribaltare 'fisso/mobile' dal pannello.
    """
    if object_type in NON_MESH_TYPES:
        return "technical", False

    if _has_kw(STRUCTURAL_PATTERNS, name.lower()):
        return "structural", False

    return "object", True


def resolve_classification(
    name: str,
    object_type: str,
    dimensions: list[float],
    override: Optional[dict] = None,
    const: Constants = CONST,
) -> tuple[str, bool]:
    """
    Come default_classification, ma se l'utente ha forzato 'fixed' quello vince.

    override (o None): dict con chiave opzionale "fixed": bool.
    """
    cat, auto_mov = default_classification(name, object_type, dimensions, const)
    if not override:
        return cat, auto_mov
    fixed = override.get("fixed")
    mov = (not fixed) if fixed is not None else auto_mov
    return cat, mov


# ---------------------------------------------------------------------------
# Confini della stanza (geometrici, con preferenza per gli strutturali)
# ---------------------------------------------------------------------------

def _volume(o: SceneObject) -> float:
    d = o.transform.dimensions
    return d[0] * d[1] * d[2]


def _footprint(o: SceneObject) -> float:
    d = o.transform.dimensions
    return d[0] * d[1]


def compute_room_bounds(objects: list[SceneObject]) -> RoomBounds:
    """
    Calcola i confini stanza.

    Strategia:
        1. Se esistono oggetti strutturali (per nome), usali (e' il caso tipico:
           una mesh-stanza, oppure muri+pavimento).
        2. Altrimenti (scena senza strutturali riconosciuti) ripiega sulla
           geometria: prende l'oggetto con l'impronta XY piu' grande come stanza.
        3. Ultimo fallback: AABB unione di tutti gli oggetti.
    z_ceiling viene dagli oggetti 'ceiling/room/soffitto/stanza' o dal massimo Z.
    """
    if not objects:
        logger.warning("Scena vuota. Confini di default +/- 5 m.")
        return RoomBounds(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0)

    structural = [o for o in objects if o.category == "structural"]
    source = structural

    if not source:
        # Nessun strutturale per nome: usa l'oggetto con l'impronta piu' grande.
        meshes = [o for o in objects if o.object_type == "MESH"] or objects
        biggest = max(meshes, key=_footprint)
        source = [biggest]
        logger.info("Nessun strutturale: stanza dedotta da '%s' (impronta maggiore).", biggest.name)

    ceiling_kws  = ("ceiling", "room", "roof", "soffitto", "stanza")
    ceiling_objs = [o for o in source if _has_kw(ceiling_kws, o.name.lower())]
    if ceiling_objs:
        z_ceiling = max(o.transform.z_range()[1] for o in ceiling_objs)
    else:
        max_z = max((o.transform.z_range()[1] for o in source), default=2.5)
        z_ceiling = max_z if max_z > 1.0 else 2.5

    # Mesh-stanza dominante (volume > 50% del totale): usa il suo AABB.
    vols = [(o, _volume(o)) for o in source]
    largest_obj, max_vol = max(vols, key=lambda x: x[1])
    total_vol = sum(v for _, v in vols)
    if total_vol > 0 and max_vol > 0.5 * total_vol and max_vol > 1.0:
        x_min, x_max, y_min, y_max = largest_obj.transform.aabb_xy(margin=0.0)
        return RoomBounds(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                          z_floor=0.0, z_ceiling=z_ceiling)

    aabbs = [o.transform.aabb_xy(margin=0.0) for o in source]
    return RoomBounds(
        x_min=min(a[0] for a in aabbs), x_max=max(a[1] for a in aabbs),
        y_min=min(a[2] for a in aabbs), y_max=max(a[3] for a in aabbs),
        z_floor=0.0, z_ceiling=z_ceiling,
    )


# ---------------------------------------------------------------------------
# Parenting MANUALE: applica una mappa {figlio: padre} decisa dall'utente
# ---------------------------------------------------------------------------

def apply_manual_parents(
    objects: list[SceneObject],
    parent_map: dict[str, str],
) -> None:
    """
    Imposta parent/children sugli oggetti a partire da una mappa {figlio: padre}
    fornita dall'utente (dal pannello). Ignora coppie non valide (padre assente,
    auto-riferimento, padre fisso). Resetta prima ogni relazione.
    """
    by_name = {o.name: o for o in objects}
    for o in objects:
        o.parent = None
        o.children = []

    for child_name, parent_name in (parent_map or {}).items():
        if not parent_name:
            continue
        child = by_name.get(child_name)
        parent = by_name.get(parent_name)
        if child is None or parent is None or child_name == parent_name:
            continue
        # Evita catene banali: il padre non puo' essere a sua volta figlio del figlio.
        if parent.parent == child_name:
            continue
        child.parent = parent_name
        parent.children.append(child_name)


# ---------------------------------------------------------------------------
# Suggerimento di grouping GEOMETRICO (opzionale, name-agnostic)
# ---------------------------------------------------------------------------

def suggest_grouping(objects: list[SceneObject]) -> dict[str, str]:
    """
    Propone una mappa {figlio: padre} basata SOLO sulla geometria (niente
    categorie semantiche): un oggetto e' figlio di un altro se gli sta sopra o
    dentro, le impronte XY si sovrappongono e il padre e' sensibilmente piu'
    grande. Usato dal bottone "Suggerisci gruppi"; l'utente poi corregge.

    Considera solo oggetti mobili e non strutturali, sia come figli sia come
    padri. Ritorna la mappa (non muta gli oggetti).
    """
    candidates = [
        o for o in objects
        if o.is_movable and o.category not in ("structural", "technical")
    ]

    result: dict[str, str] = {}

    for child in candidates:
        c_z_min, c_z_max = child.transform.z_range()
        c_h    = c_z_max - c_z_min
        c_vol  = _volume(child)
        c_area = _footprint(child)
        c_poly = child.transform.obb_corners_xy(margin=0.0)

        best_parent: Optional[str] = None
        best_score  = float("inf")

        for cand in candidates:
            if cand.name == child.name:
                continue

            cand_vol  = _volume(cand)
            cand_area = _footprint(cand)
            bigger_footprint = cand_area >= c_area * 1.05
            bigger_volume    = cand_vol  >= c_vol  * 1.20

            p_z_min, p_z_max = cand.transform.z_range()
            on_top    = -0.08 <= (c_z_min - p_z_max) <= 0.20
            is_inside = (c_z_min >= p_z_min - 0.05) and (c_z_max <= p_z_max + 0.05)
            z_overlap = max(0.0, min(c_z_max, p_z_max) - max(c_z_min, p_z_min))
            close_z   = c_h > 0 and (z_overlap / c_h) >= 0.30

            matched = False
            score   = 0.0
            if (on_top and bigger_footprint) or (is_inside and bigger_volume):
                if sat_overlap(c_poly, cand.transform.obb_corners_xy(margin=0.0)):
                    matched, score = True, abs(c_z_min - p_z_max) if on_top else 0.0
            elif close_z and bigger_volume:
                if sat_overlap(c_poly, cand.transform.obb_corners_xy(margin=0.15)):
                    cx, cy = cand.transform.geometric_center_xy()
                    bx, by = child.transform.geometric_center_xy()
                    matched, score = True, 10.0 + math.hypot(bx - cx, by - cy)

            if matched and score < best_score:
                best_score, best_parent = score, cand.name

        if best_parent is not None:
            result[child.name] = best_parent

    logger.info("Suggerimento grouping: %d relazioni proposte.", len(result))
    return result
