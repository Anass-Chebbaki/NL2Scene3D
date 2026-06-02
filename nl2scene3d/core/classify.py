# nl2scene3d/core/classify.py
"""
Classificazione oggetti, calcolo dei confini stanza, grouping padre/figlio e
regole di congelamento. Tutto PURO Python: opera su liste di SceneObject e non
tocca bpy, quindi e' testabile da riga di comando.

Differenza rispetto alla versione originale (scene_state.py):
  - compute_grouping NON legge/scrive piu' le custom properties di Blender.
    La persistenza del grouping (nl2_parent) e' responsabilita' del layer bpy
    (scene_io.py), che puo' passare qui un 'prior' gia' pronto. In questo modo
    il grouping diventa una funzione pura e deterministica.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from .geometry import sat_overlap
from .models import RoomBounds, SceneObject
from .settings import (
    CEILING_LIGHT_PATTERNS,
    Constants,
    NON_MESH_TYPES,
    STRUCTURAL_PATTERNS,
    CONST,
)

logger = logging.getLogger(__name__)


def _has_kw(keywords, text: str) -> bool:
    """True se almeno una keyword e' contenuta in text."""
    return any(k in text for k in keywords)


# ---------------------------------------------------------------------------
# Classificazione di un singolo oggetto
# ---------------------------------------------------------------------------

def classify_object(
    name: str,
    object_type: str,
    dimensions: list[float],
    const: Constants = CONST,
) -> tuple[str, bool]:
    """
    Determina la coppia (categoria, is_movable) per un oggetto.

    Ritorna:
        (category, is_movable)
    """
    name_lower = name.lower()

    # I tipi non-mesh sono sempre statici.
    if object_type in NON_MESH_TYPES:
        return "technical", False

    # Oggetti molto piccoli = decorazioni fisse (pomelli, viti, ecc.).
    max_dim = max(dimensions) if dimensions else 0.0
    if max_dim < const.min_object_dimension:
        return "decoration_small", False

    # Luci.
    if _has_kw(("lamp", "lampada", "light"), name_lower):
        if _has_kw(CEILING_LIGHT_PATTERNS, name_lower):
            return "light_ceiling", False
        return "light_floor", True

    # Pomelli e maniglie sempre fissi.
    if _has_kw(("knob", "pomello", "handle", "maniglia"), name_lower):
        return "technical", False

    # Decorazioni ed elettronica da scrivania.
    if _has_kw(
        (
            "decor", "decoration", "ornament", "book", "bottle",
            "monitor", "pc", "computer", "keyboard", "mouse", "trashbin",
        ),
        name_lower,
    ):
        return "decoration", True

    # Elementi strutturali sempre statici (dopo le decorazioni, per precedenza corretta).
    if _has_kw(STRUCTURAL_PATTERNS, name_lower):
        return "structural", False

    # Categorie principali di mobilio.
    if _has_kw(("sofa", "couch", "divano"), name_lower):
        return "seating_large", True
    if _has_kw(("chair", "sedia", "stool", "sgabello"), name_lower):
        return "seating_small", True
    if _has_kw(("table", "tavolo", "desk", "scrivania"), name_lower):
        return "table", True
    if _has_kw(("shelf", "scaffale", "bookcase", "libreria"), name_lower):
        return "storage", True
    if _has_kw(("bed", "letto", "mattress", "materasso"), name_lower):
        return "bed", True
    if _has_kw(("wardrobe", "armadio", "cabinet", "dresser"), name_lower):
        return "storage", True
    if _has_kw(("rug", "tappeto", "carpet"), name_lower):
        return "rug", True
    if _has_kw(("plant", "pianta", "vase", "vaso"), name_lower):
        return "decoration", True

    # I comodini restano sempre root (mai figli), cosi' una lampada sopra resta
    # attaccata come figlio e si muove con loro.
    if _has_kw(("nightstand", "comodino", "bedside", "bedside_table"), name_lower):
        return "furniture", True

    return "furniture", True


def resolve_classification(
    name: str,
    object_type: str,
    dimensions: list[float],
    override: Optional[dict] = None,
    const: Constants = CONST,
) -> tuple[str, bool]:
    """
    Determina (categoria, is_movable). La categoria e' sempre automatica (dedotta
    dal nome); l'utente puo' solo forzare lo stato fisso/mobile, che e' il
    controllo critico per la sicurezza (non far spostare muri/porte).

    override (oppure None) e' un dict con chiave opzionale:
        "fixed": bool  -> True = oggetto fisso, False = mobile

    Se "fixed" e' assente o None, la mobilita' resta quella automatica.
    """
    auto_cat, auto_mov = classify_object(name, object_type, dimensions, const)
    if not override:
        return auto_cat, auto_mov

    fixed = override.get("fixed")
    mov = (not fixed) if fixed is not None else auto_mov
    return auto_cat, mov


# ---------------------------------------------------------------------------
# Confini della stanza
# ---------------------------------------------------------------------------

def compute_room_bounds(objects: list[SceneObject]) -> RoomBounds:
    """
    Calcola i confini stanza dagli oggetti strutturali.

    Strategia:
        1. Se esiste una mesh-stanza dominante (volume > 50% del volume strutturale
           totale), usa le sue dimensioni come confine.
        2. Altrimenti combina gli AABB di tutti gli oggetti strutturali.
        3. z_ceiling deriva dagli oggetti con nome 'ceiling/room/roof/soffitto';
           altrimenti dal massimo Z strutturale, con fallback minimo 2.5 m.
    """
    structural = [o for o in objects if o.category == "structural"]
    if not structural:
        logger.warning("Nessun oggetto strutturale. Uso l'intero set di oggetti.")
        structural = objects
    if not structural:
        logger.warning("Scena vuota. Uso confini di default +/- 5 m.")
        return RoomBounds(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0)

    ceiling_kws  = ("ceiling", "room", "roof", "soffitto")
    ceiling_objs = [o for o in structural if _has_kw(ceiling_kws, o.name.lower())]

    if ceiling_objs:
        z_ceiling = max(o.transform.z_range()[1] for o in ceiling_objs)
    else:
        max_z = max((o.transform.z_range()[1] for o in structural), default=2.5)
        z_ceiling = max_z if max_z > 1.0 else 2.5

    # Strategia 1: mesh-stanza dominante.
    vols = [
        (o, o.transform.dimensions[0] * o.transform.dimensions[1] * o.transform.dimensions[2])
        for o in structural
    ]
    largest_obj, max_vol = max(vols, key=lambda x: x[1])
    total_vol = sum(v for _, v in vols)

    if total_vol > 0 and max_vol > 0.5 * total_vol and max_vol > 1.0:
        x_min, x_max, y_min, y_max = largest_obj.transform.aabb_xy(margin=0.0)
        logger.info(
            "Stanza dedotta da un singolo oggetto '%s' (AABB: X[%.2f, %.2f] Y[%.2f, %.2f]).",
            largest_obj.name, x_min, x_max, y_min, y_max,
        )
        return RoomBounds(
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            z_floor=0.0, z_ceiling=z_ceiling,
        )

    # Strategia 2: AABB unione di tutti gli strutturali.
    aabbs = [o.transform.aabb_xy(margin=0.0) for o in structural]
    return RoomBounds(
        x_min=min(a[0] for a in aabbs),
        x_max=max(a[1] for a in aabbs),
        y_min=min(a[2] for a in aabbs),
        y_max=max(a[3] for a in aabbs),
        z_floor=0.0, z_ceiling=z_ceiling,
    )


# ---------------------------------------------------------------------------
# Grouping padre/figlio (puro, con 'prior' opzionale per la persistenza)
# ---------------------------------------------------------------------------

def _volume(o: SceneObject) -> float:
    d = o.transform.dimensions
    return d[0] * d[1] * d[2]


def compute_grouping(
    objects: list[SceneObject],
    prior: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """
    Rileva le relazioni padre-figlio e le annota in-place su ogni SceneObject
    (campi parent / children). Ritorna anche la mappa {figlio: padre} calcolata,
    cosi' il layer bpy puo' persisterla.

    Se `prior` e' fornito e contiene almeno una relazione valida, viene usato
    cosi' com'e' (grouping stabile tra Randomize e AI Reorder), senza ricalcolo.

    Un oggetto B e' figlio di A se:
        1. Superficie: B poggia su A (z_bottom di B ~= z_top di A, +/- 15 cm)
                       E le impronte XY si sovrappongono.
        2. Contenuto:  B e' dentro il range Z di A (es. libro su scaffale)
                       E le impronte XY si sovrappongono.
        3. Prossimita': B e A hanno forte sovrapposizione Z (>= 30% dell'altezza
                       di B) E le impronte XY sono entro 60 cm (sedia sotto la
                       scrivania, PC accanto alla scrivania).
    In tutti i casi A deve avere volume >= 1.5x quello di B.
    """
    # Reset.
    for obj in objects:
        obj.parent = None
        obj.children = []

    by_name = {o.name: o for o in objects}

    # Riuso di un grouping precedente, se fornito e valido.
    if prior:
        applied = False
        for child_name, parent_name in prior.items():
            if child_name in by_name and parent_name in by_name:
                by_name[child_name].parent = parent_name
                by_name[parent_name].children.append(child_name)
                applied = True
        if applied:
            logger.info("Grouping riutilizzato da uno stato precedente.")
            return {c: p for c, p in prior.items() if c in by_name and p in by_name}

    # Calcolo da zero.
    movable = [o for o in objects if o.is_movable and o.category != "structural"]

    ALLOWED_CHILD_CATEGORIES  = {"decoration", "decoration_small", "seating_small", "light_floor"}
    ALLOWED_PARENT_CATEGORIES = {"table", "desk", "storage", "seating_large", "bed", "furniture"}

    result: dict[str, str] = {}

    for child in movable:
        if child.category not in ALLOWED_CHILD_CATEGORIES:
            continue

        child_z_min, child_z_max = child.transform.z_range()
        child_vol    = _volume(child)
        child_height = child_z_max - child_z_min
        child_poly   = child.transform.obb_corners_xy(margin=0.0)

        best_parent: Optional[str] = None
        best_score: float = float("inf")

        for candidate in movable:
            if candidate.name == child.name:
                continue
            if candidate.category not in ALLOWED_PARENT_CATEGORIES:
                continue

            cand_vol = _volume(candidate)

            child_area = child.transform.dimensions[0] * child.transform.dimensions[1]
            cand_area  = candidate.transform.dimensions[0] * candidate.transform.dimensions[1]

            parent_bigger_footprint = cand_area >= child_area * 1.05  # "poggia sopra"
            parent_bigger_volume    = cand_vol  >= child_vol  * 1.2   # "contenuto"/"prossimita'"

            par_z_min, par_z_max = candidate.transform.z_range()

            z_diff_top = child_z_min - par_z_max
            is_on_top  = -0.08 <= z_diff_top <= 0.20

            is_inside = (
                child_z_min >= par_z_min - 0.05
                and child_z_max <= par_z_max + 0.05
            )

            z_overlap     = max(0.0, min(child_z_max, par_z_max) - max(child_z_min, par_z_min))
            has_z_overlap = (child_height > 0) and (z_overlap / child_height >= 0.30)

            matched = False
            score   = 0.0

            if (is_on_top and parent_bigger_footprint) or (is_inside and parent_bigger_volume):
                par_poly = candidate.transform.obb_corners_xy(margin=0.0)
                if sat_overlap(child_poly, par_poly):
                    matched = True
                    score = abs(z_diff_top) if is_on_top else 0.0

            if not matched and has_z_overlap:
                allowed_proximity_children = {
                    "seating_small", "decoration", "decoration_small", "light_floor"
                }
                allowed_proximity_parents = {
                    "table", "desk", "storage", "seating_large", "bed", "furniture"
                }
                if (
                    child.category in allowed_proximity_children
                    and candidate.category in allowed_proximity_parents
                    and parent_bigger_volume
                ):
                    par_poly_expanded = candidate.transform.obb_corners_xy(margin=0.15)
                    if sat_overlap(child_poly, par_poly_expanded):
                        matched = True
                        cx, cy = candidate.transform.geometric_center_xy()
                        bx, by = child.transform.geometric_center_xy()
                        score  = 10.0 + math.hypot(bx - cx, by - cy)

            if not matched:
                continue

            if score < best_score:
                best_score  = score
                best_parent = candidate.name

        if best_parent is not None:
            child.parent = best_parent
            by_name[best_parent].children.append(child.name)
            result[child.name] = best_parent
            logger.debug("Grouping: '%s' -> padre '%s' (score=%.2f).", child.name, best_parent, best_score)

    n_groups   = sum(1 for o in objects if o.children)
    n_children = sum(1 for o in objects if o.parent is not None)
    logger.info("Grouping completo: %d gruppi, %d figli.", n_groups, n_children)
    return result


# ---------------------------------------------------------------------------
# Regole di congelamento (oggetti a parete alta / soffitto)
# ---------------------------------------------------------------------------

def apply_static_placement_rules(
    objects: list[SceneObject],
    room_bounds: RoomBounds,
    const: Constants = CONST,
    protected: Optional[set[str]] = None,
) -> None:
    """
    Congela (is_movable=False) gli oggetti root montati in alto sui muri
    (mensole, lampade a parete, quadri) o attaccati al soffitto, propagando lo
    stato statico a tutti i loro figli. Va chiamata DOPO compute_grouping.

    `protected` e' l'insieme dei nomi che l'utente ha dichiarato esplicitamente
    MOBILI: questi non vengono mai congelati automaticamente (la scelta manuale
    dell'utente ha la precedenza sulla regola euristica sull'altezza).
    """
    protected = protected or set()
    by_name = {o.name: o for o in objects}

    def freeze(name: str) -> None:
        o = by_name.get(name)
        if not o:
            return
        o.is_movable = False
        for c in o.children:
            freeze(c)

    threshold = const.static_height_threshold
    ceiling   = room_bounds.z_ceiling
    frozen    = 0

    for o in objects:
        if not o.is_movable or o.parent is not None:
            continue
        if o.name in protected:
            continue  # l'utente l'ha dichiarato mobile: non lo tocchiamo.

        z_min, z_max = o.transform.z_range()
        on_wall_high = z_min >= threshold
        on_ceiling   = const.freeze_ceiling_objects and z_max >= ceiling - 0.15

        if on_wall_high or on_ceiling:
            freeze(o.name)
            frozen += 1
            logger.info(
                "Oggetto congelato: '%s' (z_min=%.2f, z_max=%.2f, soffitto=%.2f).",
                o.name, z_min, z_max, ceiling,
            )

    if frozen:
        logger.info("Regole statiche: %d gruppi congelati.", frozen)
