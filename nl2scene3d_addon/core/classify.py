# nl2scene3d/core/classify.py
"""
Classificazione degli oggetti, calcolo dei confini stanza e suggerimento
di grouping geometrico per NL2Scene3D.

Filosofia di progetto:
    - Nessuna categoria semantica per i mobili (bed/table/chair/...): erano liste
      di parole fragili che non generalizzavano tra scene diverse.
    - L'utente decide cosa e' fisso/mobile e chi e' figlio di chi, dal pannello.
    - L'automatico fa solo il minimo robusto:
        * Camera/luci e oggetti con nome strutturale (muro/pavimento/stanza/...)
          sono fissi per default (l'utente puo' sempre cambiare).
        * I confini della stanza si ricavano dalla geometria.
        * Il grouping automatico e' un suggerimento geometrico opzionale
          (bottone "Suggerisci gruppi"), non viene mai imposto.

Modulo puro Python: nessuna dipendenza da bpy, testabile da riga di comando.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Optional

from .geometry import sat_overlap
from .models import RoomBounds, SceneObject
from .settings import CONST, Constants, NON_MESH_TYPES, STRUCTURAL_PATTERNS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility sui token del nome
# ---------------------------------------------------------------------------

def _tokens(text: str) -> set[str]:
    """
    Spezza un nome in parole separate (su _, -, spazi, numeri, ecc.).

    Esempio: 'decor_doorknob' -> {'decor', 'doorknob'}
    """
    return {t for t in re.split(r"[^a-z]+", text.lower()) if t}


def _has_kw(keywords, text: str) -> bool:
    """
    True se almeno una keyword compare come parola intera in text.

    Il match avviene su token interi (separati da _, -, spazi, numeri),
    non come sottostringa generica. Cosi' 'doorknob' non viene confuso con
    'door', ne' 'window_blind' con 'window'.
    """
    toks = _tokens(text)
    return any(k in toks for k in keywords)


# ---------------------------------------------------------------------------
# Classificazione minima (strutturale vs resto)
# ---------------------------------------------------------------------------

def default_classification(
    name:        str,
    object_type: str,
    dimensions:  list[float],
    const:       Constants = CONST,
) -> tuple[str, bool]:
    """
    Classificazione di default, senza categorie di mobili.

    Restituisce (category, is_movable) dove category e' una delle tre:
        "technical"  -> camera/luci/non-mesh  (fisso)
        "structural" -> muro/pavimento/stanza  (fisso)
        "object"     -> tutto il resto         (mobile)

    La category serve solo alle collisioni (strutturale vs resto) e al
    report Inspect. L'utente puo' sempre ribaltare fisso/mobile dal pannello.
    """
    if object_type in NON_MESH_TYPES:
        return "technical", False

    if _has_kw(STRUCTURAL_PATTERNS, name.lower()):
        return "structural", False

    return "object", True


def resolve_classification(
    name:        str,
    object_type: str,
    dimensions:  list[float],
    override:    Optional[dict] = None,
    const:       Constants = CONST,
) -> tuple[str, bool]:
    """
    Come default_classification, ma rispetta l'override manuale dell'utente.

    Se l'override contiene la chiave "fixed", il suo valore booleano vince
    sulla stima automatica di is_movable.

    Args:
        override: dizionario opzionale con chiave "fixed": bool.
    """
    cat, auto_mov = default_classification(name, object_type, dimensions, const)

    if not override:
        return cat, auto_mov

    fixed = override.get("fixed")
    mov   = (not fixed) if fixed is not None else auto_mov
    return cat, mov


# ---------------------------------------------------------------------------
# Calcolo dei confini della stanza
# ---------------------------------------------------------------------------

def _volume(o: SceneObject) -> float:
    """Volume dell'oggetto (prodotto delle tre dimensioni)."""
    d = o.transform.dimensions
    return d[0] * d[1] * d[2]


def _footprint(o: SceneObject) -> float:
    """Impronta XY dell'oggetto (larghezza x profondita')."""
    d = o.transform.dimensions
    return d[0] * d[1]


def compute_room_bounds(objects: list[SceneObject]) -> RoomBounds:
    """
    Calcola i confini spaziali della stanza a partire dagli oggetti della scena.

    Strategia:
        1. Se esistono oggetti strutturali (riconosciuti per nome), li usa come
           riferimento (caso tipico: una mesh-stanza, oppure muri + pavimento).
        2. Se non ci sono strutturali riconosciuti, ripiegpa sulla geometria:
           usa l'oggetto con l'impronta XY piu' grande come approssimazione
           della stanza.
        3. Ultimo fallback: AABB unione di tutti gli oggetti.

    z_ceiling viene dedotto dagli oggetti 'ceiling/room/soffitto/stanza'
    oppure dal massimo Z rilevato tra gli strutturali.
    """
    if not objects:
        logger.warning("Scena vuota. Confini di default +/- 5 m.")
        return RoomBounds(x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0)

    structural = [o for o in objects if o.category == "structural"]
    source     = structural

    if not source:
        # Nessun strutturale riconosciuto per nome: usa l'oggetto con l'impronta maggiore.
        meshes  = [o for o in objects if o.object_type == "MESH"] or objects
        biggest = max(meshes, key=_footprint)
        source  = [biggest]
        logger.info(
            "Nessun strutturale: stanza dedotta da '%s' (impronta maggiore).",
            biggest.name,
        )

    # Stima z_ceiling dagli elementi che includono 'ceiling/room/soffitto/stanza'.
    ceiling_kws  = ("ceiling", "room", "roof", "soffitto", "stanza")
    ceiling_objs = [o for o in source if _has_kw(ceiling_kws, o.name.lower())]
    if ceiling_objs:
        z_ceiling = max(o.transform.z_range()[1] for o in ceiling_objs)
    else:
        max_z     = max((o.transform.z_range()[1] for o in source), default=2.5)
        z_ceiling = max_z if max_z > 1.0 else 2.5

    # Se un singolo oggetto strutturale domina il volume (> 50% del totale),
    # usa il suo AABB come confini della stanza.
    vols             = [(o, _volume(o)) for o in source]
    largest_obj, max_vol = max(vols, key=lambda x: x[1])
    total_vol        = sum(v for _, v in vols)

    if total_vol > 0 and max_vol > 0.5 * total_vol and max_vol > 1.0:
        x_min, x_max, y_min, y_max = largest_obj.transform.aabb_xy(margin=0.0)
        return RoomBounds(
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            z_floor=0.0, z_ceiling=z_ceiling,
        )

    # Fallback: AABB unione di tutti gli oggetti source.
    aabbs = [o.transform.aabb_xy(margin=0.0) for o in source]
    return RoomBounds(
        x_min=min(a[0] for a in aabbs), x_max=max(a[1] for a in aabbs),
        y_min=min(a[2] for a in aabbs), y_max=max(a[3] for a in aabbs),
        z_floor=0.0, z_ceiling=z_ceiling,
    )


# ---------------------------------------------------------------------------
# Parenting manuale
# ---------------------------------------------------------------------------

def _would_create_cycle(
    child_name:  str,
    parent_name: str,
    parent_of:   dict[str, str],
) -> bool:
    """
    True se rendere `parent_name` il padre di `child_name` chiuderebbe un ciclo,
    data la gerarchia gia' costruita finora (`parent_of`: {figlio: padre}).

    Risale la catena dei padri gia' accettati a partire da `parent_name`: se
    incontra `child_name` (o un nodo gia' visitato) la relazione creerebbe un
    anello. Lavorando sulla gerarchia PARZIALE, spezza solo l'arco che chiude
    il ciclo e conserva il resto della catena (A->B->C viene mantenuto, solo
    C->A viene scartato).
    """
    seen = {child_name}
    cur: Optional[str] = parent_name
    while cur:
        if cur in seen:
            return True
        seen.add(cur)
        cur = parent_of.get(cur)
    return False


def apply_manual_parents(
    objects:    list[SceneObject],
    parent_map: dict[str, str],
) -> None:
    """
    Imposta le relazioni parent/children sugli oggetti a partire da una mappa
    {figlio: padre} fornita dall'utente tramite il pannello.

    Ignora le coppie non valide:
        - padre assente nella scena;
        - auto-riferimento (oggetto padre di se stesso);
        - catene circolari di qualsiasi lunghezza (A->B->C->A): viene scartato
          solo l'arco che chiude l'anello, il resto della catena e' preservato,
          cosi' nessun oggetto sparisce dalla pipeline.

    Resetta tutte le relazioni precedenti prima di applicare la nuova mappa.
    """
    by_name = {o.name: o for o in objects}

    # Azzera tutte le relazioni esistenti.
    for o in objects:
        o.parent   = None
        o.children = []

    # Gerarchia accettata finora, costruita in modo incrementale.
    parent_of: dict[str, str] = {}

    for child_name, parent_name in (parent_map or {}).items():
        if not parent_name:
            continue

        child  = by_name.get(child_name)
        parent = by_name.get(parent_name)

        if child is None or parent is None or child_name == parent_name:
            continue

        # Evita catene circolari di qualsiasi lunghezza, spezzando solo
        # l'arco che chiuderebbe l'anello.
        if _would_create_cycle(child_name, parent_name, parent_of):
            logger.warning(
                "Relazione padre-figlio ignorata: '%s' -> '%s' creerebbe un ciclo.",
                child_name, parent_name,
            )
            continue

        child.parent = parent_name
        parent.children.append(child_name)
        parent_of[child_name] = parent_name


# ---------------------------------------------------------------------------
# Suggerimento di grouping geometrico (opzionale)
# ---------------------------------------------------------------------------

def suggest_grouping(objects: list[SceneObject]) -> dict[str, str]:
    """
    Propone una mappa {figlio: padre} basata esclusivamente sulla geometria.

    Nessuna categoria semantica: un oggetto e' considerato figlio di un altro se:
        - gli sta sopra o dentro (sovrapposizione Z);
        - le impronte XY si sovrappongono (verificato via SAT);
        - il potenziale padre e' sensibilmente piu' grande.

    Usato dal bottone "Suggerisci gruppi"; l'utente e' libero di correggere
    o ignorare i suggerimenti. Non muta gli oggetti in ingresso.

    Considera solo oggetti mobili e non strutturali, sia come figli sia come padri.
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
        best_score = float("inf")

        for cand in candidates:
            if cand.name == child.name:
                continue

            cand_vol  = _volume(cand)
            cand_area = _footprint(cand)

            bigger_footprint = cand_area >= c_area * CONST.grouping_footprint_ratio
            bigger_volume    = cand_vol  >= c_vol  * CONST.grouping_volume_ratio

            p_z_min, p_z_max = cand.transform.z_range()

            # on_top accetta solo oggetti che poggiano SOPRA il candidato
            # (c_z_min >= p_z_max - piccola tolleranza), non oggetti che penetrano
            # dentro di lui. La soglia inferiore era -0.08 (penetrazione ammessa)
            # che causava falsi positivi (es. sedie "figlio" del pavimento).
            # Ora: lo 0 esatto e' il limite (CONST.grouping_on_top_lo = 0.0).
            on_top    = CONST.grouping_on_top_lo <= (c_z_min - p_z_max) <= CONST.grouping_on_top_hi
            is_inside = (c_z_min >= p_z_min - 0.05) and (c_z_max <= p_z_max + 0.05)
            z_overlap = max(0.0, min(c_z_max, p_z_max) - max(c_z_min, p_z_min))
            close_z   = c_h > 0 and (z_overlap / c_h) >= CONST.grouping_z_overlap_frac

            matched = False
            score   = 0.0

            if (on_top and bigger_footprint) or (is_inside and bigger_volume):
                if sat_overlap(c_poly, cand.transform.obb_corners_xy(margin=0.0)):
                    matched = True
                    score   = abs(c_z_min - p_z_max) if on_top else 0.0

            elif close_z and bigger_volume:
                if sat_overlap(c_poly, cand.transform.obb_corners_xy(margin=0.15)):
                    cx, cy  = cand.transform.geometric_center_xy()
                    bx, by  = child.transform.geometric_center_xy()
                    matched = True
                    score   = 10.0 + math.hypot(bx - cx, by - cy)

            if matched and score < best_score:
                best_score, best_parent = score, cand.name

        if best_parent is not None:
            result[child.name] = best_parent

    logger.info("Suggerimento grouping: %d relazioni proposte.", len(result))
    return result