# nl2scene3d/utils/geometry.py
"""
Collision detection per la pipeline NL2Scene3D.

Architettura:
- I calcoli geometrici base (AABB, OBB corners, z_range) vivono su Transform.
- Questo modulo si occupa solo di:
    1. has_collision():      check binario tra un candidato e una lista di oggetti.
    2. wall_collision():     check separato per i muri (AABB + Z overlap).
    3. furniture_collision(): check SAT tra OBB per i mobili.

Collision margin:
    Ogni oggetto viene espanso di `margin` metri su tutti i lati prima del check.
    Questo garantisce che dopo il posizionamento ci siano sempre almeno
    2×margin cm di spazio tra due oggetti adiacenti, evitando l'intrecciamento
    visivo anche quando l'LLM o il randomizer li piazza molto vicini.

    Valori consigliati:
    - Randomizer:     margin = config.collision_margin  (default 0.05m = 5cm)
    - Post-LLM check: margin = 0.02m (più tollerante, l'LLM ragiona su mobili grandi)
    - Wall check:     margin = config.wall_margin       (default 0.20m)
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nl2scene3d.models import SceneObject

logger = logging.getLogger(__name__)

# Tipi Blender che non partecipano alla collision detection
_IGNORED_TYPES = frozenset({"CAMERA", "LIGHT", "EMPTY", "SPEAKER", "ARMATURE", "CURVE"})


# ---------------------------------------------------------------------------
# Utility geometriche base
# ---------------------------------------------------------------------------

def is_finite_float(val: Any) -> bool:
    """True se il valore è un float finito (non NaN o inf)."""
    try:
        f = float(val)
        return math.isfinite(f)
    except (ValueError, TypeError):
        return False


def snap_rotation_90(rz: float) -> float:
    """Snap della rotazione Z a multipli di 90° (0, 90, 180, 270)."""
    multiples = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
    return min(multiples, key=lambda m: abs(m - (rz % (2 * math.pi))))


# ---------------------------------------------------------------------------
# Check muri (AABB + Z overlap)
# ---------------------------------------------------------------------------

def wall_collision(
    candidate: "SceneObject",
    wall_objects: list["SceneObject"],
    wall_margin: float = 0.20,
) -> bool:
    """
    Verifica se il candidato penetra un muro.

    Usa AABB 2D (non SAT) perché i muri hanno geometrie complesse
    (modanature, battiscopa) che causano falsi positivi con OBB esatti.
    Il margine wall_margin viene aggiunto intorno al candidato per
    tenere i mobili distanti dai muri.

    Non controlla porte/finestre/room (sono aperture o contenitori).
    """
    # AABB del candidato espanso del margine
    c_aabb = candidate.transform.aabb_xy(margin=wall_margin)
    c_z_min, c_z_max = candidate.transform.z_range()

    for wall in wall_objects:
        name_lower = wall.name.lower()
        # Porte, finestre e la mesh-stanza non sono ostacoli AABB
        if any(k in name_lower for k in ("door", "window", "room")):
            continue

        w_aabb = wall.transform.aabb_xy(margin=0.0)

        # Check sovrapposizione XY
        x_overlap = max(0.0, min(c_aabb[1], w_aabb[1]) - max(c_aabb[0], w_aabb[0]))
        y_overlap = max(0.0, min(c_aabb[3], w_aabb[3]) - max(c_aabb[2], w_aabb[2]))

        if x_overlap <= 0.01 or y_overlap <= 0.01:
            continue

        # Check sovrapposizione Z — pavimenti e soffitti non sono muri laterali
        w_z_min, w_z_max = wall.transform.z_range()
        z_overlap = max(0.0, min(c_z_max, w_z_max) - max(c_z_min, w_z_min))

        if z_overlap > 0.01:
            logger.debug(
                "Collisione muro: '%s' ↔ '%s' (XY overlap: %.3f×%.3f, Z: %.3f).",
                candidate.name, wall.name, x_overlap, y_overlap, z_overlap,
            )
            return True

    return False


# ---------------------------------------------------------------------------
# Check mobili (SAT tra OBB)
# ---------------------------------------------------------------------------

def _sat_overlap(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
) -> bool:
    """Separating Axis Theorem per due poligoni convessi 2D."""
    def get_axes(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
        axes = []
        n = len(poly)
        for i in range(n):
            p1, p2 = poly[i], poly[(i + 1) % n]
            ex, ey = p2[0] - p1[0], p2[1] - p1[1]
            mag = math.hypot(ex, ey)
            if mag > 1e-6:
                axes.append((-ey / mag, ex / mag))
        return axes

    def project(poly: list[tuple[float, float]], axis: tuple[float, float]) -> tuple[float, float]:
        dots = [p[0] * axis[0] + p[1] * axis[1] for p in poly]
        return min(dots), max(dots)

    for axis in get_axes(poly_a) + get_axes(poly_b):
        mn_a, mx_a = project(poly_a, axis)
        mn_b, mx_b = project(poly_b, axis)
        if mx_a < mn_b or mx_b < mn_a:
            return False  # Asse separante → nessuna collisione
    return True  # Nessun asse separante → collisione


def furniture_collision(
    candidate: "SceneObject",
    furniture_objects: list["SceneObject"],
    margin: float = 0.05,
) -> bool:
    """
    Verifica se il candidato si sovrappone a un altro mobile.

    Usa SAT su OBB 2D per gestire correttamente oggetti ruotati.
    Il margin espande ogni OBB di `margin` metri, garantendo spazio fisico
    tra gli oggetti anche dopo il posizionamento.
    """
    cand_poly = candidate.transform.obb_corners_xy(margin=margin)
    cand_z_min, cand_z_max = candidate.transform.z_range()

    for other in furniture_objects:
        # Check Z overlap prima di fare SAT (ottimizzazione: evita SAT per oggetti su piani diversi)
        o_z_min, o_z_max = other.transform.z_range()
        z_overlap = max(0.0, min(cand_z_max, o_z_max) - max(cand_z_min, o_z_min))
        if z_overlap < 0.02:
            # Oggetti su piani Z molto diversi (es. oggetto su tavolo vs oggetto a terra)
            # non si scontrano nel senso fisico del randomizer
            continue

        other_poly = other.transform.obb_corners_xy(margin=margin)
        if _sat_overlap(cand_poly, other_poly):
            logger.debug(
                "Collisione SAT: '%s' ↔ '%s'.", candidate.name, other.name
            )
            return True

    return False


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def has_collision(
    candidate: "SceneObject",
    placed_objects: list["SceneObject"],
    wall_margin: float = 0.20,
    furniture_margin: float = 0.05,
    check_walls: bool = True,
) -> bool:
    """
    Verifica se il candidato ha collisioni con gli oggetti già posizionati.

    Args:
        candidate:         Oggetto da testare.
        placed_objects:    Oggetti già posizionati (inclusi strutturali e mobili).
        wall_margin:       Margine minimo dai muri in metri.
        furniture_margin:  Margine espanso su ogni OBB dei mobili in metri.
                           Con 0.05m ogni coppia di oggetti avrà ≥10cm di spazio.
        check_walls:       Se False, salta il check con i muri (utile per decorazioni
                           da appendere a parete).

    Returns:
        True se c'è almeno una collisione.
    """
    walls: list["SceneObject"] = []
    furniture: list["SceneObject"] = []

    for obj in placed_objects:
        if obj.name == candidate.name:
            continue
        if obj.object_type in _IGNORED_TYPES:
            continue
        if obj.category == "structural":
            walls.append(obj)
        else:
            furniture.append(obj)

    if check_walls and walls:
        if wall_collision(candidate, walls, wall_margin):
            return True

    if furniture:
        if furniture_collision(candidate, furniture, furniture_margin):
            return True

    return False


# ---------------------------------------------------------------------------
# Utilità per il randomizer e il post-LLM solver
# ---------------------------------------------------------------------------

def collision_score(
    candidate: "SceneObject",
    placed_objects: list["SceneObject"],
    wall_margin: float = 0.20,
    furniture_margin: float = 0.05,
    check_walls: bool = True,
) -> float:
    """
    Restituisce un punteggio di "bontà" della posizione del candidato.

    0.0 = nessuna collisione (posizione perfetta).
    > 0.0 = c'è sovrapposizione; più alto è il valore, peggio è la posizione.

    Utile nel randomizer per scegliere la posizione meno problematica quando
    non si riesce a trovarne una completamente libera entro max_attempts.

    Il calcolo usa AABB (non SAT) per velocità — è una stima, non un check esatto.
    """
    total = 0.0

    c_aabb_base = candidate.transform.aabb_xy(margin=0.0)
    c_aabb_wall = candidate.transform.aabb_xy(margin=wall_margin)
    c_aabb_furn = candidate.transform.aabb_xy(margin=furniture_margin)
    c_z_min, c_z_max = candidate.transform.z_range()

    for obj in placed_objects:
        if obj.name == candidate.name:
            continue
        if obj.object_type in _IGNORED_TYPES:
            continue

        o_aabb = obj.transform.aabb_xy(margin=0.0)
        o_z_min, o_z_max = obj.transform.z_range()

        # Z overlap check comune
        z_overlap = max(0.0, min(c_z_max, o_z_max) - max(c_z_min, o_z_min))
        if z_overlap < 0.02:
            continue

        if obj.category == "structural" and check_walls:
            name_lower = obj.name.lower()
            if any(k in name_lower for k in ("door", "window", "room")):
                continue
            ratio = aabb_overlap_ratio(c_aabb_wall, o_aabb)
            total += ratio * 2.0  # I muri pesano il doppio
        else:
            ratio = aabb_overlap_ratio(c_aabb_furn, o_aabb)
            total += ratio

    return total


def penetration_vector(
    candidate: "SceneObject",
    other: "SceneObject",
    margin: float = 0.05,
) -> tuple[float, float]:
    """
    Calcola il vettore di separazione minimo (MTV) tra due oggetti nel piano XY.

    Restituisce (dx, dy) da applicare al candidato per risolvere la sovrapposizione.
    Se non c'è sovrapposizione, restituisce (0.0, 0.0).

    Utile nel post-LLM solver per spostare intelligentemente gli oggetti
    che si sovrappongono, invece di usare jitter casuale.
    """
    c_cx, c_cy = candidate.transform.geometric_center_xy()
    o_cx, o_cy = other.transform.geometric_center_xy()

    c_aabb = candidate.transform.aabb_xy(margin=margin)
    o_aabb = other.transform.aabb_xy(margin=0.0)

    x_overlap = min(c_aabb[1], o_aabb[1]) - max(c_aabb[0], o_aabb[0])
    y_overlap = min(c_aabb[3], o_aabb[3]) - max(c_aabb[2], o_aabb[2])

    if x_overlap <= 0 or y_overlap <= 0:
        return 0.0, 0.0  # Nessuna sovrapposizione

    # Sposta lungo l'asse con la penetrazione minore (MTV standard)
    if x_overlap < y_overlap:
        dx = x_overlap + 0.01  # +1cm di buffer
        return (dx if c_cx > o_cx else -dx), 0.0
    else:
        dy = y_overlap + 0.01
        return 0.0, (dy if c_cy > o_cy else -dy)


def aabb_overlap_ratio(
    aabb_a: tuple[float, float, float, float],
    aabb_b: tuple[float, float, float, float],
) -> float:
    """
    Rapporto di sovrapposizione tra due AABB 2D.

    Returns:
        Valore in [0.0, 1.0]. 0.0 = nessuna sovrapposizione.
        Usa l'area minima al denominatore per proteggere gli oggetti piccoli.
    """
    x_overlap = max(0.0, min(aabb_a[1], aabb_b[1]) - max(aabb_a[0], aabb_b[0]))
    y_overlap = max(0.0, min(aabb_a[3], aabb_b[3]) - max(aabb_a[2], aabb_b[2]))
    intersection = x_overlap * y_overlap

    if intersection < 1e-6:
        return 0.0

    area_a = (aabb_a[1] - aabb_a[0]) * (aabb_a[3] - aabb_a[2])
    area_b = (aabb_b[1] - aabb_b[0]) * (aabb_b[3] - aabb_b[2])
    min_area = min(area_a, area_b)

    if min_area < 1e-6:
        return 0.0

    return min(intersection / min_area, 1.0)


# ---------------------------------------------------------------------------
# Funzioni di compatibilità con il codice legacy (grouping.py, scene_reorganizer)
# ---------------------------------------------------------------------------

def compute_aabb_2d(
    obj: "SceneObject",
    margin: float = 0.0,
) -> tuple[float, float, float, float]:
    """Wrapper BC: restituisce l'AABB 2D di un oggetto (x_min, x_max, y_min, y_max)."""
    return obj.transform.aabb_xy(margin=margin)


def compute_z_range(obj: "SceneObject") -> tuple[float, float]:
    """Wrapper BC: restituisce (z_min, z_max) dell'oggetto."""
    return obj.transform.z_range()


def _get_obb_corners(
    obj: "SceneObject",
    margin: float = 0.0,
) -> list[tuple[float, float]]:
    """Wrapper BC: restituisce i 4 angoli dell'OBB 2D."""
    return obj.transform.obb_corners_xy(margin=margin)


def _check_sat_overlap(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
) -> bool:
    """Wrapper BC: alias di _sat_overlap (stesso algoritmo SAT)."""
    return _sat_overlap(poly_a, poly_b)