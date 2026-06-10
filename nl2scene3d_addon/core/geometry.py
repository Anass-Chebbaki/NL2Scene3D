# nl2scene3d/core/geometry.py
"""
Rilevamento collisioni per NL2Scene3D.

Architettura:
    I calcoli geometrici di base (AABB, angoli OBB, z_range) vivono su Transform.
    Questo modulo e' l'unica fonte di verita' per la geometria di collisione:

        1. sat_overlap():              Separating Axis Theorem per poligoni 2D.
        2. wall_collision():           SAT OBB tra oggetto e muri, con check Z.
        3. furniture_collision():      SAT OBB tra oggetti mobili.
        4. check_openings_clearance(): Zona di rispetto davanti a porte/finestre.
        5. has_collision():            Entry point principale: aggrega i check sopra.
        6. collision_score():          Punteggio "bruttezza" di una posizione.
        7. penetration_vector():       Minimum Translation Vector per risolvere overlap.
        8. aabb_overlap_ratio():       Rapporto di sovrapposizione tra due AABB 2D.
        9. group_aabb_xy():            AABB combinato di un gruppo padre+figli.

Margine di collisione:
    Ogni oggetto viene espanso di `margin` metri su tutti i lati prima del check.
    Valori consigliati:
        - Randomizer:   margin = collision_margin (default 0.05 m = 5 cm)
        - Check post-LLM: margin = 0.02 m (piu' tollerante)
        - Check muri:   margin = wall_margin    (default 0.20 m)

Nota:
    In versioni precedenti _sat_overlap era duplicato in scene_io.py. Qui e'
    definito una volta sola come sat_overlap; i moduli che lo usano importano
    da qui. L'alias _sat_overlap e' mantenuto per retro-compatibilita'.

Modulo puro Python: nessuna dipendenza da bpy.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import RoomBounds, SceneObject

logger = logging.getLogger(__name__)

# Tipi di oggetto Blender che non partecipano al rilevamento collisioni.
_IGNORED_TYPES = frozenset({"CAMERA", "LIGHT", "EMPTY", "SPEAKER", "ARMATURE", "CURVE"})


# ---------------------------------------------------------------------------
# Utility geometriche di base
# ---------------------------------------------------------------------------

def is_finite_float(val: Any) -> bool:
    """True se il valore e' convertibile in un float finito (non NaN, non inf)."""
    try:
        f = float(val)
        return math.isfinite(f)
    except (ValueError, TypeError):
        return False


def snap_rotation_90(rz: float) -> float:
    """Approssima una rotazione Z al multiplo di 90 gradi piu' vicino (0, 90, 180, 270 gradi)."""
    multiples = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
    return min(multiples, key=lambda m: abs(m - (rz % (2 * math.pi))))


# ---------------------------------------------------------------------------
# Separating Axis Theorem (unica definizione del package)
# ---------------------------------------------------------------------------

def sat_overlap(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
) -> bool:
    """
    Separating Axis Theorem per due poligoni convessi 2D.

    Restituisce True se i due poligoni si sovrappongono, False se esiste
    almeno un asse separante.
    """
    def get_axes(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Calcola gli assi normali ai lati del poligono."""
        axes = []
        n    = len(poly)
        for i in range(n):
            p1, p2 = poly[i], poly[(i + 1) % n]
            ex, ey = p2[0] - p1[0], p2[1] - p1[1]
            mag    = math.hypot(ex, ey)
            if mag > 1e-6:
                axes.append((-ey / mag, ex / mag))
        return axes

    def project(
        poly: list[tuple[float, float]],
        axis: tuple[float, float],
    ) -> tuple[float, float]:
        """Proietta il poligono sull'asse dato e restituisce (min, max)."""
        dots = [p[0] * axis[0] + p[1] * axis[1] for p in poly]
        return min(dots), max(dots)

    for axis in get_axes(poly_a) + get_axes(poly_b):
        mn_a, mx_a = project(poly_a, axis)
        mn_b, mx_b = project(poly_b, axis)
        if mx_a < mn_b or mx_b < mn_a:
            return False  # Asse separante trovato: nessuna collisione.

    return True  # Nessun asse separante: i poligoni si sovrappongono.


# Alias per retro-compatibilita' con il codice che usava _sat_overlap.
_sat_overlap = sat_overlap


# ---------------------------------------------------------------------------
# Collisione con i muri
# ---------------------------------------------------------------------------

def wall_collision(
    candidate:    "SceneObject",
    wall_objects: list["SceneObject"],
    wall_margin:  float = 0.20,
) -> bool:
    """
    Verifica se il candidato (espanso di wall_margin) penetra un muro fisico.

    Usa il SAT su OBB (non solo AABB) cosi' che un mobile ruotato a 45 gradi
    non produca falsi negativi contro muri sottili. wall_margin garantisce che
    i mobili restino sempre a distanza dai muri.

    Porte, finestre e mesh "room" sono escluse da questo check perche' non
    sono ostacoli fisici da evitare.
    """
    cand_poly      = candidate.transform.obb_corners_xy(margin=wall_margin)
    c_z_min, c_z_max = candidate.transform.z_range()

    for wall in wall_objects:
        name_lower = wall.name.lower()
        if any(k in name_lower for k in ("door", "window", "room", "porta", "finestra", "stanza")):
            continue  # escludi aperture e mesh-stanza

        # Ottimizzazione: check rapido di sovrapposizione Z prima del SAT.
        w_z_min, w_z_max = wall.transform.z_range()
        z_overlap = max(0.0, min(c_z_max, w_z_max) - max(c_z_min, w_z_min))
        if z_overlap <= 0.01:
            continue

        wall_poly = wall.transform.obb_corners_xy(margin=0.0)
        if sat_overlap(cand_poly, wall_poly):
            logger.debug(
                "Collisione muro (SAT): '%s' vs '%s' (sovrapp. Z: %.3f).",
                candidate.name, wall.name, z_overlap,
            )
            return True

    return False


# ---------------------------------------------------------------------------
# Collisione tra mobili
# ---------------------------------------------------------------------------

def furniture_collision(
    candidate:         "SceneObject",
    furniture_objects: list["SceneObject"],
    margin:            float = 0.05,
) -> bool:
    """
    Verifica se il candidato si sovrappone a un altro oggetto mobile.

    Usa il SAT su OBB 2D per gestire correttamente oggetti ruotati. Il margine
    espande ogni OBB di `margin` metri per garantire gioco fisico visivo.
    La soglia Z e' 0.01 m per intercettare anche oggetti quasi complanari
    (es. tappeto sotto una sedia, oggetti sul bordo di un tavolo).
    """
    cand_poly             = candidate.transform.obb_corners_xy(margin=margin)
    cand_z_min, cand_z_max = candidate.transform.z_range()

    for other in furniture_objects:
        if other.name == candidate.name:
            continue

        o_z_min, o_z_max = other.transform.z_range()
        z_overlap = max(0.0, min(cand_z_max, o_z_max) - max(cand_z_min, o_z_min))
        if z_overlap < 0.01:
            continue

        other_poly = other.transform.obb_corners_xy(margin=margin)
        if sat_overlap(cand_poly, other_poly):
            logger.debug("Collisione SAT: '%s' vs '%s'.", candidate.name, other.name)
            return True

    return False


# ---------------------------------------------------------------------------
# Clearance davanti ad aperture (porte e finestre)
# ---------------------------------------------------------------------------

def check_openings_clearance(
    candidate:          "SceneObject",
    structural_objects: list["SceneObject"],
) -> bool:
    """
    Verifica se il candidato invade la zona di rispetto davanti a una porta o finestra.

    Restituisce True se c'e' invasione della clearance zone.

    Profondita' della zona di rispetto:
        - Porta:    0.90 m (passaggio + raggio di apertura)
        - Finestra: 0.50 m (luce + accesso)
    """
    for obj in structural_objects:
        name_lower = obj.name.lower()
        is_door    = any(k in name_lower for k in ("door", "porta"))
        is_window  = any(k in name_lower for k in ("window", "finestra"))

        if not (is_door or is_window):
            continue

        clearance_depth = 0.90 if is_door else 0.50

        cx, cy         = obj.transform.geometric_center_xy()
        rz             = obj.transform.rotation_euler[2]
        cos_z, sin_z   = math.cos(rz), math.sin(rz)
        dim            = obj.transform.dimensions

        # Estende la dimensione Y locale per includere la clearance su entrambi i lati.
        w = dim[0] / 2.0                      # semi-larghezza strutturale
        h = dim[1] / 2.0 + clearance_depth    # semi-profondita' estesa

        local_corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
        clearance_poly = [
            (cx + lx * cos_z - ly * sin_z, cy + lx * sin_z + ly * cos_z)
            for lx, ly in local_corners
        ]

        cand_poly = candidate.transform.obb_corners_xy(margin=0.02)

        if sat_overlap(clearance_poly, cand_poly):
            c_z_min, c_z_max = candidate.transform.z_range()
            o_z_min, o_z_max = obj.transform.z_range()
            z_overlap = max(0.0, min(c_z_max, o_z_max) - max(c_z_min, o_z_min))

            if is_door:
                # Per le porte basta qualsiasi sovrapposizione Z significativa.
                if z_overlap > 0.05:
                    logger.debug(
                        "Collisione porta: '%s' blocca il passaggio di '%s'.",
                        candidate.name, obj.name,
                    )
                    return True
            elif is_window:
                # Per le finestre blocca solo se l'oggetto supera il davanzale.
                if c_z_max > o_z_min + 0.10 and z_overlap > 0.05:
                    logger.debug(
                        "Collisione finestra: '%s' copre la luce di '%s'.",
                        candidate.name, obj.name,
                    )
                    return True

    return False


# ---------------------------------------------------------------------------
# Entry point principale: has_collision
# ---------------------------------------------------------------------------

def has_collision(
    candidate:        "SceneObject",
    placed_objects:   list["SceneObject"],
    wall_margin:      float = 0.20,
    furniture_margin: float = 0.05,
    check_walls:      bool  = True,
    room_bounds:      "RoomBounds | None" = None,
) -> bool:
    """
    Verifica se il candidato collide con almeno un oggetto gia' piazzato.

    Args:
        candidate:        Oggetto da testare.
        placed_objects:   Oggetti gia' piazzati (strutturali e mobili).
        wall_margin:      Distanza minima dai muri (m).
        furniture_margin: Margine di espansione OBB per i mobili (m).
                          Con 0.05 m ogni coppia avra' almeno 10 cm di gioco.
        check_walls:      Se False, salta i check sui muri (utile per decorazioni a parete).
        room_bounds:      RoomBounds opzionale. Se fornito, verifica anche che il candidato
                          sia completamente contenuto nella stanza (check piu' affidabile
                          perche' non dipende dalla presenza di mesh-muro).

    Restituisce:
        True se viene rilevata almeno una collisione.
    """
    # Check di contenimento nei confini stanza (il piu' affidabile e veloce).
    if check_walls and room_bounds is not None:
        c_aabb = candidate.transform.aabb_xy(margin=0.0)
        if not room_bounds.contains_aabb(c_aabb, margin=wall_margin):
            logger.debug(
                "Fuori dai confini: '%s' AABB %s non contenuto (margin=%.2f).",
                candidate.name, c_aabb, wall_margin,
            )
            return True

    walls:     list["SceneObject"] = []
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
        if check_openings_clearance(candidate, walls):
            return True

    if furniture:
        if furniture_collision(candidate, furniture, furniture_margin):
            return True

    return False


# ---------------------------------------------------------------------------
# Punteggio di "bruttezza" di una posizione
# ---------------------------------------------------------------------------

def collision_score(
    candidate:        "SceneObject",
    placed_objects:   list["SceneObject"],
    wall_margin:      float = 0.20,
    furniture_margin: float = 0.05,
    check_walls:      bool  = True,
    room_bounds:      "RoomBounds | None" = None,
) -> float:
    """
    Restituisce un punteggio di "bruttezza" della posizione corrente del candidato.

        0.0   = nessuna collisione (posizione perfetta).
        > 0.0 = sovrapposizione presente; valori piu' alti = posizione peggiore.

    Usato dal randomizer per scegliere la posizione meno problematica quando
    non si trova un piazzamento privo di collisioni entro max_attempts.

    Se room_bounds e' fornito, viene aggiunta una penalita' proporzionale
    all'overflow fuori dai confini della stanza.
    """
    # Ritorna 0.0 immediatamente se il SAT esatto non rileva collisioni,
    # evitando falsi positivi dovuti all'approssimazione AABB su oggetti ruotati.
    if not has_collision(
        candidate, placed_objects, wall_margin, furniture_margin, check_walls, room_bounds
    ):
        return 0.0

    total = 0.0

    c_aabb_base = candidate.transform.aabb_xy(margin=0.0)
    c_aabb_wall = candidate.transform.aabb_xy(margin=wall_margin)
    c_aabb_furn = candidate.transform.aabb_xy(margin=furniture_margin)
    c_z_min, c_z_max = candidate.transform.z_range()

    # Penalita' per oggetti fuori dai confini della stanza.
    if check_walls and room_bounds is not None:
        if not room_bounds.contains_aabb(c_aabb_base, margin=wall_margin):
            overflow = (
                max(0.0, room_bounds.x_min + wall_margin - c_aabb_base[0])
                + max(0.0, c_aabb_base[1] - (room_bounds.x_max - wall_margin))
                + max(0.0, room_bounds.y_min + wall_margin - c_aabb_base[2])
                + max(0.0, c_aabb_base[3] - (room_bounds.y_max - wall_margin))
            )
            total += 100.0 + overflow * 10.0  # penalita' bloccante proporzionale

    for obj in placed_objects:
        if obj.name == candidate.name:
            continue
        if obj.object_type in _IGNORED_TYPES:
            continue

        o_aabb             = obj.transform.aabb_xy(margin=0.0)
        o_z_min, o_z_max   = obj.transform.z_range()
        z_overlap = max(0.0, min(c_z_max, o_z_max) - max(c_z_min, o_z_min))
        if z_overlap < 0.02:
            continue

        if obj.category == "structural" and check_walls:
            name_lower = obj.name.lower()
            if any(k in name_lower for k in ("door", "window", "room", "porta", "finestra", "stanza")):
                is_door   = any(k in name_lower for k in ("door", "porta"))
                is_window = any(k in name_lower for k in ("window", "finestra"))

                clearance_depth = 0.90 if is_door else 0.50
                cx, cy          = obj.transform.geometric_center_xy()
                rz              = obj.transform.rotation_euler[2]
                cos_z, sin_z    = math.cos(rz), math.sin(rz)
                dim             = obj.transform.dimensions
                w               = dim[0] / 2.0
                h               = dim[1] / 2.0 + clearance_depth

                local_corners  = [(-w, -h), (w, -h), (w, h), (-w, h)]
                clearance_poly = [
                    (cx + lx * cos_z - ly * sin_z, cy + lx * sin_z + ly * cos_z)
                    for lx, ly in local_corners
                ]
                cand_poly = candidate.transform.obb_corners_xy(margin=0.02)

                if sat_overlap(clearance_poly, cand_poly):
                    real_z_overlap = max(0.0, min(c_z_max, o_z_max) - max(c_z_min, o_z_min))
                    if is_door and real_z_overlap > 0.05:
                        total += 50.0   # penalita' pesante per le porte
                    elif is_window and c_z_max > o_z_min + 0.10 and real_z_overlap > 0.05:
                        total += 25.0   # penalita' piu' leggera per le finestre
                continue

            ratio  = aabb_overlap_ratio(c_aabb_wall, o_aabb)
            total += ratio * 2.0  # i muri pesano il doppio dei mobili
        else:
            ratio  = aabb_overlap_ratio(c_aabb_furn, o_aabb)
            total += ratio

    return total


# ---------------------------------------------------------------------------
# Minimum Translation Vector (MTV)
# ---------------------------------------------------------------------------

def penetration_vector(
    candidate: "SceneObject",
    other:     "SceneObject",
    margin:    float = 0.05,
) -> tuple[float, float]:
    """
    Calcola il Minimum Translation Vector (MTV) tra due oggetti nel piano XY.

    Restituisce (dx, dy) da applicare al candidato per risolvere la
    sovrapposizione. Restituisce (0.0, 0.0) se non c'e' sovrapposizione.

    Usato nel solver post-LLM per spostare gli oggetti sovrapposti in modo
    intelligente invece di usare jitter casuale.
    """
    c_cx, c_cy = candidate.transform.geometric_center_xy()
    o_cx, o_cy = other.transform.geometric_center_xy()

    c_aabb = candidate.transform.aabb_xy(margin=margin)
    o_aabb = other.transform.aabb_xy(margin=0.0)

    x_overlap = min(c_aabb[1], o_aabb[1]) - max(c_aabb[0], o_aabb[0])
    y_overlap = min(c_aabb[3], o_aabb[3]) - max(c_aabb[2], o_aabb[2])

    if x_overlap <= 0 or y_overlap <= 0:
        return 0.0, 0.0  # nessuna sovrapposizione

    # Spinge lungo l'asse con la penetrazione minore (MTV standard).
    if x_overlap < y_overlap:
        dx = x_overlap + 0.01  # +1 cm di buffer aggiuntivo
        return (dx if c_cx > o_cx else -dx), 0.0
    else:
        dy = y_overlap + 0.01
        return 0.0, (dy if c_cy > o_cy else -dy)


# ---------------------------------------------------------------------------
# Rapporto di sovrapposizione AABB
# ---------------------------------------------------------------------------

def aabb_overlap_ratio(
    aabb_a: tuple[float, float, float, float],
    aabb_b: tuple[float, float, float, float],
) -> float:
    """
    Calcola il rapporto di sovrapposizione tra due AABB 2D.

    Restituisce un valore in [0.0, 1.0]:
        0.0 = nessuna sovrapposizione.
        1.0 = sovrapposizione totale.

    Usa l'area dell'AABB piu' piccolo come denominatore per proteggere
    gli oggetti piccoli da penalita' sproporzionate.
    """
    x_overlap    = max(0.0, min(aabb_a[1], aabb_b[1]) - max(aabb_a[0], aabb_b[0]))
    y_overlap    = max(0.0, min(aabb_a[3], aabb_b[3]) - max(aabb_a[2], aabb_b[2]))
    intersection = x_overlap * y_overlap

    if intersection < 1e-6:
        return 0.0

    area_a   = (aabb_a[1] - aabb_a[0]) * (aabb_a[3] - aabb_a[2])
    area_b   = (aabb_b[1] - aabb_b[0]) * (aabb_b[3] - aabb_b[2])
    min_area = min(area_a, area_b)

    if min_area < 1e-6:
        return 0.0

    return min(intersection / min_area, 1.0)


# ---------------------------------------------------------------------------
# AABB combinato di un gruppo (padre + figli)
# ---------------------------------------------------------------------------

def group_aabb_xy(
    orig_parent:    "SceneObject",
    proposed_loc:   list[float],
    proposed_rz:    float,
    orig_children:  list["SceneObject"],
    margin:         float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Calcola l'AABB XY combinato di un gruppo padre+figli a una posizione proposta.

    Usa l'AABB reale di ogni membro (inclusa rotazione e origin offset) tramite
    la classe Transform, cosi' il risultato e' sempre geometricamente accurato.

    Args:
        orig_parent:   Oggetto padre nella posa originale.
        proposed_loc:  Nuova posizione [x, y, z] proposta per il padre.
        proposed_rz:   Nuova rotazione Z proposta per il padre.
        orig_children: Figli nella loro posa originale.
        margin:        Margine da aggiungere a ogni OBB membro.

    Restituisce:
        (x_min, x_max, y_min, y_max) dell'AABB combinato del gruppo.
    """
    from .models import Transform  # import locale per evitare dipendenze circolari

    old_parent_loc = orig_parent.transform.location
    old_parent_rz  = orig_parent.transform.rotation_euler[2]
    d_rz           = proposed_rz - old_parent_rz
    cos_a, sin_a   = math.cos(d_rz), math.sin(d_rz)

    # Calcola l'AABB del padre alla posizione proposta.
    temp_parent_tf = Transform(
        location=[proposed_loc[0], proposed_loc[1], orig_parent.transform.location[2]],
        rotation_euler=[
            orig_parent.transform.rotation_euler[0],
            orig_parent.transform.rotation_euler[1],
            proposed_rz,
        ],
        dimensions=orig_parent.transform.dimensions,
        origin_offset=orig_parent.transform.origin_offset,
    )
    x_min, x_max, y_min, y_max = temp_parent_tf.aabb_xy(margin=margin)

    # Aggiunge l'AABB di ogni figlio, ruotato rigidamente col padre.
    for orig_child in orig_children:
        rel_x  = orig_child.transform.location[0] - old_parent_loc[0]
        rel_y  = orig_child.transform.location[1] - old_parent_loc[1]
        new_cx = proposed_loc[0] + rel_x * cos_a - rel_y * sin_a
        new_cy = proposed_loc[1] + rel_x * sin_a + rel_y * cos_a
        c_rz   = (orig_child.transform.rotation_euler[2] + d_rz) % (2 * math.pi)

        temp_child_tf = Transform(
            location=[new_cx, new_cy, orig_child.transform.location[2]],
            rotation_euler=[
                orig_child.transform.rotation_euler[0],
                orig_child.transform.rotation_euler[1],
                c_rz,
            ],
            dimensions=orig_child.transform.dimensions,
            origin_offset=orig_child.transform.origin_offset,
        )
        cx_min, cx_max, cy_min, cy_max = temp_child_tf.aabb_xy(margin=margin)

        x_min = min(x_min, cx_min)
        x_max = max(x_max, cx_max)
        y_min = min(y_min, cy_min)
        y_max = max(y_max, cy_max)

    return x_min, x_max, y_min, y_max