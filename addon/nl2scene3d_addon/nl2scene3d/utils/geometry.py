# src/nl2scene3d/utils/geometry.py
"""
Utility geometriche per il controllo delle collisioni e il calcolo dei bounding box.
Utilizza mathutils.bvhtree per precisione mesh-esatta in ambiente Blender.

NOTA ARCHITETTURALE:
- Per i MURI (structural): usa check AABB 2D, perché i muri hanno geometrie
  complesse (modanature, battiscopa, ecc.) che causano falsi positivi con BVH.
  Il check AABB basta per assicurarsi che un mobile non penetri un muro.
- Per i MOBILI (movable): usa BVH mesh-esatto se disponibile, con fallback AABB.
"""
from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from nl2scene3d.models import SceneObject

logger = logging.getLogger(__name__)


def compute_aabb_2d(obj: "SceneObject") -> Tuple[float, float, float, float]:
    """
    Calcola l'Axis-Aligned Bounding Box (AABB) 2D di un oggetto nel piano XY.
    Tiene conto della rotazione sull'asse Z e dell'offset dell'origine.
    """
    loc = obj.transform.location
    dim = obj.transform.dimensions
    rz = obj.transform.rotation_euler[2]
    off = obj.transform.origin_offset

    # Ruotiamo l'offset locale per trovare il centro geometrico in coordinate mondo
    cos_z_rot = math.cos(rz)
    sin_z_rot = math.sin(rz)
    world_off_x = off[0] * cos_z_rot - off[1] * sin_z_rot
    world_off_y = off[0] * sin_z_rot + off[1] * cos_z_rot

    # Centro reale dell'AABB
    center_x = loc[0] + world_off_x
    center_y = loc[1] + world_off_y

    # Ingombro dell'AABB ruotata con un piccolo margine di sicurezza (1cm)
    cos_z_abs = abs(cos_z_rot)
    sin_z_abs = abs(sin_z_rot)
    eff_x = dim[0] * cos_z_abs + dim[1] * sin_z_abs + 0.01
    eff_y = dim[0] * sin_z_abs + dim[1] * cos_z_abs + 0.01

    half_x = eff_x / 2.0
    half_y = eff_y / 2.0
    
    return (
        center_x - half_x,
        center_x + half_x,
        center_y - half_y,
        center_y + half_y,
    )


def compute_z_range(obj: "SceneObject") -> Tuple[float, float]:
    """
    Calcola il range Z dell'oggetto (quota_min, quota_max) considerando l'offset.
    """
    loc_z = obj.transform.location[2]
    off_z = obj.transform.origin_offset[2]
    half_h = obj.transform.dimensions[2] / 2.0
    
    center_z = loc_z + off_z
    return (center_z - half_h, center_z + half_h)


def aabb_overlap_ratio(
    aabb_a: Tuple[float, float, float, float],
    aabb_b: Tuple[float, float, float, float],
) -> float:
    """
    Calcola il rapporto di sovrapposizione tra due AABB 2D.
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

    return intersection / min_area


def _check_wall_collision(
    candidate: "SceneObject",
    wall_objects: List["SceneObject"],
    wall_margin: float = 0.05,
) -> bool:
    """
    Verifica se un oggetto collide con i muri della stanza usando AABB 2D.
    Un margine viene aggiunto per evitare che i mobili siano incollati ai muri.
    
    NON controlla porte/finestre (che sono aperture, non ostacoli).
    
    Args:
        candidate: Oggetto da testare.
        wall_objects: Lista di oggetti strutturali (muri).
        wall_margin: Margine aggiuntivo in metri.
        
    Returns:
        True se c'è collisione con un muro.
    """
    cand_aabb = compute_aabb_2d(candidate)
    # Espandi il candidate AABB con il margine
    cand_expanded = (
        cand_aabb[0] - wall_margin,
        cand_aabb[1] + wall_margin,
        cand_aabb[2] - wall_margin,
        cand_aabb[3] + wall_margin,
    )
    
    for wall in wall_objects:
        name_lower = wall.name.lower()
        # Porte, finestre e la stanza stessa sono aperture o contenitori, non ostacoli AABB solidi
        if "door" in name_lower or "window" in name_lower or "room" in name_lower:
            continue
        # Solo i muri veri (non pavimento/soffitto che sono in Z diverso)
        # Un muro ha tipicamente una dimensione molto sottile in una delle due direzioni XY
        # e altezza significativa
        wall_aabb = compute_aabb_2d(wall)
        
        # Check: l'oggetto si sovrappone all'area del muro nel piano XY?
        x_overlap = max(0.0, min(cand_expanded[1], wall_aabb[1]) - max(cand_expanded[0], wall_aabb[0]))
        y_overlap = max(0.0, min(cand_expanded[3], wall_aabb[3]) - max(cand_expanded[2], wall_aabb[2]))
        
        if x_overlap > 0.01 and y_overlap > 0.01:
            # Verifica aggiuntiva: il muro è effettivamente un ostacolo planare?
            # Pavimenti e soffitti hanno Z molto diversa dagli oggetti
            wall_z = wall.transform.location[2]
            cand_z = candidate.transform.location[2]
            wall_height = wall.transform.dimensions[2]
            cand_height = candidate.transform.dimensions[2]
            
            # Se il muro è al livello del pavimento/soffitto e non si
            # sovrappone verticalmente con l'oggetto, non è una collisione
            wall_z_min = wall_z - wall_height / 2.0
            wall_z_max = wall_z + wall_height / 2.0
            cand_z_min = cand_z - cand_height / 2.0
            cand_z_max = cand_z + cand_height / 2.0
            
            z_overlap = max(0.0, min(wall_z_max, cand_z_max) - max(wall_z_min, cand_z_min))
            if z_overlap > 0.01:
                logger.debug(
                    "Collisione AABB con muro '%s': overlap XY=(%.3f, %.3f), Z=%.3f",
                    wall.name, x_overlap, y_overlap, z_overlap,
                )
                return True
    
    return False


def _get_obb_corners(obj: "SceneObject", margin: float = 0.05) -> List[Tuple[float, float]]:
    """Restituisce i 4 angoli dell'Oriented Bounding Box (OBB) 2D nel piano XY."""
    loc = obj.transform.location
    dim = obj.transform.dimensions
    rz = obj.transform.rotation_euler[2]
    off = obj.transform.origin_offset
    
    # Centro mondo dell'oggetto (applicando rotazione all'offset)
    c, s = math.cos(rz), math.sin(rz)
    world_off_x = off[0] * c - off[1] * s
    world_off_y = off[0] * s + off[1] * c
    cx, cy = loc[0] + world_off_x, loc[1] + world_off_y
    
    # Dimensioni con margine (5cm)
    w, h = (dim[0] + margin) / 2.0, (dim[1] + margin) / 2.0
    
    # Angoli locali (rispetto al centro geometrico)
    local_corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
    
    # Ruota e trasla gli angoli in coordinate mondo
    world_corners = []
    for lx, ly in local_corners:
        wx = cx + (lx * c - ly * s)
        wy = cy + (lx * s + ly * c)
        world_corners.append((wx, wy))
    return world_corners


def _check_sat_overlap(poly_a: List[Tuple[float, float]], poly_b: List[Tuple[float, float]]) -> bool:
    """Implementazione del Separating Axis Theorem (SAT) per due poligoni convessi."""
    def get_axes(poly):
        axes = []
        for i in range(len(poly)):
            p1, p2 = poly[i], poly[(i + 1) % len(poly)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            normal = (-edge[1], edge[0]) # Perpendicolare
            mag = math.sqrt(normal[0]**2 + normal[1]**2)
            if mag > 1e-6:
                axes.append((normal[0]/mag, normal[1]/mag))
        return axes

    def project(poly, axis):
        dots = [p[0] * axis[0] + p[1] * axis[1] for p in poly]
        return min(dots), max(dots)

    axes = get_axes(poly_a) + get_axes(poly_b)
    for axis in axes:
        min_a, max_a = project(poly_a, axis)
        min_b, max_b = project(poly_b, axis)
        if max_a < min_b or max_b < min_a:
            return False # Trovato un asse di separazione: nessuna collisione
    return True # Nessun asse di separazione trovato: c'e' collisione


def has_collision(
    candidate: "SceneObject",
    placed_objects: List["SceneObject"],
    check_walls: bool = True,
    wall_margin: float = 0.05,
) -> bool:
    """
    Verifica se un oggetto ha collisioni con quelli gia' posizionati o con i muri.
    Usa il Separating Axis Theorem (SAT) per precisione millimetrica tra OBB ruotati.
    """
    # Filtriamo gli oggetti validi
    wall_objects = []
    furniture_objects = []
    for other in placed_objects:
        if other.name == candidate.name:
            continue
        if other.object_type in ("CAMERA", "LIGHT", "EMPTY", "SPEAKER", "ARMATURE"):
            continue
        if other.category == "structural":
            wall_objects.append(other)
        else:
            furniture_objects.append(other)
    
    # 1. Check collisioni con muri (AABB 2D + Z overlap)
    if check_walls and wall_objects:
        if _check_wall_collision(candidate, wall_objects, wall_margin):
            return True
    
    # 2. Check collisioni con mobili (SAT - Oriented Bounding Box)
    if not furniture_objects:
        return False

    cand_poly = _get_obb_corners(candidate)
    
    for other in furniture_objects:
        other_poly = _get_obb_corners(other)
        if _check_sat_overlap(cand_poly, other_poly):
            logger.debug("COLLISIONE SAT RILEVATA tra %s e %s", candidate.name, other.name)
            return True

    return False


def clear_bvh_cache() -> None:
    """Svuota la cache BVH per forzare il ricalcolo."""
    if hasattr(has_collision, "_bvh_cache"):
        has_collision._bvh_cache.clear()


def compute_scene_collision_ratio(
    candidate: "SceneObject",
    placed_objects: List["SceneObject"],
    check_walls: bool = True,
    wall_margin: float = 0.05,
) -> float:
    """
    Calcola il rapporto massimo di sovrapposizione tra un candidato e gli oggetti piazzati.
    Utile per il randomizzatore per valutare la 'bonta' di una posizione casuale.
    
    Args:
        candidate: Oggetto da testare.
        placed_objects: Lista di oggetti già piazzati.
        check_walls: Se True, include i muri nel calcolo.
        wall_margin: Margine da aggiungere ai muri.
        
    Returns:
        Rapporto di sovrapposizione massimo [0.0, 1.0].
    """
    max_ratio = 0.0
    cand_aabb = compute_aabb_2d(candidate)
    cand_z = compute_z_range(candidate)
    
    # Per i muri usiamo un AABB espanso per mantenere il margine
    cand_expanded = (
        cand_aabb[0] - wall_margin,
        cand_aabb[1] + wall_margin,
        cand_aabb[2] - wall_margin,
        cand_aabb[3] + wall_margin,
    )
    
    for other in placed_objects:
        if other.name == candidate.name:
            continue
            
        if other.category == "structural" and "room" in other.name.lower():
            continue
            
        other_aabb = compute_aabb_2d(other)
        xy_ratio = aabb_overlap_ratio(cand_aabb if other.category != "structural" else cand_expanded, other_aabb)
        
        if xy_ratio < 0.01:
            continue
            
        # Se c'e' overlap XY, verifichiamo la Z
        other_z = compute_z_range(other)
        z_overlap = max(0.0, min(cand_z[1], other_z[1]) - max(cand_z[0], other_z[0]))
        
        # Se non si toccano in verticale (almeno 2cm di overlap), ignoriamo la collisione XY
        if z_overlap < 0.02:
            continue
            
        if other.category == "structural" and check_walls:
            # Check specifico per i muri (usa l'AABB espanso)
            max_ratio = max(max_ratio, xy_ratio * 1.5)
        else:
            # Check per mobili (AABB standard)
            max_ratio = max(max_ratio, xy_ratio)
            
    return min(max_ratio, 1.0)
