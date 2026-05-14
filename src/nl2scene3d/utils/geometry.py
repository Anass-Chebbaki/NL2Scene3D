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

    # Ingombro dell'AABB ruotata
    cos_z_abs = abs(cos_z_rot)
    sin_z_abs = abs(sin_z_rot)
    eff_x = dim[0] * cos_z_abs + dim[1] * sin_z_abs
    eff_y = dim[0] * sin_z_abs + dim[1] * cos_z_abs

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


def has_collision(
    candidate: "SceneObject",
    placed_objects: List["SceneObject"],
    check_walls: bool = True,
    wall_margin: float = 0.05,
) -> bool:
    """
    Verifica se un oggetto ha collisioni con quelli gia' posizionati.
    
    Strategia a due livelli:
    1. Per gli oggetti strutturali (muri): check AABB 2D con margine
    2. Per i mobili: check BVH mesh-esatto (Blender) + fallback AABB 2D
    
    Args:
        candidate: Oggetto da testare.
        placed_objects: Lista di oggetti già piazzati.
        check_walls: Se True, verifica anche le collisioni con i muri.
        wall_margin: Margine aggiuntivo per i muri.
        
    Returns:
        True se c'è una collisione.
    """
    # Separa muri da mobili
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
    
    # 2. Check collisioni con mobili (BVH mesh + fallback AABB)
    try:
        import bpy  # noqa: PLC0415
        import mathutils
        blender_env = True
    except ImportError:
        blender_env = False

    if blender_env and furniture_objects:
        # Cache per i BVHTree per evitare ricalcoli costosi nello stesso loop
        if not hasattr(has_collision, "_bvh_cache"):
            has_collision._bvh_cache = {}
            
        def get_bvh(obj_name, state_obj, depsgraph):
            cache_key = (obj_name, tuple(state_obj.transform.location), tuple(state_obj.transform.rotation_euler))
            if cache_key in has_collision._bvh_cache:
                return has_collision._bvh_cache[cache_key]
            
            blender_obj = bpy.data.objects.get(obj_name)
            if not blender_obj or blender_obj.type != 'MESH':
                return None
                
            loc = mathutils.Vector(state_obj.transform.location)
            rot = mathutils.Euler(state_obj.transform.rotation_euler, 'XYZ')
            scale = blender_obj.scale
            mat = mathutils.Matrix.Translation(loc) @ rot.to_matrix().to_4x4()
            mat = mat @ mathutils.Matrix.Diagonal(scale.to_4d())
            
            eval_obj = blender_obj.evaluated_get(depsgraph)
            mesh = eval_obj.to_mesh()
            verts = [mat @ v.co for v in mesh.vertices]
            polys = [p.vertices for p in mesh.polygons]
            
            if polys:
                bvh = mathutils.bvhtree.BVHTree.FromPolygons(verts, polys)
                eval_obj.to_mesh_clear()
                has_collision._bvh_cache[cache_key] = bvh
                return bvh
            eval_obj.to_mesh_clear()
            return None

        depsgraph = bpy.context.evaluated_depsgraph_get()
        bvh_cand = get_bvh(candidate.name, candidate, depsgraph)
        
        if bvh_cand:
            for other in furniture_objects:
                if other.object_type != "MESH":
                    continue
                
                bvh_other = get_bvh(other.name, other, depsgraph)
                if bvh_other and bvh_cand.overlap(bvh_other):
                    return True
        else:
            # Fallback AABB 2D per oggetti non-mesh
            cand_aabb = compute_aabb_2d(candidate)
            for other in furniture_objects:
                other_aabb = compute_aabb_2d(other)
                ratio = aabb_overlap_ratio(cand_aabb, other_aabb)
                if ratio > 0.05:
                    return True
    elif furniture_objects:
        # Fuori da Blender: solo AABB 2D
        cand_aabb = compute_aabb_2d(candidate)
        for other in furniture_objects:
            other_aabb = compute_aabb_2d(other)
            ratio = aabb_overlap_ratio(cand_aabb, other_aabb)
            if ratio > 0.05:
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
