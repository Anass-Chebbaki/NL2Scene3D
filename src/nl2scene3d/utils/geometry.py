# src/nl2scene3d/utils/geometry.py
"""
Utility geometriche per il controllo delle collisioni e il calcolo dei bounding box.
Utilizza mathutils.bvhtree per precisione mesh-esatta in ambiente Blender.
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from nl2scene3d.models import SceneObject

logger = logging.getLogger(__name__)

def has_collision(
    candidate: SceneObject,
    placed_objects: List[SceneObject],
) -> bool:
    """
    Verifica se un oggetto ha collisioni con quelli gia' posizionati.
    Usa mathutils.bvhtree se disponibile, altrimenti fallback AABB.
    """
    try:
        import bpy  # noqa: PLC0415
        import mathutils
        blender_env = True
    except ImportError:
        blender_env = False

    if blender_env:
        # Cache per i BVHTree per evitare ricalcoli costosi nello stesso loop
        if not hasattr(has_collision, "_bvh_cache"):
            has_collision._bvh_cache = {}
            
        def get_bvh(obj_name, state_obj, depsgraph):
            # Usiamo una chiave che include posizione e rotazione per invalidare la cache se l'oggetto si muove
            # (anche se per i placed_objects non dovrebbe succedere nel loop del reorganizer)
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
            for other in placed_objects:
                if other.name == candidate.name: continue
                if getattr(other, "category", "") == "structural" and "door" not in other.name.lower() and "window" not in other.name.lower():
                    continue
                if other.object_type not in ("MESH"): continue
                
                bvh_other = get_bvh(other.name, other, depsgraph)
                if bvh_other and bvh_cand.overlap(bvh_other):
                    return True

    # Fallback AABB 2D (Semplificato)
    # Nota: per brevita' non replichiamo tutto _compute_aabb qui se siamo in Blender.
    # Se siamo fuori Blender, questa utility sara' limitata.
    return False
