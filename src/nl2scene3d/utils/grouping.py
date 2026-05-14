# src/nl2scene3d/utils/grouping.py
"""
Utility per il raggruppamento spaziale degli oggetti (Rigid Body Groups).

Rileva relazioni genitore-figlio basandosi su coordinate e dimensioni (AABB)
per garantire che oggetti "appoggiati" su altri si muovano insieme.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Dict, List, Set

if TYPE_CHECKING:
    from nl2scene3d.models import SceneObject

logger = logging.getLogger(__name__)


def find_object_groups(objects: List["SceneObject"], z_tolerance: float = 0.15) -> Dict[str, List[str]]:
    """
    Rileva le relazioni spaziali tra gli oggetti, identificando i "gruppi".
    Un oggetto e' figlio di un altro se e' appoggiato sopra di esso.

    Restituisce un dizionario contenente solo le radici (root parents)
    e la lista piatta di tutti i loro discendenti (figli, nipoti, ecc.).

    Args:
        objects: Lista degli oggetti della scena.
        z_tolerance: Tolleranza massima in Z per considerare un oggetto "appoggiato".

    Returns:
        Dizionario `root_name -> list[descendant_names]`.
    """
    from nl2scene3d.utils.geometry import compute_aabb_2d, compute_z_range, aabb_overlap_ratio

    obj_data = {}
    for obj in objects:
        obj_data[obj.name] = {
            "obj": obj,
            "aabb": compute_aabb_2d(obj),
            "z_range": compute_z_range(obj),
            "area": obj.transform.dimensions[0] * obj.transform.dimensions[1]
        }

    direct_groups: Dict[str, List[str]] = {obj.name: [] for obj in objects}
    child_to_parent: Dict[str, str] = {}

    for child_name, child_info in obj_data.items():
        if child_info["obj"].category == "structural":
            continue

        best_parent = None
        max_overlap_score = 0.0
        min_z_diff = float("inf")

        for parent_name, parent_info in obj_data.items():
            if child_name == parent_name:
                continue

            if parent_info["obj"].category == "structural":
                continue

            # Il genitore deve avere un'area maggiore
            if parent_info["area"] <= child_info["area"]:
                continue

            child_z = child_info["obj"].transform.location[2]
            child_h = child_info["obj"].transform.dimensions[2]
            child_off_z = child_info["obj"].transform.origin_offset[2]
            
            parent_z = parent_info["obj"].transform.location[2]
            parent_h = parent_info["obj"].transform.dimensions[2]
            parent_off_z = parent_info["obj"].transform.origin_offset[2]

            # Calcola le quote reali di base e top usando gli offset
            child_bottom = child_z + child_off_z - (child_h / 2.0)
            parent_top = parent_z + parent_off_z + (parent_h / 2.0)
            
            # Il figlio deve essere appoggiato sopra il genitore (tolleranza 10cm)
            z_diff = child_bottom - parent_top
            if abs(z_diff) > 0.1:
                # Caso speciale: se il figlio è leggermente penetrato (z_diff < 0) 
                # o leggermente fluttuante (z_diff > 0)
                if z_diff < -0.15 or z_diff > 0.25:
                    continue

            overlap = aabb_overlap_ratio(child_info["aabb"], parent_info["aabb"])
            
            if overlap > 0.5:  # Almeno il 50% del figlio deve essere sopra il genitore
                score = overlap
                # Selezioniamo il genitore con Z piu' vicino (minimo gap)
                if abs(z_diff) < min_z_diff:
                    min_z_diff = abs(z_diff)
                    max_overlap_score = score
                    best_parent = parent_name

        if best_parent:
            direct_groups[best_parent].append(child_name)
            child_to_parent[child_name] = best_parent

    # Identifica i nodi radice (quelli che non sono figli di nessuno)
    roots = [name for name in obj_data.keys() if name not in child_to_parent]
    
    root_groups: Dict[str, List[str]] = {}
    
    def get_all_descendants(parent_name: str) -> List[str]:
        descendants = []
        for child in direct_groups.get(parent_name, []):
            descendants.append(child)
            descendants.extend(get_all_descendants(child))
        return descendants

    for root in roots:
        descendants = get_all_descendants(root)
        if descendants:
            # Eliminiamo eventuali duplicati, mantenendo l'ordine
            unique_desc = list(dict.fromkeys(descendants))
            root_groups[root] = unique_desc
            logger.debug("Gruppo trovato: %s -> %s", root, unique_desc)

    return root_groups


def apply_group_transform(
    child: "SceneObject",
    old_parent_loc: List[float],
    old_parent_rot: List[float],
    new_parent_loc: List[float],
    new_parent_rot: List[float]
) -> None:
    """
    Applica una trasformazione rigida a un oggetto figlio in base allo
    spostamento e rotazione subiti dal genitore.

    Args:
        child: L'oggetto figlio da trasformare (modificato in-place).
        old_parent_loc: Posizione originale [x, y, z] del genitore.
        old_parent_rot: Rotazione originale [rx, ry, rz] del genitore.
        new_parent_loc: Nuova posizione [x, y, z] del genitore.
        new_parent_rot: Nuova rotazione [rx, ry, rz] del genitore.
    """
    # Calcolo delta
    dx = new_parent_loc[0] - old_parent_loc[0]
    dy = new_parent_loc[1] - old_parent_loc[1]
    dz = new_parent_loc[2] - old_parent_loc[2]

    d_rz = new_parent_rot[2] - old_parent_rot[2]

    # Vettore dal vecchio genitore al figlio
    rel_x = child.transform.location[0] - old_parent_loc[0]
    rel_y = child.transform.location[1] - old_parent_loc[1]

    # Rotazione del vettore relativo attorno all'asse Z del genitore
    cos_a = math.cos(d_rz)
    sin_a = math.sin(d_rz)

    rot_x = rel_x * cos_a - rel_y * sin_a
    rot_y = rel_x * sin_a + rel_y * cos_a

    # Nuova posizione del figlio (il centro del genitore + vettore ruotato)
    child.transform.location[0] = new_parent_loc[0] + rot_x
    child.transform.location[1] = new_parent_loc[1] + rot_y
    child.transform.location[2] += dz

    # Nuova rotazione del figlio
    child.transform.rotation_euler[2] = (child.transform.rotation_euler[2] + d_rz) % (2 * math.pi)
