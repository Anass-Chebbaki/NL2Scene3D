# src/nl2scene3d/randomizer.py
"""
Randomizzazione controllata degli oggetti di una scena 3D.

Disorganizza artificialmente il layout di una scena pre-esistente per creare
uno stato caotico su cui l'LLM possa intervenire.

La randomizzazione e' volutamente "plausibile": gli oggetti rimangono
all'interno dei bounds della stanza e mantengono la loro quota Z,
mentre le posizioni X/Y e la rotazione Z vengono perturbate in modo casuale.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any, List, Optional

from nl2scene3d.models import ObjectTransform, RoomBounds, SceneObject, SceneState

logger = logging.getLogger(__name__)

def has_keyword(keywords: tuple[str, ...], text: str) -> bool:
    return any(k in text for k in keywords)


@dataclass
class RandomizerConfig:
    """
    Parametri di configurazione per la randomizzazione.

    Attributes:
        seed: Seed per il generatore di numeri casuali (0 = non deterministico).
        jitter_ratio: Frazione della dimensione della stanza usata come jitter
            massimo nella posizione (es. 0.8 = fino all'80% della larghezza).
        rotate_z_only: Se True, ruota solo l'asse Z (yaw).
        check_overlaps: Se True, verifica le sovrapposizioni AABB e ritenta.
        wall_margin: Margine minimo dai muri in metri.
        max_overlap_ratio: Rapporto massimo di sovrapposizione consentito.
        max_placement_attempts: Numero massimo di tentativi di posizionamento per oggetto.
    """

    seed: int = 0
    jitter_ratio: float = 0.8
    rotate_z_only: bool = True
    check_overlaps: bool = True
    wall_margin: float = 0.2
    max_overlap_ratio: float = 0.05
    max_placement_attempts: int = 200


def _compute_aabb(obj: SceneObject) -> tuple[float, float, float, float]:
    """
    Calcola l'Axis-Aligned Bounding Box (AABB) 2D di un oggetto nel piano XY.
    Tiene conto della rotazione sull'asse Z per calcolare l'ingombro reale.

    Args:
        obj: Oggetto di cui calcolare l'AABB.

    Returns:
        Tupla (x_min, x_max, y_min, y_max).
    """
    loc = obj.transform.location
    dim = obj.transform.dimensions
    rz = obj.transform.rotation_euler[2]

    # Calcolo della dimensione effettiva (AABB) sul piano XY dopo la rotazione Z
    cos_z = abs(math.cos(rz))
    sin_z = abs(math.sin(rz))
    
    eff_x = dim[0] * cos_z + dim[1] * sin_z
    eff_y = dim[0] * sin_z + dim[1] * cos_z

    half_x = eff_x / 2.0
    half_y = eff_y / 2.0
    return (
        loc[0] - half_x,
        loc[0] + half_x,
        loc[1] - half_y,
        loc[1] + half_y,
    )


def _compute_overlap_ratio(
    aabb_a: tuple[float, float, float, float],
    aabb_b: tuple[float, float, float, float],
) -> float:
    """
    Calcola il rapporto di sovrapposizione tra due AABB 2D.

    Il rapporto e' calcolato come area di intersezione divisa per l'area
    minima tra i due bounding box. 
    Nota: L'asimmetria del calcolo (uso del min) è intenzionale per proteggere 
    gli oggetti piccoli. Se un oggetto piccolo si sovrappone a uno grande, 
    l'area minima sarà quella dell'oggetto piccolo, restituendo un rapporto alto, 
    segnalando correttamente la sovrapposizione.

    Args:
        aabb_a: AABB del primo oggetto (x_min, x_max, y_min, y_max).
        aabb_b: AABB del secondo oggetto (x_min, x_max, y_min, y_max).

    Returns:
        Valore in [0.0, 1.0] dove 0.0 indica nessuna sovrapposizione.
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


def _has_excessive_overlap(
    candidate: SceneObject,
    placed_objects: list[SceneObject],
    max_overlap_ratio: float,
) -> bool:
    """
    Verifica se un oggetto ha sovrapposizioni eccessive con quelli gia' posizionati.
    Delega al modulo geometry.has_collision che gestisce sia muri (AABB) che mobili (BVH).
    """
    from nl2scene3d.utils.geometry import has_collision
    
    # Il nuovo has_collision gestisce correttamente:
    # - Muri: AABB 2D + Z-overlap (non li salta più!)
    # - Mobili: BVH mesh-esatto + fallback AABB
    return has_collision(candidate, placed_objects, check_walls=True)


class SceneRandomizer:
    """
    Disorganizza artificialmente il layout di una scena 3D.

    Sposta e ruota gli oggetti movibili della scena in modo casuale,
    rispettando i bounds della stanza e limitando le sovrapposizioni.

    Attributes:
        config: Parametri di randomizzazione.
        _rng: Istanza del generatore di numeri casuali con seed controllato.
    """

    def __init__(self, config: Optional[RandomizerConfig] = None) -> None:
        """
        Inizializza il randomizer.

        Args:
            config: Configurazione della randomizzazione.
                    Se None, vengono usati i valori di default.
        """
        self.config = config or RandomizerConfig()
        effective_seed: Optional[int] = (
            self.config.seed if self.config.seed != 0 else None
        )
        self._rng = random.Random(effective_seed)
        logger.info(
            "SceneRandomizer inizializzato. Seed: %s, jitter_ratio: %.2f.",
            effective_seed,
            self.config.jitter_ratio,
        )

    def _is_surface(self, obj: SceneObject) -> bool:
        """Determina se un oggetto puo' ospitare altri oggetti sopra di se'."""
        name = obj.name.lower()
        cat = (obj.category or "").lower()
        return any(k in name or k in cat for k in ["table", "desk", "bed", "shelf", "nightstand", "counter", "structural"])

    def _is_snappable(self, obj: SceneObject) -> bool:
        """Determina se un oggetto deve essere appoggiato su una superficie."""
        name = obj.name.lower()
        cat = (obj.category or "").lower()
        return any(k in name or k in cat for k in ["decor", "electronics", "book", "lamp", "monitor", "keyboard", "mouse", "bottle", "bin"])

    def _get_surface_z_at(self, x: float, y: float, surfaces: List[SceneObject]) -> float:
        """Trova la quota Z della superficie piu' alta in posizione (x,y)."""
        from nl2scene3d.utils.geometry import compute_aabb_2d
        max_z = 0.0 # Default: pavimento
        for surf in surfaces:
            surf_aabb = compute_aabb_2d(surf)
            # Se il centro del candidato cade dentro la superficie
            if (surf_aabb[0] <= x <= surf_aabb[1] and surf_aabb[2] <= y <= surf_aabb[3]):
                surf_z_top = surf.transform.location[2] + (surf.transform.dimensions[2] / 2.0)
                if surf_z_top > max_z:
                    max_z = surf_z_top
        return max_z

    def _randomize_location(
        self,
        original_location: list[float],
        dimensions: list[float],
        room_bounds: RoomBounds,
        origin_offset: list[float] = None,
        rotation_z: float = 0.0,
    ) -> list[float]:
        if origin_offset is None:
            origin_offset = [0.0, 0.0, 0.0]

        half_x = dimensions[0] / 2.0
        half_y = dimensions[1] / 2.0
        margin = self.config.wall_margin
        
        cos_z_rot = math.cos(rotation_z)
        sin_z_rot = math.sin(rotation_z)
        world_off_x = origin_offset[0] * cos_z_rot - origin_offset[1] * sin_z_rot
        world_off_y = origin_offset[0] * sin_z_rot + origin_offset[1] * cos_z_rot

        safe_x_min = room_bounds.x_min + half_x + margin
        safe_x_max = room_bounds.x_max - half_x - margin
        safe_y_min = room_bounds.y_min + half_y + margin
        safe_y_max = room_bounds.y_max - half_y - margin

        # Apply jitter
        if self.config.jitter_ratio > 0.0:
            jitter_x = (room_bounds.x_max - room_bounds.x_min) * self.config.jitter_ratio
            jitter_y = (room_bounds.y_max - room_bounds.y_min) * self.config.jitter_ratio
            
            # Use original geometric center to calculate jitter range
            orig_center_x = original_location[0] + world_off_x
            orig_center_y = original_location[1] + world_off_y
            
            safe_x_min = max(safe_x_min, orig_center_x - jitter_x)
            safe_x_max = min(safe_x_max, orig_center_x + jitter_x)
            safe_y_min = max(safe_y_min, orig_center_y - jitter_y)
            safe_y_max = min(safe_y_max, orig_center_y + jitter_y)
        else:
            # No jitter, keep original center
            orig_center_x = original_location[0] + world_off_x
            orig_center_y = original_location[1] + world_off_y
            safe_x_min = safe_x_max = max(safe_x_min, min(safe_x_max, orig_center_x))
            safe_y_min = safe_y_max = max(safe_y_min, min(safe_y_max, orig_center_y))

        if safe_x_max < safe_x_min:
            center_x = (room_bounds.x_min + room_bounds.x_max) / 2.0
        else:
            center_x = self._rng.uniform(safe_x_min, safe_x_max)

        if safe_y_max < safe_y_min:
            center_y = (room_bounds.y_min + room_bounds.y_max) / 2.0
        else:
            center_y = self._rng.uniform(safe_y_min, safe_y_max)

        new_x = center_x - world_off_x
        new_y = center_y - world_off_y

        return [new_x, new_y, original_location[2]]

    def _randomize_rotation(self, original_rotation: list[float]) -> list[float]:
        multiples = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
        new_z = original_rotation[2] + self._rng.choice(multiples)
        return [original_rotation[0], original_rotation[1], new_z]

    def randomize(self, state: SceneState) -> SceneState:
        """
        Applica la randomizzazione a una copia della scena.

        Non modifica lo SceneState originale. Restituisce un nuovo
        SceneState con le trasformazioni randomizzate applicate.

        Args:
            state: Stato originale della scena da disorganizzare.

        Returns:
            Nuovo SceneState con layout randomizzato.

        Raises:
            ValueError: Se la scena non ha room_bounds definiti.
        """
        if state.room_bounds is None:
            raise ValueError(
                "La scena non ha room_bounds definiti. "
                "Assicurarsi di aver estratto correttamente lo stato."
            )

        room_bounds = state.room_bounds

        logger.info(
            "Avvio randomizzazione scena '%s'. Oggetti movibili: %d.",
            state.scene_name,
            len(state.movable_objects),
        )

        new_objects: list[SceneObject] = []
        placed_objects: list[SceneObject] = []

        for obj in state.static_objects:
            new_obj = obj.copy()
            new_objects.append(new_obj)
            placed_objects.append(new_obj)

        randomized_count = 0
        failed_count = 0

        # Calcola i gruppi di oggetti per mantenere mobili figli insieme ai genitori
        from nl2scene3d.utils.grouping import find_object_groups
        groups = find_object_groups(state.objects)
        grouped_children = set()
        for children in groups.values():
            grouped_children.update(children)
        
        # Identifichiamo le superfici disponibili per lo snapping
        potential_surfaces = [obj for obj in state.objects if self._is_surface(obj)]

        # Consideriamo solo gli oggetti "radice" per la randomizzazione principale.
        # I figli verranno spostati automaticamente insieme al genitore.
        movable_roots = [obj for obj in state.movable_objects if obj.name not in grouped_children]
        
        movable_objects = list(movable_roots)
        # Ordiniamo gli oggetti per volume decrescente: i pezzi grossi (letti, armadi) 
        # vengono piazzati per primi, rendendo piu' facile incastrare i piccoli dopo.
        movable_objects.sort(
            key=lambda o: o.transform.dimensions[0] * o.transform.dimensions[1] * o.transform.dimensions[2], 
            reverse=True
        )

        for obj in movable_objects:
            new_obj = obj.copy()
            placed = False
            
            # Se l'oggetto e' piu' grande della stanza, marchiamolo come non piazzabile
            # idealmente, ma la pipeline deve continuare.
            half_x = obj.transform.dimensions[0] / 2.0
            half_y = obj.transform.dimensions[1] / 2.0
            margin = self.config.wall_margin
            
            # Se non c'e' spazio fisico per l'oggetto nella stanza
            if (room_bounds.x_max - room_bounds.x_min < obj.transform.dimensions[0] + 2 * margin or 
                room_bounds.y_max - room_bounds.y_min < obj.transform.dimensions[1] + 2 * margin):
                logger.warning("Oggetto '%s' troppo grande per la stanza. Fallback al centro.", obj.name)
                
                # Applichiamo una rotazione casuale
                new_rotation = self._randomize_rotation(obj.transform.rotation_euler)
                rz_diff = new_rotation[2] - obj.transform.rotation_euler[2]
                cos_z = abs(math.cos(rz_diff))
                sin_z = abs(math.sin(rz_diff))
                eff_x = obj.transform.dimensions[0] * cos_z + obj.transform.dimensions[1] * sin_z
                eff_y = obj.transform.dimensions[0] * sin_z + obj.transform.dimensions[1] * cos_z
                eff_dimensions = [eff_x, eff_y, obj.transform.dimensions[2]]

                new_obj.transform.rotation_euler = new_rotation
                new_obj.transform.location = self._randomize_location(
                    obj.transform.location, eff_dimensions, room_bounds,
                    origin_offset=obj.transform.origin_offset, rotation_z=new_rotation[2]
                )
                new_objects.append(new_obj)
                placed_objects.append(new_obj)
                failed_count += 1
                randomized_count += 1
                continue

            for attempt in range(500):
                # Generiamo PRIMA la rotazione (solo a step di 90 gradi)
                candidate_rotation = self._randomize_rotation(
                    obj.transform.rotation_euler
                )
                
                # Calcoliamo le dimensioni EFFICACI dopo la rotazione
                rz_diff = candidate_rotation[2] - obj.transform.rotation_euler[2]
                cos_z = abs(math.cos(rz_diff))
                sin_z = abs(math.sin(rz_diff))
                eff_x = obj.transform.dimensions[0] * cos_z + obj.transform.dimensions[1] * sin_z
                eff_y = obj.transform.dimensions[0] * sin_z + obj.transform.dimensions[1] * cos_z
                eff_dimensions = [eff_x, eff_y, obj.transform.dimensions[2]]

                # Usiamo le dimensioni efficaci per trovare una posizione che rispetti i bounds
                candidate_location = self._randomize_location(
                    obj.transform.location, eff_dimensions, room_bounds, 
                    origin_offset=obj.transform.origin_offset, rotation_z=candidate_rotation[2]
                )
                
                candidate_transform = ObjectTransform(
                    location=candidate_location,
                    rotation_euler=candidate_rotation,
                    dimensions=obj.transform.dimensions,
                    origin_offset=list(obj.transform.origin_offset),
                )
                new_obj.transform = candidate_transform

                # Nessuno snapping verticale. L'oggetto mantiene la sua quota Z originale
                # come richiesto: "tutti gli oggetti devono avere la stessa altezza"
                # (la coordinata Z è già stata mantenuta da _randomize_location)

                if not self.config.check_overlaps:
                    placed = True
                    break

                # Usa il controllo di collisione mesh-esatto (BVH) se possibile
                from nl2scene3d.utils.geometry import has_collision
                
                collision = has_collision(new_obj, placed_objects)
                
                # INTEGRAZIONE GROUPING: Controlla che anche i figli non collidano
                root_children_names = groups.get(obj.name, [])
                if not collision and root_children_names:
                    from nl2scene3d.utils.grouping import apply_group_transform
                    for child_name in root_children_names:
                        child_orig = state.get_object_by_name(child_name)
                        if child_orig and child_orig.is_movable:
                            test_child = child_orig.copy()
                            apply_group_transform(
                                test_child,
                                obj.transform.location,
                                obj.transform.rotation_euler,
                                new_obj.transform.location,
                                new_obj.transform.rotation_euler
                            )
                            # Verifica collisioni con altri oggetti
                            if has_collision(test_child, placed_objects):
                                collision = True
                                break
                            
                            # Verifica che il figlio non sia finito fuori dai muri
                            from nl2scene3d.utils.geometry import compute_aabb_2d
                            c_aabb = compute_aabb_2d(test_child)
                            if (c_aabb[0] < room_bounds.x_min or c_aabb[1] > room_bounds.x_max or
                                c_aabb[2] < room_bounds.y_min or c_aabb[3] > room_bounds.y_max):
                                collision = True
                                break
                
                if not collision:
                    placed = True
                    break

            if placed:
                new_objects.append(new_obj)
                placed_objects.append(new_obj)
                randomized_count += 1

                # --- SPOSTA I FIGLI INSIEME AL GENITORE ---
                if obj.name in groups:
                    from nl2scene3d.utils.grouping import apply_group_transform
                    for child_name in groups[obj.name]:
                        child_orig = state.get_object_by_name(child_name)
                        if child_orig:
                            new_child = child_orig.copy()
                            apply_group_transform(
                                new_child,
                                obj.transform.location,
                                obj.transform.rotation_euler,
                                new_obj.transform.location,
                                new_obj.transform.rotation_euler
                            )
                            new_objects.append(new_child)
                            placed_objects.append(new_child)
                            logger.debug("Figlio '%s' spostato con genitore '%s'.", child_name, obj.name)
            else:
                logger.warning(
                    "Oggetto '%s': impossibile trovare posizione perfetta dopo %d tentativi. Mantenuta posizione originale per evitare collisioni.",
                    obj.name,
                    self.config.max_placement_attempts
                )
                # Se fallisce, rimettiamo l'originale (e i suoi figli nella loro pos originale)
                orig_copy = obj.copy()
                new_objects.append(orig_copy)
                placed_objects.append(orig_copy)
                
                if obj.name in groups:
                    for child_name in groups[obj.name]:
                        child_orig = state.get_object_by_name(child_name)
                        if child_orig:
                            child_copy = child_orig.copy()
                            new_objects.append(child_copy)
                            placed_objects.append(child_copy)
                
                failed_count += 1

        logger.info(
            "Randomizzazione completata: %d oggetti spostati, %d con sovrapposizioni residue.",
            randomized_count,
            failed_count,
        )

        return SceneState(
            scene_name=state.scene_name,
            objects=new_objects,
            room_bounds=room_bounds,
            pipeline_step="randomized",
            metadata={
                "randomizer_seed": self.config.seed,
                "randomized_count": randomized_count,
                "failed_placements": failed_count,
                "grouped_children": list(grouped_children),
            },
        )