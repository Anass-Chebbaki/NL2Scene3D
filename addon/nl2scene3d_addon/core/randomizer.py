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
from typing import Optional

from core.models import ObjectTransform, RoomBounds, SceneObject, SceneState

logger = logging.getLogger(__name__)


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
    max_placement_attempts: int = 50


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
    from core.utils.geometry import has_collision
    
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
        from core.utils.geometry import compute_aabb_2d
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
            original_dimensions: list[float],
            room_bounds: RoomBounds,
    ) -> list[float]:
        """
        Genera una nuova posizione casuale per un oggetto all'interno dei bounds.

        Tiene conto delle dimensioni dell'oggetto: il centro viene posizionato
        in modo che i bordi dell'AABB restino dentro i bounds della stanza,
        meno il margine dai muri.

        Args:
            original_location: Posizione originale [x, y, z].
            original_dimensions: Dimensioni AABB [w, d, h] dell'oggetto.
            room_bounds: Bounds della stanza.

        Returns:
            Nuova posizione [x, y, z] con coordinata Z invariata.
        """
        half_x = original_dimensions[0] / 2.0
        half_y = original_dimensions[1] / 2.0
        margin = self.config.wall_margin

        # I bounds effettivi per il CENTRO dell'oggetto sono ridotti di
        # mezza dimensione + margine, in modo che l'AABB completo resti dentro.
        x_lo = room_bounds.x_min + half_x + margin
        x_hi = room_bounds.x_max - half_x - margin
        y_lo = room_bounds.y_min + half_y + margin
        y_hi = room_bounds.y_max - half_y - margin

        # Se l'oggetto e' piu' grande della stanza, fallback al centro.
        if self.config.jitter_ratio > 0.0:
            jitter_x = (room_bounds.x_max - room_bounds.x_min) * self.config.jitter_ratio
            jitter_y = (room_bounds.y_max - room_bounds.y_min) * self.config.jitter_ratio
            x_lo = max(x_lo, original_location[0] - jitter_x)
            x_hi = min(x_hi, original_location[0] + jitter_x)
            y_lo = max(y_lo, original_location[1] - jitter_y)
            y_hi = min(y_hi, original_location[1] + jitter_y)
        else:
            # Se jitter_ratio e' 0.0, l'oggetto non deve muoversi affatto dalla sua posizione.
            # Tuttavia, dobbiamo comunque rispettare i bounds (clamp).
            x_lo = x_hi = max(room_bounds.x_min + half_x + margin, min(room_bounds.x_max - half_x - margin, original_location[0]))
            y_lo = y_hi = max(room_bounds.y_min + half_y + margin, min(room_bounds.y_max - half_y - margin, original_location[1]))

        if x_lo >= x_hi:
            new_x = (x_lo + x_hi) / 2.0
        else:
            new_x = self._rng.uniform(x_lo, x_hi)

        if y_lo >= y_hi:
            new_y = (y_lo + y_hi) / 2.0
        else:
            new_y = self._rng.uniform(y_lo, y_hi)

        return [new_x, new_y, original_location[2]]

    def _randomize_rotation(
        self,
        original_rotation: list[float],
    ) -> list[float]:
        """
        Genera una nuova rotazione casuale per un oggetto.

        Solo l'asse Z (yaw) viene ruotato in multipli di 90 gradi.
        Mantiene le rotazioni X e Y invariate.

        Args:
            original_rotation: Rotazione originale [rx, ry, rz] in radianti.

        Returns:
            Nuova rotazione [rx, ry, rz] con Z modificata.
        """
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

        # Identifichiamo le superfici disponibili per lo snapping
        potential_surfaces = [obj for obj in state.objects if self._is_surface(obj)]

        movable_objects = list(state.movable_objects)
        # Ordiniamo gli oggetti per volume decrescente: i pezzi grossi (letti, armadi) 
        # vengono piazzati per primi, rendendo piu' facile incastrare i piccoli dopo.
        movable_objects.sort(
            key=lambda o: o.transform.dimensions[0] * o.transform.dimensions[1] * o.transform.dimensions[2], 
            reverse=True
        )

        for obj in movable_objects:
            new_obj = obj.copy()
            placed = False
            best_transform = None
            min_max_overlap = float('inf')

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
                    obj.transform.location, eff_dimensions, room_bounds
                )
                new_objects.append(new_obj)
                placed_objects.append(new_obj)
                failed_count += 1
                randomized_count += 1
                continue

            for attempt in range(self.config.max_placement_attempts):
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
                    obj.transform.location, eff_dimensions, room_bounds
                )
                
                candidate_transform = ObjectTransform(
                    location=candidate_location,
                    rotation_euler=candidate_rotation,
                    dimensions=obj.transform.dimensions,
                )
                new_obj.transform = candidate_transform

                # --- VERTICAL PROJECTION / SURFACE SNAPPING ---
                if self._is_snappable(obj):
                    target_z = self._get_surface_z_at(candidate_location[0], candidate_location[1], potential_surfaces)
                    # Appoggiamo l'oggetto: Z = superficie_top + mezza_altezza_oggetto
                    new_obj.transform.location[2] = target_z + (obj.transform.dimensions[2] / 2.0)
                # ----------------------------------------------

                if not self.config.check_overlaps:
                    placed = True
                    break

                # Usa il controllo di collisione esatto
                from core.utils.geometry import compute_scene_collision_ratio
                overlap_ratio = compute_scene_collision_ratio(new_obj, placed_objects)
                
                if overlap_ratio < 0.001: # Quasi zero
                    placed = True
                    break
                
                # Se non è perfetto, teniamo traccia del "meno peggio"
                if overlap_ratio < min_max_overlap:
                    min_max_overlap = overlap_ratio
                    best_transform = candidate_transform.copy()

                logger.debug(
                    "Oggetto '%s': tentativo %d fallito per sovrapposizione (%.2f).",
                    obj.name,
                    attempt + 1,
                    overlap_ratio
                )

            if not placed:
                if best_transform:
                    logger.warning(
                        "Oggetto '%s': posizione perfetta non trovata. Uso la migliore (overlap: %.2f).",
                        obj.name,
                        min_max_overlap,
                    )
                    new_obj.transform = best_transform
                else:
                    logger.warning(
                        "Oggetto '%s': impossibile spostare. Mantenuta posizione originale.",
                        obj.name,
                    )
                    new_obj.transform = obj.transform.copy()
                failed_count += 1

            new_objects.append(new_obj)
            placed_objects.append(new_obj)
            randomized_count += 1

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
            },
        )