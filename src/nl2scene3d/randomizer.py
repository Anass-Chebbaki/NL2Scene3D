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

from nl2scene3d.models import ObjectTransform, RoomBounds, SceneObject, SceneState

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
    wall_margin: float = 0.1
    max_overlap_ratio: float = 0.5
    max_placement_attempts: int = 10


def _compute_aabb(obj: SceneObject) -> tuple[float, float, float, float]:
    """
    Calcola l'Axis-Aligned Bounding Box (AABB) 2D di un oggetto nel piano XY.

    Args:
        obj: Oggetto di cui calcolare l'AABB.

    Returns:
        Tupla (x_min, x_max, y_min, y_max).
    """
    loc = obj.transform.location
    dim = obj.transform.dimensions
    half_x = dim[0] / 2.0
    half_y = dim[1] / 2.0
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

    Args:
        candidate: Oggetto da verificare.
        placed_objects: Lista degli oggetti gia' posizionati.
        max_overlap_ratio: Soglia massima di sovrapposizione consentita.

    Returns:
        True se almeno una sovrapposizione supera la soglia.
    """
    candidate_aabb = _compute_aabb(candidate)
    for other in placed_objects:
        if other.name == candidate.name:
            continue
        other_aabb = _compute_aabb(other)
        ratio = _compute_overlap_ratio(candidate_aabb, other_aabb)
        if ratio > max_overlap_ratio:
            logger.debug(
                "Sovrapposizione eccessiva: '%s' vs '%s' = %.2f.",
                candidate.name,
                other.name,
                ratio,
            )
            return True
    return False


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

    def _randomize_location(
        self,
        original_location: list[float],
        room_bounds: RoomBounds,
    ) -> list[float]:
        """
        Genera una nuova posizione casuale per un oggetto all'interno dei bounds.

        Args:
            original_location: Posizione originale [x, y, z].
            room_bounds: Bounds della stanza.

        Returns:
            Nuova posizione [x, y, z] con coordinata Z invariata.
        """
        new_x = self._rng.uniform(
            room_bounds.x_min + self.config.wall_margin,
            room_bounds.x_max - self.config.wall_margin,
        )
        new_y = self._rng.uniform(
            room_bounds.y_min + self.config.wall_margin,
            room_bounds.y_max - self.config.wall_margin,
        )
        return [new_x, new_y, original_location[2]]

    def _randomize_rotation(
        self,
        original_rotation: list[float],
    ) -> list[float]:
        """
        Genera una nuova rotazione casuale per un oggetto.

        Solo l'asse Z (yaw) viene randomizzato se rotate_z_only e' True,
        in modo da mantenere gli oggetti verticali e realistici.

        Args:
            original_rotation: Rotazione originale [rx, ry, rz] in radianti.

        Returns:
            Nuova rotazione [rx, ry, rz] con Z randomizzata in [0, 2*pi].
        """
        if self.config.rotate_z_only:
            new_z = self._rng.uniform(0.0, 2.0 * math.pi)
            return [original_rotation[0], original_rotation[1], new_z]

        return [
            self._rng.uniform(0.0, 2.0 * math.pi),
            self._rng.uniform(0.0, 2.0 * math.pi),
            self._rng.uniform(0.0, 2.0 * math.pi),
        ]

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

        for obj in state.movable_objects:
            new_obj = obj.copy()
            placed = False

            for attempt in range(self.config.max_placement_attempts):
                candidate_location = self._randomize_location(
                    obj.transform.location, room_bounds
                )
                candidate_rotation = self._randomize_rotation(
                    obj.transform.rotation_euler
                )
                new_obj.transform = ObjectTransform(
                    location=candidate_location,
                    rotation_euler=candidate_rotation,
                    dimensions=obj.transform.dimensions,
                )

                if not self.config.check_overlaps or not _has_excessive_overlap(
                    new_obj, placed_objects, self.config.max_overlap_ratio
                ):
                    placed = True
                    break

                logger.debug(
                    "Oggetto '%s': tentativo %d fallito per sovrapposizione.",
                    obj.name,
                    attempt + 1,
                )

            if not placed:
                logger.warning(
                    "Oggetto '%s': posizione senza sovrapposizioni non trovata "
                    "dopo %d tentativi. Viene usata l'ultima posizione calcolata.",
                    obj.name,
                    self.config.max_placement_attempts,
                )
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
                "overlap_failures": failed_count,
            },
        )