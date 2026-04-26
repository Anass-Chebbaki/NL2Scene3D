"""
Test unitari per il modulo di randomizzazione.

Verifica che la randomizzazione rispetti i bounds della stanza,
non modifichi gli oggetti statici e produca coordinate valide.
"""

from __future__ import annotations

import math

import pytest

from nl2scene3d.models import ObjectTransform, RoomBounds, SceneObject, SceneState
from nl2scene3d.randomizer import (
    RandomizerConfig,
    SceneRandomizer,
    _compute_aabb,
    _compute_overlap_ratio,
    _has_excessive_overlap,
)

def _make_object(
    name: str,
    location: list[float],
    dimensions: list[float],
    is_movable: bool = True,
) -> SceneObject:
    """Factory per creare SceneObject di test."""
    return SceneObject(
        name=name,
        object_type="MESH",
        transform=ObjectTransform(
            location=location,
            rotation_euler=[0.0, 0.0, 0.0],
            dimensions=dimensions,
        ),
        category="furniture" if is_movable else "structural",
        is_movable=is_movable,
    )

def _make_test_state() -> SceneState:
    """Crea uno SceneState di test con oggetti movibili e statici."""
    objects = [
        _make_object("sofa", [0.0, 0.0, 0.0], [2.0, 0.8, 0.9], is_movable=True),
        _make_object("chair", [2.0, 0.0, 0.0], [0.6, 0.6, 0.9], is_movable=True),
        _make_object("table", [-1.0, 1.0, 0.0], [1.2, 0.6, 0.45], is_movable=True),
        _make_object("wall", [0.0, 5.0, 1.5], [10.0, 0.2, 3.0], is_movable=False),
    ]
    return SceneState(
        scene_name="test_room",
        objects=objects,
        room_bounds=RoomBounds(-4.0, 4.0, -3.0, 3.0),
        pipeline_step="original",
    )

class TestAabbFunctions:
    """Test per le funzioni di calcolo AABB."""

    def test_compute_aabb_symmetric(self) -> None:
        """Verifica il calcolo dell'AABB per un oggetto centrato."""
        obj = _make_object("test", [0.0, 0.0, 0.0], [2.0, 1.0, 0.5])
        aabb = _compute_aabb(obj)
        assert aabb == (-1.0, 1.0, -0.5, 0.5)

    def test_compute_aabb_offset(self) -> None:
        """Verifica il calcolo dell'AABB per un oggetto con offset."""
        obj = _make_object("test", [1.0, 2.0, 0.0], [2.0, 2.0, 1.0])
        aabb = _compute_aabb(obj)
        assert aabb == (0.0, 2.0, 1.0, 3.0)

    def test_no_overlap(self) -> None:
        """Verifica che AABB non adiacenti abbiano overlap 0."""
        obj_a = _make_object("a", [-2.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        obj_b = _make_object("b", [2.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        aabb_a = _compute_aabb(obj_a)
        aabb_b = _compute_aabb(obj_b)
        ratio = _compute_overlap_ratio(aabb_a, aabb_b)
        assert ratio == pytest.approx(0.0)

    def test_full_overlap(self) -> None:
        """Verifica che oggetti identici abbiano overlap 1."""
        obj_a = _make_object("a", [0.0, 0.0, 0.0], [2.0, 2.0, 1.0])
        obj_b = _make_object("b", [0.0, 0.0, 0.0], [2.0, 2.0, 1.0])
        aabb_a = _compute_aabb(obj_a)
        aabb_b = _compute_aabb(obj_b)
        ratio = _compute_overlap_ratio(aabb_a, aabb_b)
        assert ratio == pytest.approx(1.0)

class TestSceneRandomizer:
    """Test per la classe SceneRandomizer."""

    def test_static_objects_not_moved(self) -> None:
        """Verifica che gli oggetti statici non vengano spostati."""
        config = RandomizerConfig(seed=42, check_overlaps=False)
        randomizer = SceneRandomizer(config=config)
        state = _make_test_state()

        randomized = randomizer.randomize(state)

        original_wall = state.get_object_by_name("wall")
        randomized_wall = randomized.get_object_by_name("wall")

        assert original_wall is not None
        assert randomized_wall is not None
        assert randomized_wall.transform.location == original_wall.transform.location

    def test_movable_objects_relocated(self) -> None:
        """Verifica che almeno un oggetto movibile venga spostato."""
        config = RandomizerConfig(seed=42, check_overlaps=False)
        randomizer = SceneRandomizer(config=config)
        state = _make_test_state()

        randomized = randomizer.randomize(state)

        # Con seed fisso, almeno qualche oggetto deve essere stato spostato
        any_moved = False
        for original_obj in state.movable_objects:
            rand_obj = randomized.get_object_by_name(original_obj.name)
            if rand_obj and rand_obj.transform.location != original_obj.transform.location:
                any_moved = True
                break

        assert any_moved, "Nessun oggetto movibile e' stato spostato."

    def test_locations_within_bounds(self) -> None:
        """Verifica che tutte le posizioni randomizzate siano nei bounds."""
        config = RandomizerConfig(seed=42, check_overlaps=False)
        randomizer = SceneRandomizer(config=config)
        state = _make_test_state()

        randomized = randomizer.randomize(state)
        bounds = randomized.room_bounds

        for obj in randomized.movable_objects:
            loc = obj.transform.location
            assert bounds.x_min <= loc[0] <= bounds.x_max, (
                f"Oggetto '{obj.name}': X={loc[0]} fuori da [{bounds.x_min}, {bounds.x_max}]"
            )
            assert bounds.y_min <= loc[1] <= bounds.y_max, (
                f"Oggetto '{obj.name}': Y={loc[1]} fuori da [{bounds.y_min}, {bounds.y_max}]"
            )

    def test_z_coordinate_preserved(self) -> None:
        """Verifica che la coordinata Z venga preservata."""
        config = RandomizerConfig(seed=42, check_overlaps=False)
        randomizer = SceneRandomizer(config=config)
        state = _make_test_state()

        randomized = randomizer.randomize(state)

        for original_obj in state.movable_objects:
            rand_obj = randomized.get_object_by_name(original_obj.name)
            assert rand_obj is not None
            assert rand_obj.transform.location[2] == pytest.approx(
                original_obj.transform.location[2]
            ), f"Z modificata per oggetto '{original_obj.name}'"

    def test_object_count_preserved(self) -> None:
        """Verifica che il numero di oggetti rimanga invariato."""
        config = RandomizerConfig(seed=42)
        randomizer = SceneRandomizer(config=config)
        state = _make_test_state()

        randomized = randomizer.randomize(state)
        assert len(randomized.objects) == len(state.objects)

    def test_pipeline_step_updated(self) -> None:
        """Verifica che lo step della pipeline venga aggiornato."""
        config = RandomizerConfig(seed=42)
        randomizer = SceneRandomizer(config=config)
        state = _make_test_state()

        randomized = randomizer.randomize(state)
        assert randomized.pipeline_step == "randomized"

    def test_rotation_z_only(self) -> None:
        """Verifica che solo l'asse Z venga ruotato quando rotate_z_only=True."""
        config = RandomizerConfig(seed=42, rotate_z_only=True, check_overlaps=False)
        randomizer = SceneRandomizer(config=config)
        state = _make_test_state()

        randomized = randomizer.randomize(state)

        for original_obj in state.movable_objects:
            rand_obj = randomized.get_object_by_name(original_obj.name)
            assert rand_obj is not None
            # X e Y della rotazione devono rimanere invariati
            assert rand_obj.transform.rotation_euler[0] == pytest.approx(
                original_obj.transform.rotation_euler[0]
            )
            assert rand_obj.transform.rotation_euler[1] == pytest.approx(
                original_obj.transform.rotation_euler[1]
            )

    def test_raises_without_bounds(self) -> None:
        """Verifica che ValueError venga sollevato senza room_bounds."""
        config = RandomizerConfig(seed=42)
        randomizer = SceneRandomizer(config=config)
        state = SceneState(
            scene_name="no_bounds",
            objects=[],
            room_bounds=None,
            pipeline_step="original",
        )
        with pytest.raises(ValueError, match="room_bounds"):
            randomizer.randomize(state)

    def test_jitter_ratio_zero(self) -> None:
        """Verifica che con jitter_ratio=0.0 gli oggetti non si muovano (Bug 5.4)."""
        config = RandomizerConfig(seed=42, jitter_ratio=0.0, check_overlaps=False)
        randomizer = SceneRandomizer(config=config)
        state = _make_test_state()
        
        randomized = randomizer.randomize(state)
        
        for original_obj in state.movable_objects:
            rand_obj = randomized.get_object_by_name(original_obj.name)
            assert rand_obj.transform.location == original_obj.transform.location

    def test_object_bigger_than_room(self) -> None:
        """Verifica che un oggetto piu' grande della stanza non mandi in crash (Bug 5.4)."""
        config = RandomizerConfig(seed=42, max_placement_attempts=2)
        randomizer = SceneRandomizer(config=config)
        
        # Oggetto 10x10 in una stanza 2x2
        objects = [_make_object("big", [0, 0, 0], [10, 10, 1])]
        state = SceneState(
            scene_name="small_room",
            objects=objects,
            room_bounds=RoomBounds(-1, 1, -1, 1),
            pipeline_step="original"
        )
        
        # Dovrebbe fallire il posizionamento ma restituire comunque lo stato (con l'oggetto nella posizione originale o clampata)
        randomized = randomizer.randomize(state)
        assert len(randomized.objects) == 1
        assert randomized.metadata["failed_placements"] == 1
