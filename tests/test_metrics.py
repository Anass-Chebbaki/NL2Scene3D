"""
Test unitari per il modulo di calcolo delle metriche.
"""

from __future__ import annotations

import math

import pytest

from nl2scene3d.metrics import (
    SceneMetrics,
    _angular_difference_z,
    _euclidean_distance_2d,
    compute_metrics,
)
from nl2scene3d.models import ObjectTransform, RoomBounds, SceneObject, SceneState

def _make_obj(name: str, location: list[float], is_movable: bool = True) -> SceneObject:
    """Factory per oggetti di test."""
    return SceneObject(
        name=name,
        object_type="MESH",
        transform=ObjectTransform(location, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        category="furniture" if is_movable else "structural",
        is_movable=is_movable,
    )

def _make_state(
    objects: list[SceneObject],
    step: str = "original",
) -> SceneState:
    """Factory per SceneState di test."""
    return SceneState(
        scene_name="test",
        objects=objects,
        room_bounds=RoomBounds(-5.0, 5.0, -5.0, 5.0),
        pipeline_step=step,
    )

class TestDistanceFunctions:
    """Test per le funzioni di calcolo delle distanze."""

    def test_same_location_distance_zero(self) -> None:
        """Due punti identici hanno distanza zero."""
        loc = [1.0, 2.0, 0.0]
        assert _euclidean_distance_2d(loc, loc) == pytest.approx(0.0)

    def test_distance_ignores_z(self) -> None:
        """La distanza 2D deve ignorare la componente Z."""
        loc_a = [0.0, 0.0, 0.0]
        loc_b = [0.0, 0.0, 100.0]
        assert _euclidean_distance_2d(loc_a, loc_b) == pytest.approx(0.0)

    def test_distance_unit_step(self) -> None:
        """Distanza di un passo unitario lungo X."""
        assert _euclidean_distance_2d([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_distance_diagonal(self) -> None:
        """Distanza diagonale 3-4-5."""
        assert _euclidean_distance_2d([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]) == pytest.approx(5.0)

class TestAngularDifference:
    """Test per il calcolo della differenza angolare."""

    def test_same_angle_zero_difference(self) -> None:
        """Angoli identici hanno differenza zero."""
        rot = [0.0, 0.0, 1.0]
        assert _angular_difference_z(rot, rot) == pytest.approx(0.0)

    def test_supplementary_angles_normalized(self) -> None:
        """Angoli supplementari (0 e 2*pi) devono dare differenza minima."""
        rot_a = [0.0, 0.0, 0.0]
        rot_b = [0.0, 0.0, 2 * math.pi]
        diff = _angular_difference_z(rot_a, rot_b)
        assert diff == pytest.approx(0.0, abs=1e-9)

    def test_ninety_degree_difference(self) -> None:
        """Differenza di 90 gradi deve dare pi/2."""
        rot_a = [0.0, 0.0, 0.0]
        rot_b = [0.0, 0.0, math.pi / 2]
        diff = _angular_difference_z(rot_a, rot_b)
        assert diff == pytest.approx(math.pi / 2)

    def test_max_difference_is_pi(self) -> None:
        """La differenza massima non deve superare pi."""
        rot_a = [0.0, 0.0, 0.0]
        rot_b = [0.0, 0.0, math.pi]
        diff = _angular_difference_z(rot_a, rot_b)
        assert diff <= math.pi + 1e-9

class TestComputeMetrics:
    """Test per la funzione compute_metrics."""

    def test_identical_states_zero_delta(self) -> None:
        """Stati identici devono avere delta nullo."""
        objects = [_make_obj("sofa", [1.0, 2.0, 0.0])]
        state = _make_state(objects)
        metrics = compute_metrics(state, state)
        assert metrics.mean_position_delta_meters == pytest.approx(0.0)

    def test_moved_object_has_positive_delta(self) -> None:
        """Un oggetto spostato deve produrre un delta positivo."""
        original = _make_state([_make_obj("sofa", [0.0, 0.0, 0.0])])
        moved = _make_state(
            [_make_obj("sofa", [3.0, 4.0, 0.0])],
            step="reordered",
        )
        metrics = compute_metrics(original, moved)
        assert metrics.mean_position_delta_meters == pytest.approx(5.0)

    def test_improvement_score_one_when_perfectly_reordered(self) -> None:
        """Se lo stato riordinato e' identico all'originale, l'improvement deve essere 1.0."""
        original_objs = [_make_obj("sofa", [0.0, 0.0, 0.0])]
        disordered_objs = [_make_obj("sofa", [5.0, 0.0, 0.0])]

        original = _make_state(original_objs, step="original")
        disordered = _make_state(disordered_objs, step="randomized")
        # Il riordinato e' identico all'originale
        reordered = _make_state(original_objs, step="reordered")

        metrics = compute_metrics(original, reordered, disordered)
        assert metrics.improvement_score == pytest.approx(1.0)

    def test_improvement_score_zero_when_unchanged(self) -> None:
        """Se il riordinato e' uguale al disordinato, l'improvement deve essere 0.0."""
        original_objs = [_make_obj("sofa", [0.0, 0.0, 0.0])]
        disordered_objs = [_make_obj("sofa", [4.0, 0.0, 0.0])]

        original = _make_state(original_objs)
        disordered = _make_state(disordered_objs, step="randomized")
        # Il riordinato e' identico al disordinato (nessun miglioramento)
        reordered = _make_state(disordered_objs, step="reordered")

        metrics = compute_metrics(original, reordered, disordered)
        assert metrics.improvement_score == pytest.approx(0.0)

    def test_no_movable_objects_returns_zero_metrics(self) -> None:
        """Senza oggetti movibili le metriche devono essere zero."""
        objects = [_make_obj("wall", [0.0, 5.0, 1.5], is_movable=False)]
        reference = _make_state(objects)
        evaluated = _make_state(objects, step="reordered")
        metrics = compute_metrics(reference, evaluated)
        assert metrics.object_count_movable == 0
        assert metrics.mean_position_delta_meters == pytest.approx(0.0)

    def test_per_object_details_populated(self) -> None:
        """I dettagli per oggetto devono essere popolati."""
        original = _make_state([_make_obj("sofa", [0.0, 0.0, 0.0])])
        moved = _make_state([_make_obj("sofa", [1.0, 0.0, 0.0])], step="reordered")
        metrics = compute_metrics(original, moved)
        assert "sofa" in metrics.per_object_details
        assert "position_delta_meters" in metrics.per_object_details["sofa"]
