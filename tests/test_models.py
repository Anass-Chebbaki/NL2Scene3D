"""
Test unitari per i modelli dati del pacchetto.

Verifica la corretta serializzazione/deserializzazione JSON,
i vincoli di validazione e i metodi di utilita'.
"""

from __future__ import annotations

import json
import math

import pytest

from nl2scene3d.models import (
    LLMCorrection,
    ObjectTransform,
    RoomBounds,
    SceneObject,
    SceneState,
    degrees_to_radians,
    radians_to_degrees,
)

class TestObjectTransform:
    """Test per la classe ObjectTransform."""

    def test_valid_creation(self) -> None:
        """Verifica la creazione con valori validi."""
        t = ObjectTransform(
            location=[1.0, 2.0, 0.0],
            rotation_euler=[0.0, 0.0, 1.57],
            dimensions=[1.0, 0.5, 0.8],
        )
        assert t.location == [1.0, 2.0, 0.0]
        assert t.rotation_euler == [0.0, 0.0, 1.57]
        assert t.dimensions == [1.0, 0.5, 0.8]

    def test_invalid_location_length(self) -> None:
        """Verifica che location con lunghezza errata sollevi ValueError."""
        with pytest.raises(ValueError, match="location"):
            ObjectTransform(
                location=[1.0, 2.0],
                rotation_euler=[0.0, 0.0, 0.0],
                dimensions=[1.0, 1.0, 1.0],
            )

    def test_serialization_roundtrip(self) -> None:
        """Verifica che serializzazione e deserializzazione siano inverse."""
        original = ObjectTransform(
            location=[1.5, -0.8, 0.0],
            rotation_euler=[0.0, 0.0, math.pi / 2],
            dimensions=[2.3, 0.9, 0.8],
        )
        data = original.to_dict()
        restored = ObjectTransform.from_dict(data)

        assert restored.location == original.location
        assert restored.rotation_euler == original.rotation_euler
        assert restored.dimensions == original.dimensions

class TestRoomBounds:
    """Test per la classe RoomBounds."""

    def test_properties(self) -> None:
        """Verifica il calcolo delle proprieta' derivate."""
        bounds = RoomBounds(
            x_min=-3.0, x_max=3.0,
            y_min=-2.0, y_max=2.0,
            z_floor=0.0, z_ceiling=2.8,
        )
        assert bounds.width == pytest.approx(6.0)
        assert bounds.depth == pytest.approx(4.0)
        assert bounds.height == pytest.approx(2.8)

    def test_clamp_within_bounds(self) -> None:
        """Verifica che clamp_location non modifichi coordinate gia' nei bounds."""
        bounds = RoomBounds(x_min=-3.0, x_max=3.0, y_min=-2.0, y_max=2.0)
        location = [1.0, 1.0, 0.0]
        clamped = bounds.clamp_location(location)
        assert clamped == [1.0, 1.0, 0.0]

    def test_clamp_outside_bounds(self) -> None:
        """Verifica che clamp_location corregga coordinate fuori dai bounds."""
        bounds = RoomBounds(x_min=-3.0, x_max=3.0, y_min=-2.0, y_max=2.0)
        location = [5.0, -4.0, 0.0]
        clamped = bounds.clamp_location(location)
        assert clamped == [3.0, -2.0, 0.0]

    def test_clamp_preserves_z(self) -> None:
        """Verifica che clamp_location non modifichi la coordinata Z."""
        bounds = RoomBounds(x_min=-3.0, x_max=3.0, y_min=-2.0, y_max=2.0)
        location = [1.0, 1.0, 0.75]
        clamped = bounds.clamp_location(location)
        assert clamped[2] == pytest.approx(0.75)

    def test_serialization_roundtrip(self) -> None:
        """Verifica serializzazione e deserializzazione."""
        original = RoomBounds(
            x_min=-4.0, x_max=4.0,
            y_min=-3.0, y_max=3.0,
            z_floor=0.0, z_ceiling=3.0,
        )
        restored = RoomBounds.from_dict(original.to_dict())
        assert restored.x_min == original.x_min
        assert restored.x_max == original.x_max
        assert restored.z_ceiling == original.z_ceiling

class TestSceneObject:
    """Test per la classe SceneObject."""

    def test_to_dict_contains_required_keys(self) -> None:
        """Verifica che la serializzazione includa tutti i campi richiesti."""
        transform = ObjectTransform(
            location=[0.0, 0.0, 0.0],
            rotation_euler=[0.0, 0.0, 0.0],
            dimensions=[1.0, 1.0, 1.0],
        )
        obj = SceneObject(
            name="sofa",
            object_type="MESH",
            transform=transform,
            category="seating_large",
            is_movable=True,
        )
        data = obj.to_dict()
        required_keys = {"name", "type", "location", "rotation_euler", "dimensions",
                         "category", "is_movable"}
        assert required_keys.issubset(data.keys())

    def test_from_dict_roundtrip(self) -> None:
        """Verifica la deserializzazione da dizionario."""
        data = {
            "name": "coffee_table",
            "type": "MESH",
            "location": [0.0, 1.0, 0.0],
            "rotation_euler": [0.0, 0.0, 0.785],
            "dimensions": [1.2, 0.6, 0.45],
            "category": "table",
            "is_movable": True,
        }
        obj = SceneObject.from_dict(data)
        assert obj.name == "coffee_table"
        assert obj.transform.location == [0.0, 1.0, 0.0]
        assert obj.is_movable is True

class TestSceneState:
    """Test per la classe SceneState."""

    def _make_sample_state(self) -> SceneState:
        """Crea uno SceneState di esempio per i test."""
        objects = [
            SceneObject(
                name="sofa",
                object_type="MESH",
                transform=ObjectTransform([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.8, 0.9]),
                category="seating_large",
                is_movable=True,
            ),
            SceneObject(
                name="wall_north",
                object_type="MESH",
                transform=ObjectTransform([0.0, 5.0, 1.5], [0.0, 0.0, 0.0], [10.0, 0.2, 3.0]),
                category="structural",
                is_movable=False,
            ),
        ]
        return SceneState(
            scene_name="test_room",
            objects=objects,
            room_bounds=RoomBounds(-4.0, 4.0, -3.0, 3.0),
            pipeline_step="original",
        )

    def test_movable_objects_filter(self) -> None:
        """Verifica il filtraggio degli oggetti movibili."""
        state = self._make_sample_state()
        movable = state.movable_objects
        assert len(movable) == 1
        assert movable[0].name == "sofa"

    def test_static_objects_filter(self) -> None:
        """Verifica il filtraggio degli oggetti statici."""
        state = self._make_sample_state()
        static = state.static_objects
        assert len(static) == 1
        assert static[0].name == "wall_north"

    def test_get_object_by_name(self) -> None:
        """Verifica la ricerca per nome."""
        state = self._make_sample_state()
        found = state.get_object_by_name("sofa")
        assert found is not None
        assert found.name == "sofa"

        not_found = state.get_object_by_name("nonexistent")
        assert not_found is None

    def test_serialization_roundtrip(self) -> None:
        """Verifica la serializzazione completa."""
        original = self._make_sample_state()
        data = original.to_dict()
        restored = SceneState.from_dict(data)

        assert restored.scene_name == original.scene_name
        assert restored.pipeline_step == original.pipeline_step
        assert len(restored.objects) == len(original.objects)
        assert restored.room_bounds is not None
        assert restored.room_bounds.x_min == original.room_bounds.x_min

class TestConversions:
    """Test per le funzioni di conversione angolare."""

    def test_degrees_to_radians(self) -> None:
        """Verifica la conversione gradi -> radianti."""
        assert degrees_to_radians(0.0) == pytest.approx(0.0)
        assert degrees_to_radians(90.0) == pytest.approx(math.pi / 2)
        assert degrees_to_radians(180.0) == pytest.approx(math.pi)
        assert degrees_to_radians(360.0) == pytest.approx(2 * math.pi)

    def test_radians_to_degrees(self) -> None:
        """Verifica la conversione radianti -> gradi."""
        assert radians_to_degrees(0.0) == pytest.approx(0.0)
        assert radians_to_degrees(math.pi / 2) == pytest.approx(90.0)
        assert radians_to_degrees(math.pi) == pytest.approx(180.0)
