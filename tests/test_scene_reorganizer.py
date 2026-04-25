"""
Test unitari per il modulo scene_reorganizer.

Verifica la validazione dell'output LLM, il clamp delle coordinate,
la gestione degli oggetti mancanti e la costruzione dei prompt.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nl2scene3d.models import ObjectTransform, RoomBounds, SceneObject, SceneState
from nl2scene3d.scene_reorganizer import (
    _build_scene_json_for_llm,
    _validate_and_sanitize_llm_output,
)

def _make_movable_obj(
    name: str,
    location: list[float],
    dimensions: list[float] | None = None,
) -> SceneObject:
    """Factory per oggetti movibili di test."""
    return SceneObject(
        name=name,
        object_type="MESH",
        transform=ObjectTransform(
            location=location,
            rotation_euler=[0.0, 0.0, 0.0],
            dimensions=dimensions or [1.0, 1.0, 1.0],
        ),
        category="furniture",
        is_movable=True,
    )

def _make_static_obj(name: str, location: list[float]) -> SceneObject:
    """Factory per oggetti statici di test."""
    return SceneObject(
        name=name,
        object_type="MESH",
        transform=ObjectTransform(
            location=location,
            rotation_euler=[0.0, 0.0, 0.0],
            dimensions=[5.0, 0.2, 3.0],
        ),
        category="structural",
        is_movable=False,
    )

def _make_test_state() -> SceneState:
    """Crea uno SceneState di test per il reorganizer."""
    return SceneState(
        scene_name="test_room",
        objects=[
            _make_movable_obj("sofa", [0.0, 0.0, 0.0]),
            _make_movable_obj("table", [2.0, 0.0, 0.0]),
            _make_static_obj("wall_n", [0.0, 4.0, 1.5]),
        ],
        room_bounds=RoomBounds(-3.0, 3.0, -2.0, 2.0),
        pipeline_step="randomized",
    )

class TestBuildSceneJsonForLlm:
    """Test per la costruzione del JSON da inviare all'LLM."""

    def test_json_contains_all_objects(self) -> None:
        """Il JSON deve contenere tutti gli oggetti della scena."""
        state = _make_test_state()
        scene_json_str = _build_scene_json_for_llm(state)
        data = json.loads(scene_json_str)

        assert "objects" in data
        assert len(data["objects"]) == 3

    def test_json_contains_required_fields(self) -> None:
        """Ogni oggetto nel JSON deve avere i campi richiesti."""
        state = _make_test_state()
        scene_json_str = _build_scene_json_for_llm(state)
        data = json.loads(scene_json_str)

        for obj_data in data["objects"]:
            assert "name" in obj_data
            assert "location" in obj_data
            assert "rotation_euler" in obj_data

    def test_json_is_valid(self) -> None:
        """Il JSON generato deve essere valido."""
        state = _make_test_state()
        scene_json_str = _build_scene_json_for_llm(state)
        # Non deve sollevare eccezioni
        parsed = json.loads(scene_json_str)
        assert parsed is not None

class TestValidateAndSanitizeLlmOutput:
    """Test per la validazione e sanitizzazione dell'output LLM."""

    def test_valid_output_applied(self) -> None:
        """Un output LLM valido deve essere applicato correttamente."""
        state = _make_test_state()
        llm_output = {
            "objects": [
                {
                    "name": "sofa",
                    "location": [1.0, 1.0, 0.0],
                    "rotation_euler": [0.0, 0.0, math.pi / 2],
                },
                {
                    "name": "table",
                    "location": [-1.0, -1.0, 0.0],
                    "rotation_euler": [0.0, 0.0, 0.0],
                },
                {
                    "name": "wall_n",
                    "location": [0.0, 4.0, 1.5],
                    "rotation_euler": [0.0, 0.0, 0.0],
                },
            ]
        }

        result = _validate_and_sanitize_llm_output(llm_output, state)
        sofa = result.get_object_by_name("sofa")
        assert sofa is not None
        assert sofa.transform.location[0] == pytest.approx(1.0)
        assert sofa.transform.location[1] == pytest.approx(1.0)

    def test_coordinates_outside_bounds_clamped(self) -> None:
        """Coordinate fuori dai bounds devono essere clampate."""
        state = _make_test_state()
        llm_output = {
            "objects": [
                {
                    "name": "sofa",
                    "location": [100.0, -200.0, 0.0],  # Molto fuori dai bounds
                    "rotation_euler": [0.0, 0.0, 0.0],
                },
                {
                    "name": "table",
                    "location": [1.0, 1.0, 0.0],
                    "rotation_euler": [0.0, 0.0, 0.0],
                },
                {
                    "name": "wall_n",
                    "location": [0.0, 4.0, 1.5],
                    "rotation_euler": [0.0, 0.0, 0.0],
                },
            ]
        }

        result = _validate_and_sanitize_llm_output(llm_output, state)
        sofa = result.get_object_by_name("sofa")
        bounds = state.room_bounds

        assert sofa is not None
        assert sofa.transform.location[0] <= bounds.x_max
        assert sofa.transform.location[1] >= bounds.y_min

    def test_z_coordinate_preserved_after_llm(self) -> None:
        """L'LLM non deve poter modificare la coordinata Z."""
        state = _make_test_state()
        original_z = state.get_object_by_name("sofa").transform.location[2]

        llm_output = {
            "objects": [
                {
                    "name": "sofa",
                    "location": [1.0, 1.0, 999.0],  # Z errata dell'LLM
                    "rotation_euler": [0.0, 0.0, 0.0],
                },
                {
                    "name": "table",
                    "location": [1.0, 1.0, 0.0],
                    "rotation_euler": [0.0, 0.0, 0.0],
                },
                {
                    "name": "wall_n",
                    "location": [0.0, 4.0, 1.5],
                    "rotation_euler": [0.0, 0.0, 0.0],
                },
            ]
        }

        result = _validate_and_sanitize_llm_output(llm_output, state)
        sofa = result.get_object_by_name("sofa")
        assert sofa.transform.location[2] == pytest.approx(original_z)

    def test_missing_object_uses_original_position(self) -> None:
        """Un oggetto mancante nella risposta LLM usa la posizione originale."""
        state = _make_test_state()
        original_table_loc = list(
            state.get_object_by_name("table").transform.location
        )

        # L'LLM non include 'table' nella risposta
        llm_output = {
            "objects": [
                {
                    "name": "sofa",
                    "location": [1.0, 1.0, 0.0],
                    "rotation_euler": [0.0, 0.0, 0.0],
                },
                {
                    "name": "wall_n",
                    "location": [0.0, 4.0, 1.5],
                    "rotation_euler": [0.0, 0.0, 0.0],
                },
            ]
        }

        result = _validate_and_sanitize_llm_output(llm_output, state)
        table = result.get_object_by_name("table")
        assert table is not None
        assert table.transform.location == pytest.approx(original_table_loc)

    def test_static_objects_not_modified_by_llm(self) -> None:
        """Gli oggetti statici non devono essere modificati anche se l'LLM lo suggerisce."""
        state = _make_test_state()
        original_wall_loc = list(
            state.get_object_by_name("wall_n").transform.location
        )

        llm_output = {
            "objects": [
                {
                    "name": "sofa",
                    "location": [1.0, 1.0, 0.0],
                    "rotation_euler": [0.0, 0.0, 0.0],
                },
                {
                    "name": "table",
                    "location": [-1.0, -1.0, 0.0],
                    "rotation_euler": [0.0, 0.0, 0.0],
                },
                {
                    "name": "wall_n",
                    "location": [5.0, 5.0, 1.5],  # LLM tenta di spostare il muro
                    "rotation_euler": [0.0, 0.0, 1.57],
                },
            ]
        }

        result = _validate_and_sanitize_llm_output(llm_output, state)
        wall = result.get_object_by_name("wall_n")
        assert wall is not None
        assert wall.transform.location == pytest.approx(original_wall_loc)

    def test_invalid_location_format_uses_original(self) -> None:
        """Un formato di location non valido deve usare la posizione originale."""
        state = _make_test_state()
        original_sofa_loc = list(
            state.get_object_by_name("sofa").transform.location
        )

        llm_output = {
            "objects": [
                {
                    "name": "sofa",
                    "location": "invalid_string",  # Formato errato
                    "rotation_euler": [0.0, 0.0, 0.0],
                },
                {
                    "name": "table",
                    "location": [1.0, 1.0, 0.0],
                    "rotation_euler": [0.0, 0.0, 0.0],
                },
                {
                    "name": "wall_n",
                    "location": [0.0, 4.0, 1.5],
                    "rotation_euler": [0.0, 0.0, 0.0],
                },
            ]
        }

        result = _validate_and_sanitize_llm_output(llm_output, state)
        sofa = result.get_object_by_name("sofa")
        assert sofa is not None
        assert sofa.transform.location == pytest.approx(original_sofa_loc)

    def test_pipeline_step_set_to_reordered(self) -> None:
        """Lo step della pipeline deve essere impostato a 'reordered'."""
        state = _make_test_state()
        llm_output = {
            "objects": [
                {"name": "sofa", "location": [0.0, 0.0, 0.0], "rotation_euler": [0.0, 0.0, 0.0]},
                {"name": "table", "location": [1.0, 1.0, 0.0], "rotation_euler": [0.0, 0.0, 0.0]},
                {"name": "wall_n", "location": [0.0, 4.0, 1.5], "rotation_euler": [0.0, 0.0, 0.0]},
            ]
        }
        result = _validate_and_sanitize_llm_output(llm_output, state)
        assert result.pipeline_step == "reordered"
