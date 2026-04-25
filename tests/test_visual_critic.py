"""
Test unitari per il modulo visual_critic.

Verifica il parsing delle correzioni dall'LLM, l'applicazione
delle correzioni alla scena e la logica di fallback.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nl2scene3d.models import LLMCorrection, ObjectTransform, RoomBounds, SceneObject, SceneState
from nl2scene3d.visual_critic import (
    MAX_CORRECTIONS_TO_APPLY,
    _apply_corrections_to_state,
    _parse_corrections_from_llm,
)

def _make_movable_obj(name: str, location: list[float]) -> SceneObject:
    """Factory per oggetti movibili di test."""
    return SceneObject(
        name=name,
        object_type="MESH",
        transform=ObjectTransform(location, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        category="furniture",
        is_movable=True,
    )

def _make_static_obj(name: str, location: list[float]) -> SceneObject:
    """Factory per oggetti statici di test."""
    return SceneObject(
        name=name,
        object_type="MESH",
        transform=ObjectTransform(location, [0.0, 0.0, 0.0], [5.0, 0.2, 3.0]),
        category="structural",
        is_movable=False,
    )

def _make_test_state() -> SceneState:
    """Crea uno SceneState di test per il visual critic."""
    return SceneState(
        scene_name="test_room",
        objects=[
            _make_movable_obj("sofa", [0.5, 0.5, 0.0]),
            _make_movable_obj("chair", [2.0, 1.0, 0.0]),
            _make_static_obj("wall_n", [0.0, 4.0, 1.5]),
        ],
        room_bounds=RoomBounds(-3.0, 3.0, -2.0, 2.0),
        pipeline_step="reordered",
    )

class TestParseCorrectionFromLlm:
    """Test per il parsing dell'output LLM Vision."""

    def test_parse_valid_output_with_corrections(self) -> None:
        """Un output valido con correzioni deve essere parsato correttamente."""
        llm_output = {
            "quality_assessment": "Good layout with minor issues",
            "score": 7,
            "corrections": [
                {
                    "object_name": "sofa",
                    "action": "move",
                    "new_location": [1.0, 1.0, 0.0],
                    "new_rotation_euler": None,
                    "reason": "Move sofa closer to wall",
                }
            ],
        }
        score, assessment, corrections = _parse_corrections_from_llm(llm_output)

        assert score == 7
        assert "Good layout" in assessment
        assert len(corrections) == 1
        assert corrections[0].object_name == "sofa"
        assert corrections[0].action == "move"

    def test_parse_output_no_corrections(self) -> None:
        """Un output senza correzioni deve produrre una lista vuota."""
        llm_output = {
            "quality_assessment": "Layout is well organized",
            "score": 9,
            "corrections": [],
        }
        score, assessment, corrections = _parse_corrections_from_llm(llm_output)

        assert score == 9
        assert len(corrections) == 0

    def test_parse_invalid_type_returns_empty(self) -> None:
        """Un tipo di output non valido deve restituire dati di default."""
        score, assessment, corrections = _parse_corrections_from_llm("invalid")
        assert len(corrections) == 0

    def test_parse_list_input_as_direct_corrections(self) -> None:
        """Una lista diretta viene interpretata come lista di correzioni."""
        llm_output = [
            {
                "object_name": "chair",
                "action": "rotate",
                "new_location": None,
                "new_rotation_euler": [0.0, 0.0, 1.57],
                "reason": "Rotate chair toward table",
            }
        ]
        score, assessment, corrections = _parse_corrections_from_llm(llm_output)
        assert len(corrections) == 1
        assert corrections[0].object_name == "chair"

    def test_invalid_correction_items_are_skipped(self) -> None:
        """Correzioni con formato non valido vengono ignorate."""
        llm_output = {
            "score": 6,
            "quality_assessment": "Ok",
            "corrections": [
                "this_is_not_a_dict",
                {"object_name": "sofa", "action": "move", "new_location": [1.0, 0.0, 0.0]},
                None,
            ],
        }
        score, assessment, corrections = _parse_corrections_from_llm(llm_output)
        # Solo la correzione valida deve essere inclusa
        assert len(corrections) == 1

class TestApplyCorrectionsToState:
    """Test per l'applicazione delle correzioni alla scena."""

    def test_move_correction_applied(self) -> None:
        """Una correzione 'move' deve aggiornare la posizione."""
        state = _make_test_state()
        corrections = [
            LLMCorrection(
                object_name="sofa",
                action="move",
                new_location=[2.0, 1.5, 0.0],
                reason="Move sofa",
            )
        ]
        result = _apply_corrections_to_state(state, corrections, state.room_bounds)
        sofa = result.get_object_by_name("sofa")
        assert sofa is not None
        assert sofa.transform.location[0] == pytest.approx(2.0)
        assert sofa.transform.location[1] == pytest.approx(1.5)

    def test_rotate_correction_applied(self) -> None:
        """Una correzione 'rotate' deve aggiornare la rotazione."""
        state = _make_test_state()
        corrections = [
            LLMCorrection(
                object_name="chair",
                action="rotate",
                new_rotation_euler=[0.0, 0.0, math.pi / 2],
                reason="Rotate chair",
            )
        ]
        result = _apply_corrections_to_state(state, corrections, state.room_bounds)
        chair = result.get_object_by_name("chair")
        assert chair is not None
        assert chair.transform.rotation_euler[2] == pytest.approx(math.pi / 2)

    def test_static_objects_not_corrected(self) -> None:
        """Le correzioni non devono modificare gli oggetti statici."""
        state = _make_test_state()
        original_wall_loc = list(
            state.get_object_by_name("wall_n").transform.location
        )
        corrections = [
            LLMCorrection(
                object_name="wall_n",
                action="move",
                new_location=[5.0, 5.0, 1.5],
                reason="This should be ignored",
            )
        ]
        result = _apply_corrections_to_state(state, corrections, state.room_bounds)
        wall = result.get_object_by_name("wall_n")
        assert wall.transform.location == pytest.approx(original_wall_loc)

    def test_correction_for_nonexistent_object_skipped(self) -> None:
        """Correzioni per oggetti inesistenti vengono ignorate senza errori."""
        state = _make_test_state()
        corrections = [
            LLMCorrection(
                object_name="nonexistent_object",
                action="move",
                new_location=[1.0, 1.0, 0.0],
                reason="Should not crash",
            )
        ]
        result = _apply_corrections_to_state(state, corrections, state.room_bounds)
        assert result.metadata["skipped_corrections"] == 1

    def test_z_coordinate_preserved_in_correction(self) -> None:
        """Le correzioni non devono modificare la coordinata Z."""
        state = _make_test_state()
        original_z = state.get_object_by_name("sofa").transform.location[2]

        corrections = [
            LLMCorrection(
                object_name="sofa",
                action="move",
                new_location=[1.0, 1.0, 999.0],  # Z errata
                reason="Move sofa",
            )
        ]
        result = _apply_corrections_to_state(state, corrections, state.room_bounds)
        sofa = result.get_object_by_name("sofa")
        assert sofa.transform.location[2] == pytest.approx(original_z)

    def test_max_corrections_limit_enforced(self) -> None:
        """Non devono essere applicate piu' correzioni del limite configurato."""
        state = _make_test_state()
        # Crea molte piu' correzioni del limite
        corrections = [
            LLMCorrection(
                object_name="sofa",
                action="move",
                new_location=[float(i) * 0.1, 0.0, 0.0],
                reason=f"Correction {i}",
            )
            for i in range(MAX_CORRECTIONS_TO_APPLY + 10)
        ]
        # Non deve sollevare eccezioni
        result = _apply_corrections_to_state(state, corrections, state.room_bounds)
        # Il contatore delle applicazioni non deve superare il limite
        assert result.metadata["applied_corrections"] <= MAX_CORRECTIONS_TO_APPLY

    def test_pipeline_step_set_to_refined(self) -> None:
        """Lo step della pipeline deve essere impostato a 'refined'."""
        state = _make_test_state()
        result = _apply_corrections_to_state(state, [], state.room_bounds)
        assert result.pipeline_step == "refined"
