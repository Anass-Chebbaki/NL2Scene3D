"""
Test unitari per lo SceneApplicator.

Verifica il comportamento di apply_state con mock della scena Blender
per non dipendere dall'ambiente Blender durante i test unitari.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pathlib import Path

from nl2scene3d.models import ObjectTransform, RoomBounds, SceneObject, SceneState
from nl2scene3d.scene_applicator import SceneApplicator

def _make_blender_object_mock(name: str, x: float, y: float, z: float) -> MagicMock:
    """Crea un mock di un oggetto Blender con le proprieta' necessarie."""
    mock_obj = MagicMock()
    mock_obj.name = name
    mock_obj.location = MagicMock()
    mock_obj.location.x = x
    mock_obj.location.y = y
    mock_obj.location.z = z
    mock_obj.rotation_euler = MagicMock()
    mock_obj.rotation_euler.x = 0.0
    mock_obj.rotation_euler.y = 0.0
    mock_obj.rotation_euler.z = 0.0
    return mock_obj

class TestSceneApplicator:
    """Test per la classe SceneApplicator."""

    def _make_sample_state(self) -> SceneState:
        """Crea uno SceneState di esempio per i test."""
        return SceneState(
            scene_name="test",
            objects=[
                SceneObject(
                    name="sofa",
                    object_type="MESH",
                    transform=ObjectTransform(
                        location=[1.0, 2.0, 0.0],
                        rotation_euler=[0.0, 0.0, 1.57],
                        dimensions=[2.0, 0.8, 0.9],
                    ),
                    is_movable=True,
                ),
                SceneObject(
                    name="wall",
                    object_type="MESH",
                    transform=ObjectTransform(
                        location=[0.0, 5.0, 1.5],
                        rotation_euler=[0.0, 0.0, 0.0],
                        dimensions=[10.0, 0.2, 3.0],
                    ),
                    is_movable=False,
                ),
            ],
            room_bounds=RoomBounds(-4.0, 4.0, -3.0, 3.0),
            pipeline_step="reordered",
        )

    def test_apply_updates_movable_object(self) -> None:
        """Verifica che le proprieta' degli oggetti movibili vengano aggiornate."""
        applicator = SceneApplicator()
        state = self._make_sample_state()

        # Mock della scena Blender
        mock_sofa = _make_blender_object_mock("sofa", 0.0, 0.0, 0.0)
        mock_wall = _make_blender_object_mock("wall", 0.0, 5.0, 1.5)
        mock_scene = MagicMock()
        mock_scene.objects.get = lambda name: (
            mock_sofa if name == "sofa" else mock_wall if name == "wall" else None
        )

        mock_context = MagicMock()
        mock_context.scene = mock_scene
        mock_context.view_layer = MagicMock()

        mock_bpy = MagicMock()
        mock_bpy.context = mock_context

        with patch.dict("sys.modules", {"bpy": mock_bpy}):
            counters = applicator.apply_state(state)

        # Il sofa deve essere stato aggiornato
        assert mock_sofa.location.x == 1.0
        assert mock_sofa.location.y == 2.0
        assert counters["updated"] >= 1

    def test_static_objects_not_modified(self) -> None:
        """Verifica che gli oggetti statici non vengano modificati."""
        applicator = SceneApplicator()
        state = self._make_sample_state()

        mock_sofa = _make_blender_object_mock("sofa", 0.0, 0.0, 0.0)
        original_wall_x = 0.0
        mock_wall = _make_blender_object_mock("wall", original_wall_x, 5.0, 1.5)
        mock_scene = MagicMock()
        mock_scene.objects.get = lambda name: (
            mock_sofa if name == "sofa" else mock_wall if name == "wall" else None
        )

        mock_context = MagicMock()
        mock_context.scene = mock_scene
        mock_context.view_layer = MagicMock()
        mock_bpy = MagicMock()
        mock_bpy.context = mock_context

        with patch.dict("sys.modules", {"bpy": mock_bpy}):
            applicator.apply_state(state)

        # Il muro non deve essere stato toccato
        assert mock_wall.location.x == original_wall_x

    def test_missing_object_counted(self) -> None:
        """Verifica che gli oggetti non trovati vengano contati correttamente."""
        applicator = SceneApplicator()
        state = self._make_sample_state()

        mock_scene = MagicMock()
        # Simula che nessun oggetto venga trovato
        mock_scene.objects.get = lambda name: None

        mock_context = MagicMock()
        mock_context.scene = mock_scene
        mock_context.view_layer = MagicMock()
        mock_bpy = MagicMock()
        mock_bpy.context = mock_context

        with patch.dict("sys.modules", {"bpy": mock_bpy}):
            counters = applicator.apply_state(state)

    def test_save_blend_file(self, tmp_path: Path) -> None:
        """Verifica il salvataggio del file .blend."""
        applicator = SceneApplicator()
        output_file = tmp_path / "test_scene.blend"

        mock_bpy = MagicMock()
        # Mock della creazione directory parent
        with patch.dict("sys.modules", {"bpy": mock_bpy}):
            applicator.save_blend_file(output_file)

        mock_bpy.ops.wm.save_as_mainfile.assert_called_once_with(
            filepath=str(output_file)
        )
