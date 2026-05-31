"""
Test unitari per il modulo scene_loader.

Verifica la classificazione degli oggetti, il calcolo dei bounds
e la serializzazione/deserializzazione dello SceneState da JSON,
senza dipendere dall'ambiente Blender.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nl2scene3d.models import ObjectTransform, RoomBounds, SceneObject, SceneState
from nl2scene3d.config import PipelineConfig
from nl2scene3d.scene_loader import (
    SceneLoader,
    extract_room_bounds_from_objects,
)

class TestClassifyObject:
    """Test per la funzione di classificazione degli oggetti."""

    @pytest.fixture
    def loader(self) -> SceneLoader:
        config = PipelineConfig(
            scenes_dir=Path("scenes"),
            outputs_dir=Path("outputs"),
            max_movable_objects=20,
            randomizer_seed=42,
            min_object_dimension=0.05,
            non_mesh_types=frozenset(["CAMERA", "LIGHT", "SPEAKER", "ARMATURE", "EMPTY", "CURVE"]),
            structural_patterns=["wall", "floor", "ceiling", "room", "door", "window"],
            ceiling_light_patterns=["ceiling", "pendant", "chandelier"]
        )
        return SceneLoader(config)

    def test_camera_is_not_movable(self, loader: SceneLoader) -> None:
        """Le camere non devono essere movibili."""
        category, is_movable = loader._classify_object("Camera", "CAMERA", [0.1, 0.1, 0.1])
        assert is_movable is False
        assert category == "technical"

    def test_light_is_not_movable(self, loader: SceneLoader) -> None:
        """Le luci sono di tipo tecnico e non movibili."""
        category, is_movable = loader._classify_object("Sun", "LIGHT", [0.0, 0.0, 0.0])
        assert is_movable is False
        assert category == "technical"

    def test_wall_is_structural(self, loader: SceneLoader) -> None:
        """I muri devono essere classificati come strutturali e non movibili."""
        category, is_movable = loader._classify_object("wall_north", "MESH", [5.0, 0.2, 3.0])
        assert category == "structural"
        assert is_movable is False

    def test_floor_is_structural(self, loader: SceneLoader) -> None:
        """Il pavimento deve essere classificato come strutturale."""
        category, is_movable = loader._classify_object("Floor", "MESH", [6.0, 5.0, 0.05])
        assert category == "structural"
        assert is_movable is False

    def test_sofa_is_movable(self, loader: SceneLoader) -> None:
        """Il divano deve essere movibile e classificato come seating_large."""
        category, is_movable = loader._classify_object("sofa_01", "MESH", [2.0, 0.8, 0.9])
        assert category == "seating_large"
        assert is_movable is True

    def test_chair_is_movable(self, loader: SceneLoader) -> None:
        """La sedia deve essere movibile."""
        category, is_movable = loader._classify_object("chair_arm", "MESH", [0.6, 0.6, 0.9])
        assert category == "seating_small"
        assert is_movable is True

    def test_table_is_movable(self, loader: SceneLoader) -> None:
        """Il tavolo deve essere movibile."""
        category, is_movable = loader._classify_object("dining_table", "MESH", [1.5, 0.9, 0.75])
        assert category == "table"
        assert is_movable is True

    def test_ceiling_lamp_is_not_movable(self, loader: SceneLoader) -> None:
        """Le lampade da soffitto non devono essere movibili."""
        category, is_movable = loader._classify_object(
            "ceiling_lamp_pendant", "MESH", [0.3, 0.3, 0.5]
        )
        assert is_movable is False
        assert category == "light_ceiling"

    def test_floor_lamp_is_movable(self, loader: SceneLoader) -> None:
        """Le lampade da terra devono essere movibili."""
        category, is_movable = loader._classify_object(
            "floor_lamp_stand", "MESH", [0.3, 0.3, 1.6]
        )
        assert category == "light_floor"
        assert is_movable is True

    def test_tiny_object_is_not_movable(self, loader: SceneLoader) -> None:
        """Oggetti troppo piccoli non devono essere movibili."""
        small_dim = loader.config.min_object_dimension / 2
        category, is_movable = loader._classify_object(
            "screw_01", "MESH", [small_dim, small_dim, small_dim]
        )
        assert is_movable is False
        assert category == "decoration_small"

    def test_unknown_mesh_is_furniture(self, loader: SceneLoader) -> None:
        """Oggetti MESH non riconosciuti vengono classificati come furniture."""
        category, is_movable = loader._classify_object(
            "object_xyz_42", "MESH", [1.0, 1.0, 1.0]
        )
        assert category == "furniture"
        assert is_movable is True

    def test_classification_is_case_insensitive(self, loader: SceneLoader) -> None:
        """La classificazione deve essere case-insensitive."""
        cat1, mov1 = loader._classify_object("WALL_SOUTH", "MESH", [5.0, 0.2, 3.0])
        cat2, mov2 = loader._classify_object("Wall_South", "MESH", [5.0, 0.2, 3.0])
        assert cat1 == cat2 == "structural"
        assert mov1 == mov2 is False

class TestExtractRoomBounds:
    """Test per il calcolo automatico dei bounds della stanza."""

    def _make_obj(
        self, name: str, location: list[float], dimensions: list[float],
        is_movable: bool = False,
    ) -> SceneObject:
        """Factory per oggetti di test."""
        return SceneObject(
            name=name,
            object_type="MESH",
            transform=ObjectTransform(location, [0.0, 0.0, 0.0], dimensions),
            category="structural" if not is_movable else "furniture",
            is_movable=is_movable,
        )

    def test_bounds_from_structural_objects(self) -> None:
        """I bounds devono essere calcolati dagli oggetti strutturali."""
        objects = [
            self._make_obj("wall_north", [0.0, 3.0, 1.5], [6.0, 0.2, 3.0]),
            self._make_obj("wall_south", [0.0, -3.0, 1.5], [6.0, 0.2, 3.0]),
            self._make_obj("wall_east", [3.0, 0.0, 1.5], [0.2, 6.0, 3.0]),
            self._make_obj("wall_west", [-3.0, 0.0, 1.5], [0.2, 6.0, 3.0]),
        ]
        bounds = extract_room_bounds_from_objects(objects)
        assert bounds.x_min < 0.0
        assert bounds.x_max > 0.0
        assert bounds.y_min < 0.0
        assert bounds.y_max > 0.0

    def test_bounds_have_padding(self) -> None:
        """I bounds devono includere un margine di sicurezza."""
        objects = [
            self._make_obj("wall_n", [0.0, 3.0, 1.5], [6.0, 0.2, 3.0]),
            self._make_obj("wall_s", [0.0, -3.0, 1.5], [6.0, 0.2, 3.0]),
        ]
        bounds = extract_room_bounds_from_objects(objects)
        # I bounds devono essere leggermente piu' larghi delle posizioni strutturali
        assert bounds.y_max >= 3.0
        assert bounds.y_min <= -3.0

    def test_bounds_main_room_strategy(self) -> None:
        """Verifica la strategia dell'oggetto stanza principale (Bug 1.3/2.2)."""
        objects = [
            self._make_obj("room_mesh", [0.0, 0.0, 1.5], [10.0, 8.0, 3.0]),
            self._make_obj("small_decor", [0.0, 0.0, 0.0], [0.1, 0.1, 0.1], is_movable=True),
        ]
        bounds = extract_room_bounds_from_objects(objects)
        assert bounds.x_min == -5.0
        assert bounds.x_max == 5.0
        assert bounds.y_min == -4.0
        assert bounds.y_max == 4.0

    def test_bounds_no_structural_objects(self) -> None:
        """Verifica il fallback se non ci sono oggetti strutturali."""
        objects = [
            self._make_obj("sofa", [0.0, 0.0, 0.0], [2.0, 1.0, 1.0], is_movable=True),
        ]
        bounds = extract_room_bounds_from_objects(objects)
        assert bounds.x_min == -1.0
        assert bounds.x_max == 1.0

    def test_bounds_empty_scene(self) -> None:
        """Verifica bounds di default per scena vuota."""
        bounds = extract_room_bounds_from_objects([])
        assert bounds.x_min == -5.0
        assert bounds.z_ceiling == 3.0

class TestSceneLoaderJsonIO:
    """Test per le operazioni di I/O JSON dello SceneLoader."""

    def _make_sample_state(self) -> SceneState:
        """Crea uno SceneState completo per i test di I/O."""
        return SceneState(
            scene_name="test_io_scene",
            objects=[
                SceneObject(
                    name="sofa",
                    object_type="MESH",
                    transform=ObjectTransform(
                        [1.0, -0.5, 0.0],
                        [0.0, 0.0, math.pi / 4],
                        [2.0, 0.8, 0.9],
                    ),
                    category="seating_large",
                    is_movable=True,
                ),
                SceneObject(
                    name="wall_n",
                    object_type="MESH",
                    transform=ObjectTransform(
                        [0.0, 4.0, 1.5],
                        [0.0, 0.0, 0.0],
                        [8.0, 0.2, 3.0],
                    ),
                    category="structural",
                    is_movable=False,
                ),
            ],
            room_bounds=RoomBounds(-4.0, 4.0, -3.0, 3.0, 0.0, 3.0),
            pipeline_step="original",
        )

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Verifica che save + load riproduca lo stato originale."""
        config = PipelineConfig(
            scenes_dir=Path("scenes"),
            outputs_dir=tmp_path,
            max_movable_objects=20,
            randomizer_seed=42
        )
        loader = SceneLoader(config)
        state = self._make_sample_state()

        json_path = tmp_path / "test_state.json"
        loader.save_state_to_json(state, json_path)

        assert json_path.exists()

        restored = SceneLoader.load_state_from_json(json_path)
        assert restored.scene_name == state.scene_name
        assert restored.pipeline_step == state.pipeline_step
        assert len(restored.objects) == len(state.objects)

        sofa_restored = restored.get_object_by_name("sofa")
        sofa_original = state.get_object_by_name("sofa")
        assert sofa_restored is not None
        assert sofa_original is not None
        assert sofa_restored.transform.location == pytest.approx(
            sofa_original.transform.location
        )

    def test_load_nonexistent_file_raises(self) -> None:
        """Il caricamento di un file inesistente deve sollevare FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            SceneLoader.load_state_from_json(Path("/nonexistent/path/state.json"))

    def test_saved_json_is_valid(self, tmp_path: Path) -> None:
        """Il JSON salvato deve essere leggibile e avere la struttura corretta."""
        config = PipelineConfig(
            scenes_dir=Path("scenes"),
            outputs_dir=tmp_path,
            max_movable_objects=20,
            randomizer_seed=42
        )
        loader = SceneLoader(config)
        state = self._make_sample_state()

        json_path = tmp_path / "valid_json.json"
        loader.save_state_to_json(state, json_path)

        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)

        assert "scene_name" in data
        assert "objects" in data
        assert "room_bounds" in data
        assert isinstance(data["objects"], list)
        assert len(data["objects"]) == 2

    def test_output_directory_created_automatically(self, tmp_path: Path) -> None:
        """La directory di output deve essere creata automaticamente se mancante."""
        config = PipelineConfig(
            scenes_dir=Path("scenes"),
            outputs_dir=tmp_path,
            max_movable_objects=20,
            randomizer_seed=42
        )
        loader = SceneLoader(config)
        state = self._make_sample_state()

        nested_path = tmp_path / "deep" / "nested" / "dir" / "state.json"
        assert not nested_path.parent.exists()

        loader.save_state_to_json(state, nested_path)
        assert nested_path.exists()

class TestSceneLoaderExtraction:
    """Test per l'estrazione dello stato tramite bpy mocks."""

    def test_extract_scene_state(self) -> None:
        """Verifica l'estrazione dello stato da oggetti Blender mockati."""
        config = PipelineConfig(
            scenes_dir=Path("scenes"),
            outputs_dir=Path("outputs"),
            max_movable_objects=2,
            randomizer_seed=42
        )
        loader = SceneLoader(config)
        
        mock_bpy = MagicMock()
        mock_obj = MagicMock()
        mock_obj.name = "test_chair"
        mock_obj.type = "MESH"
        mock_obj.location = MagicMock(x=1.0, y=2.0, z=0.0)
        mock_obj.rotation_euler = MagicMock(x=0.0, y=0.0, z=0.0)
        mock_obj.dimensions = MagicMock(x=0.5, y=0.5, z=0.9)
        
        mock_bpy.context.scene.objects = [mock_obj]
        mock_bpy.context.scene.name = "Scene"
        
        with patch.dict("sys.modules", {"bpy": mock_bpy}):
            state = loader.extract_scene_state()
            assert state.scene_name == "Scene"
            assert len(state.objects) == 1
            assert state.objects[0].name == "test_chair"
            assert state.objects[0].is_movable is True

    def test_extract_scene_state_movable_limit(self) -> None:
        """Verifica il rispetto del limite massimo di oggetti movibili."""
        config = PipelineConfig(
            scenes_dir=Path("scenes"),
            outputs_dir=Path("outputs"),
            max_movable_objects=1,
            randomizer_seed=42
        )
        loader = SceneLoader(config)
        
        mock_bpy = MagicMock()
        mock_obj1 = MagicMock(name="Chair1")
        mock_obj1.name = "Chair1"
        mock_obj1.type = "MESH"
        mock_obj1.location = MagicMock(x=1.0, y=1.0, z=0.0)
        mock_obj1.rotation_euler = MagicMock(x=0.0, y=0.0, z=0.0)
        mock_obj1.dimensions = MagicMock(x=1, y=1, z=1)
        
        mock_obj2 = MagicMock(name="Chair2")
        mock_obj2.name = "Chair2"
        mock_obj2.type = "MESH"
        mock_obj2.location = MagicMock(x=2.0, y=2.0, z=0.0)
        mock_obj2.rotation_euler = MagicMock(x=0.0, y=0.0, z=0.0)
        mock_obj2.dimensions = MagicMock(x=1, y=1, z=1)
        
        mock_bpy.context.scene.objects = [mock_obj1, mock_obj2]
        
        with patch.dict("sys.modules", {"bpy": mock_bpy}):
            state = loader.extract_scene_state()
            movable = [o for o in state.objects if o.is_movable]
            assert len(movable) == 1
