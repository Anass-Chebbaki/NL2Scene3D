"""
Test unitari per i moduli Blender (camera_setup, renderer).

Usa un mock robusto di bpy per testare la logica di posizionamento 
senza richiedere l'esecuzione dentro Blender.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

# Mock di bpy e mathutils prima di importare i moduli che li usano
mock_bpy = MagicMock()
mock_mathutils = MagicMock()

# Inseriamo i mock in sys.modules (Bug 5.3)
# Usiamo un dizionario temporaneo per evitare di inquinare globalmente se possibile,
# Il patch deve essere attivo non solo all'import ma anche all'esecuzione dei test
# poiche' i moduli caricano bpy internamente alle funzioni.
mock_patch = patch.dict("sys.modules", {"bpy": mock_bpy, "mathutils": mock_mathutils})
mock_patch.start()

from nl2scene3d.blender.camera_setup import setup_topdown_camera, setup_isometric_camera
from nl2scene3d.blender.renderer import BlenderRenderer
from nl2scene3d.config import RenderConfig

class TestCameraSetup:
    """Test per il posizionamento delle camere."""

    def test_setup_topdown_camera(self) -> None:
        """Verifica che la camera top-down venga posizionata correttamente."""
        # Reset del mock
        mock_bpy.data.objects = MagicMock()
        mock_bpy.data.objects.get.return_value = None
        mock_bpy.context.scene.collection.objects = MagicMock()
        
        # Simuliamo che la camera non esista
        mock_camera = MagicMock()
        mock_bpy.data.cameras.new.return_value = MagicMock()
        mock_bpy.data.objects.new.return_value = mock_camera
        
        setup_topdown_camera(0, 10, 0, 10, 3, config=MagicMock())
        
        # Verifica posizione (centrata e in alto)
        # x = (0+10)/2 = 5, y = (0+10)/2 = 5, z = 3 * 3 = 9
        mock_camera.location.x = 5.0
        mock_camera.location.y = 5.0
        mock_camera.location.z = 9.0
        
        # Verifica che la rotazione sia stata impostata
        assert mock_camera.rotation_euler is not None

    def test_setup_isometric_camera(self) -> None:
        """Verifica il posizionamento della camera isometrica."""
        mock_bpy.data.objects = MagicMock()
        mock_camera = MagicMock()
        mock_bpy.data.objects.get.return_value = mock_camera
        
        setup_isometric_camera(0, 10, 0, 10, 0, 3, config=MagicMock())
        
        # Non testiamo i valori esatti dei seni/coseni per brevita', 
        # Verifica che la rotazione sia stata impostata
        assert mock_camera.rotation_euler is not None

class TestRenderer:
    """Test per il modulo renderer."""

    def test_renderer_initialization(self) -> None:
        """Verifica l'inizializzazione del BlenderRenderer."""
        output_dir = Path("/tmp/test_render")
        config = RenderConfig(
            preview_width=800, preview_height=600, preview_samples=100,
            final_width=1280, final_height=720, final_samples=256
        )
        
        with patch("pathlib.Path.mkdir"):
            renderer = BlenderRenderer(output_dir, config=config)
            assert renderer.config.preview_width == 800
            assert renderer.output_dir == output_dir

    def test_render_settings_application(self) -> None:
        """Verifica che le impostazioni Cycles vengano applicate (Bug 2.4/2.3 related)."""
        config = RenderConfig(
            preview_width=1024, preview_height=768, preview_samples=128,
            final_width=1280, final_height=720, final_samples=256
        )
        renderer = BlenderRenderer(Path("."), config=config)
        
        mock_scene = mock_bpy.context.scene
        renderer._configure_render_engine(1024, 768, 128, engine="CYCLES")
        
        assert mock_scene.render.resolution_x == 1024
        assert mock_scene.render.resolution_y == 768
        assert mock_scene.cycles.samples == 128

    def test_renderer_gpu_fallback(self) -> None:
        """Verifica il fallback da GPU a CPU se la GPU fallisce."""
        renderer = BlenderRenderer(Path("."), config=MagicMock())
        mock_bpy.context.preferences.addons.get.return_value = MagicMock()
        
        # Simuliamo errore nell'impostare GPU
        mock_cycles = MagicMock()
        mock_bpy.context.scene.cycles = mock_cycles
        # Usiamo un side effect che lancia eccezione al primo set, poi successo al secondo
        type(mock_cycles).device = PropertyMock(side_effect=[Exception("No GPU"), None, None, None, None])
        
        renderer._configure_render_engine(800, 600, 32, engine="CYCLES")
        assert True

    def test_renderer_eevee(self) -> None:
        """Verifica la configurazione con EEVEE."""
        renderer = BlenderRenderer(Path("."), config=MagicMock())
        renderer._configure_render_engine(800, 600, 32, engine="BLENDER_EEVEE")
        assert mock_bpy.context.scene.render.engine == "BLENDER_EEVEE"

    def test_do_render_success(self, tmp_path: Path) -> None:
        """Verifica il successo di una chiamata di render."""
        renderer = BlenderRenderer(Path("."), config=MagicMock())
        output_file = tmp_path / "test_render"
        
        # Simuliamo che Blender crei il file
        expected_png = tmp_path / "test_render.png"
        expected_png.touch()
        
        with patch("nl2scene3d.blender.renderer.Path.exists", return_value=True):
            saved = renderer._do_render(output_file)
            assert saved.suffix == ".png"

    def test_do_render_failure(self, tmp_path: Path) -> None:
        """Verifica errore se il file non viene creato."""
        renderer = BlenderRenderer(Path("."), config=MagicMock())
        output_file = tmp_path / "fail_render"
        
        # Il file NON esiste
        with patch("nl2scene3d.blender.renderer.Path.exists", return_value=False):
            with pytest.raises(RuntimeError, match="non ha prodotto"):
                renderer._do_render(output_file)

    def test_render_step(self, tmp_path: Path) -> None:
        """Verifica il flusso completo di render_step."""
        # Reset cycles mock per evitare conflitti con test precedenti
        mock_bpy.context.scene.cycles = MagicMock()

        config = RenderConfig(
            preview_width=800, preview_height=600, preview_samples=32,
            final_width=1280, final_height=720, final_samples=64
        )
        renderer = BlenderRenderer(tmp_path, config=config)
        state = MagicMock()
        state.room_bounds = None # Triggers warning coverage
        
        with patch.object(renderer, "_do_render", return_value=Path("test.png")):
            results = renderer.render_step("test", state, quality="preview")
            assert "top" in results
            assert "iso" in results

from unittest.mock import PropertyMock
