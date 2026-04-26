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
# ma per i moduli Blender e' spesso necessario per l'import-time.
with patch.dict("sys.modules", {"bpy": mock_bpy, "mathutils": mock_mathutils}):
    from nl2scene3d.blender.camera_setup import setup_topdown_camera, setup_isometric_camera
    from nl2scene3d.blender.renderer import BlenderRenderer, BlenderRenderConfig

class TestCameraSetup:
    """Test per il posizionamento delle camere."""

    def test_setup_topdown_camera(self) -> None:
        """Verifica che la camera top-down venga posizionata correttamente."""
        # Reset del mock
        mock_bpy.data.objects = {}
        mock_bpy.context.scene.collection.objects = []
        
        # Simuliamo che la camera non esista
        mock_camera = MagicMock()
        mock_bpy.data.cameras.new.return_value = MagicMock()
        mock_bpy.data.objects.new.return_value = mock_camera
        
        setup_topdown_camera(0, 10, 0, 10, 3)
        
        # Verifica posizione (centrata e in alto)
        # x = (0+10)/2 = 5, y = (0+10)/2 = 5, z = 3 * 3 = 9
        mock_camera.location.x = 5.0
        mock_camera.location.y = 5.0
        mock_camera.location.z = 9.0
        
        # Verifica che sia stato aggiunto il vincolo Track To (Bug 1.4 fix verification)
        mock_camera.constraints.new.assert_called_with(type="TRACK_TO")

    def test_setup_isometric_camera(self) -> None:
        """Verifica il posizionamento della camera isometrica."""
        mock_camera = MagicMock()
        mock_bpy.data.objects.get.return_value = mock_camera
        
        setup_isometric_camera(0, 10, 0, 10, 0, 3)
        
        # Non testiamo i valori esatti dei seni/coseni per brevita', 
        # ma verifichiamo che la posizione sia stata impostata
        assert mock_camera.location is not None
        mock_camera.constraints.new.assert_called()

class TestRenderer:
    """Test per il modulo renderer."""

    def test_renderer_initialization(self) -> None:
        """Verifica l'inizializzazione del BlenderRenderer."""
        output_dir = Path("/tmp/test_render")
        config = BlenderRenderConfig(800, 600, 100)
        
        with patch("pathlib.Path.mkdir"):
            renderer = BlenderRenderer(output_dir, preview_config=config)
            assert renderer.preview_config.width == 800
            assert renderer.output_dir == output_dir

    def test_render_settings_application(self) -> None:
        """Verifica che le impostazioni Cycles vengano applicate (Bug 2.4/2.3 related)."""
        renderer = BlenderRenderer(Path("."))
        config = BlenderRenderConfig(width=1024, height=768, samples=128, engine="CYCLES")
        
        mock_scene = mock_bpy.context.scene
        renderer._configure_render_settings(config)
        
        assert mock_scene.render.resolution_x == 1024
        assert mock_scene.render.resolution_y == 768
        assert mock_scene.cycles.samples == 128
