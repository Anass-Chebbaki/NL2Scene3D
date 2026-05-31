# src/nl2scene3d/blender/renderer.py
"""
Automated rendering system for the NL2Scene3D pipeline.

Manages rendering of top-down, isometric (two angles), and front views
for each scene configuration. Separate settings are used for preview
renders (low quality) and the final render (high quality).

After the first render call, the camera is frozen to guarantee identical
framing across all pipeline steps, preventing zoom or aspect-ratio drift
in the final output.

Must be executed inside Blender's embedded Python environment.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

from nl2scene3d.blender.camera_setup import (
    get_frozen_state,
    reset_frozen_state,
    setup_front_camera,
    setup_isometric_camera,
    setup_isometric_camera_angle2,
    setup_topdown_camera,
)
from nl2scene3d.config import RenderConfig
from nl2scene3d.models import RoomBounds, SceneState

logger = logging.getLogger(__name__)

RenderView    = Literal["top", "iso", "iso2", "front"]
RenderQuality = Literal["preview", "final"]


# ---------------------------------------------------------------------------
# Renderer class
# ---------------------------------------------------------------------------

class BlenderRenderer:
    """
    Executes rendering of the current Blender scene.

    Manages render engine configuration, camera placement, and image
    saving for each scene state.

    Attributes:
        output_dir: Base directory for rendered output files.
        config:     Full render configuration.
    """

    def __init__(
        self,
        output_dir: Path,
        config: RenderConfig,
    ) -> None:
        """
        Initialize the renderer.

        Args:
            output_dir: Directory where rendered images will be saved.
            config:     Render configuration loaded from TOML.
        """
        self.output_dir = output_dir.resolve()
        self.config = config
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Reset camera state at the start of each pipeline run.
        reset_frozen_state()

        logger.info(
            "BlenderRenderer initialized. Output directory: %s.", self.output_dir
        )

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _configure_render_engine(
        self,
        width: int,
        height: int,
        samples: int,
        engine: str = "CYCLES",
    ) -> None:
        """
        Apply render settings to the current Blender scene.

        Args:
            width:   Render width in pixels.
            height:  Render height in pixels.
            samples: Number of samples for Cycles rendering.
            engine:  Blender render engine ('CYCLES' or 'BLENDER_EEVEE').

        Raises:
            ImportError: If bpy is not available.
        """
        try:
            import bpy  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Module 'bpy' requires the Blender environment."
            ) from exc

        scene = bpy.context.scene
        render = scene.render
        render.engine = engine
        render.resolution_x = width
        render.resolution_y = height
        render.resolution_percentage = 100

        if engine == "CYCLES":
            scene.cycles.samples = samples
            scene.cycles.use_denoising = True

            preferences = bpy.context.preferences
            cycles_addon = preferences.addons.get("cycles")
            if cycles_addon is not None:
                try:
                    bpy.context.scene.cycles.device = "GPU"
                except Exception:  # noqa: BLE001
                    bpy.context.scene.cycles.device = "CPU"
                    logger.debug("GPU not available. Falling back to CPU for Cycles.")

        logger.debug(
            "Render configured: %dx%d, engine=%s, samples=%d.",
            width,
            height,
            engine,
            samples,
        )

    def _do_render(self, output_path: Path) -> Path:
        """
        Execute the render and save the image as PNG.

        Args:
            output_path: Output file path without extension.

        Returns:
            Actual path of the saved file (with .png extension).

        Raises:
            ImportError:  If bpy is not available.
            RuntimeError: If the render does not produce a valid output file.
        """
        try:
            import bpy  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Module 'bpy' requires the Blender environment."
            ) from exc

        scene = bpy.context.scene
        render = scene.render
        render.filepath = str(output_path)
        render.image_settings.file_format = "PNG"
        render.image_settings.color_mode  = "RGBA"
        render.image_settings.color_depth = "8"

        bpy.ops.render.render(write_still=True)

        saved_path = output_path.with_suffix(".png")
        if not saved_path.exists():
            saved_path = Path(str(output_path) + ".png")

        if not saved_path.exists():
            raise RuntimeError(
                f"Render did not produce a valid output file. "
                f"Expected path: {saved_path}"
            )

        logger.info("Render saved: %s.", saved_path)
        return saved_path

    def _get_bounds_args(self, room_bounds: RoomBounds) -> dict:
        """Build the common keyword arguments for camera setup functions."""
        return {
            "scene_x_min":   room_bounds.x_min,
            "scene_x_max":   room_bounds.x_max,
            "scene_y_min":   room_bounds.y_min,
            "scene_y_max":   room_bounds.y_max,
            "scene_z_min":   room_bounds.z_floor,
            "scene_z_ceiling": room_bounds.z_ceiling,
            "config":        self.config,
        }

    # ---------------------------------------------------------------------------
    # Public rendering API
    # ---------------------------------------------------------------------------

    def render_step(
        self,
        step_name: str,
        state: SceneState,
        quality: RenderQuality = "preview",
        multi_view: bool = False,
    ) -> dict[str, Path]:
        """
        Render the requested views for a given scene state.

        With multi_view=False (default): generates top-down + isometric (2 views).
        With multi_view=True: generates top-down + iso + iso2 + front (4 views).

        After the first render_step call the camera is frozen; all subsequent
        steps reuse the same framing.

        Args:
            step_name:  Identifier for the configuration (e.g. 'original', 'randomized').
            state:      Current scene state.
            quality:    Render quality ('preview' or 'final').
            multi_view: If True, generate 4 views for the visual critic.

        Returns:
            Dictionary with keys 'top', 'iso', and optionally 'iso2', 'front'.
        """
        if quality == "final":
            width   = self.config.final_width
            height  = self.config.final_height
            samples = self.config.final_samples
        else:
            width   = self.config.preview_width
            height  = self.config.preview_height
            samples = self.config.preview_samples

        self._configure_render_engine(width, height, samples)

        room_bounds = state.room_bounds
        if room_bounds is None:
            logger.warning(
                "room_bounds not defined for scene '%s'. Using default bounds.",
                state.scene_name,
            )
            room_bounds = RoomBounds(
                x_min=-5.0, x_max=5.0,
                y_min=-5.0, y_max=5.0,
                z_floor=0.0, z_ceiling=3.0,
            )

        render_paths: dict[str, Path] = {}
        bounds_args = self._get_bounds_args(room_bounds)

        # Top-down args do not include scene_z_min.
        topdown_args = {
            "scene_x_min":    room_bounds.x_min,
            "scene_x_max":    room_bounds.x_max,
            "scene_y_min":    room_bounds.y_min,
            "scene_y_max":    room_bounds.y_max,
            "scene_z_ceiling": room_bounds.z_ceiling,
            "config":         self.config,
        }

        # Top-down view.
        setup_topdown_camera(**topdown_args)
        top_path = self.output_dir / f"render_{step_name}_top"
        render_paths["top"] = self._do_render(top_path)

        # Primary isometric view.
        setup_isometric_camera(**bounds_args)
        iso_path = self.output_dir / f"render_{step_name}_iso"
        render_paths["iso"] = self._do_render(iso_path)

        if multi_view:
            # Secondary isometric view (opposite angle).
            setup_isometric_camera_angle2(**bounds_args)
            iso2_path = self.output_dir / f"render_{step_name}_iso2"
            render_paths["iso2"] = self._do_render(iso2_path)

            # Low-elevation front view.
            setup_front_camera(**bounds_args)
            front_path = self.output_dir / f"render_{step_name}_front"
            render_paths["front"] = self._do_render(front_path)

        # Freeze the camera after the first render step.
        frozen_state = get_frozen_state()
        if not frozen_state.is_frozen:
            frozen_state.freeze()
            logger.info(
                "Camera frozen after first render step '%s'. "
                "All subsequent renders will use the same framing.",
                step_name,
            )

        views_str = ", ".join(f"{k}={v}" for k, v in render_paths.items())
        logger.info("Render for '%s' complete: %s.", step_name, views_str)

        return render_paths