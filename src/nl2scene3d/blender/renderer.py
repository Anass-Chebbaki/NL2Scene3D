# src/nl2scene3d/blender/renderer.py
"""
Sistema di rendering automatico per la pipeline NL2Scene3D.

Gestisce il rendering delle viste top-down e isometrica per ogni
configurazione della scena, con impostazioni separate per i render di
anteprima (bassa qualita') e il render finale (alta qualita').

Deve essere eseguito all'interno dell'ambiente Python di Blender.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

from nl2scene3d.blender.camera_setup import setup_isometric_camera, setup_topdown_camera
from nl2scene3d.models import RoomBounds, SceneState
from nl2scene3d.config import RenderConfig

logger = logging.getLogger(__name__)

RenderView = Literal["top", "iso"]
RenderQuality = Literal["preview", "final"]


class BlenderRenderer:
    """
    Esegue il rendering della scena corrente in Blender.

    Gestisce la configurazione del motore di rendering, il posizionamento
    delle camere e il salvataggio delle immagini per ogni stato della scena.

    Attributes:
        output_dir: Directory di base per i file renderizzati.
        config: Configurazione completa del rendering.
    """

    def __init__(
        self,
        output_dir: Path,
        config: RenderConfig,
    ) -> None:
        """
        Inizializza il renderer.

        Args:
            output_dir: Directory dove salvare le immagini renderizzate.
            config: Configurazione del rendering caricata dal TOML.
        """
        self.output_dir = output_dir
        self.config = config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "BlenderRenderer inizializzato. Output dir: %s.", self.output_dir
        )

    def _configure_render_engine(
        self,
        width: int,
        height: int,
        samples: int,
        engine: str = "CYCLES",
    ) -> None:
        """
        Applica la configurazione di rendering alla scena Blender corrente.

        Args:
            width: Larghezza del render in pixel.
            height: Altezza del render in pixel.
            samples: Numero di campioni per il rendering Cycles.
            engine: Motore di rendering Blender ('CYCLES' o 'BLENDER_EEVEE').

        Raises:
            ImportError: Se bpy non e' disponibile.
        """
        try:
            import bpy  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Il modulo 'bpy' richiede l'ambiente Blender."
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
                    logger.debug("GPU non disponibile. Uso CPU per Cycles.")

        logger.debug(
            "Render configurato: %dx%d, engine=%s, samples=%d.",
            width,
            height,
            engine,
            samples,
        )

    def _do_render(self, output_path: Path) -> Path:
        """
        Esegue il rendering e salva l'immagine in formato PNG.

        Args:
            output_path: Percorso del file di output senza estensione.

        Returns:
            Percorso effettivo del file salvato (con estensione .png).

        Raises:
            ImportError: Se bpy non e' disponibile.
            RuntimeError: Se il render non produce un file di output valido.
        """
        try:
            import bpy  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Il modulo 'bpy' richiede l'ambiente Blender."
            ) from exc

        scene = bpy.context.scene
        render = scene.render

        render.filepath = str(output_path)
        render.image_settings.file_format = "PNG"
        render.image_settings.color_mode = "RGBA"
        render.image_settings.color_depth = "8"

        bpy.ops.render.render(write_still=True)

        saved_path = output_path.with_suffix(".png")
        if not saved_path.exists():
            saved_path = Path(str(output_path) + ".png")

        if not saved_path.exists():
            raise RuntimeError(
                f"Il render non ha prodotto un file di output valido. "
                f"Percorso atteso: {saved_path}"
            )

        logger.info("Render salvato: %s.", saved_path)
        return saved_path

    def render_step(
        self,
        step_name: str,
        state: SceneState,
        quality: RenderQuality = "preview",
    ) -> dict[str, Path]:
        """
        Esegue entrambe le viste (top-down e isometrica) per uno stato della scena.

        Args:
            step_name: Identificativo della configurazione (es. 'original', 'randomized').
            state: Stato corrente della scena.
            quality: Qualita' del render ('preview' o 'final').

        Returns:
            Dizionario con chiavi 'top' e 'iso' e i percorsi delle immagini generate.
        """
        if quality == "final":
            width = self.config.final_width
            height = self.config.final_height
            samples = self.config.final_samples
        else:
            width = self.config.preview_width
            height = self.config.preview_height
            samples = self.config.preview_samples

        self._configure_render_engine(width, height, samples)

        room_bounds = state.room_bounds
        if room_bounds is None:
            logger.warning(
                "room_bounds non definiti per la scena '%s'. "
                "Uso bounds di default.",
                state.scene_name,
            )
            room_bounds = RoomBounds(
                x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0,
                z_floor=0.0, z_ceiling=3.0,
            )

        render_paths: dict[str, Path] = {}

        setup_topdown_camera(
            scene_x_min=room_bounds.x_min,
            scene_x_max=room_bounds.x_max,
            scene_y_min=room_bounds.y_min,
            scene_y_max=room_bounds.y_max,
            scene_z_ceiling=room_bounds.z_ceiling,
            config=self.config,
        )
        top_path = self.output_dir / f"render_{step_name}_top"
        render_paths["top"] = self._do_render(top_path)

        setup_isometric_camera(
            scene_x_min=room_bounds.x_min,
            scene_x_max=room_bounds.x_max,
            scene_y_min=room_bounds.y_min,
            scene_y_max=room_bounds.y_max,
            scene_z_min=room_bounds.z_floor,
            scene_z_ceiling=room_bounds.z_ceiling,
            config=self.config,
        )
        iso_path = self.output_dir / f"render_{step_name}_iso"
        render_paths["iso"] = self._do_render(iso_path)

        logger.info(
            "Render per '%s' completato: top=%s, iso=%s.",
            step_name,
            render_paths["top"],
            render_paths["iso"],
        )

        return render_paths