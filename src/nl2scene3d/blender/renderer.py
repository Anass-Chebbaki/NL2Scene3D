"""
Sistema di rendering automatico per la pipeline NL2Scene3D.

Gestisce il rendering delle viste top-down e isometrica per ogni configurazione
della scena, con impostazioni separate per i render di preview
(bassa qualita') e il render finale (alta qualita').

Deve essere eseguito all'interno dell'ambiente Python di Blender.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

from nl2scene3d.blender.camera_setup import setup_isometric_camera, setup_topdown_camera
from nl2scene3d.models import RoomBounds, SceneState

logger = logging.getLogger(__name__)

# Tipo delle viste di rendering disponibili
RenderView = Literal["top", "iso"]

# Modalita' di qualita' del render
RenderQuality = Literal["preview", "final"]

class RenderConfig:
    """
    Configurazione del rendering per una modalita' specifica.

    Attributes:
        width: Larghezza in pixel dell'immagine renderizzata.
        height: Altezza in pixel dell'immagine renderizzata.
        samples: Numero di campioni Cycles per pixel.
        engine: Engine di rendering Blender da usare.
    """

    def __init__(
        self,
        width: int,
        height: int,
        samples: int,
        engine: str = "CYCLES",
    ) -> None:
        self.width = width
        self.height = height
        self.samples = samples
        self.engine = engine


# Configurazioni predefinite
PREVIEW_CONFIG = RenderConfig(width=512, height=512, samples=64)
FINAL_CONFIG = RenderConfig(width=1280, height=720, samples=256)

class BlenderRenderer:
    """
    Esegue il rendering della scena corrente in Blender.

    Gestisce la configurazione del motore di rendering, il posizionamento
    delle camere e il salvataggio delle immagini per ogni stato della scena.

    Attributes:
        output_dir: Directory di base per i file renderizzati.
        preview_config: Configurazione per i render di preview.
        final_config: Configurazione per il render finale.
    """

    def __init__(
        self,
        output_dir: Path,
        preview_config: Optional[RenderConfig] = None,
        final_config: Optional[RenderConfig] = None,
    ) -> None:
        """
        Inizializza il renderer.

        Args:
            output_dir: Directory dove salvare le immagini renderizzate.
            preview_config: Configurazione per i render di preview.
                            Se None, usa PREVIEW_CONFIG.
            final_config: Configurazione per il render finale.
                          Se None, usa FINAL_CONFIG.
        """
        self.output_dir = output_dir
        self.preview_config = preview_config or PREVIEW_CONFIG
        self.final_config = final_config or FINAL_CONFIG
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "BlenderRenderer inizializzato. Output dir: %s", self.output_dir
        )

    def _configure_render_settings(
        self,
        config: RenderConfig,
    ) -> None:
        """
        Applica la configurazione di rendering alla scena Blender corrente.

        Args:
            config: Configurazione da applicare.

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

        render.engine = config.engine
        render.resolution_x = config.width
        render.resolution_y = config.height
        render.resolution_percentage = 100

        # Configura Cycles se selezionato
        if config.engine == "CYCLES":
            scene.cycles.samples = config.samples
            scene.cycles.use_denoising = True
            # Usa la GPU se disponibile per accelerare il rendering
            preferences = bpy.context.preferences
            cycles_prefs = preferences.addons.get("cycles")
            if cycles_prefs:
                try:
                    bpy.context.scene.cycles.device = "GPU"
                except Exception:  # noqa: BLE001
                    bpy.context.scene.cycles.device = "CPU"
                    logger.debug("GPU non disponibile, uso CPU per Cycles.")

        logger.debug(
            "Render configurato: %dx%d, engine=%s, samples=%d",
            config.width,
            config.height,
            config.engine,
            config.samples,
        )

    def _do_render(self, output_path: Path) -> Path:
        """
        Esegue il rendering e salva l'immagine.

        Args:
            output_path: Percorso completo del file di output (senza estensione).

        Returns:
            Percorso effettivo del file salvato (con estensione .png).

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

        # Imposta il percorso di output e il formato
        render.filepath = str(output_path)
        render.image_settings.file_format = "PNG"
        render.image_settings.color_mode = "RGBA"
        render.image_settings.color_depth = "8"

        # Esegue il rendering e salva il file
        bpy.ops.render.render(write_still=True)

        saved_path = output_path.with_suffix(".png")
        if not saved_path.exists():
            # Blender aggiunge automaticamente .png se non presente
            saved_path = Path(str(output_path) + ".png")

        logger.info("Render salvato: %s", saved_path)
        return saved_path

    def render_step(
        self,
        step_name: str,
        state: SceneState,
        quality: RenderQuality = "preview",
    ) -> dict[str, Path]:
        """
        Esegue entrambe le viste (top-down e isometrica) per una configurazione della scena.

        Args:
            step_name: Identificativo della configurazione (es. 'original', 'randomized',
                       'reordered', 'refined').
            state: Stato corrente della scena (usato per configurare le camere).
            quality: Qualita' del render ('preview' o 'final').

        Returns:
            Dizionario con chiavi 'top' e 'iso' e i percorsi delle immagini.
        """
        config = self.final_config if quality == "final" else self.preview_config
        self._configure_render_settings(config)

        room_bounds = state.room_bounds
        if room_bounds is None:
            # Usa bounds di default se non disponibili
            room_bounds = RoomBounds(
                x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0,
                z_floor=0.0, z_ceiling=3.0,
            )

        render_paths: dict[str, Path] = {}

        # Vista top-down
        setup_topdown_camera(
            scene_x_min=room_bounds.x_min,
            scene_x_max=room_bounds.x_max,
            scene_y_min=room_bounds.y_min,
            scene_y_max=room_bounds.y_max,
            scene_z_ceiling=room_bounds.z_ceiling,
        )
        top_path = self.output_dir / f"render_{step_name}_top"
        render_paths["top"] = self._do_render(top_path)

        # Vista isometrica
        setup_isometric_camera(
            scene_x_min=room_bounds.x_min,
            scene_x_max=room_bounds.x_max,
            scene_y_min=room_bounds.y_min,
            scene_y_max=room_bounds.y_max,
            scene_z_min=room_bounds.z_floor,
            scene_z_ceiling=room_bounds.z_ceiling,
        )
        iso_path = self.output_dir / f"render_{step_name}_iso"
        render_paths["iso"] = self._do_render(iso_path)

        logger.info(
            "Render per '%s' completato: top=%s, iso=%s",
            step_name,
            render_paths["top"],
            render_paths["iso"],
        )
        return render_paths
