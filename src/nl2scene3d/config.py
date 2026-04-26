# src/nl2scene3d/config.py
"""
Gestione della configurazione centralizzata del progetto.

Carica le variabili d'ambiente dal file .env e le espone come attributi
tipizzati di un oggetto di configurazione. La configurazione viene
costruita con priorita': variabile d'ambiente > file TOML > valore default.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as _exc:
        raise ModuleNotFoundError(
            "Il pacchetto 'tomli' e' richiesto per Python < 3.11. "
            "Installarlo con: pip install tomli"
        ) from _exc

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Carica .env dalla root del progetto se presente.
load_dotenv()

# Istanza singleton della configurazione; None finche' non viene caricata.
_config_instance: Optional["AppConfig"] = None
_config_lock: threading.Lock = threading.Lock()


def _get_project_root() -> Path:
    """Restituisce la directory radice del progetto."""
    return Path(__file__).parent.parent.parent


def _load_toml_config() -> dict[str, Any]:
    """
    Carica il file settings.toml dalla directory config/.

    Returns:
        Dizionario con i dati di configurazione, vuoto se il file non esiste.

    Raises:
        tomllib.TOMLDecodeError: Se il file TOML non e' sintatticamente valido.
    """
    config_path = _get_project_root() / "config" / "settings.toml"
    if not config_path.exists():
        logger.debug("File settings.toml non trovato in '%s'. Uso valori default.", config_path)
        return {}
    with open(config_path, "rb") as fh:
        return tomllib.load(fh)


# ---------------------------------------------------------------------------
# Dataclass di configurazione
# ---------------------------------------------------------------------------


@dataclass
class GeminiConfig:
    """Parametri di configurazione per il client Gemini."""

    api_key: str = field(repr=False)
    model_primary: str
    model_fallback: str
    max_retries: int
    timeout_seconds: int
    temperature: float = 0.2
    max_output_tokens: int = 4096

    def __post_init__(self) -> None:
        """Valida i parametri di configurazione."""
        if not self.api_key:
            raise ValueError("api_key non puo' essere vuota.")
        if self.max_retries < 0:
            raise ValueError("max_retries deve essere >= 0.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds deve essere > 0.")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature deve essere compresa tra 0.0 e 2.0.")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens deve essere > 0.")

    @classmethod
    def from_config(cls, toml_data: dict[str, Any]) -> "GeminiConfig":
        """
        Costruisce la configurazione con priorita' Env > TOML > Default.

        Args:
            toml_data: Dizionario caricato dal file TOML.

        Returns:
            Istanza di GeminiConfig.

        Raises:
            EnvironmentError: Se GEMINI_API_KEY non e' impostata.
        """
        gemini_toml = toml_data.get("gemini", {})

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "La variabile d'ambiente GEMINI_API_KEY non e' impostata. "
                "Copiare .env.example in .env e inserire la chiave API."
            )

        return cls(
            api_key=api_key,
            model_primary=os.environ.get(
                "GEMINI_MODEL_PRIMARY",
                gemini_toml.get("model_primary", "gemini-3-flash-preview"),
            ),
            model_fallback=os.environ.get(
                "GEMINI_MODEL_FALLBACK",
                gemini_toml.get("model_fallback", "gemini-2.5-flash"),
            ),
            max_retries=int(
                os.environ.get(
                    "GEMINI_MAX_RETRIES",
                    str(gemini_toml.get("max_retries", 3)),
                )
            ),
            timeout_seconds=int(
                os.environ.get(
                    "GEMINI_TIMEOUT_SECONDS",
                    str(gemini_toml.get("timeout_seconds", 120)),
                )
            ),
            temperature=float(
                os.environ.get(
                    "GEMINI_TEMPERATURE",
                    str(gemini_toml.get("temperature", 0.2)),
                )
            ),
            max_output_tokens=int(
                os.environ.get(
                    "GEMINI_MAX_OUTPUT_TOKENS",
                    str(gemini_toml.get("max_output_tokens", 4096)),
                )
            ),
        )


@dataclass
class RenderConfig:
    """Parametri di configurazione per il sistema di rendering."""

    preview_width: int
    preview_height: int
    preview_samples: int
    final_width: int
    final_height: int
    final_samples: int
    isometric_elevation: float = 45.0
    isometric_azimuth: float = 45.0
    isometric_focal_length: float = 35.0
    topdown_height_multiplier: float = 3.0
    topdown_ortho_scale_padding: float = 1.10
    pipeline_camera_prefix: str = "NL2Scene3D_Camera"

    @classmethod
    def from_config(cls, toml_data: dict[str, Any]) -> "RenderConfig":
        """
        Costruisce la configurazione con priorita' Env > TOML > Default.

        Args:
            toml_data: Dizionario caricato dal file TOML.

        Returns:
            Istanza di RenderConfig.
        """
        render_toml = toml_data.get("render", {})
        preview_toml = render_toml.get("preview", {})
        final_toml = render_toml.get("final", {})
        camera_toml = render_toml.get("camera", {})

        return cls(
            preview_width=int(
                os.environ.get("RENDER_PREVIEW_WIDTH", str(preview_toml.get("width", 512)))
            ),
            preview_height=int(
                os.environ.get("RENDER_PREVIEW_HEIGHT", str(preview_toml.get("height", 512)))
            ),
            preview_samples=int(
                os.environ.get("RENDER_PREVIEW_SAMPLES", str(preview_toml.get("samples", 64)))
            ),
            final_width=int(
                os.environ.get("RENDER_FINAL_WIDTH", str(final_toml.get("width", 1280)))
            ),
            final_height=int(
                os.environ.get("RENDER_FINAL_HEIGHT", str(final_toml.get("height", 720)))
            ),
            final_samples=int(
                os.environ.get("RENDER_FINAL_SAMPLES", str(final_toml.get("samples", 256)))
            ),
            isometric_elevation=float(
                camera_toml.get("isometric_elevation_degrees", 45.0)
            ),
            isometric_azimuth=float(
                camera_toml.get("isometric_azimuth_degrees", 45.0)
            ),
            isometric_focal_length=float(
                camera_toml.get("isometric_focal_length_mm", 35.0)
            ),
            topdown_height_multiplier=float(
                camera_toml.get("topdown_height_multiplier", 3.0)
            ),
            topdown_ortho_scale_padding=float(
                camera_toml.get("topdown_ortho_scale_padding", 1.10)
            ),
            pipeline_camera_prefix=camera_toml.get(
                "pipeline_camera_prefix", "NL2Scene3D_Camera"
            ),
        )


@dataclass
class RandomizerConfig:
    """
    Parametri di configurazione per la randomizzazione della scena.

    Attributes:
        seed: Seed per il generatore di numeri casuali (0 = non deterministico).
        jitter_ratio: Frazione della dimensione della stanza usata come jitter massimo.
        rotate_z_only: Se True, ruota solo l'asse Z (yaw).
        check_overlaps: Se True, verifica le sovrapposizioni AABB e ritenta.
        wall_margin: Margine minimo dai muri in metri.
        max_overlap_ratio: Rapporto massimo di sovrapposizione consentito.
        max_placement_attempts: Numero massimo di tentativi di posizionamento per oggetto.
    """

    seed: int = 0
    jitter_ratio: float = 0.8
    rotate_z_only: bool = True
    check_overlaps: bool = True
    wall_margin: float = 0.1
    max_overlap_ratio: float = 0.5
    max_placement_attempts: int = 10


@dataclass
class PipelineConfig:
    """Parametri generali della pipeline."""

    scenes_dir: Path
    outputs_dir: Path
    max_movable_objects: int
    randomizer_seed: int
    min_object_dimension: float = 0.05
    wall_margin: float = 0.10
    max_overlap_ratio: float = 0.50
    max_placement_attempts: int = 10
    min_quality_score: int = 7
    max_corrections: int = 5
    non_mesh_types: frozenset[str] = field(default_factory=frozenset)
    structural_patterns: list[str] = field(default_factory=list)
    ceiling_light_patterns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Valida i parametri di configurazione e normalizza i tipi."""
        if self.max_movable_objects <= 0:
            raise ValueError("max_movable_objects deve essere > 0.")
        if self.min_object_dimension < 0:
            raise ValueError("min_object_dimension deve essere >= 0.")
        if not isinstance(self.non_mesh_types, frozenset):
            self.non_mesh_types = frozenset(self.non_mesh_types)

    @classmethod
    def from_config(cls, toml_data: dict[str, Any]) -> "PipelineConfig":
        """
        Costruisce la configurazione con priorita' Env > TOML > Default.

        Args:
            toml_data: Dizionario caricato dal file TOML.

        Returns:
            Istanza di PipelineConfig.
        """
        pipeline_toml = toml_data.get("pipeline", {})
        paths_toml = toml_data.get("paths", {})
        randomizer_toml = toml_data.get("randomizer", {})
        class_toml = toml_data.get("object_classification", {})

        return cls(
            scenes_dir=Path(
                os.environ.get(
                    "SCENES_DIR", paths_toml.get("scenes_dir", "scenes/originals")
                )
            ),
            outputs_dir=Path(
                os.environ.get(
                    "OUTPUTS_DIR", paths_toml.get("outputs_dir", "scenes/outputs")
                )
            ),
            max_movable_objects=int(
                os.environ.get(
                    "MAX_MOVABLE_OBJECTS",
                    str(pipeline_toml.get("max_movable_objects", 20)),
                )
            ),
            randomizer_seed=int(
                os.environ.get(
                    "RANDOMIZER_SEED",
                    str(randomizer_toml.get("seed", 0)),
                )
            ),
            min_object_dimension=float(
                pipeline_toml.get("min_object_dimension_meters", 0.05)
            ),
            wall_margin=float(pipeline_toml.get("wall_margin_meters", 0.10)),
            max_overlap_ratio=float(pipeline_toml.get("max_overlap_ratio", 0.50)),
            max_placement_attempts=int(
                pipeline_toml.get("max_placement_attempts", 10)
            ),
            min_quality_score=int(
                pipeline_toml.get("min_quality_score_for_corrections", 7)
            ),
            max_corrections=int(
                pipeline_toml.get("max_corrections_to_apply", 5)
            ),
            non_mesh_types=frozenset(
                class_toml.get(
                    "non_mesh_types",
                    ["CAMERA", "LIGHT", "SPEAKER", "ARMATURE", "EMPTY", "CURVE"],
                )
            ),
            structural_patterns=class_toml.get(
                "structural_name_patterns",
                ["wall", "floor", "ceiling", "room", "door", "window"],
            ),
            ceiling_light_patterns=class_toml.get(
                "ceiling_light_patterns",
                ["ceiling", "pendant", "chandelier"],
            ),
        )


@dataclass
class LoggingConfig:
    """Configurazione del sistema di logging."""

    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    write_to_file: bool = False
    log_file: Optional[Path] = None

    @classmethod
    def from_config(cls, toml_data: dict[str, Any]) -> "LoggingConfig":
        """
        Costruisce la configurazione dal TOML.

        Args:
            toml_data: Dizionario caricato dal file TOML.

        Returns:
            Istanza di LoggingConfig.
        """
        log_toml = toml_data.get("logging", {})
        log_file_raw = log_toml.get("log_file")
        return cls(
            level=log_toml.get("level", "INFO"),
            format=log_toml.get(
                "format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            ),
            datefmt=log_toml.get("datefmt", "%Y-%m-%d %H:%M:%S"),
            write_to_file=log_toml.get("write_to_file", False),
            log_file=Path(log_file_raw) if log_file_raw else None,
        )


@dataclass
class AppConfig:
    """
    Configurazione completa dell'applicazione.

    Aggrega tutte le sotto-configurazioni in un unico punto di accesso.
    """

    gemini: GeminiConfig
    render: RenderConfig
    pipeline: PipelineConfig
    logging: LoggingConfig

    @classmethod
    def load(cls) -> "AppConfig":
        """
        Carica l'intera configurazione dalle variabili d'ambiente e dal file TOML.

        Returns:
            Istanza di AppConfig completamente inizializzata.
        """
        toml_data = _load_toml_config()
        return cls(
            gemini=GeminiConfig.from_config(toml_data),
            render=RenderConfig.from_config(toml_data),
            pipeline=PipelineConfig.from_config(toml_data),
            logging=LoggingConfig.from_config(toml_data),
        )


def get_config() -> AppConfig:
    """
    Restituisce l'istanza singleton della configurazione.

    Alla prima chiamata carica la configurazione; le chiamate successive
    restituiscono la stessa istanza senza rileggere il file.
    L'inizializzazione e' protetta da lock per garantire thread-safety.

    Returns:
        Istanza di AppConfig.
    """
    global _config_instance  # noqa: PLW0603
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = AppConfig.load()
    return _config_instance


def reset_config() -> None:
    """
    Azzera il singleton della configurazione.

    Utile esclusivamente nei test unitari per isolare i test tra loro.
    Non deve essere chiamata nel codice di produzione.
    """
    global _config_instance  # noqa: PLW0603
    _config_instance = None