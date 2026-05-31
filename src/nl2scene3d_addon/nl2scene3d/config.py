# nl2scene3d/config.py
"""
Centralized configuration for the NL2Scene3D pipeline.

Loading priority: environment variable > settings.toml > built-in default.

Inside the Blender addon, the API key is injected directly into os.environ
before get_config() is called, so no singleton or complex reset mechanism
is needed — every call to get_config() is idempotent with respect to the
current environment.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib          # type: ignore[no-redef]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install 'tomli' for Python < 3.11: pip install tomli"
        ) from exc

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_config_dir() -> Path:
    """
    Locates the config/ directory, checking the package folder first and
    falling back to the project root. Works for both bundled addon and
    standalone development usage.
    """
    pkg_config  = Path(__file__).parent / "config"
    if pkg_config.exists():
        return pkg_config

    root_config = Path(__file__).parent.parent.parent / "config"
    if root_config.exists():
        return root_config

    return pkg_config   # return the path even if it does not exist yet


def _load_toml(config_dir: Path) -> dict[str, Any]:
    settings_path = config_dir / "settings.toml"
    if not settings_path.exists():
        logger.debug("settings.toml not found at '%s'. Using built-in defaults.", settings_path)
        return {}
    with open(settings_path, "rb") as fh:
        return tomllib.load(fh)


def _env(key: str, toml_value: Any, default: Any) -> Any:
    """Returns the environment variable if set, then the TOML value, then the default."""
    return os.environ.get(key, toml_value if toml_value is not None else default)


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------

@dataclass
class GeminiConfig:
    """Connection and generation settings for the Gemini API."""

    api_key:          str   = field(repr=False)
    model_primary:    str   = "gemini-3.5-flash"
    model_fallback:   str   = "gemini-2.5-flash"
    max_retries:      int   = 3
    timeout_seconds:  int   = 300
    temperature:      float = 0.2
    max_output_tokens: int  = 32000

    def __post_init__(self) -> None:
        if not self.api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Set the environment variable or configure it in the addon preferences."
            )
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be in the range [0.0, 2.0].")

    @classmethod
    def from_toml(cls, toml: dict[str, Any]) -> "GeminiConfig":
        g       = toml.get("gemini", {})
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Set the environment variable or configure it in the addon preferences."
            )
        return cls(
            api_key=api_key,
            model_primary=_env(
                "GEMINI_MODEL_PRIMARY", g.get("model_primary"), "gemini-3.5-flash"
            ),
            model_fallback=_env(
                "GEMINI_MODEL_FALLBACK", g.get("model_fallback"), "gemini-2.5-flash"
            ),
            max_retries=int(_env("GEMINI_MAX_RETRIES",       g.get("max_retries"),       3)),
            timeout_seconds=int(_env("GEMINI_TIMEOUT_SECONDS", g.get("timeout_seconds"), 300)),
            temperature=float(_env("GEMINI_TEMPERATURE",      g.get("temperature"),      0.2)),
            max_output_tokens=int(
                _env("GEMINI_MAX_OUTPUT_TOKENS", g.get("max_output_tokens"), 32000)
            ),
        )


@dataclass
class RandomizerConfig:
    """Parameters controlling the scene layout randomization."""

    seed:                   int   = 0
    jitter_ratio:           float = 0.8     # Maximum displacement as a fraction of the room size
    wall_margin:            float = 0.20    # Minimum distance from walls, in meters
    collision_margin:       float = 0.07    # Extra clearance between objects, in meters
    max_placement_attempts: int   = 200     # Attempts to find a collision-free position

    @classmethod
    def from_toml(cls, toml: dict[str, Any]) -> "RandomizerConfig":
        r = toml.get("randomizer", {})
        p = toml.get("pipeline",   {})
        return cls(
            seed=int(_env("RANDOMIZER_SEED", r.get("seed"), 0)),
            jitter_ratio=float(r.get("jitter_ratio", 0.8)),
            wall_margin=float(p.get("wall_margin_meters", 0.20)),
            collision_margin=float(p.get("collision_margin_meters", 0.05)),
            max_placement_attempts=int(p.get("max_placement_attempts", 200)),
        )


@dataclass
class PipelineConfig:
    """General pipeline parameters and object-classification rules."""

    scenes_dir:     Path = Path("scenes/originals")
    outputs_dir:    Path = Path("scenes/outputs")

    max_movable_objects:    int   = 20
    min_object_dimension:   float = 0.05    # Objects smaller than this are treated as non-movable

    min_quality_score:          int   = 7   # Minimum score for applying visual-critic corrections
    good_quality_score:         int   = 8   # Score above which the layout is considered good (no corrections)
    max_corrections:            int   = 5   # Maximum corrections per visual-critic iteration
    max_correction_displacement: float = 1.5  # Maximum per-correction displacement, in meters

    # Object-classification patterns
    non_mesh_types: frozenset[str] = field(
        default_factory=lambda: frozenset(
            ["CAMERA", "LIGHT", "SPEAKER", "ARMATURE", "EMPTY", "CURVE"]
        )
    )
    structural_patterns: list[str] = field(
        default_factory=lambda: [
            "wall", "floor", "ceiling", "room", "door", "window",
            "muro", "parete", "pavimento", "soffitto", "porta", "finestra",
        ]
    )
    ceiling_light_patterns: list[str] = field(
        default_factory=lambda: ["ceiling", "pendant", "chandelier"]
    )

    # Rules for freezing objects mounted high on walls or attached to the ceiling
    static_height_threshold: float = 1.0   # Root objects whose base Z >= this become static
    freeze_ceiling_objects:  bool  = True   # Also freeze objects attached to the ceiling

    def __post_init__(self) -> None:
        if not isinstance(self.non_mesh_types, frozenset):
            self.non_mesh_types = frozenset(self.non_mesh_types)

    @classmethod
    def from_toml(cls, toml: dict[str, Any]) -> "PipelineConfig":
        p    = toml.get("pipeline", {})
        paths = toml.get("paths",   {})
        cls_t = toml.get("object_classification", {})
        return cls(
            scenes_dir=Path(_env("SCENES_DIR",  paths.get("scenes_dir"),  "scenes/originals")),
            outputs_dir=Path(_env("OUTPUTS_DIR", paths.get("outputs_dir"), "scenes/outputs")),
            max_movable_objects=int(
                _env("MAX_MOVABLE_OBJECTS", p.get("max_movable_objects"), 20)
            ),
            min_object_dimension=float(p.get("min_object_dimension_meters", 0.05)),
            min_quality_score=int(p.get("min_quality_score_for_corrections", 7)),
            good_quality_score=int(p.get("good_quality_score_for_protection", 8)),
            max_corrections=int(p.get("max_corrections_to_apply", 5)),
            max_correction_displacement=float(
                p.get("max_correction_displacement_meters", 1.5)
            ),
            non_mesh_types=frozenset(
                cls_t.get(
                    "non_mesh_types",
                    ["CAMERA", "LIGHT", "SPEAKER", "ARMATURE", "EMPTY", "CURVE"],
                )
            ),
            structural_patterns=cls_t.get(
                "structural_name_patterns",
                ["wall", "floor", "ceiling", "room", "door", "window"],
            ),
            ceiling_light_patterns=cls_t.get(
                "ceiling_light_patterns", ["ceiling", "pendant", "chandelier"]
            ),
            static_height_threshold=float(p.get("static_height_threshold_meters", 1.0)),
            freeze_ceiling_objects=bool(p.get("freeze_ceiling_objects", True)),
        )


@dataclass
class RenderConfig:
    """Blender rendering parameters used for preview and final images."""

    preview_width:   int   = 512
    preview_height:  int   = 512
    preview_samples: int   = 64
    final_width:     int   = 1280
    final_height:    int   = 1280
    final_samples:   int   = 256

    isometric_elevation:          float = 30.0
    isometric_azimuth:            float = 275.0
    isometric_focal_length:       float = 50.0
    isometric_distance_multiplier: float = 2.2
    topdown_height_multiplier:    float = 2.0
    topdown_ortho_scale_padding:  float = 1.05

    pipeline_camera_prefix: str = "NL2Scene3D_Camera"

    @classmethod
    def from_toml(cls, toml: dict[str, Any]) -> "RenderConfig":
        r     = toml.get("render", {})
        prev  = r.get("preview", {})
        final = r.get("final",   {})
        cam   = r.get("camera",  {})
        return cls(
            preview_width=int(_env("RENDER_PREVIEW_WIDTH",   prev.get("width"),   512)),
            preview_height=int(_env("RENDER_PREVIEW_HEIGHT", prev.get("height"),  512)),
            preview_samples=int(_env("RENDER_PREVIEW_SAMPLES", prev.get("samples"), 64)),
            final_width=int(_env("RENDER_FINAL_WIDTH",   final.get("width"),   1280)),
            final_height=int(_env("RENDER_FINAL_HEIGHT", final.get("height"),  1280)),
            final_samples=int(_env("RENDER_FINAL_SAMPLES", final.get("samples"), 256)),
            isometric_elevation=float(
                cam.get("isometric_elevation_degrees", 30.0)
            ),
            isometric_azimuth=float(
                cam.get("isometric_azimuth_degrees", 275.0)
            ),
            isometric_focal_length=float(
                cam.get("isometric_focal_length_mm", 50.0)
            ),
            isometric_distance_multiplier=float(
                cam.get("isometric_distance_multiplier", 2.2)
            ),
            topdown_height_multiplier=float(
                cam.get("topdown_height_multiplier", 2.0)
            ),
            topdown_ortho_scale_padding=float(
                cam.get("topdown_ortho_scale_padding", 1.05)
            ),
            pipeline_camera_prefix=cam.get(
                "pipeline_camera_prefix", "NL2Scene3D_Camera"
            ),
        )


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """
    Complete application configuration.

    In the Blender addon, this is rebuilt on every operator call after the
    addon has injected the API key into os.environ:

        os.environ["GEMINI_API_KEY"] = prefs.api_key
        config = AppConfig.load()
        config.gemini.model_primary = prefs.model_name
    """

    gemini:     GeminiConfig
    randomizer: RandomizerConfig
    pipeline:   PipelineConfig
    render:     RenderConfig

    @classmethod
    def load(cls) -> "AppConfig":
        """Loads configuration from the environment and settings.toml."""
        config_dir = _find_config_dir()
        toml       = _load_toml(config_dir)
        return cls(
            gemini=GeminiConfig.from_toml(toml),
            randomizer=RandomizerConfig.from_toml(toml),
            pipeline=PipelineConfig.from_toml(toml),
            render=RenderConfig.from_toml(toml),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_config() -> AppConfig:
    """
    Loads and returns the current configuration.

    Does not use a singleton: each call re-reads os.environ so that addon
    overrides (API key, model) are always reflected. TOML loading is fast
    (small file, ~1 KB).
    """
    return AppConfig.load()