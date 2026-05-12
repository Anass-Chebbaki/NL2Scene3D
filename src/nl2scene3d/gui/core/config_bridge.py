# gui/core/config_bridge.py
"""
Reads project configuration (settings.toml + .env) without importing
the full nl2scene3d package, so the GUI runs in the system Python
(not Blender's embedded Python).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Locate project root
# config_bridge.py is in src/nl2scene3d/gui/core/
# Path(__file__) -> config_bridge.py
# .parent -> core/
# .parent.parent -> gui/
# .parent.parent.parent -> nl2scene3d/
# .parent.parent.parent.parent -> src/
# .parent.parent.parent.parent.parent -> ROOT
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if sys.version_info >= (3, 11):
        import tomllib
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    else:
        try:
            import tomli
            with open(path, "rb") as fh:
                return tomli.load(fh)
        except ModuleNotFoundError:
            return {}


@dataclass
class GUIConfig:
    """Flat configuration object consumed by the GUI."""
    # Paths
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)
    scenes_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "scenes" / "originals")
    outputs_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "scenes" / "outputs")
    prompts_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "config" / "prompts")

    # Gemini
    api_key: str = ""
    model_primary: str = "gemini-2.5-flash"
    model_fallback: str = "gemini-2.5-flash"
    max_retries: int = 3
    timeout_seconds: int = 120
    temperature: float = 0.2
    max_output_tokens: int = 32768

    # Pipeline
    max_movable_objects: int = 20
    randomizer_seed: int = 0
    wall_margin: float = 0.10
    max_overlap_ratio: float = 0.05
    max_placement_attempts: int = 500
    min_quality_score: int = 7
    good_quality_score: int = 8
    max_corrections: int = 5

    # Render preview
    preview_width: int = 512
    preview_height: int = 512
    preview_samples: int = 64

    # Render final
    final_width: int = 1280
    final_height: int = 1280
    final_samples: int = 256

    # Logging
    log_level: str = "INFO"

    # Blender executable
    blender_executable: str = "blender"


def load_gui_config() -> GUIConfig:
    """Load configuration from .env and settings.toml."""
    load_dotenv(_PROJECT_ROOT / ".env")

    toml_data = _load_toml(_PROJECT_ROOT / "config" / "settings.toml")

    gemini = toml_data.get("gemini", {})
    pipeline = toml_data.get("pipeline", {})
    paths = toml_data.get("paths", {})
    randomizer = toml_data.get("randomizer", {})
    render = toml_data.get("render", {})
    preview = render.get("preview", {})
    final = render.get("final", {})
    logging_cfg = toml_data.get("logging", {})

    cfg = GUIConfig()

    # API key from env
    cfg.api_key = os.environ.get("GEMINI_API_KEY", "")

    # Paths
    scenes_raw = os.environ.get("SCENES_DIR") or paths.get("scenes_dir", "scenes/originals")
    outputs_raw = os.environ.get("OUTPUTS_DIR") or paths.get("outputs_dir", "scenes/outputs")
    cfg.scenes_dir = Path(scenes_raw) if Path(scenes_raw).is_absolute() else _PROJECT_ROOT / scenes_raw
    cfg.outputs_dir = Path(outputs_raw) if Path(outputs_raw).is_absolute() else _PROJECT_ROOT / outputs_raw
    cfg.prompts_dir = _PROJECT_ROOT / paths.get("prompts_dir", "config/prompts")

    # Gemini
    cfg.model_primary = os.environ.get("GEMINI_MODEL_PRIMARY") or gemini.get("model_primary", cfg.model_primary)
    cfg.model_fallback = os.environ.get("GEMINI_MODEL_FALLBACK") or gemini.get("model_fallback", cfg.model_fallback)
    cfg.max_retries = int(os.environ.get("GEMINI_MAX_RETRIES", str(gemini.get("max_retries", cfg.max_retries))))
    cfg.timeout_seconds = int(os.environ.get("GEMINI_TIMEOUT_SECONDS", str(gemini.get("timeout_seconds", cfg.timeout_seconds))))
    cfg.temperature = float(os.environ.get("GEMINI_TEMPERATURE", str(gemini.get("temperature", cfg.temperature))))
    cfg.max_output_tokens = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", str(gemini.get("max_output_tokens", cfg.max_output_tokens))))

    # Pipeline
    cfg.max_movable_objects = int(os.environ.get("MAX_MOVABLE_OBJECTS", str(pipeline.get("max_movable_objects", cfg.max_movable_objects))))
    cfg.randomizer_seed = int(os.environ.get("RANDOMIZER_SEED", str(randomizer.get("seed", cfg.randomizer_seed))))
    cfg.wall_margin = float(pipeline.get("wall_margin_meters", cfg.wall_margin))
    cfg.max_overlap_ratio = float(pipeline.get("max_overlap_ratio", cfg.max_overlap_ratio))
    cfg.max_placement_attempts = int(pipeline.get("max_placement_attempts", cfg.max_placement_attempts))
    cfg.min_quality_score = int(pipeline.get("min_quality_score_for_corrections", cfg.min_quality_score))
    cfg.good_quality_score = int(pipeline.get("good_quality_score_for_protection", cfg.good_quality_score))
    cfg.max_corrections = int(pipeline.get("max_corrections_to_apply", cfg.max_corrections))

    # Render
    cfg.preview_width = int(preview.get("width", cfg.preview_width))
    cfg.preview_height = int(preview.get("height", cfg.preview_height))
    cfg.preview_samples = int(preview.get("samples", cfg.preview_samples))
    cfg.final_width = int(final.get("width", cfg.final_width))
    cfg.final_height = int(final.get("height", cfg.final_height))
    cfg.final_samples = int(final.get("samples", cfg.final_samples))

    # Logging
    cfg.log_level = logging_cfg.get("level", cfg.log_level)

    # Blender executable from env
    cfg.blender_executable = os.environ.get("BLENDER_EXECUTABLE", "blender")

    return cfg