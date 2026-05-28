# nl2scene3d/config.py
"""
Configurazione centralizzata della pipeline NL2Scene3D.

Priorità di caricamento: variabile d'ambiente > settings.toml > valore default.

Nell'addon Blender la API key viene iniettata direttamente in os.environ
prima di chiamare get_config(), quindi non serve un meccanismo di reset
o un singleton complesso — ogni chiamata a get_config() è idempotente
rispetto all'ambiente corrente.
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
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Installa 'tomli' per Python < 3.11: pip install tomli"
        ) from exc

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_config_dir() -> Path:
    """
    Cerca la directory 'config/' partendo dal package, poi dalla root progetto.
    Funziona sia nell'addon bundled che in sviluppo standalone.
    """
    pkg_config = Path(__file__).parent / "config"
    if pkg_config.exists():
        return pkg_config
    root_config = Path(__file__).parent.parent.parent / "config"
    if root_config.exists():
        return root_config
    return pkg_config  # fallback: restituisce il path anche se non esiste


def _load_toml(config_dir: Path) -> dict[str, Any]:
    settings_path = config_dir / "settings.toml"
    if not settings_path.exists():
        logger.debug("settings.toml non trovato in '%s'. Uso valori default.", settings_path)
        return {}
    with open(settings_path, "rb") as fh:
        return tomllib.load(fh)


def _env(key: str, toml_value: Any, default: Any) -> Any:
    """Legge da env se presente, altrimenti usa toml_value, altrimenti default."""
    return os.environ.get(key, toml_value if toml_value is not None else default)


# ---------------------------------------------------------------------------
# Sotto-configurazioni
# ---------------------------------------------------------------------------

@dataclass
class GeminiConfig:
    api_key: str = field(repr=False)
    model_primary: str = "gemini-3.5-flash"
    model_fallback: str = "gemini-2.5-flash"
    max_retries: int = 3
    timeout_seconds: int = 300
    temperature: float = 0.2
    max_output_tokens: int = 32000

    def __post_init__(self) -> None:
        if not self.api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY non impostata. "
                "Imposta la variabile d'ambiente o configurala nelle preferenze dell'addon."
            )
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature deve essere in [0.0, 2.0].")

    @classmethod
    def from_toml(cls, toml: dict[str, Any]) -> "GeminiConfig":
        g = toml.get("gemini", {})
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY non impostata. "
                "Imposta la variabile d'ambiente o configurala nelle preferenze dell'addon."
            )
        return cls(
            api_key=api_key,
            model_primary=_env("GEMINI_MODEL_PRIMARY", g.get("model_primary"), "gemini-3.5-flash"),
            model_fallback=_env("GEMINI_MODEL_FALLBACK", g.get("model_fallback"), "gemini-2.5-flash"),
            max_retries=int(_env("GEMINI_MAX_RETRIES", g.get("max_retries"), 3)),
            timeout_seconds=int(_env("GEMINI_TIMEOUT_SECONDS", g.get("timeout_seconds"), 300)),
            temperature=float(_env("GEMINI_TEMPERATURE", g.get("temperature"), 0.2)),
            max_output_tokens=int(_env("GEMINI_MAX_OUTPUT_TOKENS", g.get("max_output_tokens"), 32000)),
        )


@dataclass
class RandomizerConfig:
    """Parametri per la randomizzazione del layout."""
    seed: int = 0
    jitter_ratio: float = 0.8          # Quanto si sposta al massimo (frazione della stanza)
    wall_margin: float = 0.20          # Distanza minima dai muri in metri
    collision_margin: float = 0.07     # Margine extra tra oggetti per evitare l'intrecciamento
    max_placement_attempts: int = 200  # Tentativi per trovare una posizione libera

    @classmethod
    def from_toml(cls, toml: dict[str, Any]) -> "RandomizerConfig":
        r = toml.get("randomizer", {})
        p = toml.get("pipeline", {})
        return cls(
            seed=int(_env("RANDOMIZER_SEED", r.get("seed"), 0)),
            jitter_ratio=float(r.get("jitter_ratio", 0.8)),
            wall_margin=float(p.get("wall_margin_meters", 0.20)),
            collision_margin=float(p.get("collision_margin_meters", 0.05)),
            max_placement_attempts=int(p.get("max_placement_attempts", 200)),
        )


@dataclass
class PipelineConfig:
    """Parametri generali della pipeline."""
    scenes_dir: Path = Path("scenes/originals")
    outputs_dir: Path = Path("scenes/outputs")
    max_movable_objects: int = 20
    min_object_dimension: float = 0.05    # Oggetti più piccoli → non movibili
    min_quality_score: int = 7            # Score minimo per applicare correzioni del visual critic
    good_quality_score: int = 8           # Score sopra il quale non si correggge (layout già buono)
    max_corrections: int = 5              # Max correzioni per iterazione del visual critic
    max_correction_displacement: float = 1.5  # Max spostamento (m) per singola correzione

    # Pattern per la classificazione degli oggetti
    non_mesh_types: frozenset[str] = field(
        default_factory=lambda: frozenset(["CAMERA", "LIGHT", "SPEAKER", "ARMATURE", "EMPTY", "CURVE"])
    )
    structural_patterns: list[str] = field(
        default_factory=lambda: [
            "wall", "floor", "ceiling", "room", "door", "window",
            "muro", "parete", "pavimento", "soffitto", "porta", "finestra"
        ]
    )
    ceiling_light_patterns: list[str] = field(
        default_factory=lambda: ["ceiling", "pendant", "chandelier"]
    )

    def __post_init__(self) -> None:
        if not isinstance(self.non_mesh_types, frozenset):
            self.non_mesh_types = frozenset(self.non_mesh_types)

    @classmethod
    def from_toml(cls, toml: dict[str, Any]) -> "PipelineConfig":
        p = toml.get("pipeline", {})
        paths = toml.get("paths", {})
        cls_toml = toml.get("object_classification", {})
        return cls(
            scenes_dir=Path(_env("SCENES_DIR", paths.get("scenes_dir"), "scenes/originals")),
            outputs_dir=Path(_env("OUTPUTS_DIR", paths.get("outputs_dir"), "scenes/outputs")),
            max_movable_objects=int(_env("MAX_MOVABLE_OBJECTS", p.get("max_movable_objects"), 20)),
            min_object_dimension=float(p.get("min_object_dimension_meters", 0.05)),
            min_quality_score=int(p.get("min_quality_score_for_corrections", 7)),
            good_quality_score=int(p.get("good_quality_score_for_protection", 8)),
            max_corrections=int(p.get("max_corrections_to_apply", 5)),
            max_correction_displacement=float(p.get("max_correction_displacement_meters", 1.5)),
            non_mesh_types=frozenset(
                cls_toml.get("non_mesh_types", ["CAMERA", "LIGHT", "SPEAKER", "ARMATURE", "EMPTY", "CURVE"])
            ),
            structural_patterns=cls_toml.get(
                "structural_name_patterns", ["wall", "floor", "ceiling", "room", "door", "window"]
            ),
            ceiling_light_patterns=cls_toml.get(
                "ceiling_light_patterns", ["ceiling", "pendant", "chandelier"]
            ),
        )


@dataclass
class RenderConfig:
    """Parametri per il rendering Blender."""
    preview_width: int = 512
    preview_height: int = 512
    preview_samples: int = 64
    final_width: int = 1280
    final_height: int = 1280
    final_samples: int = 256

    isometric_elevation: float = 30.0
    isometric_azimuth: float = 275.0
    isometric_focal_length: float = 50.0
    isometric_distance_multiplier: float = 2.2
    topdown_height_multiplier: float = 2.0
    topdown_ortho_scale_padding: float = 1.05
    pipeline_camera_prefix: str = "NL2Scene3D_Camera"

    @classmethod
    def from_toml(cls, toml: dict[str, Any]) -> "RenderConfig":
        r = toml.get("render", {})
        prev = r.get("preview", {})
        final = r.get("final", {})
        cam = r.get("camera", {})
        return cls(
            preview_width=int(_env("RENDER_PREVIEW_WIDTH", prev.get("width"), 512)),
            preview_height=int(_env("RENDER_PREVIEW_HEIGHT", prev.get("height"), 512)),
            preview_samples=int(_env("RENDER_PREVIEW_SAMPLES", prev.get("samples"), 64)),
            final_width=int(_env("RENDER_FINAL_WIDTH", final.get("width"), 1280)),
            final_height=int(_env("RENDER_FINAL_HEIGHT", final.get("height"), 1280)),
            final_samples=int(_env("RENDER_FINAL_SAMPLES", final.get("samples"), 256)),
            isometric_elevation=float(cam.get("isometric_elevation_degrees", 30.0)),
            isometric_azimuth=float(cam.get("isometric_azimuth_degrees", 275.0)),
            isometric_focal_length=float(cam.get("isometric_focal_length_mm", 50.0)),
            isometric_distance_multiplier=float(cam.get("isometric_distance_multiplier", 2.2)),
            topdown_height_multiplier=float(cam.get("topdown_height_multiplier", 2.0)),
            topdown_ortho_scale_padding=float(cam.get("topdown_ortho_scale_padding", 1.05)),
            pipeline_camera_prefix=cam.get("pipeline_camera_prefix", "NL2Scene3D_Camera"),
        )


# ---------------------------------------------------------------------------
# Configurazione principale
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """
    Configurazione completa dell'applicazione.

    Nell'addon Blender, viene ricreata ad ogni operazione dopo che
    l'addon ha iniettato la API key in os.environ:

        os.environ["GEMINI_API_KEY"] = prefs.api_key
        config = AppConfig.load()
        config.gemini.model_primary = prefs.model_name  # override UI
    """
    gemini: GeminiConfig
    randomizer: RandomizerConfig
    pipeline: PipelineConfig
    render: RenderConfig

    @classmethod
    def load(cls) -> "AppConfig":
        """Carica la configurazione dall'ambiente e dal settings.toml."""
        config_dir = _find_config_dir()
        toml = _load_toml(config_dir)
        return cls(
            gemini=GeminiConfig.from_toml(toml),
            randomizer=RandomizerConfig.from_toml(toml),
            pipeline=PipelineConfig.from_toml(toml),
            render=RenderConfig.from_toml(toml),
        )

    @property
    def prompts_dir(self) -> Path:
        """Directory dei prompt, sempre relativa al package."""
        return _find_config_dir() / "prompts"


def get_config() -> AppConfig:
    """
    Carica e restituisce la configurazione corrente.

    Non usa singleton: ogni chiamata rilegge os.environ, garantendo che
    le override dell'addon (API key, modello) siano sempre riflesse.
    Il caricamento del TOML è veloce (file piccolo, ~1KB).
    """
    return AppConfig.load()


def reset_config() -> None:
    """No-op per compatibilità con il codice che si aspetta un reset del singleton."""
    pass