"""
Gestione della configurazione centralizzata del progetto.

Carica le variabili d'ambiente dal file .env e le espone
come attributi tipizzati di un oggetto di configurazione singleton.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Carica .env dalla root del progetto se presente
load_dotenv()


@dataclass
class GeminiConfig:
    """Parametri di configurazione per il client Gemini."""

    api_key: str
    model_primary: str
    model_fallback: str
    max_retries: int
    timeout_seconds: int

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        """Carica la configurazione dalle variabili d'ambiente."""
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "La variabile d'ambiente GEMINI_API_KEY non e' impostata. "
                "Copiare .env.example in .env e inserire la chiave API."
            )
        return cls(
            api_key=api_key,
            model_primary=os.environ.get(
                "GEMINI_MODEL_PRIMARY", "gemini-2.5-flash"
            ),
            model_fallback=os.environ.get(
                "GEMINI_MODEL_FALLBACK", "gemini-2.5-flash-lite"
            ),
            max_retries=int(os.environ.get("GEMINI_MAX_RETRIES", "3")),
            timeout_seconds=int(os.environ.get("GEMINI_TIMEOUT_SECONDS", "120")),
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

    @classmethod
    def from_env(cls) -> "RenderConfig":
        """Carica la configurazione dalle variabili d'ambiente."""
        return cls(
            preview_width=int(os.environ.get("RENDER_PREVIEW_WIDTH", "512")),
            preview_height=int(os.environ.get("RENDER_PREVIEW_HEIGHT", "512")),
            preview_samples=int(os.environ.get("RENDER_PREVIEW_SAMPLES", "64")),
            final_width=int(os.environ.get("RENDER_FINAL_WIDTH", "1280")),
            final_height=int(os.environ.get("RENDER_FINAL_HEIGHT", "720")),
            final_samples=int(os.environ.get("RENDER_FINAL_SAMPLES", "256")),
        )


@dataclass
class PipelineConfig:
    """Parametri generali della pipeline."""

    scenes_dir: Path
    outputs_dir: Path
    max_movable_objects: int
    randomizer_seed: int

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Carica la configurazione dalle variabili d'ambiente."""
        return cls(
            scenes_dir=Path(os.environ.get("SCENES_DIR", "scenes/originals")),
            outputs_dir=Path(os.environ.get("OUTPUTS_DIR", "scenes/outputs")),
            max_movable_objects=int(
                os.environ.get("MAX_MOVABLE_OBJECTS", "20")
            ),
            randomizer_seed=int(os.environ.get("RANDOMIZER_SEED", "0")),
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

    @classmethod
    def load(cls) -> "AppConfig":
        """
        Carica l'intera configurazione dalle variabili d'ambiente.

        Returns:
            Istanza configurata di AppConfig.

        Raises:
            EnvironmentError: Se variabili obbligatorie sono mancanti.
        """
        return cls(
            gemini=GeminiConfig.from_env(),
            render=RenderConfig.from_env(),
            pipeline=PipelineConfig.from_env(),
        )


def get_config() -> AppConfig:
    """
    Punto di accesso globale alla configurazione.

    Returns:
        Istanza di AppConfig caricata dall'ambiente.
    """
    return AppConfig.load()