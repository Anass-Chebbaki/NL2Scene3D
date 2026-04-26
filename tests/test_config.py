"""
Test unitari per il sistema di configurazione.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest
from nl2scene3d.config import AppConfig, GeminiConfig, PipelineConfig, get_config

def test_singleton_config() -> None:
    """Verifica che get_config() restituisca sempre la stessa istanza (Bug 2.2)."""
    with patch("nl2scene3d.config._load_toml_config", return_value={}):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

def test_config_validation() -> None:
    """Verifica la validazione dei parametri di configurazione (Bug 4.2)."""
    # Max retries negativo
    with pytest.raises(ValueError, match="max_retries"):
        GeminiConfig(
            api_key="key",
            model_primary="m1",
            model_fallback="m2",
            max_retries=-1,
            timeout_seconds=30
        )
    
    # Max objects negativo
    with pytest.raises(ValueError, match="max_movable_objects"):
        PipelineConfig(
            scenes_dir=Path("."),
            outputs_dir=Path("."),
            max_movable_objects=0,
            randomizer_seed=0
        )

def test_env_override() -> None:
    """Verifica che le variabili d'ambiente abbiano la precedenza (Bug 1.1)."""
    toml_data = {
        "gemini": {"model_primary": "from-toml"},
        "pipeline": {"max_movable_objects": 10}
    }
    
    with patch.dict(os.environ, {"GEMINI_MODEL_PRIMARY": "from-env"}):
        config = AppConfig.from_dict(toml_data)
        assert config.gemini.model_primary == "from-env"
        # Questo non e' in env, quindi deve venire dal toml
        assert config.pipeline.max_movable_objects == 10

def test_toml_loading() -> None:
    """Verifica il caricamento del file TOML (Bug 2.1)."""
    fake_toml = """
    [gemini]
    model_primary = "test-model"
    
    [pipeline]
    max_movable_objects = 42
    """
    with patch("builtins.open", mock_open(read_data=fake_toml.encode())):
        with patch("nl2scene3d.config.Path.exists", return_value=True):
            from nl2scene3d.config import _load_toml_config
            data = _load_toml_config()
            assert data["gemini"]["model_primary"] == "test-model"
            assert data["pipeline"]["max_movable_objects"] == 42
