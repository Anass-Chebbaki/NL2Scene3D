"""
Test unitari per il client Gemini.

Usa mock per evitare chiamate reali alle API durante i test.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nl2scene3d.config import GeminiConfig
from nl2scene3d.gemini_client import (
    GeminiClient,
    GeminiParsingError,
    GeminiRateLimitError,
)

def _make_config() -> GeminiConfig:
    """Crea una configurazione Gemini di test."""
    return GeminiConfig(
        api_key="test_api_key_not_real",
        model_primary="gemini-2.5-flash",
        model_fallback="gemini-2.5-flash-lite",
        max_retries=2,
        timeout_seconds=30,
    )

class TestJsonExtraction:
    """Test per il parsing JSON dalla risposta dell'LLM."""

    def test_parse_direct_json(self) -> None:
        """Verifica il parsing di una risposta JSON pulita."""
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel"):
            client = GeminiClient(_make_config())
            data = {"objects": [{"name": "sofa"}]}
            result = client._extract_json_from_response(json.dumps(data))
            assert result == data

    def test_parse_json_in_code_block(self) -> None:
        """Verifica il parsing di JSON racchiuso in un blocco ```json```."""
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel"):
            client = GeminiClient(_make_config())
            data = {"objects": []}
            text = f"Here is the result:\n```json\n{json.dumps(data)}\n```"
            result = client._extract_json_from_response(text)
            assert result == data

    def test_parse_json_embedded_in_text(self) -> None:
        """Verifica il parsing di JSON embedded in testo."""
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel"):
            client = GeminiClient(_make_config())
            data = {"score": 8, "corrections": []}
            text = f"Sure! Here is my analysis:\n{json.dumps(data)}\nHope this helps."
            result = client._extract_json_from_response(text)
            assert result == data

    def test_raises_on_invalid_json(self) -> None:
        """Verifica che GeminiParsingError venga sollevato per JSON non valido."""
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel"):
            client = GeminiClient(_make_config())
            with pytest.raises(GeminiParsingError):
                client._extract_json_from_response("This is not JSON at all.")
