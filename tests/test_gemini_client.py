"""
Test unitari per il client Gemini.

Usa mock per evitare chiamate reali alle API durante i test.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import GoogleAPIError, ResourceExhausted

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
        model_primary="gemini-primary",
        model_fallback="gemini-fallback",
        max_retries=2,
        timeout_seconds=30,
        temperature=0.2,
        max_output_tokens=4096,
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

    def test_raises_on_invalid_json(self) -> None:
        """Verifica che GeminiParsingError venga sollevato per JSON non valido."""
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel"):
            client = GeminiClient(_make_config())
            with pytest.raises(GeminiParsingError):
                client._extract_json_from_response("This is not JSON at all.")

class TestRetryLogic:
    """Test per la logica di retry e fallback (Bug 5.2)."""

    def test_retry_on_rate_limit(self) -> None:
        """Verifica che il client riprovi in caso di ResourceExhausted."""
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as mock_model_class:
            
            mock_model = mock_model_class.return_value
            # Fallisce la prima volta con ResourceExhausted, poi successo
            mock_model.generate_content.side_effect = [
                ResourceExhausted("Rate limit exceeded"),
                MagicMock(text='{"success": true}')
            ]
            
            client = GeminiClient(_make_config())
            with patch("time.sleep"): # Salta l'attesa reale
                result = client.call_text("system", "user")
            
            assert result == {"success": True}
            assert mock_model.generate_content.call_count == 2

    def test_fallback_on_repeated_rate_limit(self) -> None:
        """Verifica il passaggio al modello fallback dopo vari errori di quota."""
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as mock_model_class:
            
            # Vogliamo simulare: 
            # 1. Chiamata primaria fallisce 2 volte (max_retries)
            # 2. Chiamata fallback ha successo
            mock_model_primary = MagicMock()
            mock_model_primary.generate_content.side_effect = ResourceExhausted("Quota")
            
            mock_model_fallback = MagicMock()
            mock_model_fallback.generate_content.return_value = MagicMock(text='{"fallback": true}')
            
            # Il primo init crea il primario, il secondo il fallback
            mock_model_class.side_effect = [mock_model_primary, mock_model_fallback]
            
            client = GeminiClient(_make_config())
            with patch("time.sleep"):
                result = client.call_text("system", "user")
            
            assert result == {"fallback": True}
            assert mock_model_primary.generate_content.call_count == 2
            assert mock_model_fallback.generate_content.call_count == 1

    def test_raises_after_max_retries(self) -> None:
        """Verifica che venga sollevato GeminiRateLimitError dopo esaurimento tentativi."""
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as mock_model_class:
            
            mock_model = mock_model_class.return_value
            mock_model.generate_content.side_effect = ResourceExhausted("Deadlock")
            
            client = GeminiClient(_make_config())
            # Impostiamo use_fallback=True per testare il crash finale dopo che anche il fallback fallisce
            with patch("time.sleep"), pytest.raises(GeminiRateLimitError):
                client.call_text("system", "user", use_fallback=True)
