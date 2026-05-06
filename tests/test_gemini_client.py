"""
Test unitari per il client Gemini.

Usa mock per evitare chiamate reali alle API durante i test.
"""

from __future__ import annotations

import json
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
        with patch("google.genai.Client"):
            client = GeminiClient(_make_config())
            data = {"objects": [{"name": "sofa"}]}
            result = client._extract_json_from_response(json.dumps(data))
            assert result == data

    def test_parse_json_in_code_block(self) -> None:
        """Verifica il parsing di JSON racchiuso in un blocco ```json```."""
        with patch("google.genai.Client"):
            client = GeminiClient(_make_config())
            data = {"objects": []}
            text = f"Here is the result:\n```json\n{json.dumps(data)}\n```"
            result = client._extract_json_from_response(text)
            assert result == data

    def test_raises_on_invalid_json(self) -> None:
        """Verifica che GeminiParsingError venga sollevato per JSON non valido."""
        with patch("google.genai.Client"):
            client = GeminiClient(_make_config())
            with pytest.raises(GeminiParsingError):
                client._extract_json_from_response("This is not JSON at all.")


class TestRetryLogic:
    """Test per la logica di retry e fallback."""

    def test_retry_on_rate_limit(self) -> None:
        """Verifica che il client riprovi in caso di rate limit (429)."""
        with patch("google.genai.Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            # Fallisce la prima volta con 429, poi successo
            mock_client.models.generate_content.side_effect = [
                Exception("429 Resource Exhausted"),
                MagicMock(text='{"success": true}')
            ]

            client = GeminiClient(_make_config())
            with patch("time.sleep"):
                result = client.call_text("system", "user")

            assert result == {"success": True}
            assert mock_client.models.generate_content.call_count == 2

    def test_fallback_on_repeated_rate_limit(self) -> None:
        """Verifica il passaggio al modello fallback dopo vari errori di quota."""
        with patch("google.genai.Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            
            # Chiamata al primario fallisce sempre
            # Chiamata al fallback (che viene fatta con lo stesso client ma diverso model name)
            # ha successo.
            def side_effect(model, contents, config):
                if model == "gemini-primary":
                    raise Exception("429 Quota Exceeded")
                return MagicMock(text='{"fallback": true}')

            mock_client.models.generate_content.side_effect = side_effect

            client = GeminiClient(_make_config())
            with patch("time.sleep"):
                result = client.call_text("system", "user")

            assert result == {"fallback": True}
            # 2 tentativi sul primario (max_retries=2) + 1 sul fallback
            assert mock_client.models.generate_content.call_count == 3

    def test_raises_after_max_retries(self) -> None:
        """Verifica che venga sollevato GeminiRateLimitError dopo esaurimento tentativi."""
        with patch("google.genai.Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.models.generate_content.side_effect = Exception("429 Deadlock")

            client = GeminiClient(_make_config())
            with patch("time.sleep"), pytest.raises(GeminiRateLimitError):
                client.call_text("system", "user")
