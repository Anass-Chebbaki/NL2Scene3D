# nl2scene3d/gemini_client.py
"""
Client for the Google Gemini API.

Responsibilities:
  - Text-only calls for scene reorganization
  - Vision calls (single or multi-image) for visual feedback
  - Automatic retry with exponential backoff
  - Fallback to the secondary model on persistent errors
  - Robust JSON extraction from raw model output
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from nl2scene3d.config import GeminiConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GeminiClientError(Exception):
    """Base exception for Gemini client errors."""


class GeminiParsingError(GeminiClientError):
    """Raised when the JSON response from the model cannot be parsed."""


class GeminiRateLimitError(GeminiClientError):
    """Raised when the API rate limit is hit persistently across all retries."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class GeminiClient:
    """
    Wrapper around the Google Gemini API.

    Implements retry with exponential backoff and automatic fallback to the
    secondary model when the primary model returns persistent errors.
    """

    def __init__(self, config: GeminiConfig) -> None:
        self.config  = config
        self._client = genai.Client(
            api_key=config.api_key,
            http_options={"timeout": config.timeout_seconds * 1000},
        )
        logger.info(
            "GeminiClient initialized. Timeout: %ds. Primary: %s, fallback: %s.",
            config.timeout_seconds,
            config.model_primary,
            config.model_fallback,
        )

    # ------------------------------------------------------------------
    # JSON extraction
    # ------------------------------------------------------------------

    def _extract_json_from_response(self, text: str) -> dict | list:
        """
        Extracts and parses JSON from the raw model response text.

        Tries four strategies in order:
          1. Direct parse.
          2. Extract from a markdown code fence (```json ... ```).
          3. Brute-force: find the outermost { } or [ ] block.
          4. Repair truncated JSON by appending missing closing brackets.
        """
        text = text.strip()
        logger.debug("Raw Gemini response (first 500 chars):\n%s", text[:500])

        # Strategy 1: direct parse.
        try:
            result = json.loads(text)
            logger.debug("Strategy 1 (direct parse) succeeded. Type: %s.", type(result).__name__)
            return result
        except json.JSONDecodeError as exc:
            logger.debug("Strategy 1 failed: %s.", exc)

        # Strategy 2: markdown code fence.
        pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
        match   = pattern.search(text)
        if match:
            try:
                extracted = match.group(1).strip()
                result    = json.loads(extracted)
                logger.debug("Strategy 2 (code fence) succeeded. Type: %s.", type(result).__name__)
                return result
            except json.JSONDecodeError as exc:
                logger.debug("Strategy 2 failed: %s.", exc)

        # Strategy 3: brute-force outer bracket search.
        start_dict = text.find("{")
        start_list = text.find("[")

        if start_dict != -1 and (start_list == -1 or start_dict < start_list):
            start_idx, end_char = start_dict, "}"
        elif start_list != -1:
            start_idx, end_char = start_list, "]"
        else:
            start_idx = -1

        if start_idx != -1:
            end_idx = text.rfind(end_char)
            if end_idx > start_idx:
                json_str = text[start_idx : end_idx + 1]
                try:
                    result = json.loads(json_str)
                    logger.debug("Strategy 3 (brute force) succeeded. Type: %s.", type(result).__name__)
                    return result
                except json.JSONDecodeError:
                    # Try stripping inline comments before giving up.
                    clean = re.sub(r"//.*?\n|/\*.*?\*/", "", json_str, flags=re.DOTALL)
                    try:
                        result = json.loads(clean)
                        logger.debug("Strategy 3 (comment-stripped) succeeded.")
                        return result
                    except json.JSONDecodeError as exc:
                        logger.debug("Strategy 3 failed: %s.", exc)

        # Strategy 4: repair truncated JSON.
        if start_idx != -1:
            logger.warning("JSON appears truncated. Attempting automatic closure.")
            open_braces   = text.count("{")
            close_braces  = text.count("}")
            open_brackets = text.count("[")
            close_brackets = text.count("]")

            repaired = text[start_idx:]
            if open_brackets > close_brackets:
                repaired += "]" * (open_brackets - close_brackets)
            if open_braces > close_braces:
                repaired += "}" * (open_braces - close_braces)

            try:
                result = json.loads(repaired)
                logger.debug("Strategy 4 (repair) succeeded.")
                return result
            except json.JSONDecodeError as exc:
                logger.debug("Strategy 4 failed: %s.", exc)

        logger.error("Full Gemini response:\n%s", text)
        raise GeminiParsingError(
            "Could not extract valid JSON from the model response. "
            "The response may be truncated or contain syntax errors. "
            f"Preview: {text[:200]}..."
        )

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    def _call_with_retry(
        self,
        model_name:      str,
        contents:        Any,
        system_prompt:   Optional[str]  = None,
        config_override: Optional[dict] = None,
    ) -> str:
        """
        Calls the model with automatic retry and exponential backoff.

        On the first attempt, requests JSON MIME type output. If the model
        does not support it, subsequent attempts omit that constraint.
        """
        gen_config_dict: dict = {
            "temperature":      self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
            **(config_override or {}),
        }

        use_json_mime    = True
        last_exception: Exception = GeminiClientError("No attempt was executed.")

        for attempt in range(self.config.max_retries):
            try:
                config_dict = gen_config_dict.copy()
                # Request JSON MIME type only on the very first attempt.
                if use_json_mime and attempt < 1:
                    config_dict["response_mime_type"] = "application/json"

                gen_config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    **config_dict,
                )
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=gen_config,
                )
                if not response.text:
                    raise GeminiParsingError("Empty model response.")
                return response.text

            except Exception as exc:
                exc_str = str(exc).lower()

                # The model does not support application/json MIME type.
                if "mime type" in exc_str and use_json_mime and attempt < 1:
                    logger.warning(
                        "response_mime_type='application/json' not supported by %s. Retrying without it.",
                        model_name,
                    )
                    use_json_mime = False
                    continue

                # Rate limit (429) or quota exhausted.
                if any(err in exc_str for err in ("429", "quota", "exhausted")):
                    last_exception = exc
                    wait = 10 * (2 ** attempt)
                    logger.warning(
                        "API quota exhausted (429). Waiting %ds (attempt %d/%d).",
                        wait, attempt + 1, self.config.max_retries,
                    )
                    if attempt < self.config.max_retries - 1:
                        time.sleep(wait)
                    else:
                        raise GeminiRateLimitError(
                            f"Quota exhausted after all retries: {exc}"
                        ) from exc

                # Service temporarily unavailable (503).
                elif any(err in exc_str for err in ("503", "unavailable", "demand")):
                    last_exception = exc
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        "Gemini API temporarily unavailable (503, attempt %d/%d). Waiting %ds.",
                        attempt + 1, self.config.max_retries, wait,
                    )
                    if attempt < self.config.max_retries - 1:
                        time.sleep(wait)

                # Permanent client error (400, 401, 403).
                elif any(err in exc_str for err in ("400", "invalid", "401", "403")):
                    logger.error("Permanent API client error: %s.", exc)
                    raise GeminiClientError(f"Permanent API error: {exc}") from exc

                # Any other error (500, 502, 504, network issues).
                else:
                    last_exception = exc
                    logger.error(
                        "Unexpected Gemini API error (attempt %d/%d): %s.",
                        attempt + 1, self.config.max_retries, exc,
                    )
                    if attempt < self.config.max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        raise GeminiRateLimitError(
                            f"Persistent API error: {exc}"
                        ) from exc

        raise GeminiClientError(f"All retries exhausted. Last error: {last_exception}")

    # ------------------------------------------------------------------
    # Public call methods
    # ------------------------------------------------------------------

    def call_text(
        self,
        system_prompt: str,
        user_prompt:   str,
        use_fallback:  bool = False,
    ) -> dict | list:
        """
        Sends a text-only request to Gemini and returns the parsed JSON response.

        Automatically switches to the fallback model on rate-limit errors.
        """
        model_name = self.config.model_fallback if use_fallback else self.config.model_primary
        logger.info(
            "Text call to Gemini (%s). System prompt: %d chars, user prompt: %d chars.",
            model_name, len(system_prompt), len(user_prompt),
        )

        try:
            raw   = self._call_with_retry(model_name, user_prompt, system_prompt=system_prompt)
            parsed = self._extract_json_from_response(raw)
            logger.info(
                "Gemini response parsed. Type: %s, entries: %s.",
                type(parsed).__name__,
                len(parsed) if isinstance(parsed, (list, dict)) else "N/A",
            )
            return parsed
        except GeminiRateLimitError:
            if not use_fallback:
                logger.warning("Switching to fallback model due to rate limit.")
                return self.call_text(system_prompt, user_prompt, use_fallback=True)
            raise

    def _call_vision_internal(
        self,
        model_name:  str,
        contents:    list,
        use_fallback: bool,
    ) -> dict | list:
        """Shared implementation for single- and multi-image vision calls."""
        try:
            raw = self._call_with_retry(model_name, contents)
            return self._extract_json_from_response(raw)
        except GeminiRateLimitError:
            if not use_fallback:
                return self._call_vision_internal(
                    self.config.model_fallback, contents, use_fallback=True
                )
            raise

    def call_vision(
        self,
        image_path:  Path,
        user_prompt: str,
        use_fallback: bool = False,
    ) -> dict | list:
        """
        Sends a single-image vision request to Gemini.

        Raises FileNotFoundError if the image does not exist.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Vision image not found: {image_path}")

        model_name = self.config.model_fallback if use_fallback else self.config.model_primary
        logger.info("Single-image vision call to Gemini (%s).", model_name)

        try:
            import PIL.Image
            img      = PIL.Image.open(image_path)
            contents = [img, user_prompt]
            return self._call_vision_internal(model_name, contents, use_fallback)
        except Exception as exc:
            logger.error("Vision call failed: %s.", exc)
            raise GeminiClientError(f"Vision error: {exc}") from exc

    def call_vision_multi(
        self,
        image_paths: list[Path],
        user_prompt: str,
        use_fallback: bool = False,
    ) -> dict | list:
        """
        Sends a multi-image vision request to Gemini.

        All images are included as separate content elements before the text
        prompt, giving the model a complete multi-view context in one call.

        Raises FileNotFoundError if any image does not exist.
        """
        for path in image_paths:
            if not path.exists():
                raise FileNotFoundError(f"Vision image not found: {path}")

        model_name = self.config.model_fallback if use_fallback else self.config.model_primary
        logger.info(
            "Multi-image vision call to Gemini (%s). Images: %d.",
            model_name, len(image_paths),
        )

        try:
            import PIL.Image
            contents: list = []
            for path in image_paths:
                contents.append(PIL.Image.open(path))
            contents.append(user_prompt)
            return self._call_vision_internal(model_name, contents, use_fallback)
        except Exception as exc:
            logger.error("Multi-image vision call failed: %s.", exc)
            raise GeminiClientError(f"Multi-image vision error: {exc}") from exc