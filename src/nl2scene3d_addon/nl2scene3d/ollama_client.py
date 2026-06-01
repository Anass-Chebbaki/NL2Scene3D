# nl2scene3d/ollama_client.py
"""
Local LLM backend via Ollama (no API key, no quota).

Exposes the same public methods as GeminiClient (call_text, call_vision,
call_vision_multi) and returns parsed JSON, so SceneReorganizer / VisualCritic
can use it interchangeably. Uses only the standard library (urllib) + PIL,
so nothing new needs to be vendored.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

from nl2scene3d.gemini_client import GeminiClientError, GeminiParsingError

logger = logging.getLogger(__name__)


def _extract_json(text: str):
    """Extract a JSON object/array from raw text (3 strategies)."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        end   = text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass
    raise GeminiParsingError(f"Could not extract JSON from Ollama response: {text[:200]}")


class OllamaClient:
    """Local vision/text client backed by an Ollama server."""

    def __init__(self, model, base_url="http://localhost:11434",
                 temperature=0.2, timeout=600, num_ctx=8192, max_image_edge=768):
        self.model          = model
        self.base_url       = base_url.rstrip("/")
        self.temperature    = temperature
        self.timeout        = timeout
        self.num_ctx        = num_ctx
        self.max_image_edge = max_image_edge
        logger.info("OllamaClient initialized. Model: %s, URL: %s.", model, self.base_url)

    def _image_to_b64(self, image_path: Path) -> str:
        try:
            import PIL.Image
            img  = PIL.Image.open(image_path).convert("RGB")
            w, h = img.size
            edge = max(w, h)
            if self.max_image_edge and edge > self.max_image_edge:
                s = self.max_image_edge / edge
                img = img.resize((int(w * s), int(h * s)))
            buf = BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            with open(image_path, "rb") as fh:
                return base64.b64encode(fh.read()).decode("utf-8")

    def _post_chat(self, messages: list):
        body = {
            "model":    self.model,
            "messages": messages,
            "stream":   False,
            "options":  {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        req = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise GeminiClientError(
                f"Cannot reach Ollama at {self.base_url}. Is the Ollama app running? ({exc})"
            ) from exc
        except Exception as exc:
            raise GeminiClientError(f"Ollama request failed: {exc}") from exc

        try:
            content = json.loads(raw)["message"]["content"]
        except Exception as exc:
            raise GeminiClientError(f"Unexpected Ollama response: {raw[:200]}") from exc
        return _extract_json(content)

    # --- same shape as GeminiClient ---

    def call_text(self, system_prompt, user_prompt, use_fallback=False):
        logger.info("Ollama text call (%s).", self.model)
        return self._post_chat([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ])

    def call_vision(self, image_path, user_prompt, use_fallback=False):
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Vision image not found: {image_path}")
        logger.info("Ollama single-image call (%s).", self.model)
        return self._post_chat([{
            "role": "user", "content": user_prompt,
            "images": [self._image_to_b64(Path(image_path))],
        }])

    def call_vision_multi(self, image_paths, user_prompt, use_fallback=False):
        for p in image_paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"Vision image not found: {p}")
        logger.info("Ollama multi-image call (%s). Images: %d.", self.model, len(image_paths))
        return self._post_chat([{
            "role": "user", "content": user_prompt,
            "images": [self._image_to_b64(Path(p)) for p in image_paths],
        }])