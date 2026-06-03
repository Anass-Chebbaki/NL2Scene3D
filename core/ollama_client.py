# nl2scene3d/core/ollama_client.py
"""
Client Ollama autonomo: parla con il server locale via HTTP usando SOLO la
libreria standard (urllib). Nessuna dipendenza esterna, nessun bpy.

Scelte importanti per i modelli "thinking" (es. qwen3.5) su hardware modesto:
  - think=False di default: per un compito di piazzamento JSON non serve far
    ragionare il modello, e cosi' risponde diretto e veloce (niente token sprecati
    a "pensare", niente risposte vuote).
  - streaming: la risposta arriva a pezzi (NDJSON), la connessione resta viva e
    non scatta il timeout durante una generazione lunga.
  - timeout generoso e configurabile: copre il primo caricamento del modello in
    VRAM, che su GPU piccole puo' richiedere minuti.

Le parti pure (build_payload, parse_response) restano testabili da riga di comando.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_URL = "http://localhost:11434"


class OllamaError(Exception):
    """Errore di comunicazione/uso del server Ollama, con messaggio leggibile."""


def build_payload(
    model: str,
    prompt: str,
    images: Optional[list[str]] = None,
    temperature: float = 0.2,
    stream: bool = False,
    force_json: bool = False,
    num_predict: int = 2048,
    think: bool = False,
) -> dict:
    """
    Costruisce il corpo della richiesta /api/generate. PURO e testabile.

    images:     PNG base64 (per i modelli con visione); se vuoto non viene inviato.
    force_json: vincola l'output a JSON valido (disattivo: l'estrattore e' robusto).
    num_predict: budget massimo di token in uscita.
    think:      se False disattiva il ragionamento dei modelli "thinking".
    """
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": bool(stream),
        "think": bool(think),
        "options": {
            "temperature": float(temperature),
            "num_predict": int(num_predict),
        },
    }
    if images:
        payload["images"] = list(images)
    if force_json:
        payload["format"] = "json"
    return payload


def parse_response(raw) -> str:
    """
    Estrae il campo 'response' da una risposta NON-streaming. PURO e testabile.
    Accetta bytes o str. Solleva OllamaError se non e' JSON valido.
    """
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise OllamaError(f"Risposta di Ollama non in JSON: {exc}") from exc
    if "error" in data:
        raise OllamaError(f"Ollama ha risposto con errore: {data['error']}")
    return data.get("response", "")


class OllamaClient:
    """Client minimale per il server Ollama locale (streaming, thinking off)."""

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        model: str = "qwen3.5:4b",
        temperature: float = 0.2,
        timeout: int = 300,
    ) -> None:
        self.base_url = (base_url or DEFAULT_URL).rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def is_available(self) -> bool:
        """True se il server risponde (GET /api/tags). Non solleva eccezioni."""
        try:
            req = urllib.request.Request(self.base_url + "/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return getattr(resp, "status", 200) == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        images: Optional[list[str]] = None,
        force_json: bool = False,
        think: bool = False,
    ) -> str:
        """
        Invia il prompt in STREAMING e ritorna il testo completo della risposta
        (concatenando i pezzi 'response'). I pezzi di 'thinking' eventuali non
        vengono inclusi. Solleva OllamaError con messaggio chiaro in caso di
        problemi (server spento, modello assente, timeout).
        """
        payload = build_payload(
            self.model, prompt, images, self.temperature,
            stream=True, force_json=force_json, num_predict=2048, think=think,
        )
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        chunks: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw_line in resp:  # NDJSON: un oggetto JSON per riga
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if not isinstance(obj, dict):
                        continue
                    if obj.get("error"):
                        raise OllamaError(f"Ollama ha risposto con errore: {obj['error']}")
                    piece = obj.get("response", "")
                    if piece:
                        chunks.append(piece)
                    if obj.get("done"):
                        break
        except OllamaError:
            raise
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise OllamaError(
                f"Ollama ha risposto HTTP {exc.code}. Modello '{self.model}' "
                f"installato? (ollama pull {self.model}). Dettaglio: {detail[:300]}"
            ) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise OllamaError(
                f"Timeout dopo {self.timeout}s. Il primo caricamento di '{self.model}' "
                f"in VRAM puo' essere lento su GPU piccole. Riprova (a modello caldo e' "
                f"piu' rapido) o aumenta il timeout nelle Preferences."
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise OllamaError(
                    f"Timeout dopo {self.timeout}s durante il caricamento di "
                    f"'{self.model}'. Riprova o aumenta il timeout nelle Preferences."
                ) from exc
            raise OllamaError(
                f"Ollama non raggiungibile su {self.base_url}. "
                f"Avvialo (ollama serve). Dettaglio: {reason}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise OllamaError(f"Errore imprevisto contattando Ollama: {exc}") from exc

        return "".join(chunks)
