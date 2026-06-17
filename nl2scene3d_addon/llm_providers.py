# nl2scene3d/llm_providers.py
"""
Client di rete per i provider LLM (Gemini, e in prospettiva Anthropic/OpenAI).

Questo modulo e' l'UNICO punto dell'add-on che effettua chiamate di rete.
E' volutamente tenuto FUORI da `core/` per non violare la garanzia di
"core puro e offline": qui non si importa bpy e si usa solo la stdlib
(urllib), quindi:

    * nessuna dipendenza esterna da installare nel Python di Blender
      (niente `google-generativeai`, niente `requests`): Blender include
      gia' ssl/urllib, percio' le chiamate HTTPS funzionano out-of-the-box;
    * il modulo resta importabile e testabile da riga di comando senza
      Blender (come i moduli di core/).

La logica geometrica (build_prompt, extract_json, sanitize_response) resta
in core.reorganizer: questo modulo si limita a "prendere un prompt + immagini
e restituire il testo grezzo della risposta", che poi viene dato in pasto
a reorganizer.extract_json esattamente come avveniva con il copia-incolla.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Tipi di supporto
# ---------------------------------------------------------------------------

class LLMError(RuntimeError):
    """Errore applicativo durante la chiamata al provider (auth, rete, parsing)."""


@dataclass
class LLMResult:
    """Risultato di una chiamata: testo grezzo + metadati utili per la UI/log."""
    text: str
    provider: str
    model: str
    raw: dict = field(default_factory=dict)   # risposta JSON completa (per debug)


# Provider supportati. L'enum della UI usa queste stesse chiavi.
GEMINI    = "GEMINI"
ANTHROPIC = "ANTHROPIC"
OPENAI    = "OPENAI"

# Modelli di default suggeriti per ciascun provider (giugno 2026).
DEFAULT_MODELS = {
    GEMINI:    "gemini-3.5-flash",
    ANTHROPIC: "claude-haiku-4-5",
    OPENAI:    "gpt-5.1-mini",
}

# Nome della variabile d'ambiente usata come fallback se la chiave non e'
# stata inserita nelle preferenze dell'add-on.
ENV_KEYS = {
    GEMINI:    "GEMINI_API_KEY",
    ANTHROPIC: "ANTHROPIC_API_KEY",
    OPENAI:    "OPENAI_API_KEY",
}


# ---------------------------------------------------------------------------
# Helper comuni
# ---------------------------------------------------------------------------

_LOG_PREFIX = "[NL2Scene3D]"


def _log(verbose: bool, msg: str) -> None:
    """
    Stampa un messaggio di avanzamento nella console di sistema di Blender.

    Usa print() (non logging) perche' e' l'unico canale che Blender mostra
    sempre nella System Console senza configurazione aggiuntiva. flush=True
    garantisce che il messaggio compaia subito, anche durante una chiamata
    di rete bloccante eseguita in un thread.
    """
    if verbose:
        print(f"{_LOG_PREFIX} {msg}", flush=True)


def _read_image_b64(path: str) -> tuple[str, str]:
    """
    Legge un file immagine e ne restituisce (mime_type, base64).
    Solleva LLMError se il file non e' leggibile.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        raise LLMError(f"Immagine non leggibile: {path} ({exc})") from exc

    mime = mimetypes.guess_type(path)[0] or "image/png"
    return mime, base64.b64encode(data).decode("ascii")


def _image_label(path: str) -> str:
    """
    Ricava una descrizione leggibile della vista dal suffisso del file di render
    (_cam / _top / _iso), da inserire come parte testuale prima dell'immagine.
    Cosi' il modello sa sempre quale vista sta guardando, senza dover indovinare
    dall'ordine degli allegati.
    """
    base = os.path.basename(path).lower()
    if base.endswith("_top.png"):
        return "TOP-DOWN floor plan (orthographic; includes scale bar and X/Y compass)"
    if base.endswith("_cam.png"):
        return "ANGLED perspective view (includes X/Y compass; no scale bar)"
    if base.endswith("_iso.png"):
        return "ISOMETRIC view (orthographic; includes scale bar and X/Y compass)"
    return f"Rendered view: {os.path.basename(path)}"


# Codici di stato considerati transitori: vale la pena riprovare.
#   429 = rate limit, 500/502/503/504 = sovraccarico/errore temporaneo del server.
_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})


def _backoff_delay(attempt: int) -> float:
    """Ritardo esponenziale con jitter: ~1.5, 3, 6, 12s (cap a 20s)."""
    return min(1.5 * (2 ** attempt), 20.0) + random.uniform(0.0, 0.6)


def _retry_after_or_backoff(exc: "urllib.error.HTTPError", attempt: int) -> float:
    """Usa l'header Retry-After del server se presente e valido, altrimenti il backoff."""
    try:
        ra = exc.headers.get("Retry-After") if exc.headers else None
        if ra:
            return min(float(ra), 30.0)
    except (ValueError, AttributeError):
        pass
    return _backoff_delay(attempt)


def _http_post_json(
    url: str,
    payload: dict,
    headers: dict,
    timeout: float,
    *,
    verbose: bool = False,
    max_retries: int = 4,
) -> dict:
    """
    Esegue una POST JSON e restituisce il JSON di risposta come dict.

    Gli errori transitori (429 rate limit, 503 sovraccarico, 500/502/504, e i
    blip di rete) vengono ritentati automaticamente con backoff esponenziale.
    Questo gira nel thread di rete, quindi le attese non bloccano la UI di
    Blender. Gli errori non transitori (es. 400/401/403 = chiave o richiesta
    sbagliata) NON vengono ritentati: sarebbe inutile.
    """
    body = json.dumps(payload).encode("utf-8")
    attempt = 0
    while True:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            # Prova ad estrarre il messaggio strutturato dell'API.
            msg = detail
            try:
                err = json.loads(detail)
                if isinstance(err, dict):
                    msg = (err.get("error") or {}).get("message") or err.get("error") or detail
            except Exception:
                pass

            if exc.code in _TRANSIENT_STATUS and attempt < max_retries:
                delay = _retry_after_or_backoff(exc, attempt)
                _log(
                    verbose,
                    f"  HTTP {exc.code} (sovraccarico/limite del provider). "
                    f"Riprovo tra {delay:.1f}s [tentativo {attempt + 1}/{max_retries}]...",
                )
                time.sleep(delay)
                attempt += 1
                continue

            hint = ""
            if exc.code in _TRANSIENT_STATUS:
                hint = (
                    " Il servizio e' sovraccarico anche dopo i tentativi: "
                    "riprova tra qualche minuto o cambia modello nelle preferenze."
                )
            raise LLMError(f"HTTP {exc.code} dal provider: {msg}{hint}") from exc

        except urllib.error.URLError as exc:
            if attempt < max_retries:
                delay = _backoff_delay(attempt)
                _log(
                    verbose,
                    f"  errore di rete ({exc.reason}). "
                    f"Riprovo tra {delay:.1f}s [tentativo {attempt + 1}/{max_retries}]...",
                )
                time.sleep(delay)
                attempt += 1
                continue
            raise LLMError(
                f"Errore di rete: {exc.reason}. "
                "Controlla la connessione, eventuali proxy/firewall e la chiave API."
            ) from exc

        except (TimeoutError, OSError) as exc:
            if attempt < max_retries:
                delay = _backoff_delay(attempt)
                _log(
                    verbose,
                    f"  timeout/I-O ({exc}). "
                    f"Riprovo tra {delay:.1f}s [tentativo {attempt + 1}/{max_retries}]...",
                )
                time.sleep(delay)
                attempt += 1
                continue
            raise LLMError(f"Timeout o errore I/O contattando il provider: {exc}") from exc


def _resolve_key(provider: str, api_key: str) -> str:
    """Usa la chiave passata o, se vuota, quella nella variabile d'ambiente."""
    key = (api_key or "").strip()
    if key:
        return key
    env = os.environ.get(ENV_KEYS.get(provider, ""), "").strip()
    if env:
        return env
    raise LLMError(
        f"Nessuna API key per {provider}. Inseriscila nelle preferenze "
        f"dell'add-on o nella variabile d'ambiente {ENV_KEYS.get(provider)}."
    )


# ---------------------------------------------------------------------------
# Provider: Google Gemini  (generateContent REST, v1beta)
# ---------------------------------------------------------------------------

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


def call_gemini(
    *,
    api_key: str,
    model: str,
    prompt: str,
    image_paths: list[str],
    temperature: float = 0.7,
    timeout: float = 120.0,
    verbose: bool = False,
    max_retries: int = 4,
) -> LLMResult:
    """
    Chiama Gemini con testo + immagini inline e restituisce il testo grezzo.

    Il prompt e le immagini sono inviati in un singolo turno 'user'. Si forza
    `responseMimeType=application/json` per ottenere JSON pulito; la robustezza
    e' comunque garantita a valle da reorganizer.extract_json.
    """
    key = _resolve_key(GEMINI, api_key)

    parts: list[dict] = [{"text": prompt}]
    img_bytes = 0
    for p in image_paths:
        mime, b64 = _read_image_b64(p)
        img_bytes += len(b64)
        label = _image_label(p)
        parts.append({"text": f"\n[{label}]"})
        parts.append({"inline_data": {"mime_type": mime, "data": b64}})
        _log(verbose, f"  allego immagine: {os.path.basename(p)} -> {label}")

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": float(temperature),
            "responseMimeType": "application/json",
        },
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": key,
    }
    url = GEMINI_ENDPOINT.format(model=model)

    _log(verbose, f"POST {url}")
    _log(
        verbose,
        f"  modello={model} temp={temperature} prompt={len(prompt)} char "
        f"immagini={len(image_paths)} (~{img_bytes // 1024} KB base64) timeout={timeout}s",
    )
    t0 = time.monotonic()
    data = _http_post_json(url, payload, headers, timeout, verbose=verbose, max_retries=max_retries)
    dt = time.monotonic() - t0
    _log(verbose, f"  risposta ricevuta in {dt:.1f}s")

    # Token usati (se presenti), utili per stimare costo e dimensione.
    usage = data.get("usageMetadata") or {}
    if usage and verbose:
        _log(
            verbose,
            f"  token: prompt={usage.get('promptTokenCount', '?')} "
            f"output={usage.get('candidatesTokenCount', '?')} "
            f"totale={usage.get('totalTokenCount', '?')}",
        )

    # Estrae il testo da candidates[0].content.parts[*].text, ignorando le
    # eventuali parti di "thinking" (i modelli Gemini 3 possono emetterle).
    candidates = data.get("candidates") or []
    if not candidates:
        # Blocco per safety o prompt: rendilo esplicito.
        feedback = data.get("promptFeedback") or {}
        reason = feedback.get("blockReason") or "nessun candidato restituito"
        raise LLMError(f"Gemini non ha prodotto una risposta ({reason}).")

    cand = candidates[0]
    cand_parts = (cand.get("content") or {}).get("parts") or []
    texts = [p["text"] for p in cand_parts if isinstance(p, dict) and "text" in p and not p.get("thought")]
    if not texts:  # fallback: prendi qualunque testo, anche thought
        texts = [p["text"] for p in cand_parts if isinstance(p, dict) and "text" in p]

    text = "\n".join(texts).strip()
    if not text:
        finish = cand.get("finishReason", "?")
        raise LLMError(f"Gemini ha risposto senza testo (finishReason={finish}).")

    _log(
        verbose,
        f"  finishReason={cand.get('finishReason', '?')} testo={len(text)} char",
    )
    return LLMResult(text=text, provider=GEMINI, model=model, raw=data)


# ---------------------------------------------------------------------------
# Provider: Anthropic  (Messages API)  -- pronto per il roadmap multi-provider
# ---------------------------------------------------------------------------

ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"


def call_anthropic(
    *,
    api_key: str,
    model: str,
    prompt: str,
    image_paths: list[str],
    temperature: float = 0.7,
    timeout: float = 120.0,
    max_tokens: int = 4096,
    verbose: bool = False,
    max_retries: int = 4,
) -> LLMResult:
    """Chiama l'endpoint Messages di Anthropic con testo + immagini base64."""
    key = _resolve_key(ANTHROPIC, api_key)

    content: list[dict] = [{"type": "text", "text": prompt}]
    for p in image_paths:
        mime, b64 = _read_image_b64(p)
        content.append({"type": "text", "text": f"\n[{_image_label(p)}]"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": b64},
        })

    payload = {
        "model": model,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "messages": [{"role": "user", "content": content}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    }
    _log(verbose, f"POST {ANTHROPIC_ENDPOINT} (modello={model}, immagini={len(image_paths)})")
    t0 = time.monotonic()
    data = _http_post_json(ANTHROPIC_ENDPOINT, payload, headers, timeout, verbose=verbose, max_retries=max_retries)
    _log(verbose, f"  risposta ricevuta in {time.monotonic() - t0:.1f}s")

    blocks = data.get("content") or []
    texts = [b["text"] for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
    text = "\n".join(texts).strip()
    if not text:
        raise LLMError("Anthropic ha risposto senza blocchi di testo.")
    return LLMResult(text=text, provider=ANTHROPIC, model=model, raw=data)


# ---------------------------------------------------------------------------
# Provider: OpenAI  (Chat Completions)  -- pronto per il roadmap multi-provider
# ---------------------------------------------------------------------------

OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"


def call_openai(
    *,
    api_key: str,
    model: str,
    prompt: str,
    image_paths: list[str],
    temperature: float = 0.7,
    timeout: float = 120.0,
    verbose: bool = False,
    max_retries: int = 4,
) -> LLMResult:
    """Chiama Chat Completions di OpenAI con testo + immagini (data URI)."""
    key = _resolve_key(OPENAI, api_key)

    content: list[dict] = [{"type": "text", "text": prompt}]
    for p in image_paths:
        mime, b64 = _read_image_b64(p)
        content.append({"type": "text", "text": f"\n[{_image_label(p)}]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })

    payload = {
        "model": model,
        "temperature": float(temperature),
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content}],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    _log(verbose, f"POST {OPENAI_ENDPOINT} (modello={model}, immagini={len(image_paths)})")
    t0 = time.monotonic()
    data = _http_post_json(OPENAI_ENDPOINT, payload, headers, timeout, verbose=verbose, max_retries=max_retries)
    _log(verbose, f"  risposta ricevuta in {time.monotonic() - t0:.1f}s")

    choices = data.get("choices") or []
    if not choices:
        raise LLMError("OpenAI non ha restituito 'choices'.")
    text = ((choices[0].get("message") or {}).get("content") or "").strip()
    if not text:
        raise LLMError("OpenAI ha risposto con un contenuto vuoto.")
    return LLMResult(text=text, provider=OPENAI, model=model, raw=data)


# ---------------------------------------------------------------------------
# Dispatcher unico
# ---------------------------------------------------------------------------

_PROVIDERS = {
    GEMINI:    call_gemini,
    ANTHROPIC: call_anthropic,
    OPENAI:    call_openai,
}


def call_llm(
    *,
    provider: str,
    api_key: str,
    model: str,
    prompt: str,
    image_paths: Optional[list[str]] = None,
    temperature: float = 0.7,
    timeout: float = 120.0,
    verbose: bool = False,
    max_retries: int = 4,
) -> LLMResult:
    """
    Punto d'ingresso unico per gli operatori: instrada al provider giusto.

    Args:
        provider:    una tra GEMINI / ANTHROPIC / OPENAI.
        api_key:     chiave API (vuota = usa la variabile d'ambiente).
        model:       identificatore del modello (vuoto = default del provider).
        prompt:      prompt testuale completo (gia' comprensivo del JSON scena).
        image_paths: lista di PNG da allegare (puo' essere vuota).
        temperature: creativita' del modello.
        timeout:     timeout di rete in secondi.
        verbose:     se True, stampa l'avanzamento nella console di sistema.

    Returns:
        LLMResult con il testo grezzo della risposta.
    """
    fn = _PROVIDERS.get(provider)
    if fn is None:
        raise LLMError(f"Provider non supportato: {provider!r}")
    model = (model or "").strip() or DEFAULT_MODELS[provider]
    _log(verbose, f"call_llm: provider={provider} modello={model}")
    return fn(
        api_key=api_key,
        model=model,
        prompt=prompt,
        image_paths=list(image_paths or []),
        temperature=temperature,
        timeout=timeout,
        verbose=verbose,
        max_retries=max_retries,
    )