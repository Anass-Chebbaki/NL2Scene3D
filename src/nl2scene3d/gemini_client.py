# src/nl2scene3d/gemini_client.py
"""
Client per l'interazione con le API Google Gemini.

Gestisce:
- Chiamate testuali per il riordino della scena
- Chiamate vision per il feedback visivo
- Retry automatico con backoff esponenziale
- Fallback al modello alternativo in caso di errori persistenti
- Parsing robusto dell'output JSON del modello
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


class GeminiClientError(Exception):
    """Eccezione base per errori del client Gemini."""


class GeminiParsingError(GeminiClientError):
    """Sollevata quando il parsing della risposta JSON fallisce."""


class GeminiRateLimitError(GeminiClientError):
    """Sollevata quando il rate limit viene raggiunto in modo persistente."""


class GeminiClient:
    """
    Client per le chiamate alle API Google Gemini.

    Implementa retry con backoff esponenziale e fallback automatico
    al modello secondario in caso di errori persistenti sul modello primario.
    """

    def __init__(self, config: GeminiConfig) -> None:
        """
        Inizializza il client e configura la connessione.

        Args:
            config: Oggetto di configurazione Gemini.
        """
        self.config = config
        self._client = genai.Client(
            api_key=config.api_key,
            http_options={'timeout': config.timeout_seconds * 1000}
        )
        logger.info(
            "GeminiClient inizializzato con successo (timeout: %ds). "
            "Modello primario: %s, fallback: %s",
            config.timeout_seconds,
            config.model_primary,
            config.model_fallback,
        )

    def _extract_json_from_response(self, text: str) -> dict | list:
        """
        Estrae e parsa il JSON dalla risposta testuale del modello.
        """
        # Pulizia preliminare: rimuove eventuale testo prima del primo { o [
        text = text.strip()
        
        # Strategia 1: parse diretto
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategia 2: estrazione di blocchi ```json ... ``` (piu' robusta)
        json_block_pattern = re.compile(
            r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE
        )
        match = json_block_pattern.search(text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Strategia 3: ricerca della struttura più esterna { } o [ ]
        # Gestisce il caso in cui il modello aggiunge spiegazioni prima o dopo
        start_idx_dict = text.find('{')
        start_idx_list = text.find('[')
        
        start_idx = -1
        if start_idx_dict != -1 and (start_idx_list == -1 or start_idx_dict < start_idx_list):
            start_idx = start_idx_dict
            end_char = '}'
        elif start_idx_list != -1:
            start_idx = start_idx_list
            end_char = ']'
            
        if start_idx != -1:
            end_idx = text.rfind(end_char)
            if end_idx > start_idx:
                json_str = text[start_idx:end_idx + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        raise GeminiParsingError(
            "Impossibile estrarre JSON valido dalla risposta del modello. "
            f"La risposta potrebbe essere stata troncata o contenere errori di sintassi. "
            f"Anteprima: {text[:200]}..."
        )

    def _call_with_retry(
        self,
        model_name: str,
        contents: Any,
        system_prompt: Optional[str] = None,
        config_override: Optional[dict] = None,
    ) -> str:
        """
        Esegue una chiamata al modello con retry e backoff esponenziale.
        """
        gen_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
            response_mime_type="application/json",
            **(config_override or {}),
        )

        last_exception: Exception = GeminiClientError("Nessun tentativo eseguito.")

        for attempt in range(self.config.max_retries):
            try:
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=gen_config,
                )
                if not response.text:
                    raise GeminiParsingError("Risposta del modello vuota.")
                return response.text

            except Exception as exc:
                exc_str = str(exc).lower()
                # 429 (Rate Limit) o 503 (Model Overloaded/Unavailable)
                if any(err in exc_str for err in ("429", "quota", "exhausted", "503", "unavailable", "demand")):
                    last_exception = exc
                    wait_seconds = 2 ** (attempt + 1)
                    logger.warning(
                        "API Gemini temporaneamente non disponibile (%s, tentativo %d/%d). "
                        "Attesa di %d secondi.",
                        exc,
                        attempt + 1,
                        self.config.max_retries,
                        wait_seconds,
                    )
                    if attempt < self.config.max_retries - 1:
                        time.sleep(wait_seconds)
                    else:
                        # Se abbiamo esaurito i retry per un errore temporaneo,
                        # solleviamo GeminiRateLimitError per innescare il fallback.
                        raise GeminiRateLimitError(
                            f"Modello non disponibile dopo i retry: {exc}"
                        ) from exc
                elif any(err in exc_str for err in ("400", "invalid", "401", "403")):
                    logger.error("Errore API permanente (Client Error): %s", exc)
                    raise GeminiClientError(f"Errore API permanente: {exc}") from exc
                else:
                    # Altri errori (500, 502, 504 o errori di rete)
                    last_exception = exc
                    logger.error("Errore API Gemini imprevisto (tentativo %d/%d): %s", attempt + 1, self.config.max_retries, exc)
                    if attempt < self.config.max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        # Proviamo comunque il fallback per qualsiasi errore persistente
                        # che non sia un errore 400 del client.
                        raise GeminiRateLimitError(f"Errore API persistente: {exc}") from exc

        raise GeminiClientError(f"Tentativi esauriti. Ultimo errore: {last_exception}")

    def call_text(
        self,
        system_prompt: str,
        user_prompt: str,
        use_fallback: bool = False,
    ) -> dict | list:
        """
        Esegue una chiamata testuale al modello e restituisce il JSON parsato.
        """
        model_name = (
            self.config.model_fallback if use_fallback else self.config.model_primary
        )
        logger.info(
            "Chiamata testuale a Gemini (%s).",
            model_name,
        )
        
        try:
            raw_response = self._call_with_retry(
                model_name=model_name,
                contents=user_prompt,
                system_prompt=system_prompt,
            )
            parsed = self._extract_json_from_response(raw_response)
            return parsed
        except GeminiRateLimitError:
            if not use_fallback:
                logger.warning("Switching to fallback model due to rate limit.")
                return self.call_text(system_prompt, user_prompt, use_fallback=True)
            raise

    def _call_vision_internal(self, model_name: str, contents: list, use_fallback: bool) -> dict | list:
        try:
            raw_response = self._call_with_retry(
                model_name=model_name,
                contents=contents,
            )
            return self._extract_json_from_response(raw_response)
        except GeminiRateLimitError:
            if not use_fallback:
                return self._call_vision_internal(self.config.model_fallback, contents, use_fallback=True)
            raise

    def call_vision(
        self,
        image_path: Path,
        user_prompt: str,
        use_fallback: bool = False,
    ) -> dict | list:
        """
        Esegue una chiamata vision al modello con un'immagine allegata.
        """
        if not image_path.exists():
            raise FileNotFoundError(
                f"Immagine per la chiamata vision non trovata: {image_path}"
            )

        model_name = (
            self.config.model_fallback if use_fallback else self.config.model_primary
        )

        logger.info("Chiamata vision a Gemini (%s).", model_name)

        try:
            import PIL.Image
            img = PIL.Image.open(image_path)
            contents = [img, user_prompt]
            return self._call_vision_internal(model_name, contents, use_fallback)
        except Exception as exc:
            logger.error("Errore nella chiamata vision: %s", exc)
            raise GeminiClientError(f"Errore vision: {exc}") from exc