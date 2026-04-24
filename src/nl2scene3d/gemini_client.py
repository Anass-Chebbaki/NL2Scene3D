# src/nl2scene3d/gemini_client.py
"""
Client per l'interazione con le API Google Gemini.

Gestisce:
- Chiamate testuali per il riordino della scena (Step 7)
- Chiamate vision per il feedback visivo (Step 9)
- Retry automatico con backoff esponenziale
- Fallback al modello alternativo in caso di errori
- Parsing robusto dell'output JSON dell'LLM
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError, ResourceExhausted

from nl2scene3d.config import GeminiConfig

logger = logging.getLogger(__name__)


class GeminiClientError(Exception):
    """Eccezione base per errori del client Gemini."""


class GeminiParsingError(GeminiClientError):
    """Eccezione sollevata quando il parsing della risposta JSON fallisce."""


class GeminiRateLimitError(GeminiClientError):
    """Eccezione sollevata quando il rate limit viene raggiunto."""


class GeminiClient:
    """
    Client astratto per le chiamate alle API Google Gemini.

    Implementa retry con backoff esponenziale e fallback automatico
    al modello secondario in caso di errori persistenti.

    Attributes:
        config: Configurazione Gemini (API key, modelli, limiti).
        _primary_model: Istanza del modello primario.
        _fallback_model: Istanza del modello fallback.
    """

    def __init__(self, config: GeminiConfig) -> None:
        """
        Inizializza il client e configura i modelli.

        Args:
            config: Oggetto di configurazione Gemini.
        """
        self.config = config
        genai.configure(api_key=config.api_key)

        self._primary_model = genai.GenerativeModel(config.model_primary)
        self._fallback_model = genai.GenerativeModel(config.model_fallback)

        logger.info(
            "GeminiClient inizializzato. Modello primario: %s, fallback: %s",
            config.model_primary,
            config.model_fallback,
        )

    def _extract_json_from_response(self, text: str) -> dict | list:
        """
        Estrae e parsa il JSON dalla risposta testuale dell'LLM.

        L'LLM spesso include testo aggiuntivo prima e dopo il JSON.
        Questo metodo prova diverse strategie di estrazione.

        Args:
            text: Testo grezzo della risposta dell'LLM.

        Returns:
            Struttura dati Python parsata dal JSON.

        Raises:
            GeminiParsingError: Se nessuna strategia di parsing ha successo.
        """
        # Strategia 1: parse diretto della risposta pulita
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Strategia 2: estrazione di blocchi ```json ... ```
        json_block_pattern = re.compile(
            r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE
        )
        match = json_block_pattern.search(text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Strategia 3: ricerca del primo { ... } o [ ... ] nel testo
        brace_pattern = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)
        match = brace_pattern.search(text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        raise GeminiParsingError(
            f"Impossibile estrarre JSON valido dalla risposta del modello. "
            f"Risposta ricevuta (primi 500 caratteri): {text[:500]}"
        )

    def _call_with_retry(
        self,
        model: genai.GenerativeModel,
        contents: list,
        generation_config: Optional[dict] = None,
    ) -> str:
        """
        Esegue una chiamata al modello con retry automatico.

        Usa backoff esponenziale: attende 2^attempt secondi tra un tentativo
        e il successivo.

        Args:
            model: Modello Gemini da usare.
            contents: Contenuti da inviare al modello.
            generation_config: Configurazione opzionale della generazione.

        Returns:
            Testo della risposta del modello.

        Raises:
            GeminiRateLimitError: Se il rate limit e' persistentemente raggiunto.
            GeminiClientError: Per altri errori non recuperabili.
        """
        gen_config = generation_config or {
            "temperature": 0.2,
            "max_output_tokens": 4096,
        }

        for attempt in range(self.config.max_retries):
            try:
                response = model.generate_content(
                    contents,
                    generation_config=gen_config,
                )
                return response.text

            except ResourceExhausted as exc:
                wait_seconds = 2 ** (attempt + 1)
                logger.warning(
                    "Rate limit raggiunto (tentativo %d/%d). "
                    "Attendo %d secondi prima di riprovare.",
                    attempt + 1,
                    self.config.max_retries,
                    wait_seconds,
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(wait_seconds)
                else:
                    raise GeminiRateLimitError(
                        "Rate limit persistente dopo tutti i tentativi."
                    ) from exc

            except GoogleAPIError as exc:
                logger.error(
                    "Errore API Google (tentativo %d/%d): %s",
                    attempt + 1,
                    self.config.max_retries,
                    exc,
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise GeminiClientError(
                        f"Errore API persistente: {exc}"
                    ) from exc

        # Non dovrebbe mai essere raggiunto, ma per sicurezza
        raise GeminiClientError("Tutti i tentativi esauriti senza risposta.")

    def call_text(
        self,
        system_prompt: str,
        user_prompt: str,
        use_fallback: bool = False,
    ) -> dict | list:
        """
        Esegue una chiamata testuale al modello e restituisce il JSON parsato.

        Corrisponde allo Step 7 della pipeline: riordino della scena
        tramite ragionamento testuale.

        Args:
            system_prompt: Prompt di sistema che contestualizza il task.
            user_prompt: Prompt utente con i dati della scena disordinata.
            use_fallback: Se True, usa il modello fallback invece del primario.

        Returns:
            Struttura JSON parsata dalla risposta del modello.

        Raises:
            GeminiClientError: In caso di errori API non recuperabili.
            GeminiParsingError: Se il JSON nella risposta non e' valido.
        """
        model = self._fallback_model if use_fallback else self._primary_model
        model_name = (
            self.config.model_fallback if use_fallback else self.config.model_primary
        )

        logger.info(
            "Chiamata testuale a Gemini (modello: %s). "
            "Lunghezza system_prompt: %d, user_prompt: %d",
            model_name,
            len(system_prompt),
            len(user_prompt),
        )

        # Composizione del messaggio nel formato atteso da Gemini
        contents = [
            {"role": "user", "parts": [system_prompt + "\n\n" + user_prompt]},
        ]

        try:
            raw_response = self._call_with_retry(model, contents)
            logger.debug("Risposta grezza ricevuta: %s", raw_response[:200])
            parsed = self._extract_json_from_response(raw_response)
            logger.info("JSON parsato con successo dalla risposta testuale.")
            return parsed

        except GeminiRateLimitError:
            if not use_fallback:
                logger.warning(
                    "Rate limit sul modello primario. Tentativo con modello fallback."
                )
                return self.call_text(system_prompt, user_prompt, use_fallback=True)
            raise

    def call_vision(
        self,
        image_path: Path,
        user_prompt: str,
        use_fallback: bool = False,
    ) -> dict | list:
        """
        Esegue una chiamata vision al modello con un'immagine allegata.

        Corrisponde allo Step 9 della pipeline: critica visiva del render
        per identificare imperfezioni nel layout riordinato.

        Args:
            image_path: Percorso all'immagine del render da analizzare.
            user_prompt: Prompt utente che descrive il task di critica.
            use_fallback: Se True, usa il modello fallback invece del primario.

        Returns:
            Struttura JSON con le correzioni suggerite dal modello.

        Raises:
            FileNotFoundError: Se il file immagine non esiste.
            GeminiClientError: In caso di errori API non recuperabili.
            GeminiParsingError: Se il JSON nella risposta non e' valido.
        """
        if not image_path.exists():
            raise FileNotFoundError(
                f"Immagine per la chiamata vision non trovata: {image_path}"
            )

        model = self._fallback_model if use_fallback else self._primary_model
        model_name = (
            self.config.model_fallback if use_fallback else self.config.model_primary
        )

        logger.info(
            "Chiamata vision a Gemini (modello: %s). Immagine: %s",
            model_name,
            image_path,
        )

        # Caricamento dell'immagine tramite le API File di Gemini
        uploaded_file = genai.upload_file(str(image_path))

        contents = [uploaded_file, user_prompt]

        try:
            raw_response = self._call_with_retry(model, contents)
            logger.debug(
                "Risposta vision grezza ricevuta: %s", raw_response[:200]
            )
            parsed = self._extract_json_from_response(raw_response)
            logger.info("JSON parsato con successo dalla risposta vision.")
            return parsed

        except GeminiRateLimitError:
            if not use_fallback:
                logger.warning(
                    "Rate limit sul modello primario. Tentativo con modello fallback."
                )
                return self.call_vision(image_path, user_prompt, use_fallback=True)
            raise