# nl2scene3d/gemini_client.py
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
        
        logger.debug(f"Raw response from Gemini (first 500 chars):\n{text[:500]}")
        
        # Strategia 1: parse diretto
        try:
            result = json.loads(text)
            logger.debug(f"✓ Direct JSON parse successful. Type: {type(result).__name__}")
            return result
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parse failed: {e}")
            pass

        # Strategia 2: estrazione di blocchi ```json ... ``` (piu' robusta)
        json_block_pattern = re.compile(
            r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE
        )
        match = json_block_pattern.search(text)
        if match:
            try:
                extracted = match.group(1).strip()
                logger.debug(f"Found JSON block in markdown code fence. Content:\n{extracted[:300]}")
                result = json.loads(extracted)
                logger.debug(f"✓ Markdown JSON parse successful. Type: {type(result).__name__}")
                return result
            except json.JSONDecodeError as e:
                logger.debug(f"Markdown JSON parse failed: {e}")
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
                logger.debug(f"Extracted JSON substring (brute force): {json_str[:300]}")
                try:
                    result = json.loads(json_str)
                    logger.debug(f"✓ Brute force JSON parse successful. Type: {type(result).__name__}")
                    return result
                except json.JSONDecodeError as e:
                    logger.debug(f"Brute force JSON parse failed: {e}")
                    # Se fallisce ancora, proviamo a pulire i commenti se presenti
                    json_str_clean = re.sub(r'//.*?\n|/\*.*?\*/', '', json_str, flags=re.DOTALL)
                    try:
                        result = json.loads(json_str_clean)
                        logger.debug(f"✓ Cleaned JSON parse successful. Type: {type(result).__name__}")
                        return result
                    except json.JSONDecodeError as e2:
                        logger.debug(f"Cleaned JSON parse failed: {e2}")
                        pass

        # Strategia 4: Se sembra troncato (mancano le chiusure), proviamo a chiuderlo manualmente
        if start_idx != -1:
            logger.warning("JSON appears truncated. Attempting automatic closure.")
            # Conta le parentesi aperte/chiuse
            open_braces = text.count('{')
            close_braces = text.count('}')
            open_brackets = text.count('[')
            close_brackets = text.count(']')
            
            repaired_text = text[start_idx:]
            if open_brackets > close_brackets:
                repaired_text += ']' * (open_brackets - close_brackets)
            if open_braces > close_braces:
                repaired_text += '}' * (open_braces - close_braces)
            
            logger.debug(f"Repaired JSON: {repaired_text[:300]}")
            try:
                result = json.loads(repaired_text)
                logger.debug(f"✓ Repaired JSON parse successful. Type: {type(result).__name__}")
                return result
            except json.JSONDecodeError as e:
                logger.debug(f"Repaired JSON parse failed: {e}")
                pass

        logger.error(f"FULL GEMINI RESPONSE:\n{text}")
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
        
        Nota: response_mime_type="application/json" potrebbe non essere supportato
        da tutti i modelli. Se fallisce, ritentiamo senza questo vincolo.
        """
        gen_config_dict = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
            **(config_override or {}),
        }
        
        # Prova con response_mime_type="application/json" solo come primo tentativo
        use_json_mime = True
        
        last_exception: Exception = GeminiClientError("Nessun tentativo eseguito.")

        for attempt in range(self.config.max_retries):
            try:
                config_dict = gen_config_dict.copy()
                if use_json_mime and attempt < 1:  # Solo primo tentativo
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
                    raise GeminiParsingError("Risposta del modello vuota.")
                return response.text

            except Exception as exc:
                exc_str = str(exc).lower()
                
                # Se è un errore "mime type not supported", ritenta senza
                if "mime type" in exc_str and use_json_mime and attempt < 1:
                    logger.warning(
                        "response_mime_type='application/json' non supportato dal modello %s. "
                        "Ritento senza questo vincolo.",
                        model_name,
                    )
                    use_json_mime = False
                    continue
                
                # 429 (Rate Limit) o 503 (Model Overloaded/Unavailable)
                if any(err in exc_str for err in ("429", "quota", "exhausted")):
                    last_exception = exc
                    # Backoff più aggressivo per la quota: 10, 20, 40 secondi
                    wait_seconds = 10 * (2 ** attempt)
                    logger.warning(
                        "Quota API esaurita (429). Attesa di %d secondi (tentativo %d/%d).",
                        wait_seconds,
                        attempt + 1,
                        self.config.max_retries,
                    )
                    if attempt < self.config.max_retries - 1:
                        time.sleep(wait_seconds)
                    else:
                        raise GeminiRateLimitError(f"Quota esaurita dopo i retry: {exc}") from exc
                elif any(err in exc_str for err in ("503", "unavailable", "demand")):
                    last_exception = exc
                    wait_seconds = 2 ** (attempt + 1)
                    logger.warning(
                        "API Gemini temporaneamente non disponibile (503, tentativo %d/%d). "
                        "Attesa di %d secondi.",
                        attempt + 1,
                        self.config.max_retries,
                        wait_seconds,
                    )
                    if attempt < self.config.max_retries - 1:
                        time.sleep(wait_seconds)
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
            "Chiamata testuale a Gemini (%s). System prompt length: %d, User prompt length: %d",
            model_name,
            len(system_prompt),
            len(user_prompt),
        )
        
        try:
            raw_response = self._call_with_retry(
                model_name=model_name,
                contents=user_prompt,
                system_prompt=system_prompt,
            )
            logger.debug(f"Raw response from model:\n{raw_response[:1000]}")
            
            parsed = self._extract_json_from_response(raw_response)
            logger.info(f"Successfully parsed Gemini response. Type: {type(parsed).__name__}, entries: {len(parsed) if isinstance(parsed, (list, dict)) else 'N/A'}")
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

    def call_vision_multi(
        self,
        image_paths: list[Path],
        user_prompt: str,
        use_fallback: bool = False,
    ) -> dict | list:
        """
        Esegue una chiamata vision al modello con piu' immagini allegate.
        
        Ogni immagine viene passata come elemento separato nella lista contents,
        seguita dal prompt testuale. Questo permette al modello di analizzare
        viste multiple della stessa scena in un singolo contesto.

        Args:
            image_paths: Lista di percorsi alle immagini da analizzare.
            user_prompt: Prompt testuale per l'analisi.
            use_fallback: Se True, usa il modello fallback.

        Returns:
            Output JSON parsato dal modello.
        """
        for path in image_paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"Immagine per la chiamata vision non trovata: {path}"
                )

        model_name = (
            self.config.model_fallback if use_fallback else self.config.model_primary
        )

        logger.info(
            "Chiamata vision multi-immagine a Gemini (%s). Immagini: %d.",
            model_name,
            len(image_paths),
        )

        try:
            import PIL.Image
            contents: list = []
            for i, path in enumerate(image_paths):
                img = PIL.Image.open(path)
                contents.append(img)
            contents.append(user_prompt)
            return self._call_vision_internal(model_name, contents, use_fallback)
        except Exception as exc:
            logger.error("Errore nella chiamata vision multi: %s", exc)
            raise GeminiClientError(f"Errore vision multi: {exc}") from exc