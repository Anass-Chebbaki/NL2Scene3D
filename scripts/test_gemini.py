# scripts/test_gemini.py
"""
Script di verifica rapida della connessione alle API Gemini.

Esegue una chiamata di test minimale per verificare che la chiave API
sia valida e che il modello risponda correttamente.

Utilizzo:
    python scripts/test_gemini.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from nl2scene3d.config import get_config
from nl2scene3d.gemini_client import GeminiClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nl2scene3d.test_gemini")


def main() -> None:
    """Esegue il test di connessione a Gemini."""
    logger.info("Verifica connessione a Gemini...")

    try:
        config = get_config()
    except EnvironmentError as exc:
        logger.critical("Configurazione non valida: %s", exc)
        sys.exit(1)

    client = GeminiClient(config.gemini)

    system_prompt = (
        "You are a JSON generator. Always respond with valid JSON only."
    )
    user_prompt = (
        'Return a JSON object with a single key "status" and value "ok". '
        "Do not include any other text."
    )

    logger.info("Invio chiamata testuale a Gemini...")
    result = client.call_text(system_prompt, user_prompt)
    logger.info("Risposta ricevuta: %s", json.dumps(result))

    if isinstance(result, dict) and result.get("status") == "ok":
        logger.info("Test superato. API Gemini funzionante.")
    else:
        logger.warning("Risposta ricevuta ma formato inatteso: %s", result)
        sys.exit(1)


if __name__ == "__main__":
    main()