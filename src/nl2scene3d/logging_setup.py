"""
Configurazione centralizzata del logging per NL2Scene3D.

Fornisce una funzione di setup unica da chiamare all'avvio di ogni
script o modulo principale, garantendo formattazione coerente
in tutta la pipeline.

Il modulo supporta:
- Output su stdout (sempre attivo)
- Output su file rotante (opzionale, configurabile)
- Livello di logging configurabile per modulo
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

# Formato standard usato in tutta la pipeline
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT: str = "%Y-%m-%d %H:%M:%S"

# Dimensione massima del file di log prima della rotazione (10 MB)
LOG_FILE_MAX_BYTES: int = 10 * 1024 * 1024

# Numero di backup dei file di log da mantenere
LOG_FILE_BACKUP_COUNT: int = 3

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """
    Configura il sistema di logging per la pipeline.

    Imposta un handler su stdout e, opzionalmente, un handler su file
    con rotazione automatica. La formattazione e' standardizzata in
    tutta la pipeline.

    Deve essere chiamata una sola volta all'avvio dello script principale
    (run_pipeline.py o batch_pipeline.py).

    Args:
        level: Livello di logging come stringa (es. 'INFO', 'DEBUG').
               Case-insensitive.
        log_file: Percorso opzionale al file di log. Se None, il log
                  viene scritto solo su stdout.
        logger_name: Nome del logger da restituire. Se None, usa il
                     logger root.

    Returns:
        Logger configurato pronto per l'uso.

    Raises:
        ValueError: Se il livello fornito non e' valido.

    Example:
        logger = setup_logging(level="DEBUG", log_file=Path("logs/run.log"))
        logger.info("Pipeline avviata.")
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(
            f"Livello di logging non valido: '{level}'. "
            f"Valori validi: DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)

    # Configura il logger root per intercettare tutti i messaggi
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Rimuove handler preesistenti per evitare duplicati
    # (utile quando setup_logging viene chiamato piu' volte in test)
    root_logger.handlers.clear()

    # Handler stdout: sempre presente
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(numeric_level)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # Handler file: opzionale, con rotazione automatica
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file),
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Riduce la verbosita' di librerie esterne rumorose
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    target_logger = (
        logging.getLogger(logger_name) if logger_name else root_logger
    )

    target_logger.debug(
        "Logging configurato. Livello: %s, File: %s",
        level.upper(),
        str(log_file) if log_file else "non attivo",
    )

    return target_logger
