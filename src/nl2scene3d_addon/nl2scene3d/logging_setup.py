# src/nl2scene3d/logging_setup.py
"""
Configurazione centralizzata del logging per NL2Scene3D.

Fornisce una funzione di setup unica da richiamare all'avvio di ogni
script o modulo principale, garantendo formattazione coerente
in tutta la pipeline.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from nl2scene3d.config import LoggingConfig

DEFAULT_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATEFMT: str = "%Y-%m-%d %H:%M:%S"

# Dimensione massima del file di log prima della rotazione (10 MB).
LOG_FILE_MAX_BYTES: int = 10 * 1024 * 1024

# Numero di file di backup da mantenere dopo la rotazione.
LOG_FILE_BACKUP_COUNT: int = 3

# Librerie esterne il cui livello di logging viene alzato a WARNING
# per ridurre il rumore nei log applicativi.
_NOISY_LIBRARIES: tuple[str, ...] = ("urllib3", "httpx", "google", "PIL")


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    logger_name: Optional[str] = None,
    config: Optional["LoggingConfig"] = None,
) -> logging.Logger:
    """
    Configura il sistema di logging per la pipeline.

    I parametri espliciti hanno precedenza sui valori presenti in config.
    La priorita' e': argomento esplicito > config > default.

    Args:
        level: Override del livello di logging (es. 'DEBUG', 'INFO').
        log_file: Override del percorso al file di log.
        logger_name: Nome del logger da restituire. Se None, restituisce
                     il logger root.
        config: Configurazione completa del logging (LoggingConfig).

    Returns:
        Logger configurato pronto per l'uso.
    """
    log_level_str: str = level or (config.level if config else "INFO")
    log_format: str = config.format if config else DEFAULT_FORMAT
    log_datefmt: str = config.datefmt if config else DEFAULT_DATEFMT

    effective_log_file: Optional[Path] = log_file
    if effective_log_file is None and config is not None and config.write_to_file:
        effective_log_file = config.log_file

    numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)
    formatter = logging.Formatter(fmt=log_format, datefmt=log_datefmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Rimuove gli handler precedenti per evitare duplicati in caso di
    # chiamate multiple a setup_logging durante la stessa sessione.
    root_logger.handlers.clear()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(numeric_level)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    if effective_log_file is not None:
        try:
            effective_log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(effective_log_file),
                maxBytes=LOG_FILE_MAX_BYTES,
                backupCount=LOG_FILE_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except OSError as exc:
            print(  # noqa: T201
                f"WARNING: Impossibile inizializzare il log su file "
                f"'{effective_log_file}': {exc}",
                file=sys.stderr,
            )

    for lib in _NOISY_LIBRARIES:
        logging.getLogger(lib).setLevel(logging.WARNING)

    target_logger = (
        logging.getLogger(logger_name) if logger_name else root_logger
    )

    target_logger.debug(
        "Logging configurato. Livello: %s, File: %s.",
        log_level_str.upper(),
        str(effective_log_file) if effective_log_file else "non attivo",
    )

    return target_logger