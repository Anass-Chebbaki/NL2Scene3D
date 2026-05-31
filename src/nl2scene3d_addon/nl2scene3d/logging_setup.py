# nl2scene3d/logging_setup.py
"""
Centralized logging configuration for NL2Scene3D.

Provides a single setup function to call at the entry point of any script
or main module, ensuring a consistent format across the entire pipeline.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from nl2scene3d.config import LoggingConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_FORMAT:      str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATEFMT:     str = "%Y-%m-%d %H:%M:%S"
LOG_FILE_MAX_BYTES:  int = 10 * 1024 * 1024   # Rotate at 10 MB
LOG_FILE_BACKUP_COUNT: int = 3                 # Keep up to 3 rotated files

# Third-party libraries whose log level is raised to WARNING to reduce noise.
_NOISY_LIBRARIES: tuple[str, ...] = ("urllib3", "httpx", "google", "PIL")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logging(
    level:       Optional[str]            = None,
    log_file:    Optional[Path]           = None,
    logger_name: Optional[str]            = None,
    config:      Optional["LoggingConfig"] = None,
) -> logging.Logger:
    """
    Configures the logging system for the pipeline.

    Explicit arguments take precedence over values in `config`, which in
    turn take precedence over built-in defaults. Priority:
      explicit argument > config value > default.

    Args:
        level:       Override for the log level (e.g. 'DEBUG', 'INFO').
        log_file:    Override for the log-file path.
        logger_name: Name of the logger to return. Returns the root logger
                     when None.
        config:      Full LoggingConfig object (optional).

    Returns:
        A configured Logger ready to use.
    """
    log_level_str   = level or (config.level  if config else "INFO")
    log_format      = config.format  if config else DEFAULT_FORMAT
    log_datefmt     = config.datefmt if config else DEFAULT_DATEFMT

    effective_log_file = log_file
    if effective_log_file is None and config is not None and config.write_to_file:
        effective_log_file = config.log_file

    numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)
    formatter     = logging.Formatter(fmt=log_format, datefmt=log_datefmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers to avoid duplicate output on repeated calls.
    root_logger.handlers.clear()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(numeric_level)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # Optional rotating file handler.
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
            print(
                f"WARNING: Cannot initialize log file '{effective_log_file}': {exc}",
                file=sys.stderr,
            )

    # Suppress verbose output from known noisy libraries.
    for lib in _NOISY_LIBRARIES:
        logging.getLogger(lib).setLevel(logging.WARNING)

    target_logger = (
        logging.getLogger(logger_name) if logger_name else root_logger
    )
    target_logger.debug(
        "Logging configured. Level: %s, File: %s.",
        log_level_str.upper(),
        str(effective_log_file) if effective_log_file else "disabled",
    )
    return target_logger