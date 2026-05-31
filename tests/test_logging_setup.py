"""
Test unitari per il modulo di configurazione del logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from nl2scene3d.logging_setup import setup_logging
from nl2scene3d.config import LoggingConfig

def test_setup_logging_basic() -> None:
    """Verifica il setup base del logging su console."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_root = MagicMock()
        mock_get_logger.return_value = mock_root
        
        logger = setup_logging(level="DEBUG", logger_name="test_logger")
        
        # Verifica che il livello sia stato impostato correttamente sul root (o dove richiesto)
        # setup_logging chiama logging.getLogger() per il root
        assert logger is not None
        mock_get_logger.assert_any_call()

def test_setup_logging_with_file(tmp_path: Path) -> None:
    """Verifica la creazione del file handler."""
    log_file = tmp_path / "test.log"
    config = LoggingConfig(
        level="INFO",
        format="%(message)s",
        datefmt="%Y-%m-%d",
        write_to_file=True,
        log_file=log_file
    )
    
    logger = setup_logging(config=config)
    
    # Verifica che il file sia stato creato (il logger root avra' un FileHandler)
    root_logger = logging.getLogger()
    has_file_handler = any(
        isinstance(h, logging.handlers.RotatingFileHandler) for h in root_logger.handlers
    )
    assert has_file_handler
    assert log_file.parent.exists()

def test_setup_logging_error_handling(tmp_path: Path) -> None:
    """Verifica la gestione degli errori (es. permessi negati)."""
    # Usiamo un percorso non scrivibile (es. radice su linux o una stringa invalida)
    invalid_file = Path("/invalid/path/to/log.log")
    if os_is_windows():
        invalid_file = Path("Z:/invalid/path/log.log") # Supponendo Z non esista
        
    config = LoggingConfig(
        level="INFO",
        write_to_file=True,
        log_file=invalid_file
    )
    
    # Non deve crashare
    with patch("sys.stderr.write"):
        logger = setup_logging(config=config)
    assert logger is not None

def os_is_windows() -> bool:
    import platform
    return platform.system() == "Windows"
