# gui/core/image_watcher.py
"""
Polls an output directory for new PNG files and emits a Qt signal.
Secondary detection mechanism alongside log-line parsing in PipelineRunner.
"""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal


class ImageWatcher(QThread):
    """
    Periodically scans a directory for new PNG files.

    Signals
    -------
    new_image_found(path: str)
        Emitted from the background thread — connect to a slot in the main
        thread to safely update the GUI.
    """

    new_image_found: Signal = Signal(str)

    def __init__(self, directory: Path, interval: float = 1.5, parent=None) -> None:
        super().__init__(parent)
        self._dir = directory
        self._interval = interval
        self._seen: set[Path] = set()
        self._running = False

    def stop(self) -> None:
        self._running = False

    def reset(self) -> None:
        self._seen.clear()

    def run(self) -> None:
        self._running = True
        while self._running:
            self._scan()
            time.sleep(self._interval)

    def _scan(self) -> None:
        if not self._dir.exists():
            return
        for png in sorted(self._dir.glob("*.png"), key=lambda p: p.stat().st_mtime):
            if png not in self._seen:
                self._seen.add(png)
                self.new_image_found.emit(str(png))