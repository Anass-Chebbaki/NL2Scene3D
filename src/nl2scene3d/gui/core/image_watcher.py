# gui/core/image_watcher.py
"""
Polls an output directory for new PNG files and fires a callback.
Used as a secondary detection mechanism alongside log-line parsing.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional


class ImageWatcher:
    """Periodically scans a directory for new PNG files."""

    def __init__(
        self,
        directory: Path,
        on_new_image: Callable[[Path], None],
        interval: float = 1.5,
    ) -> None:
        self._dir = directory
        self._callback = on_new_image
        self._interval = interval
        self._seen: set[Path] = set()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def reset(self) -> None:
        self._seen.clear()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._scan()
            time.sleep(self._interval)

    def _scan(self) -> None:
        if not self._dir.exists():
            return
        for png in sorted(self._dir.glob("*.png"), key=lambda p: p.stat().st_mtime):
            if png not in self._seen:
                self._seen.add(png)
                self._callback(png)