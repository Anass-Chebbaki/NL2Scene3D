# gui/core/pipeline_runner.py
"""
Launches and monitors the Blender pipeline subprocess.
Emits log lines and status updates via callbacks so the GUI stays
fully responsive (runs in a daemon thread).
"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

from nl2scene3d.gui.core.config_bridge import GUIConfig


class PipelineRunner:
    """
    Manages the lifecycle of a single pipeline execution.

    All callbacks are called from the worker thread; callers must
    marshal to the main thread if they touch GUI widgets (use
    widget.after(0, callback) in Tkinter).
    """

    def __init__(
        self,
        config: GUIConfig,
        on_log: Callable[[str, str], None],
        on_image: Callable[[Path], None],
        on_finished: Callable[[bool, Optional[str]], None],
    ) -> None:
        """
        Args:
            config:      Current GUI configuration.
            on_log:      Called with (level, message) for each log line.
            on_image:    Called with Path whenever a new PNG is detected.
            on_finished: Called with (success, error_message) on exit.
        """
        self._config = config
        self._on_log = on_log
        self._on_image = on_image
        self._on_finished = on_finished

        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Parameters set just before launch
        self.blend_file: Optional[Path] = None
        self.scene_name: str = ""
        self.output_dir: Optional[Path] = None
        self.seed: int = 0
        self.skip_vision: bool = False
        self.max_objects: int = 20
        self.model_override: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the pipeline in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Request graceful termination and kill the subprocess."""
        self._stop_event.set()
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_command(self) -> list[str]:
        cfg = self._config
        script = str(cfg.project_root / "scripts" / "run_pipeline.py")
        output_dir = str(self.output_dir or (cfg.outputs_dir / self.scene_name))
        blend = str(self.blend_file) if self.blend_file else ""

        cmd = [
            cfg.blender_executable,
            "--background",
        ]
        if blend:
            cmd += [blend]
        cmd += [
            "--python", script,
            "--",
            "--scene-name", self.scene_name,
            "--output-dir", output_dir,
            "--log-level", cfg.log_level,
            "--max-objects", str(self.max_objects),
            "--seed", str(self.seed),
            "--min-quality-score", str(self._config.min_quality_score),
            "--good-quality-score", str(self._config.good_quality_score),
        ]
        if self.skip_vision:
            cmd.append("--skip-vision")
        if self.model_override:
            cmd += ["--model", self.model_override]
        return cmd

    def _run(self) -> None:
        cmd = self._build_command()
        self._on_log("INFO", "Launching: " + " ".join(cmd))

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            self._on_log("ERROR", f"Blender executable not found: {exc}")
            self._on_finished(False, str(exc))
            return
        except Exception as exc:
            self._on_log("ERROR", f"Failed to launch pipeline: {exc}")
            self._on_finished(False, str(exc))
            return

        seen_images: set[Path] = set()

        for raw_line in self._process.stdout:  # type: ignore[union-attr]
            if self._stop_event.is_set():
                break

            line = raw_line.rstrip()
            if not line:
                continue

            level = self._classify_level(line)
            self._on_log(level, line)

            # Detect newly written PNG files mentioned in log
            self._detect_new_images(line, seen_images)

        self._process.wait()
        success = self._process.returncode == 0 and not self._stop_event.is_set()

        # Final image scan after process ends
        self._scan_output_dir(seen_images)

        error_msg = None if success else f"Pipeline exited with code {self._process.returncode}"
        self._on_finished(success, error_msg)

    @staticmethod
    def _classify_level(line: str) -> str:
        upper = line.upper()
        if "[ERROR]" in upper or "ERROR:" in upper or "CRITICAL" in upper:
            return "ERROR"
        if "[WARNING]" in upper or "WARNING:" in upper:
            return "WARNING"
        if "[DEBUG]" in upper:
            return "DEBUG"
        return "INFO"

    def _detect_new_images(self, line: str, seen: set[Path]) -> None:
        """Look for .png paths in the log line."""
        import re
        matches = re.findall(r'[^\s"\']+\.png', line, re.IGNORECASE)
        for match in matches:
            p = Path(match)
            if not p.is_absolute():
                output_dir = self.output_dir or (self._config.outputs_dir / self.scene_name)
                p = output_dir / p.name
            if p.exists() and p not in seen:
                seen.add(p)
                self._on_image(p)

    def _scan_output_dir(self, seen: set[Path]) -> None:
        output_dir = self.output_dir or (self._config.outputs_dir / self.scene_name)
        if output_dir.exists():
            for png in sorted(output_dir.glob("*.png")):
                if png not in seen:
                    seen.add(png)
                    self._on_image(png)