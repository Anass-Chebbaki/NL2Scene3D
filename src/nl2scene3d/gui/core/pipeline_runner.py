# gui/core/pipeline_runner.py
"""
Launches and monitors the Blender pipeline subprocess inside a QThread.

Signals are emitted to communicate with the GUI main thread safely —
NEVER touch any widget directly from this thread.

Signals
-------
log_emitted(level: str, message: str)
    Fires for every parsed log line.
image_detected(path: str)
    Fires whenever a new PNG render is detected in log output or on disk.
pipeline_finished(success: bool, error_msg: str)
    Fires once when the process exits (error_msg is "" on success).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

from nl2scene3d.gui.core.config_bridge import GUIConfig


class PipelineRunner(QThread):
    """
    Manages the lifecycle of a single pipeline execution.

    Usage::

        runner = PipelineRunner(config)
        runner.log_emitted.connect(log_panel.append)
        runner.image_detected.connect(viewer.add_image)
        runner.pipeline_finished.connect(on_finished)

        # Set params, then:
        runner.start()   # QThread.start() → calls run() in background
        runner.request_stop()
    """

    # ── Signals ────────────────────────────────────────────────────────────
    log_emitted: Signal = Signal(str, str)           # (level, message)
    image_detected: Signal = Signal(str)             # absolute path string
    pipeline_finished: Signal = Signal(bool, str)    # (success, error_msg)

    def __init__(self, config: GUIConfig, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._process: Optional[subprocess.Popen] = None
        self._stop_requested = False

        # Parameters set just before start()
        self.blend_file: Optional[Path] = None
        self.scene_name: str = ""
        self.output_dir: Optional[Path] = None
        self.seed: int = 0
        self.skip_vision: bool = False
        self.max_objects: int = 20
        self.model_override: str = ""

    # ── Public API ──────────────────────────────────────────────────────────

    def request_stop(self) -> None:
        """Signal the worker to stop and terminate the subprocess."""
        self._stop_requested = True
        proc = self._process
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    # ── QThread entry point ────────────────────────────────────────────────

    def run(self) -> None:
        """Called in the background thread by QThread.start()."""
        self._stop_requested = False
        cmd = self._build_command()
        self.log_emitted.emit("INFO", "Launching: " + " ".join(cmd))

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
            self.log_emitted.emit("ERROR", f"Blender executable not found: {exc}")
            self.pipeline_finished.emit(False, str(exc))
            return
        except Exception as exc:
            self.log_emitted.emit("ERROR", f"Failed to launch pipeline: {exc}")
            self.pipeline_finished.emit(False, str(exc))
            return

        seen_images: set[Path] = set()

        assert self._process.stdout is not None
        for raw_line in self._process.stdout:
            if self._stop_requested:
                break
            line = raw_line.rstrip()
            if not line:
                continue
            level = self._classify_level(line)
            self.log_emitted.emit(level, line)
            self._detect_new_images(line, seen_images)

        self._process.wait()
        success = self._process.returncode == 0 and not self._stop_requested

        # Final pass — pick up any images not mentioned in logs
        self._scan_output_dir(seen_images)

        if success:
            self.pipeline_finished.emit(True, "")
        else:
            msg = f"Pipeline exited with code {self._process.returncode}"
            self.pipeline_finished.emit(False, msg)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _build_command(self) -> list[str]:
        cfg = self._config
        script = str(cfg.project_root / "scripts" / "run_pipeline.py")
        output_dir = str(self.output_dir or (cfg.outputs_dir / self.scene_name))
        blend = str(self.blend_file) if self.blend_file else ""

        cmd = [cfg.blender_executable, "--background"]
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
            "--min-quality-score", str(cfg.min_quality_score),
            "--good-quality-score", str(cfg.good_quality_score),
        ]
        if self.skip_vision:
            cmd.append("--skip-vision")
        if self.model_override:
            cmd += ["--model", self.model_override]
        return cmd

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
        matches = re.findall(r'[^\s"\']+\.png', line, re.IGNORECASE)
        for match in matches:
            p = Path(match)
            if not p.is_absolute():
                output_dir = self.output_dir or (self._config.outputs_dir / self.scene_name)
                p = output_dir / p.name
            if p.exists() and p not in seen:
                seen.add(p)
                self.image_detected.emit(str(p))

    def _scan_output_dir(self, seen: set[Path]) -> None:
        output_dir = self.output_dir or (self._config.outputs_dir / self.scene_name)
        if output_dir.exists():
            for png in sorted(output_dir.glob("*.png")):
                if png not in seen:
                    seen.add(png)
                    self.image_detected.emit(str(png))