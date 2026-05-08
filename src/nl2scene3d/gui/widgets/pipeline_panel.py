# gui/widgets/pipeline_panel.py
"""
Pipeline control panel — scene selection, run / stop buttons,
and a status indicator with step progress.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Optional

import customtkinter as ctk

from gui.core.config_bridge import GUIConfig

_PIPELINE_STEPS = [
    "Extracting scene state",
    "Rendering original",
    "Randomizing scene",
    "Rendering randomized",
    "Calling model (reorder)",
    "Applying reordered state",
    "Rendering reordered",
    "Visual critique (vision model)",
    "Applying visual corrections",
    "Final render",
    "Computing metrics",
    "Done",
]

_STEP_KEYWORDS = {
    "Extracting":         0,
    "Render originale":   1,
    "original":           1,
    "Randomizzazione":    2,
    "Randomization":      2,
    "randomizing":        2,
    "Render scena dis":   3,
    "randomized":         3,
    "Chiamata LLM":       4,
    "Calling model":      4,
    "reorder":            4,
    "Applicazione coord": 5,
    "Applying reorder":   5,
    "Render":             6,
    "reordered":          6,
    "Critica visiva":     7,
    "Visual critique":    7,
    "vision":             7,
    "correzioni":         8,
    "corrections":        8,
    "finale":             9,
    "final":              9,
    "metriche":           10,
    "metrics":            10,
    "completata":         11,
    "completed":          11,
}


class PipelinePanel(ctk.CTkFrame):
    """Scene selector, progress tracker, and run/stop controls."""

    def __init__(
        self,
        master: ctk.CTkBaseClass,
        config: GUIConfig,
        on_run: Callable[[], None],
        on_stop: Callable[[], None],
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self._config = config
        self._on_run = on_run
        self._on_stop = on_stop

        self._blend_file: Optional[Path] = None
        self._current_step = -1

        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        # Title
        ctk.CTkLabel(
            self, text="Pipeline Control",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(10, 4))

        sep = ctk.CTkFrame(self, height=1, fg_color="#374151")
        sep.pack(fill="x", padx=10, pady=(0, 8))

        # Scene selection
        scene_frame = ctk.CTkFrame(self, fg_color="transparent")
        scene_frame.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(scene_frame, text="Scene (.blend):", width=130, anchor="w").pack(side="left")

        self._blend_var = tk.StringVar(value="No file selected")
        self._blend_entry = ctk.CTkEntry(
            scene_frame, textvariable=self._blend_var, state="disabled", width=240,
        )
        self._blend_entry.pack(side="left", padx=4)

        ctk.CTkButton(
            scene_frame, text="Browse", width=80, height=30,
            command=self._browse_blend,
            fg_color="#374151", hover_color="#4B5563",
        ).pack(side="left", padx=4)

        # Scene name override
        name_frame = ctk.CTkFrame(self, fg_color="transparent")
        name_frame.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(name_frame, text="Scene Name:", width=130, anchor="w").pack(side="left")
        self._name_var = tk.StringVar()
        ctk.CTkEntry(name_frame, textvariable=self._name_var, width=200).pack(side="left", padx=4)
        ctk.CTkLabel(
            name_frame, text="(auto-filled from filename)",
            text_color="#6B7280", font=ctk.CTkFont(size=10),
        ).pack(side="left", padx=6)

        # Output dir override
        out_frame = ctk.CTkFrame(self, fg_color="transparent")
        out_frame.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(out_frame, text="Output Directory:", width=130, anchor="w").pack(side="left")
        self._outdir_var = tk.StringVar()
        ctk.CTkEntry(out_frame, textvariable=self._outdir_var, width=200).pack(side="left", padx=4)

        ctk.CTkButton(
            out_frame, text="Browse", width=80, height=30,
            command=self._browse_outdir,
            fg_color="#374151", hover_color="#4B5563",
        ).pack(side="left", padx=4)

        # Status bar
        status_frame = ctk.CTkFrame(self, fg_color="#1F2937", corner_radius=6)
        status_frame.pack(fill="x", padx=10, pady=(10, 4))

        self._status_icon = ctk.CTkLabel(
            status_frame, text="  ", width=24,
            font=ctk.CTkFont(size=14),
        )
        self._status_icon.pack(side="left", padx=4, pady=6)

        self._status_label = ctk.CTkLabel(
            status_frame, text="Ready",
            font=ctk.CTkFont(size=12),
            anchor="w",
        )
        self._status_label.pack(side="left", fill="x", expand=True)

        # Progress bar
        self._progress = ctk.CTkProgressBar(self, height=6, corner_radius=3)
        self._progress.pack(fill="x", padx=10, pady=(0, 4))
        self._progress.set(0)

        # Step tracker
        self._step_label = ctk.CTkLabel(
            self, text="", text_color="#9CA3AF",
            font=ctk.CTkFont(size=10), anchor="w",
        )
        self._step_label.pack(fill="x", padx=10)

        # Action buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)

        self._run_btn = ctk.CTkButton(
            btn_frame,
            text="Run Pipeline",
            width=140, height=36,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_run,
            fg_color="#2563EB",
            hover_color="#1D4ED8",
        )
        self._run_btn.pack(side="left")

        self._stop_btn = ctk.CTkButton(
            btn_frame,
            text="Stop",
            width=100, height=36,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_stop,
            fg_color="#7F1D1D",
            hover_color="#991B1B",
            state="disabled",
        )
        self._stop_btn.pack(side="left", padx=8)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def blend_file(self) -> Optional[Path]:
        return self._blend_file

    @property
    def scene_name(self) -> str:
        return self._name_var.get().strip()

    @property
    def output_dir(self) -> Optional[Path]:
        raw = self._outdir_var.get().strip()
        return Path(raw) if raw else None

    def set_running(self, running: bool) -> None:
        self._run_btn.configure(state="disabled" if running else "normal")
        self._stop_btn.configure(state="normal" if running else "disabled")
        if not running:
            self._progress.set(0)
            self._current_step = -1

    def update_from_log(self, line: str) -> None:
        """Update progress step from a log line."""
        lower = line.lower()
        for keyword, step_idx in _STEP_KEYWORDS.items():
            if keyword.lower() in lower:
                if step_idx > self._current_step:
                    self._current_step = step_idx
                    self._update_step_ui(step_idx)
                break

    def set_status(self, text: str, color: str = "#D1D5DB", icon: str = "") -> None:
        self._status_label.configure(text=text, text_color=color)
        self._status_icon.configure(text=icon)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _browse_blend(self) -> None:
        path = filedialog.askopenfilename(
            title="Select .blend file",
            filetypes=[("Blender files", "*.blend"), ("All files", "*.*")],
            initialdir=str(self._config.scenes_dir),
        )
        if path:
            self._blend_file = Path(path)
            self._blend_var.set(path)
            if not self._name_var.get():
                self._name_var.set(self._blend_file.stem)
            if not self._outdir_var.get():
                self._outdir_var.set(
                    str(self._config.outputs_dir / self._blend_file.stem)
                )

    def _browse_outdir(self) -> None:
        path = filedialog.askdirectory(
            title="Select output directory",
            initialdir=str(self._config.outputs_dir),
        )
        if path:
            self._outdir_var.set(path)

    def _update_step_ui(self, step_idx: int) -> None:
        total = len(_PIPELINE_STEPS)
        progress = min((step_idx + 1) / total, 1.0)
        self._progress.set(progress)

        label = _PIPELINE_STEPS[step_idx] if step_idx < total else _PIPELINE_STEPS[-1]
        self._step_label.configure(
            text=f"Step {step_idx + 1}/{total}: {label}"
        )