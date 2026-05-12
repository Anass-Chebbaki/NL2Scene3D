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

from nl2scene3d.gui.core.config_bridge import GUIConfig

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
        # --- SESSION SECTION ---
        session_card = ctk.CTkFrame(self, fg_color="#334155", corner_radius=10)
        session_card.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            session_card, text="SESSION SETUP",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#94A3B8",
        ).pack(anchor="w", padx=12, pady=(10, 5))

        # Scene selection
        scene_row = ctk.CTkFrame(session_card, fg_color="transparent")
        scene_row.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(scene_row, text="Scene File", width=80, anchor="w", font=ctk.CTkFont(size=12)).pack(side="left")
        self._blend_var = tk.StringVar(value="No file selected")
        ctk.CTkEntry(scene_row, textvariable=self._blend_var, state="disabled", height=28, fg_color="#1E293B", border_width=0).pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkButton(scene_row, text="...", width=32, height=28, command=self._browse_blend, fg_color="#475569").pack(side="left")

        # Name selection
        name_row = ctk.CTkFrame(session_card, fg_color="transparent")
        name_row.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(name_row, text="Name ID", width=80, anchor="w", font=ctk.CTkFont(size=12)).pack(side="left")
        self._name_var = tk.StringVar()
        ctk.CTkEntry(name_row, textvariable=self._name_var, height=28, fg_color="#1E293B", border_width=1, border_color="#475569").pack(side="left", fill="x", expand=True, padx=4)

        # Output selection
        out_row = ctk.CTkFrame(session_card, fg_color="transparent")
        out_row.pack(fill="x", padx=10, pady=(2, 12))
        ctk.CTkLabel(out_row, text="Output", width=80, anchor="w", font=ctk.CTkFont(size=12)).pack(side="left")
        self._outdir_var = tk.StringVar()
        ctk.CTkEntry(out_row, textvariable=self._outdir_var, height=28, fg_color="#1E293B", border_width=0).pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkButton(out_row, text="...", width=32, height=28, command=self._browse_outdir, fg_color="#475569").pack(side="left")

        # --- PROGRESS SECTION ---
        progress_card = ctk.CTkFrame(self, fg_color="#1E293B", border_width=1, border_color="#334155", corner_radius=10)
        progress_card.pack(fill="both", expand=True, padx=10, pady=5)

        ctk.CTkLabel(
            progress_card, text="PIPELINE STATUS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#94A3B8",
        ).pack(anchor="w", padx=12, pady=(10, 0))

        # Status & Icon
        status_line = ctk.CTkFrame(progress_card, fg_color="transparent")
        status_line.pack(fill="x", padx=12, pady=5)
        self._status_icon = ctk.CTkLabel(status_line, text="●", text_color="#475569", font=ctk.CTkFont(size=16))
        self._status_icon.pack(side="left")
        self._status_label = ctk.CTkLabel(status_line, text="System Ready", font=ctk.CTkFont(size=13, weight="bold"))
        self._status_label.pack(side="left", padx=8)

        # Progress bar
        self._progress = ctk.CTkProgressBar(progress_card, height=8, corner_radius=4, progress_color="#6366F1", fg_color="#0F172A")
        self._progress.pack(fill="x", padx=12, pady=8)
        self._progress.set(0)

        # Active Step
        self._step_label = ctk.CTkLabel(
            progress_card, text="Waiting for initiation...",
            text_color="#94A3B8", font=ctk.CTkFont(size=11, slant="italic"),
        )
        self._step_label.pack(fill="x", padx=12, pady=(0, 10))

        # --- ACTIONS SECTION ---
        action_card = ctk.CTkFrame(self, fg_color="transparent")
        action_card.pack(fill="x", padx=10, pady=(5, 10))

        self._run_btn = ctk.CTkButton(
            action_card, text="START PIPELINE",
            height=45, font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_run, fg_color="#6366F1", hover_color="#4F46E5",
        )
        self._run_btn.pack(fill="x", side="top", pady=2)

        self._stop_btn = ctk.CTkButton(
            action_card, text="ABORT MISSION",
            height=32, font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_stop, fg_color="#991B1B", hover_color="#B91C1C",
            state="disabled",
        )
        self._stop_btn.pack(fill="x", side="top", pady=2)

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