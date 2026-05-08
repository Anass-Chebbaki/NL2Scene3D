# gui/widgets/config_panel.py
"""
Configuration panel — exposes all pipeline parameters as editable fields.
Writes back to a mutable GUIConfig object; does NOT write to disk.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Optional

import customtkinter as ctk

from gui.core.config_bridge import GUIConfig


class ConfigPanel(ctk.CTkScrollableFrame):
    """Scrollable panel containing all pipeline configuration fields."""

    def __init__(
        self,
        master: ctk.CTkBaseClass,
        config: GUIConfig,
        on_change: Optional[Callable[[], None]] = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self._config = config
        self._on_change = on_change
        self._vars: dict[str, tk.Variable] = {}
        self._build()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _section(self, title: str) -> None:
        ctk.CTkLabel(
            self, text=title,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#60A5FA",
        ).pack(anchor="w", padx=6, pady=(12, 2))
        sep = ctk.CTkFrame(self, height=1, fg_color="#374151")
        sep.pack(fill="x", padx=6, pady=(0, 6))

    def _row(
        self,
        label: str,
        var: tk.Variable,
        tooltip: str = "",
        entry_width: int = 220,
    ) -> ctk.CTkEntry:
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=2)

        lbl = ctk.CTkLabel(row, text=label, width=180, anchor="w")
        lbl.pack(side="left")

        entry = ctk.CTkEntry(row, textvariable=var, width=entry_width)
        entry.pack(side="left", padx=(4, 0))

        if tooltip:
            ctk.CTkLabel(
                row, text=tooltip, text_color="#6B7280",
                font=ctk.CTkFont(size=10), anchor="w",
            ).pack(side="left", padx=6)

        var.trace_add("write", lambda *_: self._on_change and self._on_change())
        return entry

    def _path_row(self, label: str, var: tk.StringVar, is_file: bool = False) -> None:
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=2)

        ctk.CTkLabel(row, text=label, width=180, anchor="w").pack(side="left")
        ctk.CTkEntry(row, textvariable=var, width=180).pack(side="left", padx=(4, 0))

        def browse() -> None:
            if is_file:
                p = filedialog.askopenfilename(
                    filetypes=[("Blender files", "*.blend"), ("All files", "*.*")]
                )
            else:
                p = filedialog.askdirectory()
            if p:
                var.set(p)

        ctk.CTkButton(
            row, text="Browse", width=70, height=28,
            command=browse,
            fg_color="#374151", hover_color="#4B5563",
        ).pack(side="left", padx=4)

        var.trace_add("write", lambda *_: self._on_change and self._on_change())

    def _check_row(self, label: str, var: tk.Variable, tooltip: str = "") -> None:
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=2)
        ctk.CTkCheckBox(row, text=label, variable=var).pack(side="left")
        if tooltip:
            ctk.CTkLabel(
                row, text=tooltip, text_color="#6B7280",
                font=ctk.CTkFont(size=10),
            ).pack(side="left", padx=8)
        var.trace_add("write", lambda *_: self._on_change and self._on_change())

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        cfg = self._config

        # ---- API ----
        self._section("Gemini API")
        self._v("api_key", tk.StringVar(value=cfg.api_key))
        self._row("API Key", self._vars["api_key"], entry_width=260)

        self._v("model_primary", tk.StringVar(value=cfg.model_primary))
        self._row("Primary Model", self._vars["model_primary"])

        self._v("model_fallback", tk.StringVar(value=cfg.model_fallback))
        self._row("Fallback Model", self._vars["model_fallback"])

        self._v("temperature", tk.StringVar(value=str(cfg.temperature)))
        self._row("Temperature", self._vars["temperature"], "0.0 – 2.0")

        self._v("max_output_tokens", tk.StringVar(value=str(cfg.max_output_tokens)))
        self._row("Max Output Tokens", self._vars["max_output_tokens"])

        self._v("max_retries", tk.StringVar(value=str(cfg.max_retries)))
        self._row("Max Retries", self._vars["max_retries"])

        self._v("timeout_seconds", tk.StringVar(value=str(cfg.timeout_seconds)))
        self._row("Timeout (s)", self._vars["timeout_seconds"])

        # ---- Pipeline ----
        self._section("Pipeline")
        self._v("max_movable_objects", tk.StringVar(value=str(cfg.max_movable_objects)))
        self._row("Max Movable Objects", self._vars["max_movable_objects"])

        self._v("randomizer_seed", tk.StringVar(value=str(cfg.randomizer_seed)))
        self._row("Randomizer Seed", self._vars["randomizer_seed"], "0 = random")

        self._v("wall_margin", tk.StringVar(value=str(cfg.wall_margin)))
        self._row("Wall Margin (m)", self._vars["wall_margin"])

        self._v("max_overlap_ratio", tk.StringVar(value=str(cfg.max_overlap_ratio)))
        self._row("Max Overlap Ratio", self._vars["max_overlap_ratio"])

        self._v("max_placement_attempts", tk.StringVar(value=str(cfg.max_placement_attempts)))
        self._row("Max Placement Attempts", self._vars["max_placement_attempts"])

        self._v("min_quality_score", tk.StringVar(value=str(cfg.min_quality_score)))
        self._row("Min Quality Score", self._vars["min_quality_score"], "1 – 10")

        self._v("max_corrections", tk.StringVar(value=str(cfg.max_corrections)))
        self._row("Max Visual Corrections", self._vars["max_corrections"])

        self._v("skip_vision", tk.BooleanVar(value=False))
        self._check_row("Skip Visual Critique", self._vars["skip_vision"],
                        "Disable the second LLM (vision) call")

        # ---- Render ----
        self._section("Render — Preview")
        self._v("preview_width", tk.StringVar(value=str(cfg.preview_width)))
        self._row("Width (px)", self._vars["preview_width"])
        self._v("preview_height", tk.StringVar(value=str(cfg.preview_height)))
        self._row("Height (px)", self._vars["preview_height"])
        self._v("preview_samples", tk.StringVar(value=str(cfg.preview_samples)))
        self._row("Samples", self._vars["preview_samples"])

        self._section("Render — Final")
        self._v("final_width", tk.StringVar(value=str(cfg.final_width)))
        self._row("Width (px)", self._vars["final_width"])
        self._v("final_height", tk.StringVar(value=str(cfg.final_height)))
        self._row("Height (px)", self._vars["final_height"])
        self._v("final_samples", tk.StringVar(value=str(cfg.final_samples)))
        self._row("Samples", self._vars["final_samples"])

        # ---- Paths ----
        self._section("Paths")
        self._v("blender_executable", tk.StringVar(value=cfg.blender_executable))
        self._row("Blender Executable", self._vars["blender_executable"], entry_width=240)

        self._v("scenes_dir", tk.StringVar(value=str(cfg.scenes_dir)))
        self._path_row("Scenes Directory", self._vars["scenes_dir"])  # type: ignore

        self._v("outputs_dir", tk.StringVar(value=str(cfg.outputs_dir)))
        self._path_row("Outputs Directory", self._vars["outputs_dir"])  # type: ignore

        # ---- Logging ----
        self._section("Logging")
        self._v("log_level", tk.StringVar(value=cfg.log_level))
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=2)
        ctk.CTkLabel(row, text="Log Level", width=180, anchor="w").pack(side="left")
        ctk.CTkOptionMenu(
            row,
            variable=self._vars["log_level"],  # type: ignore
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            width=140,
        ).pack(side="left")
        self._vars["log_level"].trace_add("write", lambda *_: self._on_change and self._on_change())

    def _v(self, key: str, var: tk.Variable) -> None:
        self._vars[key] = var

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_to_config(self) -> None:
        """Write widget values back into the shared GUIConfig object."""
        cfg = self._config
        v = self._vars

        def _str(key: str) -> str:
            return v[key].get()  # type: ignore[union-attr]

        def _int(key: str, default: int = 0) -> int:
            try:
                return int(_str(key))
            except ValueError:
                return default

        def _float(key: str, default: float = 0.0) -> float:
            try:
                return float(_str(key))
            except ValueError:
                return default

        cfg.api_key = _str("api_key")
        cfg.model_primary = _str("model_primary")
        cfg.model_fallback = _str("model_fallback")
        cfg.temperature = _float("temperature", cfg.temperature)
        cfg.max_output_tokens = _int("max_output_tokens", cfg.max_output_tokens)
        cfg.max_retries = _int("max_retries", cfg.max_retries)
        cfg.timeout_seconds = _int("timeout_seconds", cfg.timeout_seconds)

        cfg.max_movable_objects = _int("max_movable_objects", cfg.max_movable_objects)
        cfg.randomizer_seed = _int("randomizer_seed", cfg.randomizer_seed)
        cfg.wall_margin = _float("wall_margin", cfg.wall_margin)
        cfg.max_overlap_ratio = _float("max_overlap_ratio", cfg.max_overlap_ratio)
        cfg.max_placement_attempts = _int("max_placement_attempts", cfg.max_placement_attempts)
        cfg.min_quality_score = _int("min_quality_score", cfg.min_quality_score)
        cfg.max_corrections = _int("max_corrections", cfg.max_corrections)

        cfg.preview_width = _int("preview_width", cfg.preview_width)
        cfg.preview_height = _int("preview_height", cfg.preview_height)
        cfg.preview_samples = _int("preview_samples", cfg.preview_samples)
        cfg.final_width = _int("final_width", cfg.final_width)
        cfg.final_height = _int("final_height", cfg.final_height)
        cfg.final_samples = _int("final_samples", cfg.final_samples)

        cfg.blender_executable = _str("blender_executable")
        cfg.scenes_dir = Path(_str("scenes_dir"))
        cfg.outputs_dir = Path(_str("outputs_dir"))
        cfg.log_level = _str("log_level")

    def get_skip_vision(self) -> bool:
        return bool(self._vars["skip_vision"].get())  # type: ignore[union-attr]