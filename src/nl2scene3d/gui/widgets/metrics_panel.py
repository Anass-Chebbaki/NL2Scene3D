# gui/widgets/metrics_panel.py
"""
Metrics panel — displays pipeline quality metrics from metrics.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import customtkinter as ctk


_METRIC_LABELS = {
    "mean_position_delta_meters": "Mean Position Delta",
    "mean_rotation_delta_radians": "Mean Rotation Delta",
    "object_count_movable": "Movable Objects",
    "improvement_score": "Improvement Score",
}

_STEP_COLORS = {
    "randomized": "#F59E0B",
    "reordered":  "#60A5FA",
    "refined":    "#34D399",
}


class MetricsPanel(ctk.CTkScrollableFrame):
    """Displays pipeline quality metrics in a structured table."""

    def __init__(self, master: ctk.CTkBaseClass, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._empty_label: ctk.CTkLabel
        self._content_frame: Optional[ctk.CTkFrame] = None
        self._build_header()

    def _build_header(self) -> None:
        ctk.CTkLabel(
            self, text="Quality Metrics",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=6, pady=(8, 2))

        sep = ctk.CTkFrame(self, height=1, fg_color="#374151")
        sep.pack(fill="x", padx=6, pady=(0, 6))

        self._empty_label = ctk.CTkLabel(
            self, text="Metrics will appear here after the pipeline completes.",
            text_color="#4B5563",
        )
        self._empty_label.pack(pady=20)

    def load_from_file(self, metrics_path: Path) -> None:
        """Load and render metrics from a metrics.json file."""
        if not metrics_path.exists():
            return
        try:
            with open(metrics_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return

        self._empty_label.pack_forget()

        if self._content_frame:
            self._content_frame.destroy()

        self._content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._content_frame.pack(fill="x", padx=6, pady=4)

        for step_name, step_data in data.items():
            self._render_step(step_name, step_data)

    def clear(self) -> None:
        if self._content_frame:
            self._content_frame.destroy()
            self._content_frame = None
        self._empty_label.pack(pady=20)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _render_step(self, step_name: str, data: dict) -> None:
        color = _STEP_COLORS.get(step_name, "#9CA3AF")

        # Step header
        step_frame = ctk.CTkFrame(
            self._content_frame,  # type: ignore[arg-type]
            fg_color="#1F2937",
            corner_radius=8,
        )
        step_frame.pack(fill="x", pady=4)

        ctk.CTkLabel(
            step_frame,
            text=f"  {step_name.upper()}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=color,
            anchor="w",
        ).pack(fill="x", padx=8, pady=(8, 4))

        # Metric rows
        for key, label in _METRIC_LABELS.items():
            val = data.get(key)
            if val is None:
                continue

            row = ctk.CTkFrame(step_frame, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=1)

            ctk.CTkLabel(
                row, text=label, width=200, anchor="w",
                text_color="#9CA3AF", font=ctk.CTkFont(size=11),
            ).pack(side="left")

            formatted = self._format_value(key, val)
            ctk.CTkLabel(
                row, text=formatted, anchor="w",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="#E5E7EB",
            ).pack(side="left")

        # Progress bar for improvement_score
        score = data.get("improvement_score")
        if score is not None:
            bar_row = ctk.CTkFrame(step_frame, fg_color="transparent")
            bar_row.pack(fill="x", padx=8, pady=(2, 8))

            ctk.CTkLabel(
                bar_row, text="Improvement", width=200, anchor="w",
                text_color="#9CA3AF", font=ctk.CTkFont(size=11),
            ).pack(side="left")

            bar = ctk.CTkProgressBar(bar_row, width=160, height=8, corner_radius=4)
            bar.pack(side="left", padx=4)
            bar.set(float(score))

    @staticmethod
    def _format_value(key: str, value) -> str:
        if value is None:
            return "N/A"
        if key == "mean_position_delta_meters":
            return f"{float(value):.3f} m"
        if key == "mean_rotation_delta_radians":
            return f"{math.degrees(float(value)):.1f} deg"
        if key == "improvement_score":
            return f"{float(value):.3f}"
        return str(value)