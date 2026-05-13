# gui/widgets/metrics_panel.py
"""
Metrics visualization panel.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QFrame,
    QScrollArea,
    QHBoxLayout,
    QProgressBar,
)

_STEP_COLORS = {
    "initial": "#94A3B8",
    "reorganized": "#6366F1",
    "final": "#34D399",
}

_METRIC_LABELS = {
    "mean_position_delta_meters": "Mean Position Delta",
    "mean_rotation_delta_radians": "Mean Rotation Delta",
    "max_position_delta_meters": "Max Position Delta",
    "improvement_score": "Improvement Score",
    "collision_count": "Collision Count",
    "out_of_bounds_count": "Out of Bounds",
}

class MetricsPanel(QScrollArea):
    """
    Panel for visualizing pipeline performance metrics and quality analysis.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        
        self._container = QWidget()
        self.setWidget(self._container)
        
        self._setup_ui()

    def _setup_ui(self) -> None:
        self._main_layout = QVBoxLayout(self._container)
        self._main_layout.setContentsMargins(20, 20, 20, 20)
        self._main_layout.setSpacing(16)

        header_layout = QHBoxLayout()
        title = QLabel("PIPELINE PERFORMANCE & ANALYSIS")
        title.setObjectName("section_header")
        header_layout.addWidget(title)
        self._main_layout.addLayout(header_layout)

        self._empty_label = QLabel("No metrics loaded yet. Run the pipeline to see results.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #64748B; font-style: italic; margin-top: 40px;")
        self._main_layout.addWidget(self._empty_label)

        self._content_frame = QWidget()
        self._content_layout = QVBoxLayout(self._content_frame)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(12)
        self._main_layout.addWidget(self._content_frame)
        self._content_frame.hide()

        self._main_layout.addStretch()

    def clear(self) -> None:
        while self._content_layout.count():
            child = self._content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self._content_frame.hide()
        self._empty_label.show()

    def load_from_file(self, path: Path) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._render_metrics(data)
        except Exception as e:
            self.clear()
            self._empty_label.setText(f"Error loading metrics: {e}")

    def _render_metrics(self, data: Dict[str, Any]) -> None:
        self.clear()
        self._empty_label.hide()
        self._content_frame.show()

        for step_name, step_data in data.items():
            if isinstance(step_data, dict):
                self._render_step(step_name, step_data)

    def _render_step(self, step_name: str, data: Dict[str, Any]) -> None:
        color = _STEP_COLORS.get(step_name, "#9CA3AF")
        
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        
        header = QLabel(f"  {step_name.upper()}")
        header.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: bold; border-left: 3px solid {color};")
        layout.addWidget(header)
        layout.addSpacing(10)

        for key, label in _METRIC_LABELS.items():
            val = data.get(key)
            if val is None:
                continue

            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 2, 0, 2)
            
            k_lbl = QLabel(label)
            k_lbl.setStyleSheet("color: #94A3B8; font-size: 11px;")
            
            v_lbl = QLabel(self._format_value(key, val))
            v_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            v_lbl.setStyleSheet("color: #E5E7EB; font-weight: bold; font-size: 11px;")
            
            row_l.addWidget(k_lbl)
            row_l.addWidget(v_lbl)
            layout.addWidget(row_w)

            if key == "improvement_score":
                bar = QProgressBar()
                bar.setRange(0, 100)
                bar.setValue(int(float(val) * 100))
                bar.setTextVisible(False)
                bar.setFixedHeight(6)
                layout.addWidget(bar)
                layout.addSpacing(4)

        self._content_layout.addWidget(card)

    @staticmethod
    def _format_value(key: str, value: Any) -> str:
        try:
            if value is None:
                return "N/A"
            if key.endswith("_meters"):
                return f"{float(value):.3f} m"
            if key.endswith("_radians"):
                return f"{math.degrees(float(value)):.1f}°"
            if key == "improvement_score":
                return f"{float(value):.3f}"
            return str(value)
        except Exception:
            return str(value)