# gui/widgets/config_panel.py
"""
Configuration settings panel.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QLabel,
    QScrollArea,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
)

from nl2scene3d.gui.core.config_bridge import GUIConfig

class ConfigPanel(QScrollArea):
    """
    Settings panel containing all configurable parameters for the pipeline.
    """

    def __init__(self, config: GUIConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        
        self._container = QWidget()
        self.setWidget(self._container)
        
        self._setup_ui()
        self._load_config_to_ui()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self._container)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(24)

        # Gemini API Section
        api_group, api_content_layout = self._create_section("GEMINI API")
        api_f = QFormLayout()
        api_content_layout.addLayout(api_f)
        
        self._api_key = QLineEdit()
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        api_f.addRow("API Key:", self._api_key)
        
        self._model_primary = QLineEdit()
        api_f.addRow("Primary Model:", self._model_primary)
        
        self._model_fallback = QLineEdit()
        api_f.addRow("Fallback Model:", self._model_fallback)
        
        self._temp = QDoubleSpinBox()
        self._temp.setRange(0.0, 2.0)
        self._temp.setSingleStep(0.1)
        api_f.addRow("Temperature (0.0 - 2.0):", self._temp)
        
        self._max_tokens = QSpinBox()
        self._max_tokens.setRange(100, 32000)
        self._max_tokens.setSingleStep(100)
        api_f.addRow("Max Output Tokens:", self._max_tokens)
        
        self._max_retries = QSpinBox()
        self._max_retries.setRange(0, 10)
        api_f.addRow("Max Retries:", self._max_retries)
        
        self._timeout = QSpinBox()
        self._timeout.setRange(10, 600)
        api_f.addRow("Timeout (s):", self._timeout)
        
        main_layout.addWidget(api_group)

        # Pipeline & Physics Section
        pipe_group, pipe_content_layout = self._create_section("PIPELINE & PHYSICS")
        pipe_f = QFormLayout()
        pipe_content_layout.addLayout(pipe_f)
        
        self._max_objects = QSpinBox()
        self._max_objects.setRange(1, 100)
        pipe_f.addRow("Max Movable Objects:", self._max_objects)
        
        self._seed = QSpinBox()
        self._seed.setRange(0, 999999)
        pipe_f.addRow("Randomizer Seed (0=rand):", self._seed)
        
        self._wall_margin = QDoubleSpinBox()
        self._wall_margin.setRange(0.0, 5.0)
        self._wall_margin.setSingleStep(0.05)
        pipe_f.addRow("Wall Margin (m):", self._wall_margin)
        
        self._max_overlap = QDoubleSpinBox()
        self._max_overlap.setRange(0.0, 1.0)
        self._max_overlap.setSingleStep(0.01)
        pipe_f.addRow("Max Overlap Ratio:", self._max_overlap)
        
        self._placement_attempts = QSpinBox()
        self._placement_attempts.setRange(1, 1000)
        pipe_f.addRow("Max Placement Attempts:", self._placement_attempts)
        
        self._min_quality = QSpinBox()
        self._min_quality.setRange(1, 10)
        pipe_f.addRow("Min Quality Score (1-10):", self._min_quality)
        
        self._good_quality = QSpinBox()
        self._good_quality.setRange(1, 10)
        pipe_f.addRow("Good Quality Score (Threshold):", self._good_quality)
        
        self._max_corrections = QSpinBox()
        self._max_corrections.setRange(0, 10)
        pipe_f.addRow("Max Visual Corrections:", self._max_corrections)

        self._skip_vision = QCheckBox("Skip Visual Critique")
        self._skip_vision.setToolTip("Disable the second LLM (vision) call for faster testing")
        pipe_f.addRow("", self._skip_vision)
        
        main_layout.addWidget(pipe_group)

        # Render Settings Section
        render_group, render_content_layout = self._create_section("RENDER SETTINGS")
        render_f = QFormLayout()
        render_content_layout.addLayout(render_f)
        
        # Preview
        prev_l = QHBoxLayout()
        self._prev_w = QSpinBox()
        self._prev_w.setRange(64, 4096)
        self._prev_w.setSingleStep(64)
        self._prev_h = QSpinBox()
        self._prev_h.setRange(64, 4096)
        self._prev_h.setSingleStep(64)
        self._prev_s = QSpinBox()
        self._prev_s.setRange(1, 1024)
        prev_l.addWidget(QLabel("W:"))
        prev_l.addWidget(self._prev_w)
        prev_l.addWidget(QLabel("H:"))
        prev_l.addWidget(self._prev_h)
        prev_l.addWidget(QLabel("Samples:"))
        prev_l.addWidget(self._prev_s)
        render_f.addRow("Preview (px/smp):", prev_l)
        
        # Final
        final_l = QHBoxLayout()
        self._final_w = QSpinBox()
        self._final_w.setRange(64, 8192)
        self._final_w.setSingleStep(64)
        self._final_h = QSpinBox()
        self._final_h.setRange(64, 8192)
        self._final_h.setSingleStep(64)
        self._final_s = QSpinBox()
        self._final_s.setRange(1, 8192)
        final_l.addWidget(QLabel("W:"))
        final_l.addWidget(self._final_w)
        final_l.addWidget(QLabel("H:"))
        final_l.addWidget(self._final_h)
        final_l.addWidget(QLabel("Samples:"))
        final_l.addWidget(self._final_s)
        render_f.addRow("Final (px/smp):", final_l)
        
        main_layout.addWidget(render_group)

        # Paths & Logging Section
        sys_group, sys_content_layout = self._create_section("PATHS & LOGGING")
        sys_f = QFormLayout()
        sys_content_layout.addLayout(sys_f)
        
        self._blender_exe = self._add_path_row(sys_f, "Blender Executable:", True)
        self._scenes_dir = self._add_path_row(sys_f, "Scenes Directory:", False)
        self._outputs_dir = self._add_path_row(sys_f, "Outputs Directory:", False)
        
        self._log_level = QComboBox()
        self._log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        sys_f.addRow("Log Level:", self._log_level)
        
        main_layout.addWidget(sys_group)
        main_layout.addStretch()

    def _create_section(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        """Creates a styled card with a title and returns the inner layout."""
        f = QFrame()
        f.setObjectName("card")
        l = QVBoxLayout(f)
        l.setContentsMargins(16, 16, 16, 16)
        l.setSpacing(12)
        
        h = QLabel(title)
        h.setObjectName("section_header")
        l.addWidget(h)
        
        return f, l

    def _add_path_row(self, form: QFormLayout, label: str, is_file: bool) -> QLineEdit:
        edit = QLineEdit()
        btn = QPushButton("...")
        btn.setObjectName("btn_browse")
        btn.setFixedWidth(32)
        btn.clicked.connect(lambda: (self._browse_file if is_file else self._browse_dir)(edit))
        row = QHBoxLayout()
        row.addWidget(edit)
        row.addWidget(btn)
        form.addRow(label, row)
        return edit

    def _load_config_to_ui(self) -> None:
        c = self._config
        self._api_key.setText(c.api_key)
        self._model_primary.setText(c.model_primary)
        self._model_fallback.setText(c.model_fallback)
        self._temp.setValue(c.temperature)
        self._max_tokens.setValue(c.max_output_tokens)
        self._max_retries.setValue(c.max_retries)
        self._timeout.setValue(c.timeout_seconds)
        self._max_objects.setValue(c.max_movable_objects)
        self._seed.setValue(c.randomizer_seed)
        self._wall_margin.setValue(c.wall_margin)
        self._max_overlap.setValue(c.max_overlap_ratio)
        self._placement_attempts.setValue(c.max_placement_attempts)
        self._min_quality.setValue(c.min_quality_score)
        self._good_quality.setValue(c.good_quality_score)
        self._max_corrections.setValue(c.max_corrections)
        self._skip_vision.setChecked(c.skip_vision)
        self._prev_w.setValue(c.preview_width)
        self._prev_h.setValue(c.preview_height)
        self._prev_s.setValue(c.preview_samples)
        self._final_w.setValue(c.final_width)
        self._final_h.setValue(c.final_height)
        self._final_s.setValue(c.final_samples)
        self._blender_exe.setText(c.blender_executable)
        self._scenes_dir.setText(str(c.scenes_dir))
        self._outputs_dir.setText(str(c.outputs_dir))
        idx = self._log_level.findText(c.log_level)
        if idx >= 0:
            self._log_level.setCurrentIndex(idx)

    def apply_to_config(self) -> None:
        c = self._config
        c.api_key = self._api_key.text()
        c.model_primary = self._model_primary.text()
        c.model_fallback = self._model_fallback.text()
        c.temperature = self._temp.value()
        c.max_output_tokens = self._max_tokens.value()
        c.max_retries = self._max_retries.value()
        c.timeout_seconds = self._timeout.value()
        c.max_movable_objects = self._max_objects.value()
        c.randomizer_seed = self._seed.value()
        c.wall_margin = self._wall_margin.value()
        c.max_overlap_ratio = self._max_overlap.value()
        c.max_placement_attempts = self._placement_attempts.value()
        c.min_quality_score = self._min_quality.value()
        c.good_quality_score = self._good_quality.value()
        c.max_corrections = self._max_corrections.value()
        c.skip_vision = self._skip_vision.isChecked()
        c.preview_width = self._prev_w.value()
        c.preview_height = self._prev_h.value()
        c.preview_samples = self._prev_s.value()
        c.final_width = self._final_w.value()
        c.final_height = self._final_h.value()
        c.final_samples = self._final_s.value()
        c.blender_executable = self._blender_exe.text()
        c.scenes_dir = Path(self._scenes_dir.text())
        c.outputs_dir = Path(self._outputs_dir.text())
        c.log_level = self._log_level.currentText()
        c.save()

    def _browse_file(self, e: QLineEdit) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Select File")
        if p:
            e.setText(p)

    def _browse_dir(self, e: QLineEdit) -> None:
        p = QFileDialog.getExistingDirectory(self, "Select Directory")
        if p:
            e.setText(p)