# gui/widgets/pipeline_panel.py
"""
Pipeline control panel widget.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QHBoxLayout,
    QFrame,
    QFileDialog,
)

from nl2scene3d.gui.core.config_bridge import GUIConfig

class PipelinePanel(QFrame):
    """
    Sidebar widget for controlling the pipeline execution.
    """

    def __init__(
        self,
        config: GUIConfig,
        on_run: Callable[[], None],
        on_stop: Callable[[], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._on_run = on_run
        self._on_stop = on_stop
        
        self._blend_file: Optional[Path] = None
        self._output_dir: Optional[Path] = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(20)

        # Inputs Section
        input_group = QWidget()
        form = QFormLayout(input_group)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setSpacing(12)
        
        self._blend_edit = QLineEdit()
        self._blend_edit.setPlaceholderText("Select .blend file...")
        self._blend_edit.setReadOnly(True)
        
        browse_btn = QPushButton("Browse")
        browse_btn.setObjectName("btn_browse")
        browse_btn.clicked.connect(self._browse_blend)
        
        blend_layout = QHBoxLayout()
        blend_layout.addWidget(self._blend_edit)
        blend_layout.addWidget(browse_btn)
        form.addRow("Blend File:", blend_layout)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. MyScene_01")
        form.addRow("Scene Name:", self._name_edit)
        
        layout.addWidget(input_group)

        sep = QFrame()
        sep.setObjectName("separator")
        layout.addWidget(sep)

        # Controls Section
        self._run_btn = QPushButton("RUN PIPELINE")
        self._run_btn.setObjectName("btn_run")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        self._stop_btn = QPushButton("STOP")
        self._stop_btn.setObjectName("btn_stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        layout.addWidget(self._stop_btn)

        # Status Section
        status_card = QFrame()
        status_card.setObjectName("card_dark")
        status_layout = QVBoxLayout(status_card)
        
        header = QLabel("STATUS")
        header.setObjectName("section_header")
        status_layout.addWidget(header)
        
        self._status_label = QLabel("Ready")
        self._status_label.setWordWrap(True)
        status_layout.addWidget(self._status_label)
        
        layout.addWidget(status_card)
        
        self._step_label = QLabel("")
        self._step_label.setStyleSheet("color: #94A3B8; font-size: 11px;")
        layout.addWidget(self._step_label)

    @property
    def blend_file(self) -> Optional[Path]:
        return self._blend_file

    @property
    def scene_name(self) -> str:
        return self._name_edit.text().strip()

    @property
    def output_dir(self) -> Optional[Path]:
        return self._output_dir

    def set_blend_file(self, path: Path) -> None:
        self._blend_file = path
        self._blend_edit.setText(path.name)
        if not self._name_edit.text().strip():
            self._name_edit.setText(path.stem)

    def set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._blend_edit.setEnabled(not running)
        self._name_edit.setEnabled(not running)

    def set_status(self, text: str, color: str = "#F8FAFC") -> None:
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: 500;")

    def update_from_log(self, message: str) -> None:
        msg = message.upper()
        if "REORGANIZING" in msg:
            self._step_label.setText("Step: AI Reorganization")
        elif "RENDERING" in msg:
            self._step_label.setText("Step: Rendering")
        elif "PIPELINE COMPLETED" in msg:
            self._step_label.setText("Step: Finished")

    def _browse_blend(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open .blend", str(self._config.scenes_dir), "Blender (*.blend)")
        if path:
            self.set_blend_file(Path(path))