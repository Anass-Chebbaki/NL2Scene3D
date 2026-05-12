# gui/app.py
"""
NL2Scene3D — Main application window.
"""
from __future__ import annotations

import sys
import platform
import subprocess
import ctypes
from pathlib import Path

import PySide6
import PySide6.QtSvg
import PySide6.QtSvgWidgets
from PySide6.QtCore import Qt, Slot, QSize, QCoreApplication
import os

# --- Fix for SVG/Image plugins in virtual environments ---
_plugins_path = os.path.join(os.path.dirname(PySide6.__file__), "plugins")
QCoreApplication.addLibraryPath(_plugins_path)
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(_plugins_path, "platforms")
# ---------------------------------------------------------
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QTabWidget,
    QFrame,
    QLabel,
    QMenuBar,
    QMenu,
    QMessageBox,
    QFileDialog,
)
from PySide6.QtGui import QAction, QIcon, QFont, QCloseEvent

from nl2scene3d.gui.core.config_bridge import GUIConfig, load_gui_config
from nl2scene3d.gui.core.pipeline_runner import PipelineRunner
from nl2scene3d.gui.core.image_watcher import ImageWatcher
from nl2scene3d.gui.widgets.config_panel import ConfigPanel
from nl2scene3d.gui.widgets.image_viewer import ImageViewer
from nl2scene3d.gui.widgets.log_panel import LogPanel
from nl2scene3d.gui.widgets.metrics_panel import MetricsPanel
from nl2scene3d.gui.widgets.pipeline_panel import PipelinePanel
from nl2scene3d.gui.theme import STYLESHEET, C_ACCENT, C_SUBTEXT

_APP_TITLE = "NL2Scene3D — Scene Reorganization Pipeline"
_APP_VERSION = "0.2.0"

def set_windows_dark_mode(window_id: int) -> None:
    """Apply immersive dark mode to the Windows title bar."""
    if platform.system() != "Windows":
        return
    try:
        # DWMWA_USE_IMMERSIVE_DARK_MODE = 20 (Windows 11 and later) or 19 (older Windows 10)
        # We'll try 20 first as it's the standard for modern systems.
        dwm = ctypes.windll.dwmapi
        rendering_policy = ctypes.c_int(1)
        dwm.DwmSetWindowAttribute(window_id, 20, ctypes.byref(rendering_policy), ctypes.sizeof(rendering_policy))
    except Exception:
        pass

class NL2Scene3DApp(QMainWindow):
    """
    Main application window managing the pipeline lifecycle and user interface.
    """

    def __init__(self) -> None:
        super().__init__()
        self._config: GUIConfig = load_gui_config()
        
        self.setWindowTitle(f"{_APP_TITLE}  v{_APP_VERSION}")
        self.setMinimumSize(1100, 720)
        self.resize(1400, 860)
        self.setStyleSheet(STYLESHEET)

        self._runner = PipelineRunner(self._config, parent=self)
        self._runner.log_emitted.connect(self._on_log)
        self._runner.image_detected.connect(self._on_image)
        self._runner.pipeline_finished.connect(self._handle_finished)

        self._watcher: ImageWatcher | None = None

        self._build_ui()
        self._on_log("INFO", "Application initialized.")
        
        # Apply dark title bar
        set_windows_dark_mode(self.winId())

    def _build_ui(self) -> None:
        self._central = QWidget()
        self.setCentralWidget(self._central)
        layout = QVBoxLayout(self._central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._build_menu()

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self._splitter)

        self._sidebar = QFrame()
        self._sidebar.setObjectName("sidebar")
        self._sidebar.setFixedWidth(340)
        self._build_sidebar(self._sidebar)
        self._splitter.addWidget(self._sidebar)

        self._main_content = QFrame()
        content_layout = QVBoxLayout(self._main_content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        content_layout.addWidget(self._tabs)
        self._build_tabs()
        self._splitter.addWidget(self._main_content)

    def _build_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("Open .blend file...", self)
        open_action.triggered.connect(self._menu_open_blend)
        file_menu.addAction(open_action)
        
        out_action = QAction("Open Output Directory", self)
        out_action.triggered.connect(self._open_output_dir)
        file_menu.addAction(out_action)
        
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        run_menu = menubar.addMenu("&Run")
        run_menu.addAction("Run Pipeline", self._run_pipeline, "F5")
        run_menu.addAction("Stop", self._stop_pipeline)

    def _build_sidebar(self, parent: QFrame) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        brand = QFrame()
        brand.setFixedHeight(100)
        brand.setStyleSheet("background-color: #0F172A; border-bottom: 1px solid #334155;")
        bl = QVBoxLayout(brand)
        
        t = QLabel("NL2Scene3D")
        t.setObjectName("label_accent")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bl.addWidget(t)
        
        s = QLabel("AI SCENE REORGANIZER")
        s.setObjectName("label_subtext")
        s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bl.addWidget(s)
        
        layout.addWidget(brand)

        self._pipeline_panel = PipelinePanel(self._config, on_run=self._run_pipeline, on_stop=self._stop_pipeline)
        layout.addWidget(self._pipeline_panel)
        layout.addStretch()

    def _build_tabs(self) -> None:
        p_tab = QWidget()
        pl = QHBoxLayout(p_tab)
        self._p_splitter = QSplitter(Qt.Orientation.Horizontal)
        pl.addWidget(self._p_splitter)
        
        self._viewer_mini = ImageViewer()
        self._p_splitter.addWidget(self._viewer_mini)
        
        self._log_mini = LogPanel()
        self._p_splitter.addWidget(self._log_mini)
        
        self._p_splitter.setStretchFactor(0, 3)
        self._p_splitter.setStretchFactor(1, 2)
        self._tabs.addTab(p_tab, "Pipeline")

        self._log_full = LogPanel()
        self._tabs.addTab(self._log_full, "Full Log")

        self._viewer_full = ImageViewer()
        self._tabs.addTab(self._viewer_full, "Renders Gallery")

        self._metrics_panel = MetricsPanel()
        self._tabs.addTab(self._metrics_panel, "Metrics")

        self._config_panel = ConfigPanel(self._config)
        self._tabs.addTab(self._config_panel, "Settings")

    @Slot(str, str)
    def _on_log(self, level: str, message: str) -> None:
        self._log_mini.append(level, message)
        self._log_full.append(level, message)
        self._pipeline_panel.update_from_log(message)

    @Slot(str)
    def _on_image(self, path: str) -> None:
        self._viewer_mini.add_image(path)
        self._viewer_full.add_image(path)

    @Slot(bool, str)
    def _handle_finished(self, success: bool, error_msg: str) -> None:
        self._pipeline_panel.set_running(False)
        if self._watcher:
            self._watcher.stop()
        
        if success:
            self._pipeline_panel.set_status("Pipeline completed successfully.", "#34D399")
            self._load_metrics()
            self._tabs.setCurrentIndex(3)
        else:
            self._pipeline_panel.set_status(error_msg or "Pipeline failed.", "#EF4444")

    def _run_pipeline(self) -> None:
        if self._runner.isRunning():
            return
            
        self._config_panel.apply_to_config()

        blend = self._pipeline_panel.blend_file
        scene_name = self._pipeline_panel.scene_name
        
        if not scene_name:
            QMessageBox.warning(self, "Validation Error", "Scene name is required.")
            return
            
        if not self._config.api_key:
            QMessageBox.critical(self, "Config Error", "API Key missing.")
            return

        output_dir = self._pipeline_panel.output_dir or (self._config.outputs_dir / scene_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._log_mini.clear()
        self._log_full.clear()
        self._viewer_mini.clear()
        self._viewer_full.clear()
        self._metrics_panel.clear()

        self._runner.blend_file = blend
        self._runner.scene_name = scene_name
        self._runner.output_dir = output_dir
        self._runner.seed = self._config.randomizer_seed
        self._runner.skip_vision = self._config.skip_vision
        self._runner.max_objects = self._config.max_movable_objects

        self._watcher = ImageWatcher(output_dir)
        self._watcher.new_image_found.connect(self._on_image)
        self._watcher.start()

        self._pipeline_panel.set_running(True)
        self._pipeline_panel.set_status("Running...", "#6366F1")
        self._tabs.setCurrentIndex(0)
        self._runner.start()

    def _stop_pipeline(self) -> None:
        self._runner.request_stop()
        if self._watcher:
            self._watcher.stop()
        self._pipeline_panel.set_running(False)
        self._pipeline_panel.set_status("Stopped.", "#F59E0B")

    def closeEvent(self, event: QCloseEvent) -> None:
        self._runner.request_stop()
        if self._watcher:
            self._watcher.stop()
        event.accept()

    def _load_metrics(self) -> None:
        scene_name = self._pipeline_panel.scene_name
        out = self._pipeline_panel.output_dir or (self._config.outputs_dir / scene_name)
        p = out / "metrics.json"
        if p.exists():
            self._metrics_panel.load_from_file(p)

    def _menu_open_blend(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Open Blender File", str(self._config.scenes_dir), "Blender Files (*.blend)")
        if p:
            self._pipeline_panel.set_blend_file(Path(p))

    def _open_output_dir(self) -> None:
        scene_name = self._pipeline_panel.scene_name
        out = self._pipeline_panel.output_dir or (self._config.outputs_dir / scene_name)
        out.mkdir(parents=True, exist_ok=True)
        if platform.system() == "Windows":
            subprocess.Popen(["explorer", str(out)])
        else:
            subprocess.Popen(["open" if platform.system() == "Darwin" else "xdg-open", str(out)])

def main() -> None:
    # Force SVG plugin loading
    _ = PySide6.QtSvg.QSvgRenderer
    
    app = QApplication(sys.argv)
    app.setApplicationName("NL2Scene3D")
    app.setFont(QFont("Inter", 10))
    window = NL2Scene3DApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()