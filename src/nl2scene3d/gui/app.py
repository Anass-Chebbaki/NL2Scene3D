# gui/app.py
"""
NL2Scene3D — Main application window.

Entry point:
    python gui/app.py
    python -m gui.app
"""
from __future__ import annotations

import json
import sys
import threading
import tkinter as tk
from pathlib import Path

import customtkinter as ctk

# Resolve project root so relative imports work regardless of cwd
_GUI_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _GUI_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gui.core.config_bridge import GUIConfig, load_gui_config
from gui.core.image_watcher import ImageWatcher
from gui.core.pipeline_runner import PipelineRunner
from gui.widgets.config_panel import ConfigPanel
from gui.widgets.image_viewer import ImageViewer
from gui.widgets.log_panel import LogPanel
from gui.widgets.metrics_panel import MetricsPanel
from gui.widgets.pipeline_panel import PipelinePanel

# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_APP_TITLE = "NL2Scene3D — Scene Reorganization Pipeline"
_APP_VERSION = "0.1.0"


class NL2Scene3DApp(ctk.CTk):
    """Root application window."""

    def __init__(self) -> None:
        super().__init__()

        self.title(f"{_APP_TITLE}  v{_APP_VERSION}")
        self.geometry("1400x860")
        self.minsize(1100, 680)

        # Shared configuration object
        self._config: GUIConfig = load_gui_config()

        # Pipeline runner
        self._runner = PipelineRunner(
            config=self._config,
            on_log=self._on_log,
            on_image=self._on_image,
            on_finished=self._on_finished,
        )

        # Image watcher (secondary detection)
        self._watcher: ImageWatcher | None = None

        self._build_ui()
        self._apply_config_to_ui()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ---- Menu bar ----
        self._build_menu()

        # ---- Root paned layout ----
        # Left column: pipeline control + config
        # Right column: log (top) + viewer (bottom-left) + metrics (bottom-right)

        root_pane = ctk.CTkFrame(self, fg_color="transparent")
        root_pane.pack(fill="both", expand=True, padx=6, pady=6)
        root_pane.columnconfigure(1, weight=1)
        root_pane.rowconfigure(0, weight=1)

        # Left sidebar
        left = ctk.CTkFrame(root_pane, width=310, fg_color="#111827", corner_radius=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        left.pack_propagate(False)
        self._build_left_sidebar(left)

        # Right area — notebook
        self._notebook = ctk.CTkTabview(root_pane, corner_radius=8)
        self._notebook.grid(row=0, column=1, sticky="nsew")

        self._notebook.add("Pipeline")
        self._notebook.add("Log")
        self._notebook.add("Renders")
        self._notebook.add("Metrics")
        self._notebook.add("Settings")

        self._build_pipeline_tab(self._notebook.tab("Pipeline"))
        self._build_log_tab(self._notebook.tab("Log"))
        self._build_renders_tab(self._notebook.tab("Renders"))
        self._build_metrics_tab(self._notebook.tab("Metrics"))
        self._build_settings_tab(self._notebook.tab("Settings"))

    def _build_menu(self) -> None:
        menubar = tk.Menu(self, bg="#1F2937", fg="#D1D5DB",
                          activebackground="#374151", activeforeground="white",
                          relief="flat", bd=0)
        self.configure(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0, bg="#1F2937", fg="#D1D5DB",
                            activebackground="#374151")
        file_menu.add_command(label="Open .blend file...", command=self._menu_open_blend)
        file_menu.add_separator()
        file_menu.add_command(label="Open Output Directory", command=self._open_output_dir)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        run_menu = tk.Menu(menubar, tearoff=0, bg="#1F2937", fg="#D1D5DB",
                           activebackground="#374151")
        run_menu.add_command(label="Run Pipeline", command=self._run_pipeline)
        run_menu.add_command(label="Stop", command=self._stop_pipeline)
        menubar.add_cascade(label="Run", menu=run_menu)

        help_menu = tk.Menu(menubar, tearoff=0, bg="#1F2937", fg="#D1D5DB",
                            activebackground="#374151")
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

    def _build_left_sidebar(self, parent: ctk.CTkFrame) -> None:
        # App branding
        brand = ctk.CTkFrame(parent, fg_color="#0F172A", corner_radius=0)
        brand.pack(fill="x")

        ctk.CTkLabel(
            brand,
            text="NL2Scene3D",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#60A5FA",
        ).pack(pady=(14, 0))

        ctk.CTkLabel(
            brand,
            text="Scene Reorganization via MLLM",
            font=ctk.CTkFont(size=10),
            text_color="#6B7280",
        ).pack(pady=(0, 14))

        sep = ctk.CTkFrame(brand, height=1, fg_color="#1F2937")
        sep.pack(fill="x")

        # Pipeline panel
        self._pipeline_panel = PipelinePanel(
            parent,
            config=self._config,
            on_run=self._run_pipeline,
            on_stop=self._stop_pipeline,
        )
        self._pipeline_panel.pack(fill="both", expand=True)

    def _build_pipeline_tab(self, parent) -> None:
        # This tab shows a combined view: top image + live log
        parent.columnconfigure(0, weight=3)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=1)

        self._viewer_mini = ImageViewer(parent)
        self._viewer_mini.grid(row=0, column=0, sticky="nsew", padx=(4, 2), pady=4)

        self._log_mini = LogPanel(parent)
        self._log_mini.grid(row=0, column=1, sticky="nsew", padx=(2, 4), pady=4)

    def _build_log_tab(self, parent) -> None:
        self._log_full = LogPanel(parent)
        self._log_full.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_renders_tab(self, parent) -> None:
        self._viewer_full = ImageViewer(parent)
        self._viewer_full.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_metrics_tab(self, parent) -> None:
        self._metrics_panel = MetricsPanel(parent)
        self._metrics_panel.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_settings_tab(self, parent) -> None:
        self._config_panel = ConfigPanel(
            parent,
            config=self._config,
            on_change=self._on_config_changed,
        )
        self._config_panel.pack(fill="both", expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------
    # Pipeline callbacks (called from worker thread → marshal to main)
    # ------------------------------------------------------------------

    def _on_log(self, level: str, message: str) -> None:
        # Forward to both log panels
        self._log_mini.append(level, message)
        self._log_full.append(level, message)
        # Update pipeline panel step tracker
        self.after(0, self._pipeline_panel.update_from_log, message)

    def _on_image(self, path: Path) -> None:
        self._viewer_mini.add_image(path)
        self._viewer_full.add_image(path)
        # Switch to Pipeline tab to show the new image immediately
        self.after(0, lambda: self._notebook.set("Pipeline"))

    def _on_finished(self, success: bool, error_msg: str | None) -> None:
        self.after(0, self._handle_finished, success, error_msg)

    def _handle_finished(self, success: bool, error_msg: str | None) -> None:
        self._pipeline_panel.set_running(False)
        if self._watcher:
            self._watcher.stop()

        if success:
            self._pipeline_panel.set_status(
                "Pipeline completed successfully.", "#34D399", "✓"
            )
            self._load_metrics()
            self._notebook.set("Metrics")
        else:
            msg = error_msg or "Pipeline failed."
            self._pipeline_panel.set_status(msg, "#EF4444", "✗")
            self._log_full.append("ERROR", f"Pipeline finished with error: {msg}")

    # ------------------------------------------------------------------
    # Run / Stop
    # ------------------------------------------------------------------

    def _run_pipeline(self) -> None:
        if self._runner.is_running:
            return

        self._config_panel.apply_to_config()

        blend = self._pipeline_panel.blend_file
        scene_name = self._pipeline_panel.scene_name

        if not scene_name:
            self._show_error("Scene name is required. Select a .blend file or enter a name.")
            return

        if not self._config.api_key:
            self._show_error(
                "GEMINI_API_KEY is not set.\n"
                "Add it to the .env file in the project root or set it as an environment variable."
            )
            return

        output_dir = self._pipeline_panel.output_dir or (
            self._config.outputs_dir / scene_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Reset UI
        self._log_mini.clear()
        self._log_full.clear()
        self._viewer_mini.clear()
        self._viewer_full.clear()
        self._metrics_panel.clear()

        # Configure runner
        self._runner._config = self._config
        self._runner.blend_file = blend
        self._runner.scene_name = scene_name
        self._runner.output_dir = output_dir
        self._runner.seed = self._config.randomizer_seed
        self._runner.skip_vision = self._config_panel.get_skip_vision()
        self._runner.max_objects = self._config.max_movable_objects
        self._runner.model_override = ""

        # Start watcher
        self._watcher = ImageWatcher(output_dir, on_new_image=self._on_image)
        self._watcher.start()

        # Start runner
        self._pipeline_panel.set_running(True)
        self._pipeline_panel.set_status("Running...", "#60A5FA", "...")
        self._notebook.set("Pipeline")
        self._runner.start()

    def _stop_pipeline(self) -> None:
        self._runner.stop()
        if self._watcher:
            self._watcher.stop()
        self._pipeline_panel.set_running(False)
        self._pipeline_panel.set_status("Stopped by user.", "#F59E0B", "⏹")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _on_config_changed(self) -> None:
        # Live sync without blocking
        pass

    def _apply_config_to_ui(self) -> None:
        pass  # ConfigPanel reads directly from self._config on build

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _load_metrics(self) -> None:
        scene_name = self._pipeline_panel.scene_name
        output_dir = self._pipeline_panel.output_dir or (
            self._config.outputs_dir / scene_name
        )
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            self._metrics_panel.load_from_file(metrics_path)

    # ------------------------------------------------------------------
    # Menu actions
    # ------------------------------------------------------------------

    def _menu_open_blend(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Open .blend file",
            filetypes=[("Blender files", "*.blend"), ("All files", "*.*")],
            initialdir=str(self._config.scenes_dir),
        )
        if path:
            p = Path(path)
            self._pipeline_panel._blend_file = p
            self._pipeline_panel._blend_var.set(path)
            if not self._pipeline_panel._name_var.get():
                self._pipeline_panel._name_var.set(p.stem)

    def _open_output_dir(self) -> None:
        import subprocess, platform
        scene_name = self._pipeline_panel.scene_name
        out = self._pipeline_panel.output_dir or (self._config.outputs_dir / scene_name)
        out.mkdir(parents=True, exist_ok=True)
        if platform.system() == "Windows":
            subprocess.Popen(["explorer", str(out)])
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(out)])
        else:
            subprocess.Popen(["xdg-open", str(out)])

    def _show_about(self) -> None:
        win = ctk.CTkToplevel(self)
        win.title("About NL2Scene3D")
        win.geometry("400x240")
        win.resizable(False, False)
        win.grab_set()

        ctk.CTkLabel(
            win, text="NL2Scene3D",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#60A5FA",
        ).pack(pady=(24, 4))

        ctk.CTkLabel(
            win,
            text="Scene Reorganization via Multimodal Language Models",
            font=ctk.CTkFont(size=11),
            text_color="#9CA3AF",
        ).pack()

        ctk.CTkLabel(
            win, text=f"Version {_APP_VERSION}",
            text_color="#6B7280",
        ).pack(pady=(8, 0))

        ctk.CTkButton(
            win, text="Close", command=win.destroy,
            width=100, height=32,
        ).pack(pady=24)

    def _show_error(self, message: str) -> None:
        win = ctk.CTkToplevel(self)
        win.title("Error")
        win.geometry("440x180")
        win.resizable(False, False)
        win.grab_set()

        ctk.CTkLabel(
            win, text="Error",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#EF4444",
        ).pack(pady=(20, 4))

        ctk.CTkLabel(
            win, text=message,
            wraplength=400,
            text_color="#D1D5DB",
        ).pack(padx=20)

        ctk.CTkButton(win, text="OK", command=win.destroy, width=80).pack(pady=16)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        if self._runner.is_running:
            self._runner.stop()
        if self._watcher:
            self._watcher.stop()
        self.destroy()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    app = NL2Scene3DApp()
    app.mainloop()


if __name__ == "__main__":
    main()