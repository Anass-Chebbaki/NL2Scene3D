# gui/widgets/log_panel.py
"""
Scrollable log panel with colour-coded severity levels.
"""
from __future__ import annotations

import tkinter as tk
from typing import Optional

import customtkinter as ctk

# Colour palette (dark-theme friendly)
_LEVEL_COLORS = {
    "DEBUG":    "#6B7280",  # gray
    "INFO":     "#D1D5DB",  # near-white
    "WARNING":  "#F59E0B",  # amber
    "ERROR":    "#EF4444",  # red
    "CRITICAL": "#DC2626",  # darker red
}
_MAX_LINES = 5_000


class LogPanel(ctk.CTkFrame):
    """Thread-safe, colour-coded log viewer."""

    def __init__(self, master: ctk.CTkBaseClass, **kwargs) -> None:
        super().__init__(master, **kwargs)

        self._build_toolbar()
        self._build_text()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=6, pady=(6, 0))

        ctk.CTkLabel(bar, text="Pipeline Log", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")

        ctk.CTkButton(
            bar, text="Clear", width=70, height=26,
            command=self.clear,
            fg_color="#374151", hover_color="#4B5563",
        ).pack(side="right", padx=(4, 0))

        self._auto_scroll_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            bar, text="Auto-scroll", variable=self._auto_scroll_var,
            height=26, checkbox_width=16, checkbox_height=16,
        ).pack(side="right", padx=4)

        # Level filter
        self._filter_var = tk.StringVar(value="ALL")
        ctk.CTkLabel(bar, text="Filter:").pack(side="right", padx=(8, 2))
        ctk.CTkOptionMenu(
            bar,
            variable=self._filter_var,
            values=["ALL", "DEBUG", "INFO", "WARNING", "ERROR"],
            width=90, height=26,
            command=self._on_filter_change,
        ).pack(side="right")

    def _build_text(self) -> None:
        frame = ctk.CTkFrame(self, fg_color="#111827", corner_radius=6)
        frame.pack(fill="both", expand=True, padx=6, pady=6)

        self._text = tk.Text(
            frame,
            state="disabled",
            wrap="none",
            bg="#111827",
            fg="#D1D5DB",
            insertbackground="#D1D5DB",
            selectbackground="#374151",
            font=("Consolas", 10) if tk.TkVersion >= 8.5 else ("Courier", 10),
            relief="flat",
            borderwidth=0,
        )

        # Tag colours
        for level, color in _LEVEL_COLORS.items():
            self._text.tag_configure(level, foreground=color)

        v_scroll = ctk.CTkScrollbar(frame, command=self._text.yview)
        h_scroll = ctk.CTkScrollbar(frame, orientation="horizontal", command=self._text.xview)

        self._text.configure(
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set,
        )

        h_scroll.pack(side="bottom", fill="x")
        v_scroll.pack(side="right", fill="y")
        self._text.pack(fill="both", expand=True)

        # Internal store for filter support
        self._all_entries: list[tuple[str, str]] = []  # (level, text)

    # ------------------------------------------------------------------
    # Public API (thread-safe via after())
    # ------------------------------------------------------------------

    def append(self, level: str, message: str) -> None:
        """Append a log line. Safe to call from any thread."""
        self._text.after(0, self._append_main, level, message)

    def clear(self) -> None:
        self._all_entries.clear()
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _append_main(self, level: str, message: str) -> None:
        """Must be called on the main thread."""
        self._all_entries.append((level, message))
        if len(self._all_entries) > _MAX_LINES:
            self._all_entries = self._all_entries[-_MAX_LINES:]

        filt = self._filter_var.get()
        if filt != "ALL" and level != filt:
            return

        self._insert_line(level, message)

    def _insert_line(self, level: str, message: str) -> None:
        tag = level if level in _LEVEL_COLORS else "INFO"
        self._text.configure(state="normal")
        self._text.insert("end", message + "\n", tag)

        # Trim to max lines
        line_count = int(self._text.index("end-1c").split(".")[0])
        if line_count > _MAX_LINES:
            self._text.delete("1.0", f"{line_count - _MAX_LINES}.0")

        self._text.configure(state="disabled")
        if self._auto_scroll_var.get():
            self._text.see("end")

    def _on_filter_change(self, _value: str) -> None:
        """Rebuild visible content when the filter changes."""
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")

        filt = self._filter_var.get()
        for level, message in self._all_entries:
            if filt == "ALL" or level == filt:
                self._insert_line(level, message)