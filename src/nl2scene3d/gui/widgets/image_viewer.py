# gui/widgets/image_viewer.py
"""
Image viewer panel — shows rendered PNG files as they are produced,
with a filmstrip thumbnail strip and a large preview pane.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Optional

import customtkinter as ctk
from PIL import Image, ImageTk


_THUMB_SIZE = (96, 96)
_STEP_LABELS = {
    "original":   "Original",
    "randomized": "Randomized",
    "reordered":  "Reordered",
    "refined":    "Refined / Final",
    "final":      "Final",
}


class ImageViewer(ctk.CTkFrame):
    """Displays rendered images with a thumbnail strip and full preview."""

    def __init__(self, master: ctk.CTkBaseClass, **kwargs) -> None:
        super().__init__(master, **kwargs)

        self._images: list[Path] = []
        self._current_index: int = -1
        self._photo_refs: list[ImageTk.PhotoImage] = []  # prevent GC
        self._thumb_buttons: list[ctk.CTkButton] = []

        self._build_ui()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=6, pady=(6, 0))
        ctk.CTkLabel(
            header, text="Render Viewer",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left")

        self._step_label = ctk.CTkLabel(
            header, text="", text_color="#9CA3AF",
            font=ctk.CTkFont(size=11),
        )
        self._step_label.pack(side="left", padx=8)

        ctk.CTkButton(
            header, text="Clear", width=70, height=26,
            command=self.clear,
            fg_color="#374151", hover_color="#4B5563",
        ).pack(side="right")

        # Main preview area
        preview_frame = ctk.CTkFrame(self, fg_color="#0F172A", corner_radius=6)
        preview_frame.pack(fill="both", expand=True, padx=6, pady=4)

        self._preview_label = ctk.CTkLabel(
            preview_frame, text="No images yet.",
            text_color="#4B5563",
            font=ctk.CTkFont(size=13),
        )
        self._preview_label.pack(fill="both", expand=True)

        # Navigation buttons
        nav = ctk.CTkFrame(self, fg_color="transparent")
        nav.pack(fill="x", padx=6, pady=(0, 2))

        self._prev_btn = ctk.CTkButton(
            nav, text="Previous", width=90, height=28,
            command=self._show_prev,
            fg_color="#374151", hover_color="#4B5563",
            state="disabled",
        )
        self._prev_btn.pack(side="left")

        self._next_btn = ctk.CTkButton(
            nav, text="Next", width=90, height=28,
            command=self._show_next,
            fg_color="#374151", hover_color="#4B5563",
            state="disabled",
        )
        self._next_btn.pack(side="left", padx=4)

        self._index_label = ctk.CTkLabel(nav, text="", text_color="#6B7280")
        self._index_label.pack(side="left", padx=8)

        # Thumbnail strip (horizontal scroll)
        thumb_outer = ctk.CTkFrame(self, fg_color="#1F2937", corner_radius=6, height=112)
        thumb_outer.pack(fill="x", padx=6, pady=(0, 6))
        thumb_outer.pack_propagate(False)

        self._thumb_canvas = tk.Canvas(
            thumb_outer, bg="#1F2937", highlightthickness=0, height=108,
        )
        h_scroll = tk.Scrollbar(thumb_outer, orient="horizontal",
                                command=self._thumb_canvas.xview)
        self._thumb_canvas.configure(xscrollcommand=h_scroll.set)

        h_scroll.pack(side="bottom", fill="x")
        self._thumb_canvas.pack(fill="both", expand=True)

        self._thumb_frame = tk.Frame(self._thumb_canvas, bg="#1F2937")
        self._thumb_canvas.create_window((0, 0), window=self._thumb_frame, anchor="nw")
        self._thumb_frame.bind(
            "<Configure>",
            lambda e: self._thumb_canvas.configure(
                scrollregion=self._thumb_canvas.bbox("all")
            ),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_image(self, path: Path) -> None:
        """Add a new image. Safe to call from any thread."""
        self._preview_label.after(0, self._add_image_main, path)

    def clear(self) -> None:
        self._images.clear()
        self._photo_refs.clear()
        self._current_index = -1
        for btn in self._thumb_buttons:
            btn.destroy()
        self._thumb_buttons.clear()
        self._preview_label.configure(image=None, text="No images yet.")
        self._step_label.configure(text="")
        self._update_nav()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_image_main(self, path: Path) -> None:
        if path in self._images:
            return
        self._images.append(path)
        self._add_thumbnail(path, len(self._images) - 1)
        self._show_index(len(self._images) - 1)

    def _add_thumbnail(self, path: Path, index: int) -> None:
        try:
            img = Image.open(path).convert("RGBA")
            # Use CTkImage for scaling support
            photo = ctk.CTkImage(light_image=img, dark_image=img, size=_THUMB_SIZE)
            self._photo_refs.append(photo)
        except Exception:
            return

        btn = ctk.CTkButton(
            self._thumb_frame,
            image=photo,  # type: ignore[arg-type]
            text="",
            width=_THUMB_SIZE[0] + 4,
            height=_THUMB_SIZE[1] + 4,
            corner_radius=4,
            fg_color="#374151",
            hover_color="#4B5563",
            command=lambda i=index: self._show_index(i),
        )
        btn.pack(side="left", padx=3, pady=4)
        self._thumb_buttons.append(btn)

    def _show_index(self, index: int) -> None:
        if not self._images or index < 0 or index >= len(self._images):
            return

        self._current_index = index
        path = self._images[index]

        try:
            img = Image.open(path).convert("RGBA")
            # Scale to fit the preview frame
            preview_w = max(self._preview_label.winfo_width(), 400)
            preview_h = max(self._preview_label.winfo_height(), 300)
            
            photo = ctk.CTkImage(light_image=img, dark_image=img, size=(preview_w - 8, preview_h - 8))
        except Exception:
            return

        self._preview_label.configure(image=photo, text="")

        # Step label from filename
        stem = path.stem  # e.g. render_original_iso
        step = _STEP_LABELS.get(
            next((k for k in _STEP_LABELS if k in stem), ""), stem
        )
        view_map = {
            "_top": "Top-Down",
            "_iso": "Isometric",
            "_iso2": "Isometric 2",
            "_front": "Frontal"
        }
        view = next((v for k, v in view_map.items() if stem.endswith(k)), "")
        self._step_label.configure(text=f"{step}  {view}".strip())

        self._update_nav()

    def _show_prev(self) -> None:
        self._show_index(self._current_index - 1)

    def _show_next(self) -> None:
        self._show_index(self._current_index + 1)

    def _update_nav(self) -> None:
        n = len(self._images)
        i = self._current_index
        self._prev_btn.configure(state="normal" if i > 0 else "disabled")
        self._next_btn.configure(state="normal" if 0 <= i < n - 1 else "disabled")
        if n > 0:
            self._index_label.configure(text=f"{i + 1} / {n}")
        else:
            self._index_label.configure(text="")