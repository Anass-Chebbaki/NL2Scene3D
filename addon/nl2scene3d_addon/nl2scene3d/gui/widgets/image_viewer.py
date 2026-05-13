# gui/widgets/image_viewer.py
"""
Image viewer panel — shows rendered PNG files as they are produced.

Layout
------
┌─────────────────────────────────────────────┐
│  Render Viewer          [step label] [Clear] │
├─────────────────────────────────────────────┤
│                                             │
│           Main preview (scalable)           │
│           Aspect-ratio preserved            │
│                                             │
├─────────────────────────────────────────────┤
│  [◀ Prev]  [Next ▶]   3 / 7                 │
├─────────────────────────────────────────────┤
│  [thumb][thumb][thumb][thumb] ← scrollable  │
└─────────────────────────────────────────────┘

Thread safety: add_image() is a Qt Slot — safe to connect from any thread.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QSize, Slot
from PySide6.QtGui import QPixmap, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

_THUMB_SIZE = QSize(100, 100)

_STEP_LABELS: dict[str, str] = {
    "original":   "Original",
    "randomized": "Randomized",
    "reordered":  "Reordered",
    "refined":    "Refined / Final",
    "final":      "Final",
}
_VIEW_SUFFIXES: dict[str, str] = {
    "_top":   "Top-Down",
    "_iso2":  "Isometric 2",
    "_iso":   "Isometric",
    "_front": "Frontal",
}


class _PreviewLabel(QLabel):
    """
    A QLabel that scales its pixmap while preserving aspect ratio,
    reacting fluidly to any window resize.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._source_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setMinimumSize(200, 150)
        self.setText("No images yet.")
        self.setStyleSheet("color: #4B5563; font-size: 14px;")

    def set_pixmap_scaled(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self._refresh()

    def clear_image(self) -> None:
        self._source_pixmap = None
        self.setPixmap(QPixmap())
        self.setText("No images yet.")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        self.setText("")
        scaled = self._source_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class ImageViewer(QWidget):
    """Displays rendered images with a thumbnail strip and full preview."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._images: list[Path] = []
        self._current_index: int = -1
        self._build_ui()

    # ── Construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ── Header ──
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Render Viewer")
        title.setStyleSheet("font-weight: 700; font-size: 13px;")
        header_layout.addWidget(title)

        self._step_label = QLabel("")
        self._step_label.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        header_layout.addWidget(self._step_label)
        header_layout.addStretch()

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedSize(72, 28)
        clear_btn.clicked.connect(self.clear)
        header_layout.addWidget(clear_btn)

        layout.addWidget(header)

        # ── Preview area ──
        preview_frame = QFrame()
        preview_frame.setObjectName("card_dark")
        preview_frame.setFrameShape(QFrame.Shape.NoFrame)
        preview_frame_layout = QVBoxLayout(preview_frame)
        preview_frame_layout.setContentsMargins(4, 4, 4, 4)

        self._preview = _PreviewLabel()
        preview_frame_layout.addWidget(self._preview)

        layout.addWidget(preview_frame, stretch=1)

        # ── Navigation ──
        nav = QWidget()
        nav_layout = QHBoxLayout(nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(6)

        self._prev_btn = QPushButton("◀  Prev")
        self._prev_btn.setFixedWidth(90)
        self._prev_btn.setEnabled(False)
        self._prev_btn.clicked.connect(self._show_prev)
        nav_layout.addWidget(self._prev_btn)

        self._next_btn = QPushButton("Next  ▶")
        self._next_btn.setFixedWidth(90)
        self._next_btn.setEnabled(False)
        self._next_btn.clicked.connect(self._show_next)
        nav_layout.addWidget(self._next_btn)

        self._index_label = QLabel("")
        self._index_label.setStyleSheet("color: #6B7280; font-size: 12px;")
        nav_layout.addWidget(self._index_label)
        nav_layout.addStretch()

        layout.addWidget(nav)

        # ── Thumbnail strip ──
        self._thumb_list = QListWidget()
        self._thumb_list.setFlow(QListWidget.Flow.LeftToRight)
        self._thumb_list.setFixedHeight(120)
        self._thumb_list.setIconSize(_THUMB_SIZE)
        self._thumb_list.setSpacing(4)
        self._thumb_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self._thumb_list.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._thumb_list.setStyleSheet("""
            QListWidget {
                background-color: #1F2937;
                border: 1px solid #334155;
                border-radius: 6px;
            }
            QListWidget::item {
                border: 2px solid transparent;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                border: 2px solid #6366F1;
                background-color: #1E293B;
            }
        """)
        self._thumb_list.currentRowChanged.connect(self._show_index)
        layout.addWidget(self._thumb_list)

    # ── Public API ─────────────────────────────────────────────────────────

    @Slot(str)
    def add_image(self, path_str: str) -> None:
        """Add a new image. Safe to call from any thread via Qt signal."""
        path = Path(path_str)
        if path in self._images:
            return
        self._images.append(path)

        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return
        thumb_pixmap = pixmap.scaled(
            _THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        from PySide6.QtGui import QIcon
        item = QListWidgetItem(QIcon(thumb_pixmap), "")
        item.setSizeHint(QSize(_THUMB_SIZE.width() + 8, _THUMB_SIZE.height() + 8))
        self._thumb_list.addItem(item)

        # Auto-select the latest image
        self._thumb_list.setCurrentRow(len(self._images) - 1)

    def clear(self) -> None:
        self._images.clear()
        self._current_index = -1
        self._thumb_list.clear()
        self._preview.clear_image()
        self._step_label.setText("")
        self._update_nav()

    # ── Internal ───────────────────────────────────────────────────────────

    @Slot(int)
    def _show_index(self, index: int) -> None:
        if not self._images or index < 0 or index >= len(self._images):
            return
        self._current_index = index
        path = self._images[index]

        pixmap = QPixmap(str(path))
        if not pixmap.isNull():
            self._preview.set_pixmap_scaled(pixmap)

        # Step + view label from filename stem
        stem = path.stem
        step_key = next((k for k in _STEP_LABELS if k in stem), "")
        step_str = _STEP_LABELS.get(step_key, stem)
        view_str = next((v for k, v in _VIEW_SUFFIXES.items() if stem.endswith(k)), "")
        self._step_label.setText(f"{step_str}  {view_str}".strip())

        self._update_nav()

    def _show_prev(self) -> None:
        self._thumb_list.setCurrentRow(self._current_index - 1)

    def _show_next(self) -> None:
        self._thumb_list.setCurrentRow(self._current_index + 1)

    def _update_nav(self) -> None:
        n = len(self._images)
        i = self._current_index
        self._prev_btn.setEnabled(i > 0)
        self._next_btn.setEnabled(0 <= i < n - 1)
        self._index_label.setText(f"{i + 1} / {n}" if n > 0 else "")