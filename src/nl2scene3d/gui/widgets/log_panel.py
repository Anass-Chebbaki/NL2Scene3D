# gui/widgets/log_panel.py
"""
Scrollable log panel with colour-coded severity levels.

Key improvements over the CustomTkinter version:
  - QTextEdit with word-wrap enabled → no horizontal overflow
  - Rich-text (HTML) colour tagging is applied per-line, not per-character
  - Filter dropdown hides/shows lines without clearing the internal buffer
  - Thread-safe: append() is a Qt slot, safe to connect to signals from
    any thread via an auto-connection (Qt::AutoConnection)
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor, QTextOption
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ── Colour palette ────────────────────────────────────────────────────────────
_LEVEL_HTML_COLORS: dict[str, str] = {
    "DEBUG":    "#6B7280",
    "INFO":     "#D1D5DB",
    "WARNING":  "#F59E0B",
    "ERROR":    "#EF4444",
    "CRITICAL": "#DC2626",
}
_MAX_LINES = 5_000


class LogPanel(QWidget):
    """Thread-safe, colour-coded log viewer with severity filter."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._all_entries: list[tuple[str, str]] = []  # [(level, text), …]
        self._build_ui()

    # ── Construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # ── Toolbar ──
        toolbar = QWidget()
        tbar_layout = QHBoxLayout(toolbar)
        tbar_layout.setContentsMargins(6, 6, 6, 0)
        tbar_layout.setSpacing(8)

        title = QLabel("Pipeline Log")
        title.setStyleSheet("font-weight: 700; font-size: 13px;")
        tbar_layout.addWidget(title)
        tbar_layout.addStretch()

        filter_label = QLabel("Filter:")
        filter_label.setStyleSheet("color: #94A3B8;")
        tbar_layout.addWidget(filter_label)

        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["ALL", "DEBUG", "INFO", "WARNING", "ERROR"])
        self._filter_combo.setFixedWidth(110)
        self._filter_combo.currentTextChanged.connect(self._on_filter_change)
        tbar_layout.addWidget(self._filter_combo)

        self._auto_scroll_cb = QCheckBox("Auto-scroll")
        self._auto_scroll_cb.setChecked(True)
        tbar_layout.addWidget(self._auto_scroll_cb)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(72)
        clear_btn.clicked.connect(self.clear)
        tbar_layout.addWidget(clear_btn)

        layout.addWidget(toolbar)

        # ── Text area ──
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setWordWrapMode(QTextOption.WrapAnywhere)  # key fix: word wrap!
        self._text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._text.setStyleSheet("""
            QTextEdit {
                background-color: #111827;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px;
                font-family: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
                font-size: 12px;
                color: #D1D5DB;
            }
        """)
        layout.addWidget(self._text)

    # ── Public API ─────────────────────────────────────────────────────────

    @Slot(str, str)
    def append(self, level: str, message: str) -> None:
        """
        Append a log line.

        This is a Qt Slot — when connected via a queued or auto-connection
        from another thread, Qt will marshal the call to the GUI thread
        automatically (no widget.after() equivalent needed).
        """
        self._all_entries.append((level, message))
        if len(self._all_entries) > _MAX_LINES:
            self._all_entries = self._all_entries[-_MAX_LINES:]

        filt = self._filter_combo.currentText()
        if filt != "ALL" and level != filt:
            return
        self._insert_line(level, message)

    def clear(self) -> None:
        self._all_entries.clear()
        self._text.clear()

    # ── Internal ───────────────────────────────────────────────────────────

    def _insert_line(self, level: str, message: str) -> None:
        color_hex = _LEVEL_HTML_COLORS.get(level, _LEVEL_HTML_COLORS["INFO"])

        # Build a QTextCharFormat for this severity
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color_hex))

        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Insert newline before the line (except the very first)
        if not self._text.document().isEmpty():
            cursor.insertText("\n")
        cursor.insertText(message, fmt)

        # Trim to _MAX_LINES if necessary
        doc = self._text.document()
        while doc.blockCount() > _MAX_LINES:
            cursor2 = QTextCursor(doc.begin())
            cursor2.select(QTextCursor.SelectionType.BlockUnderCursor)
            cursor2.removeSelectedText()
            cursor2.deleteChar()  # remove the trailing newline

        if self._auto_scroll_cb.isChecked():
            self._text.verticalScrollBar().setValue(
                self._text.verticalScrollBar().maximum()
            )

    def _on_filter_change(self, _value: str) -> None:
        """Rebuild visible content when the filter changes."""
        self._text.clear()
        filt = self._filter_combo.currentText()
        # Temporarily suppress auto-scroll during bulk re-insert
        old_autoscroll = self._auto_scroll_cb.isChecked()
        self._auto_scroll_cb.setChecked(False)
        for level, message in self._all_entries:
            if filt == "ALL" or level == filt:
                self._insert_line(level, message)
        self._auto_scroll_cb.setChecked(old_autoscroll)
        if old_autoscroll:
            self._text.verticalScrollBar().setValue(
                self._text.verticalScrollBar().maximum()
            )