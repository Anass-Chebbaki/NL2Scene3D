# gui/theme.py
"""
Centralized QSS dark theme for NL2Scene3D.
All colors are Slate/Indigo palette, matching the original design intent
but fully rendered via Qt stylesheets (no system-native rendering quirks).
"""

# Color tokens
C_BG        = "#0F172A"   # Slate 900 - window background
C_SURFACE   = "#1E293B"   # Slate 800 - panels, sidebar
C_SURFACE2  = "#334155"   # Slate 700 - borders, separators
C_ACCENT    = "#6366F1"   # Indigo 500
C_ACCENT_HV = "#4F46E5"   # Indigo 600 - hover
C_DANGER    = "#DC2626"   # Red 600
C_DANGER_HV = "#B91C1C"   # Red 700
C_TEXT      = "#F8FAFC"   # Slate 50
C_SUBTEXT   = "#94A3B8"   # Slate 400
C_DISABLED  = "#475569"   # Slate 600
C_INPUT_BG  = "#0F172A"   # same as background
C_SUCCESS   = "#34D399"   # Emerald 400
C_WARNING   = "#F59E0B"   # Amber 400
C_ERROR     = "#EF4444"   # Red 400

STYLESHEET = f"""
/* ─── Global ──────────────────────────────────────────────────── */
* {{
    font-family: "Inter", "Segoe UI", "Roboto", "Arial", sans-serif;
    font-size: 13px;
    color: {C_TEXT};
    outline: none;
}}

QMainWindow, QDialog {{
    background-color: {C_BG};
}}

QWidget {{
    background-color: transparent;
}}

/* ─── MenuBar ─────────────────────────────────────────────────── */
QMenuBar {{
    background-color: {C_SURFACE};
    border-bottom: 1px solid {C_SURFACE2};
    padding: 2px 4px;
    spacing: 2px;
}}

QMenuBar::item {{
    background: transparent;
    padding: 5px 12px;
    border-radius: 4px;
    color: {C_TEXT};
}}

QMenuBar::item:selected,
QMenuBar::item:pressed {{
    background-color: {C_ACCENT};
    color: white;
}}

QMenu {{
    background-color: {C_SURFACE};
    border: 1px solid {C_SURFACE2};
    border-radius: 6px;
    padding: 4px;
}}

QMenu::item {{
    padding: 7px 28px 7px 14px;
    border-radius: 4px;
    color: {C_TEXT};
}}

QMenu::item:selected {{
    background-color: {C_ACCENT};
    color: white;
}}

QMenu::separator {{
    height: 1px;
    background: {C_SURFACE2};
    margin: 4px 8px;
}}

/* ─── QPushButton ─────────────────────────────────────────────── */
QPushButton {{
    background-color: {C_SURFACE2};
    color: {C_TEXT};
    border: none;
    border-radius: 6px;
    padding: 7px 16px;
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: {C_DISABLED};
}}

QPushButton:pressed {{
    background-color: {C_SURFACE2};
}}

QPushButton:disabled {{
    background-color: {C_SURFACE};
    color: {C_DISABLED};
}}

QPushButton#btn_run {{
    background-color: {C_ACCENT};
    font-size: 14px;
    font-weight: 700;
    padding: 12px 16px;
    letter-spacing: 0.5px;
}}

QPushButton#btn_run:hover {{
    background-color: {C_ACCENT_HV};
}}

QPushButton#btn_run:disabled {{
    background-color: #3730a3;
    color: {C_DISABLED};
}}

QPushButton#btn_stop {{
    background-color: {C_DANGER};
    font-weight: 700;
    padding: 9px 16px;
}}

QPushButton#btn_stop:hover {{
    background-color: {C_DANGER_HV};
}}

QPushButton#btn_stop:disabled {{
    background-color: #7f1d1d;
    color: {C_DISABLED};
}}

QPushButton#btn_browse {{
    background-color: {C_DISABLED};
    padding: 5px 10px;
    font-size: 12px;
}}

QPushButton#btn_browse:hover {{
    background-color: {C_SURFACE2};
}}

/* ─── QLineEdit / QSpinBox / QDoubleSpinBox ───────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {C_SURFACE};
    border: 1px solid {C_SURFACE2};
    border-radius: 5px;
    padding: 5px 8px;
    color: {C_TEXT};
    selection-background-color: {C_ACCENT};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {C_ACCENT};
}}

QLineEdit:disabled {{
    color: {C_DISABLED};
    background-color: {C_BG};
}}

QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: {C_SURFACE2};
    border: none;
    border-radius: 3px;
    width: 16px;
}}

QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {C_ACCENT};
}}

/* ─── QComboBox ───────────────────────────────────────────────── */
QComboBox {{
    background-color: {C_SURFACE};
    border: 1px solid {C_SURFACE2};
    border-radius: 5px;
    padding: 5px 8px;
    color: {C_TEXT};
    min-width: 120px;
}}

QComboBox:focus {{
    border: 1px solid {C_ACCENT};
}}

QComboBox::drop-down {{
    border: none;
    background-color: transparent;
    width: 20px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid {C_SUBTEXT};
    width: 0;
    height: 0;
    margin-right: 10px;
}}

QComboBox QAbstractItemView {{
    background-color: {C_SURFACE};
    border: 1px solid {C_SURFACE2};
    border-radius: 5px;
    selection-background-color: {C_ACCENT};
    selection-color: white;
    padding: 4px;
    outline: none;
}}

/* ─── QCheckBox ───────────────────────────────────────────────── */
QCheckBox {{
    spacing: 8px;
    color: {C_TEXT};
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {C_SURFACE2};
    border-radius: 3px;
    background-color: {C_SURFACE};
}}

QCheckBox::indicator:checked {{
    background-color: {C_ACCENT};
    border-color: {C_ACCENT};
    image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjQiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PHBvbHlsaW5lIHBvaW50cz0iMjAgNiA5IDE3IDQgMTIiLz48L3N2Zz4=");
}}

QCheckBox::indicator:hover {{
    border-color: {C_ACCENT};
}}

/* ─── QTabWidget ──────────────────────────────────────────────── */
QTabWidget::pane {{
    background-color: {C_BG};
    border: 1px solid {C_SURFACE2};
    border-radius: 8px;
    top: -1px;
}}

QTabBar::tab {{
    background-color: transparent;
    color: {C_SUBTEXT};
    padding: 8px 20px;
    border-bottom: 2px solid transparent;
    font-weight: 500;
}}

QTabBar::tab:selected {{
    color: {C_TEXT};
    border-bottom: 2px solid {C_ACCENT};
}}

QTabBar::tab:hover:!selected {{
    color: {C_TEXT};
    background-color: {C_SURFACE};
    border-radius: 6px 6px 0 0;
}}

/* ─── QSplitter ───────────────────────────────────────────────── */
QSplitter::handle {{
    background-color: {C_SURFACE2};
    width: 4px;
    border-radius: 2px;
    margin: 8px 0;
}}

QSplitter::handle:hover {{
    background-color: {C_ACCENT};
}}

/* ─── QScrollBar ──────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {C_SURFACE2};
    border-radius: 4px;
    min-height: 24px;
}}

QScrollBar::handle:vertical:hover {{
    background: {C_DISABLED};
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: transparent;
    height: 0;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: {C_SURFACE2};
    border-radius: 4px;
    min-width: 24px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {C_DISABLED};
}}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {{
    background: transparent;
    width: 0;
}}

/* ─── QTextEdit ───────────────────────────────────────────────── */
QTextEdit {{
    background-color: #111827;
    border: 1px solid {C_SURFACE2};
    border-radius: 6px;
    padding: 6px;
    color: {C_TEXT};
    font-family: "JetBrains Mono", "Cascadia Code", "Consolas", "Courier New", monospace;
    font-size: 12px;
    line-height: 1.5;
}}

/* ─── QProgressBar ────────────────────────────────────────────── */
QProgressBar {{
    background-color: {C_BG};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {C_ACCENT};
    border-radius: 4px;
}}

/* ─── QScrollArea ─────────────────────────────────────────────── */
QScrollArea {{
    border: none;
    background-color: transparent;
}}

QScrollArea > QWidget > QWidget {{
    background-color: transparent;
}}

/* ─── QLabel ──────────────────────────────────────────────────── */
QLabel {{
    background: transparent;
}}

QLabel#label_accent {{
    color: {C_ACCENT};
    font-size: 22px;
    font-weight: 700;
}}

QLabel#label_subtext {{
    color: {C_SUBTEXT};
    font-size: 11px;
    font-style: italic;
    font-weight: 600;
    letter-spacing: 1px;
}}

QLabel#section_header {{
    color: {C_ACCENT};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
}}

/* ─── Panel / Card frames ─────────────────────────────────────── */
QFrame#sidebar {{
    background-color: {C_SURFACE};
    border-right: 1px solid {C_SURFACE2};
    border-radius: 0;
}}

QFrame#card {{
    background-color: {C_SURFACE};
    border: 1px solid {C_SURFACE2};
    border-radius: 10px;
}}

QFrame#card_dark {{
    background-color: #1a2744;
    border: 1px solid {C_SURFACE2};
    border-radius: 10px;
}}

QFrame#separator {{
    background-color: {C_SURFACE2};
    max-height: 1px;
}}

/* ─── Tooltips ────────────────────────────────────────────────── */
QToolTip {{
    background-color: {C_SURFACE};
    color: {C_TEXT};
    border: 1px solid {C_SURFACE2};
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11px;
}}
"""