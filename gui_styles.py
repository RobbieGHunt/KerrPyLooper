# -*- coding: utf-8 -*-
"""
KerrPyLooper Parameterized GUI Styles
=====================================
Centralized theme palettes and dynamic QSS generator for the KerrPyLooper suite.
Allows easy adjustment of hex values in a single location.
"""

# ------------------------------------------------------------------------------
# THEME PALETTES (Adjust hex values here to customize dark/light/gray modes)
# ------------------------------------------------------------------------------
THEME_PALETTES = {
    "dark": {
        "bg": "#0f172a",                 # Slate 900
        "card": "#1e293b",               # Slate 800
        "border": "#334155",             # Slate 700
        "accent": "#6366f1",             # Indigo 500
        "accent_hover": "#4f46e5",       # Indigo 600
        "accent_pressed": "#4338ca",     # Indigo 700
        "text": "#f8fafc",               # Slate 50
        "text_muted": "#94a3b8",         # Slate 400
        "btn_bg": "#334155",             # Slate 700
        "btn_border": "#475569",         # Slate 600
        "btn_hover": "#475569",
        "btn_pressed": "#1e293b",
        "input_bg": "#1e293b",
        "list_bg": "#020617",            # Slate 950
        "list_item_text": "#cbd5e1",
        "console_bg": "#020617",
        "console_text": "#38bdf8",       # Sky 400
        "progress_bg": "#020617",
        "scrollbar_bg": "#020617",
        "scrollbar_handle": "#334155",
        "scrollbar_handle_hover": "#475569",
        "spine": "#475569",              # Matplotlib axis spines
    },
    "charcoal": {
        "bg": "#18181b",                 # Zinc 900 (neutral dark gray)
        "card": "#27272a",               # Zinc 800
        "border": "#3f3f46",             # Zinc 700
        "accent": "#6366f1",             # Indigo 500 (or #818cf8 for higher contrast)
        "accent_hover": "#4f46e5",
        "accent_pressed": "#3730a3",
        "text": "#f4f4f5",               # Zinc 100
        "text_muted": "#a1a1aa",         # Zinc 400
        "btn_bg": "#3f3f46",             # Zinc 700
        "btn_border": "#52525b",         # Zinc 600
        "btn_hover": "#52525b",
        "btn_pressed": "#27272a",
        "input_bg": "#27272a",
        "list_bg": "#09090b",            # Zinc 950
        "list_item_text": "#d4d4d8",
        "console_bg": "#09090b",
        "console_text": "#38bdf8",
        "progress_bg": "#09090b",
        "scrollbar_bg": "#09090b",
        "scrollbar_handle": "#3f3f46",
        "scrollbar_handle_hover": "#52525b",
        "spine": "#52525b",
    },
    "light": {
        "bg": "#f8fafc",                 # Slate 50
        "card": "#ffffff",
        "border": "#e2e8f0",             # Slate 200
        "accent": "#4f46e5",             # Indigo 600
        "accent_hover": "#4338ca",
        "accent_pressed": "#3730a3",
        "text": "#0f172a",               # Slate 900
        "text_muted": "#64748b",         # Slate 500
        "btn_bg": "#cbd5e1",             # Slate 300
        "btn_border": "#94a3b8",         # Slate 400
        "btn_hover": "#94a3b8",
        "btn_pressed": "#cbd5e1",
        "input_bg": "#ffffff",
        "list_bg": "#ffffff",
        "list_item_text": "#334155",
        "console_bg": "#f8fafc",
        "console_text": "#0284c7",
        "progress_bg": "#e2e8f0",
        "scrollbar_bg": "#f1f5f9",
        "scrollbar_handle": "#cbd5e1",
        "scrollbar_handle_hover": "#94a3b8",
        "spine": "#cbd5e1",
    }
}

# Base stylesheet template using string formatting placeholders
BASE_QSS_TEMPLATE = """
/* BASE STYLES */
QWidget {{
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    background-color: transparent;
    color: {text};
}}

QMainWindow, QWidget#CentralWidget, QWidget#MainBg, QWidget#BatchMainBg {{
    background-color: {bg};
}}

QLabel {{
    color: {text};
}}

QLabel#SuiteTitle {{
    color: {text};
    font-size: 24px;
    font-weight: bold;
}}

QLabel#SuiteSubtitle {{
    color: {text_muted};
    font-size: 13px;
}}

QLabel#SectionTitle {{
    color: {text};
    font-size: 14px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* CARDS */
QFrame#ToolCard {{
    background-color: {card};
    border: 2px solid {border};
    border-radius: 12px;
}}

QFrame#ToolCard:hover {{
    border: 2px solid {accent};
}}

QLabel#CardTitle {{
    color: {text};
    font-size: 18px;
    font-weight: bold;
}}

QLabel#CardSubtitle {{
    color: {accent};
    font-size: 11px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

QLabel#CardDescription {{
    color: {text_muted};
    font-size: 13px;
}}

/* BUTTONS */
QPushButton {{
    background-color: {btn_bg};
    color: {text};
    font-size: 12px;
    font-weight: 600;
    border: 1px solid {btn_border};
    border-radius: 6px;
    padding: 6px 14px;
}}

QPushButton:hover {{
    background-color: {btn_hover};
}}

QPushButton:pressed {{
    background-color: {btn_pressed};
}}

QPushButton:disabled {{
    background-color: {btn_pressed};
    color: {text_muted};
    border: 1px solid {border};
}}

QPushButton#LaunchButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {accent}, stop:1 #3b82f6);
    color: #ffffff;
    font-size: 13px;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
}}

QPushButton#LaunchButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {accent_hover}, stop:1 #2563eb);
}}

QPushButton#LaunchButton:pressed {{
    background: {accent_pressed};
}}

QPushButton#StopButton {{
    background-color: #ef4444;
    color: #ffffff;
    font-size: 12px;
    font-weight: bold;
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
}}

QPushButton#StopButton:hover {{
    background-color: #dc2626;
}}

QPushButton#StopButton:disabled {{
    background-color: {btn_pressed};
    color: {text_muted};
}}

/* CONTROLS (COMBO, SPINBOX, LINEEDIT) */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background-color: {input_bg};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 5px 8px;
    color: {text};
}}

QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{
    border: 1px solid {accent};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 0px;
}}

/* LISTS & CONTAINERS */
QListWidget {{
    background-color: {list_bg};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 5px;
}}

QListWidget::item {{
    padding: 6px 10px;
    border-radius: 4px;
    color: {list_item_text};
}}

QListWidget::item:hover {{
    background-color: {card};
    color: {text};
}}

QListWidget::item:selected {{
    background-color: {accent};
    color: #ffffff;
}}

QGroupBox {{
    border: 2px solid {border};
    border-radius: 8px;
    margin-top: 12px;
    font-weight: bold;
    color: {text};
    padding-top: 12px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: {accent};
}}

/* SLIDERS */
QSlider::groove:horizontal {{
    border: 1px solid {border};
    height: 6px;
    background: {list_bg};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {accent};
    border: none;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}}

QSlider::handle:horizontal:hover {{
    background: {accent_hover};
}}

/* CHECKBOXES */
QCheckBox {{
    color: {list_item_text};
    spacing: 6px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 1px solid {border};
    border-radius: 4px;
    background-color: {input_bg};
}}

QCheckBox::indicator:hover {{
    border: 1px solid {accent};
}}

QCheckBox::indicator:checked {{
    background-color: {accent};
    border: 1px solid {accent};
    image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'><polyline points='20 6 9 17 4 12'></polyline></svg>");
}}

QCheckBox::indicator:checked:disabled {{
    background-color: {border};
    border: 1px solid {border};
    image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='gray' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'><polyline points='20 6 9 17 4 12'></polyline></svg>");
}}

QCheckBox::indicator:disabled {{
    background-color: {bg};
    border: 1px solid {border};
}}

/* CONSOLE LOG */
QTextEdit#ConsoleOutput {{
    background-color: {console_bg};
    color: {console_text};
    border: 1px solid {border};
    border-radius: 8px;
    font-family: "Consolas", "Fira Code", "Courier New", monospace;
    font-size: 12px;
}}

/* PROGRESS BAR */
QProgressBar {{
    background-color: {progress_bg};
    border: 1px solid {border};
    border-radius: 6px;
    text-align: center;
    color: {text};
    height: 12px;
    font-size: 10px;
}}

QProgressBar::chunk {{
    background-color: {accent};
    border-radius: 6px;
}}

/* SPLITTER */
QSplitter::handle {{
    background-color: {border};
}}

QSplitter::handle:hover {{
    background-color: {accent};
}}

QScrollBar:vertical {{
    border: none;
    background: {scrollbar_bg};
    width: 10px;
    margin: 0px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical {{
    background: {scrollbar_handle};
    min-height: 20px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical:hover {{
    background: {scrollbar_handle_hover};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    border: none;
    background: none;
}}

/* TABS */
QTabWidget::pane {{
    border: 1px solid {border};
    border-radius: 6px;
    background-color: {bg};
    top: -1px;
}}

QTabBar::tab {{
    background-color: {btn_bg};
    color: {text_muted};
    border: 1px solid {border};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 6px 16px;
    margin-right: 2px;
}}

QTabBar::tab:selected {{
    background-color: {bg};
    color: {text};
    border: 1px solid {border};
    border-bottom: 1px solid {bg};
}}

QTabBar::tab:hover:!selected {{
    background-color: {btn_hover};
    color: {text};
}}
"""

def get_theme_colors(theme_name="dark"):
    """
    Returns the palette dictionary for the specified theme name.
    Defaults to 'dark' if theme_name is invalid.
    """
    if theme_name not in THEME_PALETTES:
        theme_name = "dark"
    return THEME_PALETTES[theme_name]

def apply_theme(widget_or_app, theme_name="dark"):
    """
    Formats the base QSS template with target theme colors and applies it.
    """
    palette = get_theme_colors(theme_name)
    qss = BASE_QSS_TEMPLATE.format(**palette)
    widget_or_app.setStyleSheet(qss)
