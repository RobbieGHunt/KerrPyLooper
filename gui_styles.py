# -*- coding: utf-8 -*-
"""
KerrPyLooper Shared GUI Styles
==============================
Contains Slate Dark and Slate Light theme stylesheets (QSS) 
for the KerrPyLooper analysis suite, ensuring visual consistency.

Created in 2026.
"""

from PyQt5.QtWidgets import QApplication

DARK_THEME_QSS = """
/* BASE STYLES */
QWidget {
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    background-color: transparent;
    color: #f8fafc; /* Slate 50 */
}

QMainWindow, QWidget#CentralWidget, QWidget#MainBg {
    background-color: #0f172a; /* Slate 900 */
}

QLabel {
    color: #f8fafc;
}

QLabel#SuiteTitle {
    color: #f8fafc;
    font-size: 24px;
    font-weight: bold;
}

QLabel#SuiteSubtitle {
    color: #94a3b8; /* Slate 400 */
    font-size: 13px;
}

QLabel#SectionTitle {
    color: #f8fafc;
    font-size: 14px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* CARDS */
QFrame#ToolCard {
    background-color: #1e293b; /* Slate 800 */
    border: 2px solid #334155; /* Slate 700 */
    border-radius: 12px;
}

QFrame#ToolCard:hover {
    border: 2px solid #6366f1; /* Indigo 500 Glow */
}

QLabel#CardTitle {
    color: #f8fafc;
    font-size: 18px;
    font-weight: bold;
}

QLabel#CardSubtitle {
    color: #6366f1; /* Indigo Accent */
    font-size: 11px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
}

QLabel#CardDescription {
    color: #94a3b8;
    font-size: 13px;
}

/* BUTTONS */
QPushButton {
    background-color: #334155; /* Slate 700 */
    color: #f8fafc;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 6px 14px;
}

QPushButton:hover {
    background-color: #475569; /* Slate 600 */
}

QPushButton:pressed {
    background-color: #1e293b;
}

QPushButton:disabled {
    background-color: #1e293b;
    color: #64748b;
    border: 1px solid #334155;
}

QPushButton#LaunchButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366f1, stop:1 #3b82f6);
    color: #ffffff;
    font-size: 13px;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
}

QPushButton#LaunchButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #2563eb);
}

QPushButton#LaunchButton:pressed {
    background: #4338ca;
}

QPushButton#StopButton {
    background-color: #ef4444; /* Red 500 */
    color: #ffffff;
    font-size: 12px;
    font-weight: bold;
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
}

QPushButton#StopButton:hover {
    background-color: #dc2626;
}

QPushButton#StopButton:disabled {
    background-color: #334155;
    color: #64748b;
}

/* CONTROLS (COMBO, SPINBOX, LINEEDIT) */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 5px 8px;
    color: #f8fafc;
}

QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border: 1px solid #6366f1;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 0px;
}

/* LISTS & CONTAINERS */
QListWidget {
    background-color: #020617; /* Slate 950 */
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 5px;
}

QListWidget::item {
    padding: 6px 10px;
    border-radius: 4px;
    color: #cbd5e1;
}

QListWidget::item:hover {
    background-color: #1e293b;
    color: #f8fafc;
}

QListWidget::item:selected {
    background-color: #6366f1;
    color: #ffffff;
}

QGroupBox {
    border: 2px solid #334155;
    border-radius: 8px;
    margin-top: 12px;
    font-weight: bold;
    color: #f8fafc;
    padding-top: 12px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: #6366f1;
}

/* SLIDERS */
QSlider::groove:horizontal {
    border: 1px solid #334155;
    height: 6px;
    background: #020617;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #6366f1;
    border: none;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #4f46e5;
}

/* CHECKBOXES */
QCheckBox {
    color: #cbd5e1;
    spacing: 6px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #334155;
    border-radius: 4px;
    background-color: #1e293b;
}

QCheckBox::indicator:hover {
    border: 1px solid #6366f1;
}

QCheckBox::indicator:checked {
    background-color: #6366f1;
    border: 1px solid #6366f1;
    /* Draw a basic CSS check indicator */
    image: url(data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>);
}

/* CONSOLE LOG */
QTextEdit#ConsoleOutput {
    background-color: #020617;
    color: #38bdf8; /* Sky Blue */
    border: 1px solid #334155;
    border-radius: 8px;
    font-family: "Consolas", "Fira Code", "Courier New", monospace;
    font-size: 12px;
}

/* PROGRESS BAR */
QProgressBar {
    background-color: #020617;
    border: 1px solid #334155;
    border-radius: 6px;
    text-align: center;
    color: #ffffff;
    height: 12px;
    font-size: 10px;
}

QProgressBar::chunk {
    background-color: #6366f1;
    border-radius: 6px;
}

/* SPLITTER */
QSplitter::handle {
    background-color: #334155;
}

QSplitter::handle:hover {
    background-color: #6366f1;
}

QScrollBar:vertical {
    border: none;
    background: #020617;
    width: 10px;
    margin: 0px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #334155;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    border: none;
    background: none;
}
"""

LIGHT_THEME_QSS = """
/* BASE STYLES */
QWidget {
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    background-color: transparent;
    color: #0f172a; /* Slate 900 */
}

QMainWindow, QWidget#CentralWidget, QWidget#MainBg {
    background-color: #f8fafc; /* Slate 50 */
}

QLabel {
    color: #0f172a;
}

QLabel#SuiteTitle {
    color: #0f172a;
    font-size: 24px;
    font-weight: bold;
}

QLabel#SuiteSubtitle {
    color: #64748b; /* Slate 500 */
    font-size: 13px;
}

QLabel#SectionTitle {
    color: #0f172a;
    font-size: 14px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* CARDS */
QFrame#ToolCard {
    background-color: #ffffff;
    border: 2px solid #e2e8f0; /* Slate 200 */
    border-radius: 12px;
}

QFrame#ToolCard:hover {
    border: 2px solid #4f46e5; /* Indigo 600 Glow */
}

QLabel#CardTitle {
    color: #0f172a;
    font-size: 18px;
    font-weight: bold;
}

QLabel#CardSubtitle {
    color: #4f46e5; /* Indigo Accent */
    font-size: 11px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
}

QLabel#CardDescription {
    color: #64748b;
    font-size: 13px;
}

/* BUTTONS */
QPushButton {
    background-color: #cbd5e1; /* Slate 300 */
    color: #0f172a;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid #94a3b8;
    border-radius: 6px;
    padding: 6px 14px;
}

QPushButton:hover {
    background-color: #94a3b8; /* Slate 400 */
}

QPushButton:pressed {
    background-color: #cbd5e1;
}

QPushButton:disabled {
    background-color: #f1f5f9;
    color: #cbd5e1;
    border: 1px solid #cbd5e1;
}

QPushButton#LaunchButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #2563eb);
    color: #ffffff;
    font-size: 13px;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
}

QPushButton#LaunchButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4338ca, stop:1 #1d4ed8);
}

QPushButton#LaunchButton:pressed {
    background: #3730a3;
}

QPushButton#StopButton {
    background-color: #ef4444; /* Red 500 */
    color: #ffffff;
    font-size: 12px;
    font-weight: bold;
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
}

QPushButton#StopButton:hover {
    background-color: #dc2626;
}

QPushButton#StopButton:disabled {
    background-color: #cbd5e1;
    color: #94a3b8;
}

/* CONTROLS (COMBO, SPINBOX, LINEEDIT) */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 5px 8px;
    color: #0f172a;
}

QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border: 1px solid #4f46e5;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 0px;
}

/* LISTS & CONTAINERS */
QListWidget {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 5px;
}

QListWidget::item {
    padding: 6px 10px;
    border-radius: 4px;
    color: #334155;
}

QListWidget::item:hover {
    background-color: #f1f5f9;
    color: #0f172a;
}

QListWidget::item:selected {
    background-color: #4f46e5;
    color: #ffffff;
}

QGroupBox {
    border: 2px solid #cbd5e1;
    border-radius: 8px;
    margin-top: 12px;
    font-weight: bold;
    color: #0f172a;
    padding-top: 12px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: #4f46e5;
}

/* SLIDERS */
QSlider::groove:horizontal {
    border: 1px solid #cbd5e1;
    height: 6px;
    background: #f1f5f9;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #4f46e5;
    border: none;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #4338ca;
}

/* CHECKBOXES */
QCheckBox {
    color: #334155;
    spacing: 6px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    background-color: #ffffff;
}

QCheckBox::indicator:hover {
    border: 1px solid #4f46e5;
}

QCheckBox::indicator:checked {
    background-color: #4f46e5;
    border: 1px solid #4f46e5;
    image: url(data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>);
}

/* CONSOLE LOG */
QTextEdit#ConsoleOutput {
    background-color: #f8fafc;
    color: #0284c7; /* Sky Blue dark */
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    font-family: "Consolas", "Fira Code", "Courier New", monospace;
    font-size: 12px;
}

/* PROGRESS BAR */
QProgressBar {
    background-color: #e2e8f0;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    text-align: center;
    color: #0f172a;
    height: 12px;
    font-size: 10px;
}

QProgressBar::chunk {
    background-color: #4f46e5;
    border-radius: 6px;
}

/* SPLITTER */
QSplitter::handle {
    background-color: #cbd5e1;
}

QSplitter::handle:hover {
    background-color: #4f46e5;
}

QScrollBar:vertical {
    border: none;
    background: #f1f5f9;
    width: 10px;
    margin: 0px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #cbd5e1;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #94a3b8;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    border: none;
    background: none;
}
"""

def apply_theme(widget_or_app, theme_name="dark"):
    """
    Applies the specified theme stylesheet to a QWidget, QMainWindow, or QApplication.
    """
    qss = DARK_THEME_QSS if theme_name == "dark" else LIGHT_THEME_QSS
    widget_or_app.setStyleSheet(qss)
