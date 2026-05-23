# -*- coding: utf-8 -*-
"""
Kerr Magnetization Vector Analysis Tool
=======================================
Calculates and visualizes 2D magnetization vector maps from MOKE image sweeps
measured in X and Y directions.

Features:
- PyQt5 GUI using customized Charcoal/Dark themes from gui_styles.py.
- NumPy-vectorized vector and angle calculations.
- Wavelet denoising using scikit-image (optional, custom scale factor).
- Interactive crop boundary selection via Matplotlib's RectangleSelector.
- Dynamic slider to browse vector maps across the sweep in real-time.
- Scale bar support via lens.json (auto-generates standard presets).
- Single and batch exports of vector maps, color wheel legends, and hysteresis data.

Created in 2026.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle, Ellipse, Polygon as MplPolygon
from matplotlib.widgets import RectangleSelector, EllipseSelector, PolygonSelector
from matplotlib.path import Path

from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QSplitter, QTextEdit, QMessageBox,
    QFileDialog, QProgressBar, QGroupBox, QFormLayout, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QTabWidget, QSlider, QLineEdit,
    QSizePolicy, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QFont
from shared_utils.image_processing import crop600


# Safe imports for theme styles
try:
    from gui_styles import apply_theme, get_theme_colors
except ImportError:
    # Minimal fallback theme engine if running outside project folder structure
    THEME_PALETTES = {
        "charcoal": {
            "bg": "#18181b", "card": "#27272a", "border": "#3f3f46", "accent": "#6366f1",
            "text": "#f4f4f5", "text_muted": "#a1a1aa", "btn_bg": "#3f3f46", "btn_border": "#52525b",
            "spine": "#52525b"
        }
    }
    def get_theme_colors(theme_name="charcoal"):
        return THEME_PALETTES.get(theme_name, THEME_PALETTES["charcoal"])
    def apply_theme(widget, theme_name="charcoal"):
        palette = get_theme_colors(theme_name)
        qss = f"QWidget {{ background-color: {palette['bg']}; color: {palette['text']}; }}"
        widget.setStyleSheet(qss)

# Safe imports for scikit-image restoration
try:
    from skimage.restoration import denoise_wavelet, estimate_sigma
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False



def load_image_to_float32(path):
    with Image.open(path) as img:
        if img.mode in ('L', 'I;16', 'I'):
            arr = np.array(img).astype(np.float32)
            if img.mode == 'I;16' or arr.max() > 255:
                arr = arr / 256.0
            return arr
        else:
            return np.array(img.convert('L')).astype(np.float32)


def normalized_for_display(arr, scale=None, contrast=1.0):
    arr = arr.astype(np.float32)
    arr = crop600(arr)
    if scale is None or scale == 0:
        scale = np.std(arr) if np.std(arr) > 0 else 1.0
    arr_disp = np.arcsinh(arr / scale)
    
    # Exclude outliers (contamination / non-magnetic signals) from biasing the contrast range
    low = np.percentile(arr_disp, 1)
    high = np.percentile(arr_disp, 99)
    if high > low:
        arr_disp = np.clip((arr_disp - low) / (high - low) * 255.0, 0, 255)
    else:
        arr_disp = np.zeros_like(arr_disp)
        
    arr_disp = 127.5 + contrast * (arr_disp - 127.5)
    arr_disp = np.clip(arr_disp, 0, 255).astype(np.uint8)
    return arr_disp


class MovieSettingsDialog(QDialog):
    def __init__(self, parent=None, default_dpi=120):
        super().__init__(parent)
        self.setWindowTitle("Make Movie Settings")
        self.resize(320, 200)
        
        # Apply dark theme stylesheet to dialog if parent has a theme
        if parent and hasattr(parent, 'theme'):
            colors = get_theme_colors(parent.theme)
            self.setStyleSheet(f"""
                QDialog {{
                    background-color: {colors['bg']};
                    color: {colors['text']};
                }}
                QLabel {{
                    color: {colors['text']};
                    font-size: 13px;
                }}
                QComboBox, QSpinBox {{
                    background-color: {colors['input_bg']};
                    color: {colors['text']};
                    border: 1px solid {colors['border']};
                    border-radius: 4px;
                    padding: 4px;
                    font-size: 13px;
                }}
                QPushButton {{
                    background-color: {colors['btn_bg']};
                    color: {colors['text']};
                    border: 1px solid {colors['btn_border']};
                    border-radius: 4px;
                    padding: 6px 12px;
                    font-size: 13px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {colors['btn_hover']};
                }}
                QPushButton:pressed {{
                    background-color: {colors['btn_pressed']};
                }}
            """)
            
        layout = QVBoxLayout(self)
        form = QFormLayout()
        
        self.combo_format = QComboBox()
        self.combo_format.addItems(["GIF Movie (*.gif)", "MP4 Video (*.mp4)"])
        
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 60)
        self.spin_fps.setValue(5)
        
        self.spin_dpi = QSpinBox()
        self.spin_dpi.setRange(50, 300)
        self.spin_dpi.setValue(default_dpi)
        
        form.addRow("Format:", self.combo_format)
        form.addRow("Frames Per Second (FPS):", self.spin_fps)
        form.addRow("Resolution (DPI):", self.spin_dpi)
        
        layout.addLayout(form)
        
        # Buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        
    def get_settings(self):
        fmt = "gif" if self.combo_format.currentIndex() == 0 else "mp4"
        return fmt, self.spin_fps.value(), self.spin_dpi.value()


class VectorAnalysisGUI(QWidget):
    def __init__(self, theme="charcoal", initial_dir="", parent=None):
        super().__init__(parent)
        self.theme = theme
        self.setObjectName("MainBg")
        self.setWindowTitle("Kerr Magnetization Vector Hysteresis Analyzer")
        
        # State variables
        self.base_dir = ""
        self.dir_x = ""
        self.dir_y = ""
        self.mapping_file = ""
        
        self.fields = None
        self.filenames = None
        
        # Hysteresis loop data
        self.mxv = None
        self.myv = None
        self.Mx = None
        self.My = None
        self.mx_mean = 0.5
        self.my_mean = 0.5
        
        self.loop_x_raw = None
        self.loop_y_raw = None
        self.x_offset = 0.0
        self.x_amp = 1.0
        self.y_offset = 0.0
        self.y_amp = 1.0
        self.alpha = 0.0
        self.beta = 0.0
        
        # UI State
        self.current_idx = 0
        self.is_analyzed = False
        self.image_shape = None  # (height, width) of raw images
        
        # Load or create lens.json
        self.load_lens_info()
        
        # Build bright_cmap colormap (hsv with reduced brightness)
        self.setup_colormap()
        
        self.init_ui()
        
        # Auto-load initial directory if specified
        if initial_dir and os.path.exists(initial_dir):
            self.txt_base_dir.setText(initial_dir)
            self.on_base_dir_changed()

    def load_lens_info(self):
        lens_path = "lens.json"
        if not os.path.exists(lens_path):
            # Create a default lens file with standard presets
            default_lenses = {
                "10": [50, 100],  # 10x lens: 50 pixels represents 100 um
                "20": [100, 50],   # 20x lens: 100 pixels represents 50 um
                "50": [250, 20],   # 50x lens: 250 pixels represents 20 um
                "100": [500, 10]   # 100x lens: 500 pixels represents 10 um
            }
            try:
                with open(lens_path, "w") as f:
                    json.dump(default_lenses, f, indent=4)
            except Exception as e:
                print(f"Warning: Could not create default lens.json: {e}")
        
        # Load file
        try:
            with open(lens_path, "r") as f:
                self.lens_info = json.load(f)
        except Exception:
            self.lens_info = {"20": [100, 50]}  # fallback

    def setup_colormap(self):
        hsv_cmap = mpl.colormaps['hsv']
        hsv_bright = hsv_cmap(np.linspace(0, 1, 256))
        brightness = 0.8
        hsv_bright[:, :3] *= brightness
        self.bright_cmap = ListedColormap(hsv_bright)
        self.bright_cmap.set_bad(color='none')

    def init_ui(self):
        # Top-level layouts
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)
        
        # Header title (tight at the top, font size 22px)
        lbl_title = QLabel("2D Magnetization Vector Mapper")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 2px 0; margin: 0;")
        lbl_title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        main_layout.addWidget(lbl_title, 0)
        
        # Horizontal Splitter between Left control panel and Right visualizations
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(8)
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # -------------------------------------------------------------
        # Left Panel (Settings & Operations)
        # -------------------------------------------------------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        # Base Directory Group Box
        grp_dir = QGroupBox("1. Directory Setup")
        dir_layout = QFormLayout()
        
        hbox_dir_btn = QHBoxLayout()
        self.txt_base_dir = QLineEdit()
        self.txt_base_dir.setReadOnly(True)
        self.txt_base_dir.setPlaceholderText("Select base sweep directory...")
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.clicked.connect(self.choose_base_dir)
        hbox_dir_btn.addWidget(self.txt_base_dir, stretch=1)
        hbox_dir_btn.addWidget(self.btn_browse)
        dir_layout.addRow("Base Dir:", hbox_dir_btn)
        
        self.cmb_mapping = QComboBox()
        dir_layout.addRow("Mapping File:", self.cmb_mapping)
        
        self.btn_load_data = QPushButton("Load Sweep Info")
        self.btn_load_data.clicked.connect(self.load_sweep_info)
        self.btn_load_data.setEnabled(False)
        dir_layout.addRow(self.btn_load_data)
        grp_dir.setLayout(dir_layout)
        left_layout.addWidget(grp_dir)
        
        # Crop Coordinates Group Box
        grp_crop = QGroupBox("2. Image Cropping Bounds")
        crop_layout = QFormLayout()
        
        self.lbl_dims = QLabel("Dimensions: N/A")
        self.lbl_dims.setStyleSheet("font-style: italic; color: #a1a1aa;")
        crop_layout.addRow(self.lbl_dims)
        
        # Selector type choice (Rectangle, Circle, Polygon)
        self.cmb_selector_type = QComboBox()
        self.cmb_selector_type.addItems(["Rectangle", "Circle", "Polygon"])
        self.cmb_selector_type.currentIndexChanged.connect(self.on_selector_type_changed)
        crop_layout.addRow("Selector Type:", self.cmb_selector_type)
        
        self.spin_y_start = QSpinBox()
        self.spin_y_start.setRange(0, 5000)
        self.spin_y_start.setValue(0)
        self.spin_y_start.valueChanged.connect(self.on_crop_spinbox_changed)
        
        self.spin_y_end = QSpinBox()
        self.spin_y_end.setRange(1, 5000)
        self.spin_y_end.setValue(300)
        self.spin_y_end.valueChanged.connect(self.on_crop_spinbox_changed)
        
        self.spin_x_start = QSpinBox()
        self.spin_x_start.setRange(0, 5000)
        self.spin_x_start.setValue(290)
        self.spin_x_start.valueChanged.connect(self.on_crop_spinbox_changed)
        
        self.spin_x_end = QSpinBox()
        self.spin_x_end.setRange(1, 5000)
        self.spin_x_end.setValue(665)
        self.spin_x_end.valueChanged.connect(self.on_crop_spinbox_changed)
        
        crop_layout.addRow("Y Start:", self.spin_y_start)
        crop_layout.addRow("Y End:", self.spin_y_end)
        crop_layout.addRow("X Start:", self.spin_x_start)
        crop_layout.addRow("X End:", self.spin_x_end)
        grp_crop.setLayout(crop_layout)
        left_layout.addWidget(grp_crop)
        
        # Denoising Settings Group Box
        grp_denoise = QGroupBox("3. Denoising Settings")
        denoise_layout = QFormLayout()
        
        self.cmb_denoise = QComboBox()
        self.cmb_denoise.addItems(["Raw (No Denoising)", "Wavelet Denoising (Default)", "Wavelet Denoising (Custom Sigma)"])
        if not SKIMAGE_AVAILABLE:
            self.cmb_denoise.setEnabled(False)
            self.cmb_denoise.setToolTip("scikit-image package not found in this environment.")
        self.cmb_denoise.currentIndexChanged.connect(self.update_plots)
        denoise_layout.addRow("Method:", self.cmb_denoise)
        
        self.spin_sf = QDoubleSpinBox()
        self.spin_sf.setRange(0.1, 10.0)
        self.spin_sf.setValue(1.5)
        self.spin_sf.setSingleStep(0.1)
        self.spin_sf.valueChanged.connect(self.update_plots)
        denoise_layout.addRow("Sigma Scale (sf):", self.spin_sf)
        grp_denoise.setLayout(denoise_layout)
        left_layout.addWidget(grp_denoise)
        
        # Quiver and Scale Bar Settings
        grp_vis = QGroupBox("4. Visualization Additions")
        vis_layout = QFormLayout()
        
        self.chk_quiver = QCheckBox("Enable Quiver Arrows")
        self.chk_quiver.setChecked(True)
        self.chk_quiver.stateChanged.connect(self.update_plots)
        vis_layout.addRow(self.chk_quiver)
        
        self.spin_quiver_skip = QSpinBox()
        self.spin_quiver_skip.setRange(2, 100)
        self.spin_quiver_skip.setValue(10)
        self.spin_quiver_skip.valueChanged.connect(self.update_plots)
        vis_layout.addRow("Arrow Skip:", self.spin_quiver_skip)
        
        self.chk_sbar = QCheckBox("Enable Scale Bar")
        self.chk_sbar.setChecked(True)
        self.chk_sbar.stateChanged.connect(self.update_plots)
        vis_layout.addRow(self.chk_sbar)
        
        self.cmb_lens = QComboBox()
        self.cmb_lens.addItems(sorted(self.lens_info.keys(), key=int))
        self.cmb_lens.currentTextChanged.connect(self.update_plots)
        vis_layout.addRow("Lens:", self.cmb_lens)
        grp_vis.setLayout(vis_layout)
        left_layout.addWidget(grp_vis)
        
        # Normalization & Calibration Settings
        grp_norm = QGroupBox("5. Normalization & Calibration")
        norm_layout = QFormLayout()
        
        self.cmb_norm = QComboBox()
        self.cmb_norm.addItems(["None (Standard)", "X-Axis Saturated", "Y-Axis Saturated"])
        self.cmb_norm.currentIndexChanged.connect(self.on_normalization_changed)
        norm_layout.addRow("Saturated Axis:", self.cmb_norm)
        grp_norm.setLayout(norm_layout)
        left_layout.addWidget(grp_norm)
        
        # Action Buttons
        self.btn_run = QPushButton("Run Vector Analysis")
        self.btn_run.setObjectName("LaunchButton")
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_vector_analysis)
        left_layout.addWidget(self.btn_run)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        left_layout.addWidget(self.progress_bar)
        
        # Exports Options
        grp_export = QGroupBox("Export Actions")
        export_layout = QVBoxLayout()
        self.btn_save_plot = QPushButton("Save Current Plot")
        self.btn_save_plot.clicked.connect(self.save_current_plot)
        self.btn_save_plot.setEnabled(False)
        export_layout.addWidget(self.btn_save_plot)
        
        self.btn_save_batch = QPushButton("Batch Save All Fields")
        self.btn_save_batch.clicked.connect(self.save_batch_plots)
        self.btn_save_batch.setEnabled(False)
        export_layout.addWidget(self.btn_save_batch)
        
        self.btn_save_loop = QPushButton("Save Loop Data (.txt)")
        self.btn_save_loop.clicked.connect(self.save_loop_data)
        self.btn_save_loop.setEnabled(False)
        export_layout.addWidget(self.btn_save_loop)
        
        self.btn_save_movie = QPushButton("Make Movie (GIF/MP4)")
        self.btn_save_movie.clicked.connect(self.make_movie)
        self.btn_save_movie.setEnabled(False)
        export_layout.addWidget(self.btn_save_movie)
        
        grp_export.setLayout(export_layout)
        left_layout.addWidget(grp_export)
        
        left_layout.addStretch()
        left_widget.setFixedWidth(270)
        splitter.addWidget(left_widget)
        
        # -------------------------------------------------------------
        # Right Panel (Tabbed Visualizations with synchronized top sliders)
        # -------------------------------------------------------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        
        self.tabs = QTabWidget()
        self.tabs.setElideMode(Qt.ElideNone)
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        self.sliders = []
        self.lbl_sliders = []
        
        def create_slider_layout():
            layout = QVBoxLayout()
            layout.setSpacing(5)
            layout.setContentsMargins(0, 5, 0, 5)
            
            lbl = QLabel("Image Sweep Index: N/A")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-size: 18px; font-weight: bold;")
            
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(0)
            slider.setValue(0)
            slider.setEnabled(False)
            slider.valueChanged.connect(self.on_slider_changed)
            
            slider_layout = QHBoxLayout()
            slider_layout.addStretch(1)
            slider_layout.addWidget(slider, stretch=4)
            slider_layout.addStretch(1)
            
            layout.addWidget(lbl)
            layout.addLayout(slider_layout)
            
            self.sliders.append(slider)
            self.lbl_sliders.append(lbl)
            return layout
        
        # Tab 1: HSV Vector Map
        tab_vector = QWidget()
        tab_vector_layout = QVBoxLayout(tab_vector)
        tab_vector_layout.setContentsMargins(0, 0, 0, 0)
        tab_vector_layout.addLayout(create_slider_layout())
        self.fig_vector = plt.figure(figsize=(10, 6), tight_layout=True)
        self.canvas_vector = FigureCanvas(self.fig_vector)
        self.toolbar_vector = NavigationToolbar(self.canvas_vector, self)
        tab_vector_layout.addWidget(self.toolbar_vector)
        tab_vector_layout.addWidget(self.canvas_vector, stretch=1)
        self.tabs.addTab(tab_vector, "Vector Map")

        
        # Tab 2: Hysteresis Loops
        tab_loops = QWidget()
        tab_loops_layout = QVBoxLayout(tab_loops)
        tab_loops_layout.setContentsMargins(0, 0, 0, 0)
        tab_loops_layout.addLayout(create_slider_layout())
        self.fig_loops = plt.figure(figsize=(10, 6), tight_layout=True)
        self.canvas_loops = FigureCanvas(self.fig_loops)
        self.toolbar_loops = NavigationToolbar(self.canvas_loops, self)
        tab_loops_layout.addWidget(self.toolbar_loops)
        tab_loops_layout.addWidget(self.canvas_loops, stretch=1)
        self.tabs.addTab(tab_loops, "Hysteresis Loops")
        
        # Tab 3: Denoising Comparison
        tab_compare = QWidget()
        tab_compare_layout = QVBoxLayout(tab_compare)
        tab_compare_layout.setContentsMargins(0, 0, 0, 0)
        tab_compare_layout.addLayout(create_slider_layout())
        self.fig_compare = plt.figure(figsize=(10, 6), tight_layout=True)
        self.canvas_compare = FigureCanvas(self.fig_compare)
        self.toolbar_compare = NavigationToolbar(self.canvas_compare, self)
        tab_compare_layout.addWidget(self.toolbar_compare)
        tab_compare_layout.addWidget(self.canvas_compare, stretch=1)
        self.tabs.addTab(tab_compare, "Denoise Comparison")
        
        # Tab 4: Original Images (Cropping Guide)
        tab_full = QWidget()
        tab_full_layout = QVBoxLayout(tab_full)
        tab_full_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_guide = QLabel("ℹ️ Drag a box on the left image to select a cropping region of interest.")
        self.lbl_guide.setStyleSheet("font-size: 16px; padding-bottom: 2px;")
        tab_full_layout.addWidget(self.lbl_guide)
        tab_full_layout.addLayout(create_slider_layout())
        
        self.fig_full = plt.figure(figsize=(10, 6), tight_layout=True)
        self.canvas_full = FigureCanvas(self.fig_full)
        self.toolbar_full = NavigationToolbar(self.canvas_full, self)
        
        # Remove pan and zoom actions to avoid collision with selector
        actions = self.toolbar_full.actions()
        for action in actions:
            text = action.text().lower()
            tooltip = action.toolTip().lower()
            if "pan" in text or "zoom" in text or "pan" in tooltip or "zoom" in tooltip:
                self.toolbar_full.removeAction(action)
                
        # Also disable default key shortcuts on crop selector figure
        try:
            self.fig_full.canvas.mpl_disconnect(self.fig_full.canvas.manager.key_press_handler_id)
        except Exception:
            pass
            
        tab_full_layout.addWidget(self.toolbar_full)
        tab_full_layout.addWidget(self.canvas_full, stretch=1)
        self.tabs.addTab(tab_full, "Original Images (Crop Selector)")
        
        right_layout.addWidget(self.tabs, stretch=1)
        
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)
        
        # Apply CSS themes
        self.apply_theme_engine()
        self.style_all_figures()

    def style_figure_background(self, fig):
        colors = get_theme_colors(self.theme)
        fig.patch.set_facecolor(colors["bg"])
        for ax in fig.axes:
            # Polar projection handles things a bit differently
            if isinstance(ax, mpl.projections.polar.PolarAxes):
                ax.set_facecolor(colors["bg"])
                ax.tick_params(colors=colors["text"])
                continue
            ax.set_facecolor(colors["bg"])
            ax.xaxis.label.set_color(colors["text"])
            ax.yaxis.label.set_color(colors["text"])
            ax.title.set_color(colors["text"])
            ax.tick_params(colors=colors["text"])
            for spine in ax.spines.values():
                spine.set_color(colors["spine"])

    def style_all_figures(self):
        self.style_figure_background(self.fig_vector)
        self.style_figure_background(self.fig_loops)
        self.style_figure_background(self.fig_compare)
        self.style_figure_background(self.fig_full)

    def apply_theme_engine(self):
        apply_theme(self, self.theme)
        
        # Standard colors
        colors = get_theme_colors(self.theme)
        lbl_style = f"color: {colors['text']};"
        self.lbl_dims.setStyleSheet(f"font-style: italic; color: {colors['text_muted']};")
        
        # Update groupboxes and general widget fonts to be larger (14px) via QSS properties
        groupbox_qss = f"""
        QWidget {{
            font-size: 14px;
        }}
        QLabel {{
            font-size: 24px;
        }}
        QPushButton {{
            font-size: 14px;
            font-weight: bold;
        }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox {{
            font-size: 14px;
        }}
        QGroupBox {{
            border: 2px solid {colors['border']};
            border-radius: 8px;
            margin-top: 20px;
            font-weight: bold;
            color: {colors['text']};
            padding-top: 20px;
            font-size: 18px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
            color: {colors['accent']};
            font-size: 23px;
        }}
        """
        tab_qss = f"""
        QTabBar {{
            spacing: 0px;
        }}
        QTabBar::tab {{
            background-color: {colors['card']};
            color: {colors['text']};
            border: 1px solid {colors['border']};
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 8px 16px;
            margin-left: 0px;
            margin-right: -1px;
        }}
        QTabBar::tab:selected {{
            background-color: {colors['bg']};
            color: {colors['text']};
            border-color: {colors['accent']};
        }}
        QTabBar::tab:hover {{
            background-color: {colors['btn_hover']};
            color: {colors['text']};
        }}
        """
        
        slider_qss = f"""
        QSlider::groove:horizontal {{
            border: 1px solid {colors['border']};
            height: 10px;
            background: {colors['list_bg']};
            border-radius: 5px;
        }}
        QSlider::handle:horizontal {{
            background: {colors['accent']};
            border: none;
            width: 20px;
            height: 20px;
            margin: -5px 0;
            border-radius: 10px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {colors['accent_hover']};
        }}
        """
        
        self.setStyleSheet(self.styleSheet() + groupbox_qss)
        self.tabs.setStyleSheet(tab_qss)
        tab_font = QFont("Segoe UI", 12, QFont.Bold)
        self.tabs.tabBar().setFont(tab_font)
        
        # Apply style sheet and force the 23px bold font directly to each QGroupBox widget
        # to bypass native OS styling engines (like Windows Vista Style) that ignore
        # parent-level stylesheet font-size rules for group boxes.
        #This one controls title font size.
        groupbox_style = f"""
        QGroupBox {{
            border: 2px solid {colors['border']};
            border-radius: 8px;
            margin-top: 20px;
            font-weight: bold;
            color: {colors['text']};
            padding-top: 20px
            font-size: 18px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
            color: {colors['accent']};
            font-size: 18px;
        }}
        """
        for grp in self.findChildren(QGroupBox):
            grp.setStyleSheet(groupbox_style)
            grp_font = grp.font()
            grp_font.setPixelSize(18)
            grp_font.setBold(True)
            grp.setFont(grp_font)
            
        for s in self.sliders:
            s.setStyleSheet(slider_qss)

        for lbl in self.lbl_sliders:
            lbl.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {colors['text']};")

        if hasattr(self, 'lbl_guide'):
            self.lbl_guide.setStyleSheet(f"font-size: 16px; color: {colors['text']}; padding-bottom: 2px;")

    def change_theme(self, theme):
        self.theme = theme
        self.apply_theme_engine()
        self.style_all_figures()
        self.update_plots()

    # -------------------------------------------------------------
    # Directory & Loading Callbacks
    # -------------------------------------------------------------
    def choose_base_dir(self):
        selected = QFileDialog.getExistingDirectory(
            self, "Select Magnetization Vector Sweep Folder",
            os.path.dirname(os.path.abspath(__file__))
        )
        if selected:
            self.txt_base_dir.setText(selected)
            self.on_base_dir_changed()

    def on_base_dir_changed(self):
        self.base_dir = self.txt_base_dir.text().strip()
        self.dir_x = os.path.join(self.base_dir, "x")
        self.dir_y = os.path.join(self.base_dir, "y")
        
        # Check subdirectories
        if not os.path.exists(self.dir_x) or not os.path.exists(self.dir_y):
            QMessageBox.critical(
                self, "Directory Error",
                "The base directory must contain both 'x' and 'y' subdirectories."
            )
            self.btn_load_data.setEnabled(False)
            self.btn_run.setEnabled(False)
            return
            
        # Scan mapping txt files
        self.cmb_mapping.clear()
        txt_files = [f for f in os.listdir(self.dir_x) if f.endswith(".txt")]
        if not txt_files:
            # Check if there are txt files in the base dir itself
            txt_files = [f for f in os.listdir(self.base_dir) if f.endswith(".txt")]
            if txt_files:
                self.dir_x = self.base_dir
                self.dir_y = self.base_dir
        
        if not txt_files:
            QMessageBox.warning(
                self, "No Mapping File",
                "Could not find any .txt loop mapping files in the sweep folders."
            )
            self.btn_load_data.setEnabled(False)
            self.btn_run.setEnabled(False)
            return
            
        self.cmb_mapping.addItems(txt_files)
        self.btn_load_data.setEnabled(True)

    def load_sweep_info(self):
        map_filename = self.cmb_mapping.currentText()
        if not map_filename:
            return
        txt_path_x = os.path.join(self.dir_x, map_filename)
        if not os.path.exists(txt_path_x):
            txt_path_x = os.path.join(self.base_dir, map_filename)
            
        txt_path_y = os.path.join(self.dir_y, map_filename)
        if not os.path.exists(txt_path_y):
            txt_path_y = os.path.join(self.base_dir, map_filename)
            
        try:
            # Parse text file using pandas (for x)
            df_x = pd.read_csv(txt_path_x, sep=None, engine="python", comment="#", skip_blank_lines=True)
            df_x.columns = [c.strip() for c in df_x.columns]
            
            # Find field and file columns
            field_col = None
            file_col = None
            for col in df_x.columns:
                if "field" in col.lower():
                    field_col = col
                elif "file" in col.lower():
                    file_col = col
                    
            if field_col is None:
                field_col = df_x.columns[0]
            if file_col is None:
                file_col = df_x.columns[-1]
                
            # Filter rows with png files, ignoring background and unprocessed files
            df_x = df_x[df_x[file_col].str.lower().str.endswith(".png", na=False)].reset_index(drop=True)
            df_x = df_x[~df_x[file_col].str.lower().str.contains("_background|_unprocessed|_unproccessed", na=False)].reset_index(drop=True)
            
            # Try to load y mapping file
            if os.path.exists(txt_path_y) and txt_path_y != txt_path_x:
                df_y = pd.read_csv(txt_path_y, sep=None, engine="python", comment="#", skip_blank_lines=True)
                df_y.columns = [c.strip() for c in df_y.columns]
                df_y = df_y[df_y[file_col].str.lower().str.endswith(".png", na=False)].reset_index(drop=True)
                df_y = df_y[~df_y[file_col].str.lower().str.contains("_background|_unprocessed|_unproccessed", na=False)].reset_index(drop=True)
            else:
                df_y = df_x.copy()

            # Find GrayLevel columns
            gray_col_x = None
            gray_col_y = None
            for col in df_x.columns:
                if "gray" in col.lower() or "level" in col.lower() or "intensity" in col.lower():
                    gray_col_x = col
                    break
            if gray_col_x is None:
                gray_col_x = df_x.columns[1] if len(df_x.columns) > 2 else df_x.columns[0]
                
            for col in df_y.columns:
                if "gray" in col.lower() or "level" in col.lower() or "intensity" in col.lower():
                    gray_col_y = col
                    break
            if gray_col_y is None:
                gray_col_y = df_y.columns[1] if len(df_y.columns) > 2 else df_y.columns[0]
                
            # Align if lengths differ
            if len(df_x) != len(df_y):
                merged = pd.merge(df_x, df_y, on=file_col, suffixes=('_x', '_y'))
                self.fields = merged[field_col + '_x'].values.astype(float)
                self.filenames = merged[file_col].str.strip().values
                self.loop_x_raw = merged[gray_col_x + '_x'].values.astype(float)
                self.loop_y_raw = merged[gray_col_y + '_y'].values.astype(float)
            else:
                self.fields = df_x[field_col].values.astype(float)
                self.filenames = df_x[file_col].str.strip().values
                self.loop_x_raw = df_x[gray_col_x].values.astype(float)
                self.loop_y_raw = df_y[gray_col_y].values.astype(float)
            
            if len(self.filenames) == 0:
                QMessageBox.warning(self, "Parse Error", "No image filenames found in mapping text file.")
                return
                
            # Open first image to setup crop boundary maximums and show preview
            first_path_x = os.path.join(self.base_dir, "x", self.filenames[0])
            if not os.path.exists(first_path_x):
                QMessageBox.critical(self, "File Not Found", f"Could not find image: {first_path_x}")
                return
                
            img_x_raw = load_image_to_float32(first_path_x)
            img_x_cropped = crop600(img_x_raw)
            height, width = img_x_cropped.shape
            self.image_shape = (height, width)
            
            # Update spinboxes boundaries
            self.spin_x_start.setRange(0, width - 1)
            self.spin_x_end.setRange(1, width)
            self.spin_y_start.setRange(0, height - 1)
            self.spin_y_end.setRange(1, height)
            
            # Adjust default crop values if they exceed image boundaries
            self.spin_x_start.setValue(min(290, width // 4))
            self.spin_x_end.setValue(min(665, width * 3 // 4))
            self.spin_y_start.setValue(0)
            self.spin_y_end.setValue(min(300, height // 2))
            
            self.lbl_dims.setText(f"Dimensions: {width} x {height} ({len(self.filenames)} fields)")
            
            # Update slider bounds for all synchronized sliders
            for s in self.sliders:
                s.blockSignals(True)
                s.setMinimum(0)
                s.setMaximum(len(self.filenames) - 1)
                s.setValue(0)
                s.setEnabled(True)
                s.blockSignals(False)
            self.current_idx = 0
            
            # Initialize normalization values
            self.update_normalized_loop_data()
            
            # Plot the initial uncropped preview in Tab 4
            self.plot_full_preview()
            
            self.btn_run.setEnabled(True)
            self.is_analyzed = False
            self.update_plots()
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to parse loop mapping file:\n{e}")
            import traceback
            traceback.print_exc()

    # -------------------------------------------------------------
    # Plotting Functions
    # -------------------------------------------------------------
    def plot_full_preview(self):
        """Draw full image in Tab 4 and setup RectangleSelector."""
        if self.filenames is None or len(self.filenames) == 0:
            return
            
        fname = self.filenames[self.current_idx]
        path_x = os.path.join(self.base_dir, "x", fname)
        path_y = os.path.join(self.base_dir, "y", fname)
        
        try:
            raw_x = load_image_to_float32(path_x)
            raw_y = load_image_to_float32(path_y)
            
            im_x_disp = normalized_for_display(raw_x, contrast=1.0)
            im_y_disp = normalized_for_display(raw_y, contrast=1.0)
            
            self.fig_full.clear()
            self.style_figure_background(self.fig_full)
            
            self.ax_full_x = self.fig_full.add_subplot(1, 2, 1)
            self.ax_full_y = self.fig_full.add_subplot(1, 2, 2)
            
            self.ax_full_x.imshow(im_x_disp, cmap='gray', vmin=0, vmax=255)
            self.ax_full_x.set_title("Full X Sweep Image", color=get_theme_colors(self.theme)["text"])
            self.ax_full_x.axis('off')
            
            self.ax_full_y.imshow(im_y_disp, cmap='gray', vmin=0, vmax=255)
            self.ax_full_y.set_title("Full Y Sweep Image", color=get_theme_colors(self.theme)["text"])
            self.ax_full_y.axis('off')
            
            # Initialize active selector type
            self.init_selector()
            
            self.update_crop_box_indicators()
            self.canvas_full.draw()
            
        except Exception as e:
            print(f"Error loading full image preview: {e}")

    def init_selector(self):
        """Initializes the active Matplotlib selector on ax_full_x."""
        if not hasattr(self, 'ax_full_x') or self.ax_full_x is None:
            return
            
        # Clean up existing selectors
        if hasattr(self, 'rect_selector') and self.rect_selector is not None:
            self.rect_selector.set_active(False)
            self.rect_selector = None
        if hasattr(self, 'ellipse_selector') and self.ellipse_selector is not None:
            self.ellipse_selector.set_active(False)
            self.ellipse_selector = None
        if hasattr(self, 'poly_selector') and self.poly_selector is not None:
            self.poly_selector.set_active(False)
            self.poly_selector = None
            
        shape = self.cmb_selector_type.currentText()
        
        if shape == "Rectangle":
            self.rect_selector = RectangleSelector(
                self.ax_full_x, self.on_crop_select,
                useblit=True, button=1,
                minspanx=5, minspany=5,
                props=dict(facecolor='green', edgecolor='green', alpha=0.3, fill=True)
            )
        elif shape == "Circle":
            self.ellipse_selector = EllipseSelector(
                self.ax_full_x, self.on_crop_select,
                useblit=True, button=1,
                minspanx=5, minspany=5,
                props=dict(facecolor='green', edgecolor='green', alpha=0.3, fill=True)
            )
        elif shape == "Polygon":
            self.poly_selector = PolygonSelector(
                self.ax_full_x, self.on_polygon_select,
                useblit=True,
                props=dict(color='lime', alpha=0.3)
            )

    def on_selector_type_changed(self):
        """Called when the region selector type dropdown changes."""
        if self.filenames is None or len(self.filenames) == 0:
            return
        self.init_selector()
        self.update_crop_box_indicators()

    def on_crop_select(self, eclick, erelease):
        """Callback from RectangleSelector or EllipseSelector click-drag."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        x_start = max(0, min(x1, x2))
        x_end = min(self.image_shape[1], max(x1, x2))
        y_start = max(0, min(y1, y2))
        y_end = min(self.image_shape[0], max(y1, y2))
        
        # Block signals briefly to avoid infinite callback loop
        self.spin_x_start.blockSignals(True)
        self.spin_x_end.blockSignals(True)
        self.spin_y_start.blockSignals(True)
        self.spin_y_end.blockSignals(True)
        
        self.spin_x_start.setValue(x_start)
        self.spin_x_end.setValue(x_end)
        self.spin_y_start.setValue(y_start)
        self.spin_y_end.setValue(y_end)
        
        self.spin_x_start.blockSignals(False)
        self.spin_x_end.blockSignals(False)
        self.spin_y_start.blockSignals(False)
        self.spin_y_end.blockSignals(False)
        
        self.update_crop_box_indicators()
        
        if self.is_analyzed:
            # Need to re-run Pass 1 since bounds changed
            self.btn_run.setStyleSheet(f"background-color: {get_theme_colors(self.theme)['accent']}; color: #ffffff;")
            self.btn_run.setText("Run Vector Analysis (Crop Changed)")

    def on_polygon_select(self, verts):
        """Callback from PolygonSelector when shape is completed."""
        if len(verts) < 3:
            return
            
        self.polygon_vertices = verts
        
        # Get bounding box of the polygon
        xs = [p[0] for p in verts]
        ys = [p[1] for p in verts]
        x_start = max(0, int(np.floor(min(xs))))
        x_end = min(self.image_shape[1], int(np.ceil(max(xs))))
        y_start = max(0, int(np.floor(min(ys))))
        y_end = min(self.image_shape[0], int(np.ceil(max(ys))))
        
        # Block signals briefly to avoid infinite callback loop
        self.spin_x_start.blockSignals(True)
        self.spin_x_end.blockSignals(True)
        self.spin_y_start.blockSignals(True)
        self.spin_y_end.blockSignals(True)
        
        self.spin_x_start.setValue(x_start)
        self.spin_x_end.setValue(x_end)
        self.spin_y_start.setValue(y_start)
        self.spin_y_end.setValue(y_end)
        
        self.spin_x_start.blockSignals(False)
        self.spin_x_end.blockSignals(False)
        self.spin_y_start.blockSignals(False)
        self.spin_y_end.blockSignals(False)
        
        self.update_crop_box_indicators()
        
        if self.is_analyzed:
            self.btn_run.setStyleSheet(f"background-color: {get_theme_colors(self.theme)['accent']}; color: #ffffff;")
            self.btn_run.setText("Run Vector Analysis (Crop Changed)")

    def compute_current_mask(self):
        """Generates a boolean mask of shape (y_end - y_start, x_end - x_start)."""
        x_start = self.spin_x_start.value()
        x_end = self.spin_x_end.value()
        y_start = self.spin_y_start.value()
        y_end = self.spin_y_end.value()
        
        ny = y_end - y_start
        nx = x_end - x_start
        if ny <= 0 or nx <= 0:
            return None
            
        shape = self.cmb_selector_type.currentText()
        if shape == "Rectangle":
            return None  # No mask, use full rectangle
            
        elif shape == "Circle":
            # Generate ellipse/circle mask within the bounding box
            y_indices, x_indices = np.ogrid[:ny, :nx]
            xc = nx / 2.0
            yc = ny / 2.0
            rx = nx / 2.0
            ry = ny / 2.0
            if rx == 0: rx = 1.0
            if ry == 0: ry = 1.0
            mask = ((x_indices - xc) / rx)**2 + ((y_indices - yc) / ry)**2 <= 1.0
            return mask
            
        elif shape == "Polygon":
            if not hasattr(self, 'polygon_vertices') or self.polygon_vertices is None or len(self.polygon_vertices) < 3:
                return None
                
            # Generate polygon mask using matplotlib Path contains_points
            y_grid, x_grid = np.mgrid[y_start:y_end, x_start:x_end]
            points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
            
            path = Path(self.polygon_vertices)
            mask = path.contains_points(points).reshape((ny, nx))
            return mask
            
        return None

    def on_crop_spinbox_changed(self):
        """Callback when crop spinbox coordinates are edited manually."""
        self.update_crop_box_indicators()
        if self.is_analyzed:
            self.btn_run.setStyleSheet(f"background-color: {get_theme_colors(self.theme)['accent']}; color: #ffffff;")
            self.btn_run.setText("Run Vector Analysis (Crop Changed)")

    def update_crop_box_indicators(self):
        """Draw green dotted indicators on full images indicating current crop bounds/shapes."""
        if not hasattr(self, 'ax_full_x') or self.filenames is None:
            return
            
        x_start = self.spin_x_start.value()
        x_end = self.spin_x_end.value()
        y_start = self.spin_y_start.value()
        y_end = self.spin_y_end.value()
        
        # Clear previous rectangles/drawings
        for patch in list(self.ax_full_x.patches):
            patch.remove()
        for patch in list(self.ax_full_y.patches):
            patch.remove()
            
        w = x_end - x_start
        h = y_end - y_start
        
        shape = self.cmb_selector_type.currentText()
        if shape == "Rectangle":
            rect_x = Rectangle((x_start, y_start), w, h, edgecolor='lime', facecolor='none', linestyle='--', linewidth=2)
            rect_y = Rectangle((x_start, y_start), w, h, edgecolor='lime', facecolor='none', linestyle='--', linewidth=2)
            self.ax_full_x.add_patch(rect_x)
            self.ax_full_y.add_patch(rect_y)
        elif shape == "Circle":
            ellipse_x = Ellipse((x_start + w/2.0, y_start + h/2.0), w, h, edgecolor='lime', facecolor='none', linestyle='--', linewidth=2)
            ellipse_y = Ellipse((x_start + w/2.0, y_start + h/2.0), w, h, edgecolor='lime', facecolor='none', linestyle='--', linewidth=2)
            self.ax_full_x.add_patch(ellipse_x)
            self.ax_full_y.add_patch(ellipse_y)
        elif shape == "Polygon":
            if hasattr(self, 'polygon_vertices') and self.polygon_vertices is not None and len(self.polygon_vertices) >= 3:
                poly_x = MplPolygon(self.polygon_vertices, edgecolor='lime', facecolor='none', linestyle='--', linewidth=2)
                poly_y = MplPolygon(self.polygon_vertices, edgecolor='lime', facecolor='none', linestyle='--', linewidth=2)
                self.ax_full_x.add_patch(poly_x)
                self.ax_full_y.add_patch(poly_y)
            else:
                # Fallback to bounding box rectangle
                rect_x = Rectangle((x_start, y_start), w, h, edgecolor='lime', facecolor='none', linestyle='--', linewidth=2)
                rect_y = Rectangle((x_start, y_start), w, h, edgecolor='lime', facecolor='none', linestyle='--', linewidth=2)
                self.ax_full_x.add_patch(rect_x)
                self.ax_full_y.add_patch(rect_y)
        
        self.canvas_full.draw_idle()

    # -------------------------------------------------------------
    # Hysteresis Operations & Execution
    # -------------------------------------------------------------
    def run_vector_analysis(self):
        """Executes Pass 1 of analysis, computing crop means and loop centering variables."""
        if self.filenames is None or len(self.filenames) == 0:
            return
            
        x_start = self.spin_x_start.value()
        x_end = self.spin_x_end.value()
        y_start = self.spin_y_start.value()
        y_end = self.spin_y_end.value()
        
        if x_start >= x_end or y_start >= y_end:
            QMessageBox.critical(self, "Invalid Bounds", "Start coordinates must be strictly less than End coordinates.")
            return
            
        # Reset progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.btn_run.setEnabled(False)
        QApplication.processEvents()
        
        N = len(self.filenames)
        self.mxv = []
        self.myv = []
        
        mask = self.compute_current_mask()
        has_valid_mask = mask is not None and np.any(mask)
        
        # Pass 1: Compute average cropped intensities for each image (no denoising in Pass 1)
        for i, fname in enumerate(self.filenames):
            path_x = os.path.join(self.base_dir, "x", fname)
            path_y = os.path.join(self.base_dir, "y", fname)
            
            try:
                # Load images (grayscale, normalized to 0-1) and crop immediately
                im_x = crop600(load_image_to_float32(path_x) / 255.0)
                im_y = crop600(load_image_to_float32(path_y) / 255.0)
                
                # Crop to sub-region
                crop_x = im_x[y_start:y_end, x_start:x_end]
                crop_y = im_y[y_start:y_end, x_start:x_end]
                
                if has_valid_mask:
                    self.mxv.append(np.mean(crop_x[mask]))
                    self.myv.append(np.mean(crop_y[mask]))
                else:
                    self.mxv.append(np.mean(crop_x))
                    self.myv.append(np.mean(crop_y))
                
            except Exception as e:
                QMessageBox.critical(self, "Processing Error", f"Error in image {fname}:\n{e}")
                self.progress_bar.setVisible(False)
                self.btn_run.setEnabled(True)
                return
                
            # Update progress
            progress = int((i + 1) / N * 100)
            self.progress_bar.setValue(progress)
            QApplication.processEvents()
            
        # Compute image-based calibration parameters
        idx_max = np.argmax(self.fields)
        idx_min = np.argmin(self.fields)
        
        self.img_mx_mean = np.mean(self.mxv)
        self.img_my_mean = np.mean(self.myv)
        
        mxv_arr = np.array(self.mxv)
        myv_arr = np.array(self.myv)
        
        # X-Axis Saturated parameters for images
        self.img_x_offset = (mxv_arr[idx_max] + mxv_arr[idx_min]) / 2.0
        self.img_x_amp = (mxv_arr[idx_max] - mxv_arr[idx_min]) / 2.0
        if self.img_x_amp == 0:
            self.img_x_amp = 1.0
        self.img_y_offset_for_x_sat = (myv_arr[idx_max] + myv_arr[idx_min]) / 2.0
        self.img_alpha = (myv_arr[idx_max] - myv_arr[idx_min]) / 2.0
        
        # Y-Axis Saturated parameters for images
        self.img_y_offset = (myv_arr[idx_max] + myv_arr[idx_min]) / 2.0
        self.img_y_amp = (myv_arr[idx_max] - myv_arr[idx_min]) / 2.0
        if self.img_y_amp == 0:
            self.img_y_amp = 1.0
        self.img_x_offset_for_y_sat = (mxv_arr[idx_max] + mxv_arr[idx_min]) / 2.0
        self.img_beta = (mxv_arr[idx_max] - mxv_arr[idx_min]) / 2.0
        
        # Update loop data
        self.update_normalized_loop_data()
        
        # Plot loops
        self.plot_hysteresis_loops()
        
        self.is_analyzed = True
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Vector Analysis")
        self.btn_run.setStyleSheet("")  # clear highlight accent
        self.progress_bar.setVisible(False)
        
        # Enable exports
        self.btn_save_plot.setEnabled(True)
        self.btn_save_batch.setEnabled(True)
        self.btn_save_loop.setEnabled(True)
        self.btn_save_movie.setEnabled(True)
        
        # Trigger redraw
        self.update_plots()

    def plot_hysteresis_loops(self):
        """Draw loops in Tab 2."""
        self.fig_loops.clear()
        self.style_figure_background(self.fig_loops)
        
        text_color = get_theme_colors(self.theme)["text"]
        
        self.ax_loop_x = self.fig_loops.add_subplot(2, 1, 1)
        self.ax_loop_y = self.fig_loops.add_subplot(2, 1, 2)
        
        # Plot loop X
        self.ax_loop_x.plot(self.fields, self.Mx, 'o-', label="Mx loop signal", color="#6366f1", markersize=4)
        self.ax_loop_x.set_title("X-Direction Hysteresis Loop", color=text_color)
        self.ax_loop_x.set_ylabel("Magnetization Mx (a.u.)", color=text_color)
        self.ax_loop_x.grid(True, linestyle=':', alpha=0.5)
        
        # Plot loop Y
        self.ax_loop_y.plot(self.fields, self.My, 'o-', label="My loop signal", color="#10b981", markersize=4)
        self.ax_loop_y.set_title("Y-Direction Hysteresis Loop", color=text_color)
        self.ax_loop_y.set_xlabel("Magnetic Field (mT)", color=text_color)
        self.ax_loop_y.set_ylabel("Magnetization My (a.u.)", color=text_color)
        self.ax_loop_y.grid(True, linestyle=':', alpha=0.5)
        
        self.canvas_loops.draw()

    def compute_local_vectors(self, im_x, im_y):
        """Vectorized calculations for denoising, centering, and angle maps."""
        crop_x = im_x[self.spin_y_start.value():self.spin_y_end.value(),
                      self.spin_x_start.value():self.spin_x_end.value()]
        crop_y = im_y[self.spin_y_start.value():self.spin_y_end.value(),
                      self.spin_x_start.value():self.spin_x_end.value()]
                      
        method = self.cmb_denoise.currentText()
        sf = self.spin_sf.value()
        
        # Denoising
        if SKIMAGE_AVAILABLE and "Wavelet Denoising" in method:
            if "Default" in method:
                cx = denoise_wavelet(crop_x)
                cy = denoise_wavelet(crop_y)
            else:  # Custom Sigma
                sx_est = estimate_sigma(crop_x, average_sigmas=True)
                sy_est = estimate_sigma(crop_y, average_sigmas=True)
                s_est = np.sqrt(sx_est**2 + sy_est**2) * sf
                cx = denoise_wavelet(crop_x, sigma=s_est, rescale_sigma=True)
                cy = denoise_wavelet(crop_y, sigma=s_est, rescale_sigma=True)
        else:
            cx = crop_x
            cy = crop_y
            
        # Local centered/calibrated magnetization components
        mode = self.cmb_norm.currentText()
        if mode == "X-Axis Saturated":
            mx_local = (cx - self.img_x_offset) / self.img_x_amp
            my_local = (cy - self.img_y_offset_for_x_sat - self.img_alpha * mx_local) / self.img_x_amp
        elif mode == "Y-Axis Saturated":
            my_local = (cy - self.img_y_offset) / self.img_y_amp
            mx_local = (cx - self.img_x_offset_for_y_sat - self.img_beta * my_local) / self.img_y_amp
        else: # "None (Standard)"
            mx_local = cx - self.img_mx_mean
            my_local = cy - self.img_my_mean
        
        # Normalize local vectors to get angles and unit components
        mag = np.sqrt(mx_local**2 + my_local**2)
        mag_safe = np.where(mag == 0, 1.0, mag)
        
        vx = mx_local / mag_safe
        vy = my_local / mag_safe
        
        vx[mag == 0] = 0
        vy[mag == 0] = 0
        
        theta = np.arctan2(my_local, mx_local)
        
        # Apply shape mask if any
        mask = self.compute_current_mask()
        if mask is not None:
            inv_mask = ~mask
            crop_x = crop_x.copy()
            crop_y = crop_y.copy()
            cx = cx.copy()
            cy = cy.copy()
            vx = vx.copy()
            vy = vy.copy()
            theta = theta.copy()
            
            crop_x[inv_mask] = 0.0
            crop_y[inv_mask] = 0.0
            cx[inv_mask] = 0.0
            cy[inv_mask] = 0.0
            vx[inv_mask] = 0.0
            vy[inv_mask] = 0.0
            theta[inv_mask] = np.nan
            
        return crop_x, crop_y, cx, cy, vx, vy, theta

    def update_plots(self):
        """Redraws the tabs with calculated vector details for current field index."""
        if self.filenames is None or len(self.filenames) == 0:
            return
            
        idx = self.current_idx
        total = len(self.filenames)
        fname = self.filenames[idx]
        field = self.fields[idx]
        
        for lbl in self.lbl_sliders:
            lbl.setText(f"Image Sweep Index: {idx + 1} / {total} | File: {fname}")
        
        # Load images for selected field
        path_x = os.path.join(self.base_dir, "x", fname)
        path_y = os.path.join(self.base_dir, "y", fname)
        
        try:
            im_x = crop600(load_image_to_float32(path_x) / 255.0)
            im_y = crop600(load_image_to_float32(path_y) / 255.0)
        except Exception as e:
            print(f"Error loading images for plot update: {e}")
            return
            
        # Draw full preview crop guide (updates selection highlights)
        if self.tabs.currentIndex() == 3:
            # Full Images tab selected, draw full preview
            self.plot_full_preview()
            
        # If not analyzed yet, we cannot plot Tabs 1, 2, 3 details
        if not self.is_analyzed:
            return
            
        # Run vector operations for this frame
        crop_rx, crop_ry, cx, cy, vx, vy, theta = self.compute_local_vectors(im_x, im_y)
        
        text_color = get_theme_colors(self.theme)["text"]
        
        # 1. Update Tab 1: Vector HSV Map & Polar Legend
        if self.tabs.currentIndex() == 0:
            self.fig_vector.clear()
            self.style_figure_background(self.fig_vector)
            
            # Map Axes (Left, 80% width)
            ax_map = self.fig_vector.add_axes([0.05, 0.05, 0.70, 0.90])
            ax_map.imshow(theta, vmin=-np.pi, vmax=np.pi, cmap=self.bright_cmap)
            ax_map.axis('off')
            
            # Quiver Arrows
            if self.chk_quiver.isChecked():
                skip = self.spin_quiver_skip.value()
                Ny, Nx = theta.shape
                xx = np.arange(0, Nx, skip)
                yy = np.arange(0, Ny, skip)
                XX, YY = np.meshgrid(xx, yy)
                # quiver specifies x component then y component
                ax_map.quiver(XX, YY, vx[::skip, ::skip], -vy[::skip, ::skip], pivot='mid', color='black', alpha=0.9)
                
            # Scale Bar
            if self.chk_sbar.isChecked():
                lens = self.cmb_lens.currentText()
                if lens in self.lens_info:
                    bar_length, bar_size = self.lens_info[lens]
                    bar_x = 20
                    bar_y = theta.shape[0] - 20
                    ax_map.plot([bar_x, bar_x + bar_length], [bar_y, bar_y], color='white', linewidth=4)
                    ax_map.text(bar_x + bar_length / 2.0, bar_y - 10, f"{bar_size} µm",
                                ha='center', va='bottom', color='white', fontsize=16, fontweight='bold')
                                
            # Polar Legend (Right, 20% width)
            ax_polar = self.fig_vector.add_axes([0.80, 0.30, 0.16, 0.40], projection="polar")
            ax_polar.set_facecolor(get_theme_colors(self.theme)["bg"])
            ax_polar.xaxis.set_ticks([])
            ax_polar.yaxis.set_ticks([])
            ax_polar.set_aspect('equal')
            ax_polar.grid(False)
            
            normc = mpl.colors.Normalize(-np.pi, np.pi)
            n_segments = 400
            phi_grid = np.linspace(-np.pi, np.pi, n_segments)
            radius_grid = np.linspace(0.5, 1.0, 2)
            _, tg = np.meshgrid(radius_grid, phi_grid)
            ax_polar.pcolormesh(phi_grid, radius_grid, tg.T, norm=normc, cmap=self.bright_cmap)
            ax_polar.spines['polar'].set_visible(False)
            ax_polar.set_title(f"{field:.2f} mT", color=text_color, y=1.1, fontsize=12, fontweight='bold')
            
            self.canvas_vector.draw()
            
        # 2. Update Tab 2: Hysteresis indicator dots
        elif self.tabs.currentIndex() == 1:
            # Remove previous current indicators if any
            if hasattr(self, 'ax_loop_x'):
                # Pop out the dots/lines if we plotted them last time
                for line in list(self.ax_loop_x.lines)[1:]: # Keep first plotted line (the loop itself)
                    line.remove()
                for line in list(self.ax_loop_y.lines)[1:]:
                    line.remove()
                    
                # Draw new vertical line and red dot indicator
                self.ax_loop_x.axvline(field, color='red', linestyle='--', alpha=0.5)
                self.ax_loop_x.plot(field, self.Mx[idx], 'ro', markersize=7)
                
                self.ax_loop_y.axvline(field, color='red', linestyle='--', alpha=0.5)
                self.ax_loop_y.plot(field, self.My[idx], 'ro', markersize=7)
                
                self.canvas_loops.draw()
                
        # 3. Update Tab 3: Denoise Comparison
        elif self.tabs.currentIndex() == 2:
            self.fig_compare.clear()
            self.style_figure_background(self.fig_compare)
            
            axs = self.fig_compare.subplots(2, 2)
            
            crop_rx_disp = normalized_for_display(crop_rx, contrast=1.0)
            cx_disp = normalized_for_display(cx, contrast=1.0)
            crop_ry_disp = normalized_for_display(crop_ry, contrast=1.0)
            cy_disp = normalized_for_display(cy, contrast=1.0)
            
            axs[0, 0].imshow(crop_rx_disp, cmap='gray', vmin=0, vmax=255)
            axs[0, 0].set_title("Raw X Image", color=text_color)
            axs[0, 0].axis('off')
            
            axs[0, 1].imshow(cx_disp, cmap='gray', vmin=0, vmax=255)
            axs[0, 1].set_title("Processed/Denoised X", color=text_color)
            axs[0, 1].axis('off')
            
            axs[1, 0].imshow(crop_ry_disp, cmap='gray', vmin=0, vmax=255)
            axs[1, 0].set_title("Raw Y Image", color=text_color)
            axs[1, 0].axis('off')
            
            axs[1, 1].imshow(cy_disp, cmap='gray', vmin=0, vmax=255)
            axs[1, 1].set_title("Processed/Denoised Y", color=text_color)
            axs[1, 1].axis('off')
            
            self.canvas_compare.draw()

    # -------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------
    def on_slider_changed(self):
        sender = self.sender()
        if sender is not None:
            self.current_idx = sender.value()
            for s in self.sliders:
                if s != sender:
                    s.blockSignals(True)
                    s.setValue(self.current_idx)
                    s.blockSignals(False)
        else:
            if self.sliders:
                self.current_idx = self.sliders[0].value()
        self.update_plots()

    def on_tab_changed(self, index):
        self.update_plots()

    def on_normalization_changed(self):
        """Called when Saturated Axis normalization dropdown selection changes."""
        self.update_normalized_loop_data()
        self.plot_hysteresis_loops()
        self.update_plots()

    def update_normalized_loop_data(self):
        if self.loop_x_raw is None or self.loop_y_raw is None or self.fields is None:
            return
            
        idx_max = np.argmax(self.fields)
        idx_min = np.argmin(self.fields)
        
        mode = self.cmb_norm.currentText()
        
        if mode == "X-Axis Saturated":
            # X-axis parameters
            self.x_offset = (self.loop_x_raw[idx_max] + self.loop_x_raw[idx_min]) / 2.0
            self.x_amp = (self.loop_x_raw[idx_max] - self.loop_x_raw[idx_min]) / 2.0
            if self.x_amp == 0:
                self.x_amp = 1.0
                
            # Y-axis parameters (crosstalk alpha)
            self.y_offset = (self.loop_y_raw[idx_max] + self.loop_y_raw[idx_min]) / 2.0
            self.alpha = (self.loop_y_raw[idx_max] - self.loop_y_raw[idx_min]) / 2.0
            
            # Compute Mx and My loops
            self.Mx = (self.loop_x_raw - self.x_offset) / self.x_amp
            self.My = (self.loop_y_raw - self.y_offset - self.alpha * self.Mx) / self.x_amp
            
        elif mode == "Y-Axis Saturated":
            # Y-axis parameters
            self.y_offset = (self.loop_y_raw[idx_max] + self.loop_y_raw[idx_min]) / 2.0
            self.y_amp = (self.loop_y_raw[idx_max] - self.loop_y_raw[idx_min]) / 2.0
            if self.y_amp == 0:
                self.y_amp = 1.0
                
            # X-axis parameters (crosstalk beta)
            self.x_offset = (self.loop_x_raw[idx_max] + self.loop_x_raw[idx_min]) / 2.0
            self.beta = (self.loop_x_raw[idx_max] - self.loop_x_raw[idx_min]) / 2.0
            
            # Compute Mx and My loops
            self.My = (self.loop_y_raw - self.y_offset) / self.y_amp
            self.Mx = (self.loop_x_raw - self.x_offset - self.beta * self.My) / self.y_amp
            
        else: # "None (Standard)"
            self.mx_mean = np.mean(self.loop_x_raw)
            self.my_mean = np.mean(self.loop_y_raw)
            self.Mx = self.loop_x_raw - self.mx_mean
            self.My = self.loop_y_raw - self.my_mean

    # -------------------------------------------------------------
    # Export Actions
    # -------------------------------------------------------------
    def save_current_plot(self):
        """Save Tab 1 vector map to disk."""
        if not self.is_analyzed:
            return
            
        field = self.fields[self.current_idx]
        default_filename = f"vector_map_{field:.2f}mT.png"
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Current Vector Map Plot", default_filename,
            "PNG Image (*.png);;All Files (*)"
        )
        if filepath:
            try:
                # Temporarily switch back Tab 1 active state to force render, then save
                self.tabs.setCurrentIndex(0)
                self.fig_vector.savefig(filepath, transparent=True, dpi=150)
                QMessageBox.information(self, "Saved", f"Vector plot saved successfully to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save current vector plot:\n{e}")

    def save_batch_plots(self):
        """Export all frames vector maps to directory."""
        if not self.is_analyzed:
            return
            
        out_dir = QFileDialog.getExistingDirectory(
            self, "Choose Output Directory for Batch Vector Map Plots",
            self.base_dir
        )
        if not out_dir:
            return
            
        target_folder = os.path.join(out_dir, f"VectorMaps_crop_{self.spin_x_start.value()}_{self.spin_y_start.value()}")
        os.makedirs(target_folder, exist_ok=True)
        
        # Save previous selected tab index
        prev_tab = self.tabs.currentIndex()
        self.tabs.setCurrentIndex(0)  # Switch to Tab 1 to render
        
        N = len(self.filenames)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        QApplication.processEvents()
        
        try:
            for i, fname in enumerate(self.filenames):
                self.current_idx = i
                for s in self.sliders:
                    s.blockSignals(True)
                    s.setValue(i)
                    s.blockSignals(False)
                self.update_plots()
                QApplication.processEvents()
                
                # Save plot file name
                field = self.fields[i]
                out_path = os.path.join(target_folder, f"vector_plot_{i:04d}_{field:.2f}mT.png")
                self.fig_vector.savefig(out_path, transparent=True, dpi=120)
                
                # Update progress
                self.progress_bar.setValue(int((i + 1) / N * 100))
                QApplication.processEvents()
                
            QMessageBox.information(
                self, "Batch Saved",
                f"Exported all {N} vector maps successfully to:\n{target_folder}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Batch Save Error", f"An error occurred during batch saving:\n{e}")
        finally:
            self.progress_bar.setVisible(False)
            self.tabs.setCurrentIndex(prev_tab)
            if self.sliders:
                self.current_idx = self.sliders[0].value()
            self.update_plots()

    def save_loop_data(self):
        """Export loop vectors list (Fields, Mx, My) to tab delimited text file."""
        if not self.is_analyzed:
            return
            
        default_filename = "vector_loop_data.txt"
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Hysteresis Loop Vector Signals", default_filename,
            "Text Files (*.txt);;All Files (*)"
        )
        if not filepath:
            return
            
        try:
            # Build data frame
            df_out = pd.DataFrame({
                "Field(mT)": self.fields,
                "Mx_average": self.Mx,
                "My_average": self.My,
                "Intensity_X_raw": self.mxv,
                "Intensity_Y_raw": self.myv
            })
            
            # Write meta headers
            with open(filepath, "w") as f:
                f.write(f"# KerrpyLooper Magnetization Vector Hysteresis Loop Signal\n")
                f.write(f"# Base Directory: {self.base_dir}\n")
                f.write(f"# Crop Boundaries: X({self.spin_x_start.value()}:{self.spin_x_end.value()}), Y({self.spin_y_start.value()}:{self.spin_y_end.value()})\n")
                mode = self.cmb_norm.currentText()
                f.write(f"# Normalization Mode: {mode}\n")
                if mode == "X-Axis Saturated":
                    f.write(f"# Calibration Parameters: x_offset={self.x_offset:.6f}, x_amp={self.x_amp:.6f}, y_offset={self.y_offset:.6f}, alpha={self.alpha:.6f}\n")
                elif mode == "Y-Axis Saturated":
                    f.write(f"# Calibration Parameters: y_offset={self.y_offset:.6f}, y_amp={self.y_amp:.6f}, x_offset={self.x_offset:.6f}, beta={self.beta:.6f}\n")
                else:
                    f.write(f"# Calibration Parameters: mx_mean={self.mx_mean:.6f}, my_mean={self.my_mean:.6f}\n")
                df_out.to_csv(f, sep='\t', index=False)
                
            QMessageBox.information(self, "Saved", f"Loop vector data saved to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save data file:\n{e}")

    def make_movie(self):
        """Export movie (GIF or MP4) combining vector maps for all field sweep indices."""
        if not self.is_analyzed:
            return
            
        # 1. Ask for settings
        dialog = MovieSettingsDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return
            
        fmt, fps, dpi = dialog.get_settings()
        
        # 2. Get file path to save
        filter_str = "GIF Movie (*.gif)" if fmt == "gif" else "MP4 Video (*.mp4)"
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Movie", self.base_dir, filter_str
        )
        if not filepath:
            return
            
        # 3. Check for FFmpeg if MP4 chosen
        if fmt == "mp4":
            import shutil
            if not shutil.which("ffmpeg"):
                QMessageBox.warning(
                    self, "FFmpeg Missing",
                    "FFmpeg executable was not found on your system PATH.\n\n"
                    "MP4 movie export requires FFmpeg. Please install FFmpeg or choose the GIF format instead."
                )
                return
                
        # 4. Compile the frames
        # Save previous selected tab index
        prev_tab = self.tabs.currentIndex()
        self.tabs.setCurrentIndex(0)  # Switch to Tab 1 (Vector Map) to render
        
        N = len(self.filenames)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        QApplication.processEvents()
        
        frames = []
        import io
        
        try:
            for i, fname in enumerate(self.filenames):
                self.current_idx = i
                for s in self.sliders:
                    s.blockSignals(True)
                    s.setValue(i)
                    s.blockSignals(False)
                self.update_plots()
                QApplication.processEvents()
                
                # Render the current vector map plot to a PIL Image using savefig with RGBA format
                # ⚡ Bolt Optimization: Writing raw RGBA bytes instead of PNG avoids compression/decompression
                # overhead, making frame capture roughly 2-3x faster without mutating the live GUI canvas.
                buf = io.BytesIO()
                self.fig_vector.savefig(buf, format='rgba', dpi=dpi, transparent=True)
                buf.seek(0)

                w_in, h_in = self.fig_vector.get_size_inches()
                w, h = int(w_in * dpi), int(h_in * dpi)
                img = Image.frombuffer('RGBA', (w, h), buf.read(), 'raw', 'RGBA', 0, 1).copy()
                frames.append(img)
                buf.close()
                
                # Update progress (first 80% is frame rendering)
                self.progress_bar.setValue(int((i + 1) / N * 80))
                QApplication.processEvents()
                
            # Now compile the frames into the output movie file
            if fmt == "gif":
                duration_ms = int(1000.0 / fps)
                frames[0].save(
                    filepath,
                    save_all=True,
                    append_images=frames[1:],
                    duration=duration_ms,
                    loop=0
                )
            else:
                import matplotlib.animation as animation
                
                # Create a simple figure matching image size in inches
                img_w, img_h = frames[0].size
                fig_anim, ax_anim = plt.subplots(figsize=(img_w/dpi, img_h/dpi), dpi=dpi)
                fig_anim.patch.set_facecolor('none')
                ax_anim.axis('off')
                fig_anim.subplots_adjust(left=0, right=1, bottom=0, top=1)
                
                im_plot = ax_anim.imshow(np.array(frames[0]))
                
                writer = animation.FFMpegWriter(fps=fps)
                with writer.saving(fig_anim, filepath, dpi=dpi):
                    for idx, img in enumerate(frames):
                        im_plot.set_data(np.array(img))
                        writer.grab_frame()
                        
                        # Update progress for the remaining 20%
                        self.progress_bar.setValue(int(80 + (idx + 1) / N * 20))
                        QApplication.processEvents()
                
                plt.close(fig_anim)
                
            QMessageBox.information(
                self, "Movie Saved",
                f"Movie exported successfully to:\n{filepath}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Movie Save Error", f"An error occurred during movie generation:\n{e}")
        finally:
            self.progress_bar.setVisible(False)
            self.tabs.setCurrentIndex(prev_tab)
            if self.sliders:
                self.current_idx = self.sliders[0].value()
            self.update_plots()


def main():
    # Parse CLI Arguments
    import argparse
    parser = argparse.ArgumentParser(description="Kerr Magnetization Vector Analysis Tool")
    parser.add_argument("img_dir", nargs="?", default="", help="Directory containing x and y loop directories")
    parser.add_argument("--theme", default="charcoal", help="GUI style theme (charcoal, dark, light)")
    args = parser.parse_args()

    # Enable High DPI scaling if supported
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
    app = QApplication(sys.argv)
    window = VectorAnalysisGUI(theme=args.theme, initial_dir=args.img_dir)
    window.resize(1100, 750)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
