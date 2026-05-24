# -*- coding: utf-8 -*-
"""
Spatial Coercivity Mapping Tool
===============================
Extracts local coercivity (Hc) for every pixel in a Kerr MOKE
hysteresis image series,
generating spatial coercivity maps.
"""

import sys
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QPushButton, QLabel, QGroupBox, QFormLayout, QComboBox,
    QFileDialog, QMessageBox, QProgressBar, QSplitter
)
from PyQt5.QtCore import Qt

from shared_utils.image_processing import crop600
from shared_utils.coercivity import compute_coercivity_map
try:
    from gui_styles import apply_theme, get_theme_colors
except ImportError:
    THEME_PALETTES = {
        "charcoal": {
            "bg": "#18181b",
            "card": "#27272a",
            "border": "#3f3f46",
            "accent": "#6366f1",
            "text": "#f4f4f5",
            "text_muted": "#a1a1aa",
            "btn_bg": "#3f3f46",
            "btn_border": "#52525b",
            "spine": "#52525b"}}

    def get_theme_colors(theme_name="charcoal"):
        return THEME_PALETTES.get(theme_name, THEME_PALETTES["charcoal"])

    def apply_theme(widget, theme_name="charcoal"):
        palette = get_theme_colors(theme_name)
        qss = f"QWidget {{ background-color: {
            palette['bg']}; color: {
            palette['text']}; }}"
        widget.setStyleSheet(qss)


class SpatialCoercivityGUI(QWidget):
    def __init__(self, theme="charcoal", parent=None):
        super().__init__(parent)
        self.theme = theme
        self.setWindowTitle("Spatial Coercivity Mapper")
        self.base_dir = ""
        self.fields = None
        self.filenames = None
        self.hc_pos_map = None
        self.hc_neg_map = None
        self.hc_avg_map = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # Header
        lbl_title = QLabel("Spatial Coercivity Mapper")
        lbl_title.setStyleSheet(
            "font-size: 18px; font-weight: bold; padding: 2px;")
        main_layout.addWidget(lbl_title)

        splitter = QSplitter(Qt.Horizontal)

        # Left Panel
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        grp_dir = QGroupBox("1. Load Dataset")
        dir_layout = QVBoxLayout()

        self.btn_browse = QPushButton("Select Hysteresis Directory")
        self.btn_browse.clicked.connect(self.choose_dir)
        dir_layout.addWidget(self.btn_browse)

        self.lbl_dir = QLabel("No directory selected.")
        self.lbl_dir.setWordWrap(True)
        dir_layout.addWidget(self.lbl_dir)

        self.cmb_mapping = QComboBox()
        dir_layout.addWidget(QLabel("Mapping File:"))
        dir_layout.addWidget(self.cmb_mapping)

        self.btn_load = QPushButton("Load Data")
        self.btn_load.clicked.connect(self.load_data)
        self.btn_load.setEnabled(False)
        dir_layout.addWidget(self.btn_load)

        grp_dir.setLayout(dir_layout)
        left_layout.addWidget(grp_dir)

        self.btn_run = QPushButton("Run Spatial Coercivity Analysis")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_analysis)
        left_layout.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        left_layout.addWidget(self.progress)

        grp_vis = QGroupBox("2. Visualization")
        vis_layout = QFormLayout()

        self.cmb_view = QComboBox()
        self.cmb_view.addItems(["Average Hc", "Hc+", "Hc-"])
        self.cmb_view.currentIndexChanged.connect(self.update_plot)
        vis_layout.addRow("Display Map:", self.cmb_view)

        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(
            ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm"])
        self.cmb_cmap.currentIndexChanged.connect(self.update_plot)
        vis_layout.addRow("Colormap:", self.cmb_cmap)

        grp_vis.setLayout(vis_layout)
        left_layout.addWidget(grp_vis)

        self.btn_export = QPushButton("Export Maps (NumPy/CSV)")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_maps)
        left_layout.addWidget(self.btn_export)

        left_layout.addStretch()
        splitter.addWidget(left_widget)

        # Right Panel
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.fig = plt.figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, stretch=1)
        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)

        self.apply_theme_engine()

    def apply_theme_engine(self):
        apply_theme(self, self.theme)
        colors = get_theme_colors(self.theme)
        self.fig.patch.set_facecolor(colors["bg"])

    def change_theme(self, theme):
        self.theme = theme
        self.apply_theme_engine()
        if self.hc_avg_map is not None:
            self.update_plot()

    def choose_dir(self):
        selected = QFileDialog.getExistingDirectory(self, "Select Directory")
        if selected:
            self.base_dir = selected
            self.lbl_dir.setText(os.path.basename(selected))
            self.cmb_mapping.clear()
            txt_files = [
                f for f in os.listdir(
                    self.base_dir) if f.endswith(".txt")]
            if txt_files:
                self.cmb_mapping.addItems(txt_files)
                self.btn_load.setEnabled(True)
            else:
                QMessageBox.warning(
                    self, "No mapping file", "No .txt mapping files found.")

    def load_data(self):
        map_filename = self.cmb_mapping.currentText()
        if not map_filename:
            return
        txt_path = os.path.join(self.base_dir, map_filename)
        try:
            df = pd.read_csv(
                txt_path,
                sep=None,
                engine="python",
                comment="#",
                skip_blank_lines=True)
            df.columns = [c.strip() for c in df.columns]

            field_col = None
            file_col = None
            for col in df.columns:
                if "field" in col.lower():
                    field_col = col
                elif "file" in col.lower():
                    file_col = col

            if not field_col:
                field_col = df.columns[0]
            if not file_col:
                file_col = df.columns[-1]

            df = df[df[file_col].str.lower().str.endswith(
                ".png", na=False)].reset_index(drop=True)

            self.fields = df[field_col].values.astype(float)
            self.filenames = df[file_col].str.strip().values
            self.btn_run.setEnabled(True)
            QMessageBox.information(
                self, "Loaded", f"Loaded {len(self.filenames)} images.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def run_analysis(self):
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.btn_run.setEnabled(False)
        QApplication.processEvents()

        try:
            # Load all images into 3D array
            images = []
            for i, fname in enumerate(self.filenames):
                path = os.path.join(self.base_dir, fname)
                img = Image.open(path).convert('L')
                arr = crop600(np.array(img).astype(np.float32))
                images.append(arr)

                self.progress.setValue(int(50 * (i + 1) / len(self.filenames)))
                QApplication.processEvents()

            data = np.stack(images, axis=0)  # (frames, h, w)

            hc_pos, hc_neg = compute_coercivity_map(data, self.fields)

            self.hc_pos_map = hc_pos
            self.hc_neg_map = hc_neg

            # Compute average Hc where valid
            self.hc_avg_map = np.full_like(hc_pos, np.nan)
            valid = ~np.isnan(hc_pos) & ~np.isnan(hc_neg)
            self.hc_avg_map[valid] = 0.5 * \
                (np.abs(hc_pos[valid]) + np.abs(hc_neg[valid]))

            self.progress.setValue(100)
            self.btn_export.setEnabled(True)
            self.update_plot()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {e}")
        finally:
            self.progress.setVisible(False)
            self.btn_run.setEnabled(True)

    def update_plot(self):
        if self.hc_avg_map is None:
            return

        self.fig.clear()
        colors = get_theme_colors(self.theme)

        ax = self.fig.add_subplot(111)
        ax.set_facecolor(colors["bg"])

        view = self.cmb_view.currentText()
        if view == "Average Hc":
            map_data = self.hc_avg_map
            title = "Average Spatial Coercivity (mT)"
        elif view == "Hc+":
            map_data = self.hc_pos_map
            title = "Hc+ Spatial Coercivity (mT)"
        else:
            map_data = self.hc_neg_map
            title = "Hc- Spatial Coercivity (mT)"

        cmap_name = self.cmb_cmap.currentText()
        im = ax.imshow(map_data, cmap=cmap_name)

        ax.set_title(title, color=colors["text"])
        ax.axis('off')

        cbar = self.fig.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color=colors["text"])
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=colors["text"])

        self.canvas.draw()

    def export_maps(self):
        if self.hc_avg_map is None:
            return

        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Directory")
        if not out_dir:
            return

        np.save(os.path.join(out_dir, "hc_pos_map.npy"), self.hc_pos_map)
        np.save(os.path.join(out_dir, "hc_neg_map.npy"), self.hc_neg_map)
        np.save(os.path.join(out_dir, "hc_avg_map.npy"), self.hc_avg_map)

        QMessageBox.information(
            self,
            "Exported",
            f"Saved coercivity maps (.npy) to:\n{out_dir}")


def main():
    app = QApplication(sys.argv)
    win = SpatialCoercivityGUI()
    win.resize(1000, 700)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
