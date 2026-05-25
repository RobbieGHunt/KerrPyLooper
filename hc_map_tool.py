# -*- coding: utf-8 -*-
import sys
import os
import concurrent.futures
import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage as ndimage
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (
    QApplication, QWidget, QFileDialog, QVBoxLayout, QPushButton,
    QListWidget, QLabel, QMessageBox, QHBoxLayout, QSlider,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSplitter, QTextEdit,
    QCheckBox, QComboBox, QSpinBox, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from shared_utils.image_processing import crop600, calculate_local_hc

from gui_styles import apply_theme

class HcCalcThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, img_stack, loop_field, bin_size):
        super().__init__()
        self.img_stack = img_stack
        self.loop_field = loop_field
        self.bin_size = bin_size

    def run(self):
        try:
            hc_map = calculate_local_hc(self.img_stack, self.loop_field, self.bin_size)
            self.finished.emit(hc_map)
        except Exception as e:
            self.error.emit(str(e))

class HcMapGUI(QWidget):
    def __init__(self, theme="dark", parent=None):
        super().__init__(parent)
        self.theme = theme
        self.setObjectName("MainBg")
        self.setWindowTitle("Local Coercivity (Hc) Mapping")

        self.img_dir = None
        self.txt_data = None
        self.loop_field = None
        self.background_array = None

        self.hc_map = None
        self.figure = None
        self.ax = None
        self.canvas = None
        self.toolbar = None

        self.init_ui()

    def change_theme(self, theme):
        self.theme = theme
        apply_theme(self, theme)
        self.replot_map()

    def init_ui(self):
        main_layout = QHBoxLayout()

        # Left Panel: Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.btn_dir = QPushButton("Select Hysteresis Image Directory")
        self.btn_dir.clicked.connect(self.choose_directory)
        left_layout.addWidget(self.btn_dir)

        self.lbl_dir_status = QLabel("No directory selected")
        left_layout.addWidget(self.lbl_dir_status)

        group_params = QGroupBox("Mapping Parameters")
        form_params = QFormLayout()

        self.spin_bin_size = QSpinBox()
        self.spin_bin_size.setRange(1, 32)
        self.spin_bin_size.setValue(4)
        form_params.addRow("Binning Size (px):", self.spin_bin_size)

        group_params.setLayout(form_params)
        left_layout.addWidget(group_params)

        self.btn_calc = QPushButton("Calculate Hc Map")
        self.btn_calc.clicked.connect(self.calculate_map)
        self.btn_calc.setEnabled(False)
        left_layout.addWidget(self.btn_calc)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        group_plot = QGroupBox("Plot Settings")
        form_plot = QFormLayout()
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["plasma", "viridis", "inferno", "magma", "cividis", "jet"])
        self.cmb_cmap.currentTextChanged.connect(self.replot_map)
        form_plot.addRow("Colormap:", self.cmb_cmap)

        self.chk_overlay = QCheckBox("Overlay on Zero-Field Image")
        self.chk_overlay.stateChanged.connect(self.replot_map)
        form_plot.addRow(self.chk_overlay)

        self.sld_alpha = QSlider(Qt.Horizontal)
        self.sld_alpha.setRange(0, 100)
        self.sld_alpha.setValue(70)
        self.sld_alpha.valueChanged.connect(self.replot_map)
        form_plot.addRow("Overlay Alpha:", self.sld_alpha)

        group_plot.setLayout(form_plot)
        left_layout.addWidget(group_plot)

        self.btn_save = QPushButton("Save Map Image")
        self.btn_save.clicked.connect(self.save_map)
        self.btn_save.setEnabled(False)
        left_layout.addWidget(self.btn_save)

        left_layout.addStretch()

        # Right Panel: Canvas
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        from gui_styles import get_theme_colors
        colors = get_theme_colors(self.theme)
        fig_bg = colors["card"]
        ax_bg = colors["bg"]

        self.figure = Figure(facecolor=fig_bg)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(ax_bg)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 3)
        self.setLayout(main_layout)

        apply_theme(self, self.theme)

    def choose_directory(self):
        d = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if d:
            self.img_dir = d
            self.load_data()

    def load_data(self):
        txt_file = None
        for fn in os.listdir(self.img_dir):
            if fn.lower().endswith('.txt'):
                txt_file = os.path.join(self.img_dir, fn)
                break

        if not txt_file:
            self.lbl_dir_status.setText("Error: No mapping .txt file found.")
            self.btn_calc.setEnabled(False)
            return

        try:
            df = pd.read_csv(txt_file, sep=None, engine='python', comment="#", skip_blank_lines=True)
            df.columns = [c.strip() for c in df.columns]
            df = df[df[df.columns[2]].str.lower().str.endswith(".png", na=False)]
            self.txt_data = df.rename(
                columns={df.columns[0]:"Field", df.columns[1]:"Intensity", df.columns[2]:"File"}
            ).reset_index(drop=True)
            self.txt_data["File"] = self.txt_data["File"].str.strip()
            self.loop_field = self.txt_data["Field"].to_numpy(dtype=np.float32)

            zero_idx = np.argmin(np.abs(self.loop_field))
            bg_file = self.txt_data.iloc[zero_idx]["File"]
            bg_path = os.path.join(self.img_dir, bg_file)
            self.background_array = crop600(np.array(Image.open(bg_path)))

            self.lbl_dir_status.setText(f"Loaded {len(self.loop_field)} images.")
            self.btn_calc.setEnabled(True)
        except Exception as e:
            self.lbl_dir_status.setText(f"Error loading data: {e}")
            self.btn_calc.setEnabled(False)

    def calculate_map(self):
        if self.txt_data is None or self.background_array is None:
            return

        self.btn_calc.setEnabled(False)
        self.btn_dir.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate

        QApplication.processEvents()

        # Load all images into memory (cropped)
        img_stack = []
        for f in self.txt_data["File"]:
            path = os.path.join(self.img_dir, f)
            img = crop600(np.array(Image.open(path)))

            # Simple drift correction to first image if needed
            if len(img_stack) == 0:
                img_stack.append(img.astype(np.float32))
            else:
                img_stack.append(img.astype(np.float32))

        img_stack = np.array(img_stack) # shape: (N, H, W)

        bin_size = self.spin_bin_size.value()

        # Run calculation in a background thread
        self.calc_thread = HcCalcThread(img_stack, self.loop_field, bin_size)
        self.calc_thread.finished.connect(self.on_calc_finished)
        self.calc_thread.error.connect(self.on_calc_error)
        self.calc_thread.start()

    def on_calc_finished(self, hc_map):
        self.hc_map = hc_map
        self.replot_map()
        self.btn_save.setEnabled(True)
        self._reset_calc_ui()

    def on_calc_error(self, err_msg):
        QMessageBox.critical(self, "Calculation Error", err_msg)
        self._reset_calc_ui()

    def _reset_calc_ui(self):
        self.btn_calc.setEnabled(True)
        self.btn_dir.setEnabled(True)
        self.progress_bar.setVisible(False)

    def replot_map(self):
        if self.hc_map is None:
            return

        self.ax.clear()

        from gui_styles import get_theme_colors
        colors = get_theme_colors(self.theme)
        fig_bg = colors["card"]
        ax_bg = colors["bg"]
        text_color = colors["text"]

        self.figure.patch.set_facecolor(fig_bg)
        self.ax.set_facecolor(ax_bg)
        self.ax.xaxis.label.set_color(text_color)
        self.ax.yaxis.label.set_color(text_color)
        if self.ax.title:
            self.ax.title.set_color(text_color)
        self.ax.tick_params(colors=text_color)

        cmap_name = self.cmb_cmap.currentText()

        if self.chk_overlay.isChecked() and self.background_array is not None:
            self.ax.imshow(self.background_array, cmap='gray')
            alpha = self.sld_alpha.value() / 100.0

            # Create masked array where Hc is NaN or zero to avoid plotting over background completely
            masked_hc = np.ma.masked_where(np.isnan(self.hc_map) | (self.hc_map == 0), self.hc_map)

            # Ensure Hc map has same spatial extent as background image
            h, w = self.background_array.shape[:2]
            im = self.ax.imshow(masked_hc, cmap=cmap_name, alpha=alpha, extent=[0, w, h, 0])
        else:
            masked_hc = np.ma.masked_where(np.isnan(self.hc_map) | (self.hc_map == 0), self.hc_map)
            im = self.ax.imshow(masked_hc, cmap=cmap_name)

        # Only add colorbar if it doesn't exist to prevent duplicates
        if not hasattr(self, 'cbar') or self.cbar is None:
            self.cbar = self.figure.colorbar(im, ax=self.ax)
            self.cbar.set_label('Local Coercivity Hc (mT)', color=text_color)
        else:
            self.cbar.update_normal(im)

        self.cbar.ax.yaxis.set_tick_params(color=text_color)
        self.cbar.ax.yaxis.set_ticklabels(self.cbar.ax.yaxis.get_ticklabels(), color=text_color)

        self.ax.set_title("Spatially Resolved Local Coercivity Map")
        self.ax.set_xlabel("X (pixels)")
        self.ax.set_ylabel("Y (pixels)")

        self.figure.tight_layout()
        self.canvas.draw()

    def save_map(self):
        if self.hc_map is None:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Hc Map Image", "", "PNG Files (*.png)"
        )
        if save_path:
            self.figure.savefig(save_path, dpi=300)
            QMessageBox.information(self, "Saved", f"Hc map saved to:\n{save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", type=str, default="dark")
    args, unknown = parser.parse_known_args()

    app = QApplication(sys.argv)
    window = HcMapGUI(theme=args.theme)
    window.show()
    sys.exit(app.exec_())
