# -*- coding: utf-8 -*-
import sys
import os
import operator
import types
# NumPy 2.x namespace compatibility shim for older dependencies (e.g. SciPy, scikit-image)
try:
    import numpy as _np
    _np.core = _np._core
    sys.modules["numpy.core"] = _np._core
    for _name, _obj in list(_np._core.__dict__.items()):
        if isinstance(_obj, type(sys)):
            sys.modules[f"numpy.core.{_name}"] = _obj
    del _name, _obj, _np
except Exception:
    pass

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
            hc_map, hr_map = calculate_local_hc(self.img_stack, self.loop_field, self.bin_size, return_hr=True)
            self.finished.emit((hc_map, hr_map))
        except Exception as e:
            self.error.emit(str(e))

class HcMapGUI(QWidget):
    def __init__(self, theme="dark", parent=None):
        super().__init__(parent)
        self.theme = theme
        self.setObjectName("MainBg")
        self.setWindowTitle("Local Mapping (Hc & Hr)")

        self.img_dir = None
        self.txt_data = None
        self.loop_field = None
        self.background_array = None

        self.hc_map = None
        self.hr_map = None
        self.img_stack = None
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

        self.btn_calc = QPushButton("Calculate Maps")
        self.btn_calc.clicked.connect(self.calculate_map)
        self.btn_calc.setEnabled(False)
        self.btn_calc.setToolTip("Select an image directory containing a mapping .txt file first")
        left_layout.addWidget(self.btn_calc)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        group_plot = QGroupBox("Plot Settings")
        form_plot = QFormLayout()

        self.cmb_map_type = QComboBox()
        self.cmb_map_type.addItems(["Coercivity (Hc)", "Remanence (Hr)"])
        self.cmb_map_type.currentTextChanged.connect(self.replot_map)
        form_plot.addRow("Map Type:", self.cmb_map_type)

        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["seismic", "plasma", "viridis", "inferno", "magma", "cividis", "jet"])
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

        self.chk_hover = QCheckBox("Hover Over Loop")
        self.chk_hover.stateChanged.connect(self.on_hover_changed)
        form_plot.addRow(self.chk_hover)

        group_plot.setLayout(form_plot)
        left_layout.addWidget(group_plot)

        self.btn_save = QPushButton("Save Map Image")
        self.btn_save.clicked.connect(self.save_map)
        self.btn_save.setEnabled(False)
        self.btn_save.setToolTip("Calculate maps first before saving")
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

        # Hover-over overlay/widget properties
        self.hover_cid = None
        self.hover_annotation = None

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 3)
        self.setLayout(main_layout)

        apply_theme(self, self.theme)

    def on_hover_changed(self, state):
        if state == Qt.Checked:
            if self.hover_cid is None:
                self.hover_cid = self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        else:
            if self.hover_cid is not None:
                self.canvas.mpl_disconnect(self.hover_cid)
                self.hover_cid = None
            if self.hover_annotation is not None:
                try:
                    self.hover_annotation.remove()
                except Exception:
                    pass
                self.hover_annotation = None
                self.canvas.draw_idle()

    def on_mouse_move(self, event):
        if event.inaxes != self.ax or self.img_stack is None:
            if self.hover_annotation is not None:
                self.hover_annotation.set_visible(False)
                self.canvas.draw_idle()
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        h, w = self.background_array.shape[:2] if self.background_array is not None else self.img_stack.shape[1:3]
        col = int(np.clip(x, 0, w - 1))
        row = int(np.clip(y, 0, h - 1))

        # Extract local loop intensity at this pixel and subtract zero-field background
        intensity = self.img_stack[:, row, col].copy()
        zero_idx = np.argmin(np.abs(self.loop_field))
        intensity = intensity - intensity[zero_idx]

        # Apply loop corrections (drift, linear Faraday, quadratic Cotton-Mouton)
        N_pts = len(intensity)
        idx = np.arange(N_pts, dtype=np.float32)
        idx_off = idx - idx.mean()
        field_off = self.loop_field - np.mean(self.loop_field)
        field_abs_max = np.max(np.abs(field_off))

        # Pass 1 - endpoint drift alignment
        drift1 = (intensity[0] - intensity[-1]) / N_pts
        intensity = intensity + drift1 * idx_off

        # Saturation fit
        sat_threshold = 0.80 * field_abs_max
        fit_mask = np.abs(field_off) > sat_threshold
        if np.sum(fit_mask) < 4:
            sat_threshold = 0.50 * field_abs_max
            fit_mask = np.abs(field_off) > sat_threshold

        linear_val = 0.0
        quad1 = 0.0
        if np.sum(fit_mask) >= 4:
            h_fit = field_off[fit_mask]
            y_fit = intensity[fit_mask]
            A = np.column_stack([h_fit, h_fit**2, np.sign(h_fit), np.ones_like(h_fit)])
            try:
                coeffs_fit, _, _, _ = np.linalg.lstsq(A, y_fit, rcond=None)
                linear_val = -coeffs_fit[0]
                quad1 = -coeffs_fit[1]
            except Exception:
                pass

        # Apply shape correction & second drift correction
        intensity = intensity + linear_val * field_off + quad1 * (field_off ** 2)
        drift2 = (intensity[0] - intensity[-1]) / N_pts
        intensity = intensity + drift2 * idx_off

        # Normalize loop to [-1, 1]
        ptp = np.ptp(intensity)
        if ptp > 0:
            intensity = (intensity - np.min(intensity)) / ptp * 2 - 1
        else:
            intensity[:] = 0.0

        # Draw / Update Hover Annotation
        if self.hover_annotation is not None:
            try:
                self.hover_annotation.remove()
            except Exception:
                pass

        # We'll embed an inset axes or offset box inside the plot
        # For simplicity and robust display, let's create a custom annotation with a mini plot inside it or draw directly using an inset axes.
        # Since creating inset axes dynamically on mouse motion can be slow, we can draw a small pop-up box with a custom plot inside it using matplotlib's BboxConnector/AnnotationBbox or an InsetAxes that we update.
        if not hasattr(self, 'inset_ax') or self.inset_ax is None:
            # Create inset axes once
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            self.inset_ax = inset_axes(self.ax, width="30%", height="30%", loc="upper right")
            self.inset_ax.tick_params(labelsize=8, colors='white')
            # Hide borders or style appropriately
            for spine in self.inset_ax.spines.values():
                spine.set_color('cyan')
        
        self.inset_ax.clear()
        self.inset_ax.set_facecolor('#1a1a1a')
        
        # Plot Loop
        self.inset_ax.plot(self.loop_field, intensity, color='#00ffcc', marker='.', markersize=2, linewidth=1)
        self.inset_ax.set_title(f"Loop at ({col},{row})", fontsize=8, color='white')
        self.inset_ax.xaxis.set_tick_params(labelbottom=False)
        self.inset_ax.yaxis.set_tick_params(labelleft=False)
        self.inset_ax.set_visible(True)
        self.canvas.draw_idle()

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
            self.btn_calc.setToolTip("Select an image directory containing a mapping .txt file first")
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
            self.btn_calc.setToolTip("Start local coercivity calculation")
        except Exception as e:
            self.lbl_dir_status.setText(f"Error loading data: {e}")
            self.btn_calc.setEnabled(False)
            self.btn_calc.setToolTip("Select an image directory containing a mapping .txt file first")

    def calculate_map(self):
        if self.txt_data is None or self.background_array is None:
            return

        self.btn_calc.setEnabled(False)
        self.btn_calc.setToolTip("Calculation in progress...")
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

        self.img_stack = np.array(img_stack) # shape: (N, H, W)

        bin_size = self.spin_bin_size.value()

        # Run calculation in a background thread
        self.calc_thread = HcCalcThread(self.img_stack, self.loop_field, bin_size)
        self.calc_thread.finished.connect(self.on_calc_finished)
        self.calc_thread.error.connect(self.on_calc_error)
        self.calc_thread.start()

    def on_calc_finished(self, result):
        self.hc_map, self.hr_map = result
        self.replot_map()
        self.btn_save.setEnabled(True)
        self.btn_save.setToolTip("Save the current map visualization")
        self._reset_calc_ui()

    def on_calc_error(self, err_msg):
        QMessageBox.critical(self, "Calculation Error", err_msg)
        self._reset_calc_ui()

    def _reset_calc_ui(self):
        self.btn_calc.setEnabled(True)
        self.btn_calc.setToolTip("Start local coercivity calculation")
        self.btn_dir.setEnabled(True)
        self.progress_bar.setVisible(False)

    def replot_map(self):
        map_type = self.cmb_map_type.currentText()
        is_hr = "Remanence" in map_type
        current_map = self.hr_map if is_hr else self.hc_map

        if current_map is None:
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

            # Create masked array where values are NaN or zero to avoid plotting over background completely
            masked_map = np.ma.masked_where(np.isnan(current_map) | (current_map == 0), current_map)

            # Ensure map has same spatial extent as background image
            h, w = self.background_array.shape[:2]
            im = self.ax.imshow(masked_map, cmap=cmap_name, alpha=alpha, extent=[0, w, h, 0])
        else:
            masked_map = np.ma.masked_where(np.isnan(current_map) | (current_map == 0), current_map)
            im = self.ax.imshow(masked_map, cmap=cmap_name)

        label_text = 'Local Remanence Hr (a.u.)' if is_hr else 'Local Coercivity Hc (mT)'
        title_text = 'Spatially Resolved Local Remanence Map' if is_hr else 'Spatially Resolved Local Coercivity Map'

        # Instead of calling self.cbar.remove() which modifies the axes grid layout and squashes the parent plot,
        # we check if colorbar exists and update its scalar mappable.
        # If the label has changed or colorbar is not initialized, we draw it.
        if not hasattr(self, 'cbar') or self.cbar is None:
            self.cbar = self.figure.colorbar(im, ax=self.ax)
            self.cbar.set_label(label_text, color=text_color)
        else:
            self.cbar.update_normal(im)
            self.cbar.set_label(label_text, color=text_color)

        try:
            self.cbar.ax.yaxis.set_tick_params(color=text_color)
            self.cbar.ax.yaxis.set_ticklabels(self.cbar.ax.yaxis.get_ticklabels(), color=text_color)
        except Exception:
            pass

        self.ax.set_title(title_text)
        self.ax.set_xlabel("X (pixels)")
        self.ax.set_ylabel("Y (pixels)")

        self.figure.tight_layout()
        self.canvas.draw()

    def save_map(self):
        map_type = self.cmb_map_type.currentText()
        is_hr = "Remanence" in map_type
        current_map = self.hr_map if is_hr else self.hc_map

        if current_map is None:
            return

        name_prefix = "Hr" if is_hr else "Hc"
        save_path, _ = QFileDialog.getSaveFileName(
            self, f"Save {name_prefix} Map Image", "", "PNG Files (*.png)"
        )
        if save_path:
            self.figure.savefig(save_path, dpi=300)
            QMessageBox.information(self, "Saved", f"{name_prefix} map saved to:\n{save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", type=str, default="dark")
    args, unknown = parser.parse_known_args()

    app = QApplication(sys.argv)
    window = HcMapGUI(theme=args.theme)
    window.show()
    sys.exit(app.exec_())
