import sys
import os
import numpy as np
import pandas as pd
from PIL import Image
import concurrent.futures

import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (
    QApplication, QWidget, QFileDialog, QVBoxLayout, QPushButton,
    QLabel, QMessageBox, QHBoxLayout, QSlider, QTabWidget,
    QGroupBox, QFormLayout, QSplitter, QCheckBox, QComboBox, QSpinBox, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from shared_utils.image_processing import crop600
from gui_styles import apply_theme

class DmdhCalcThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, img_dir, txt_data, smooth_frames):
        super().__init__()
        self.img_dir = img_dir
        self.txt_data = txt_data
        self.smooth_frames = smooth_frames

    def run(self):
        try:
            # We assume image stack might be large, load all into memory if possible
            # Similar to hc_map_tool, we crop to 600x600.
            img_stack = []
            files = self.txt_data["File"].tolist()
            fields = self.txt_data["Field"].to_numpy(dtype=np.float32)
            N = len(files)

            if N < 2:
                self.error.emit("Not enough images to calculate dM/dH.")
                return

            def load_img(f):
                path = os.path.join(self.img_dir, f)
                return crop600(np.array(Image.open(path))).astype(np.float32)

            # Load images
            for i, f in enumerate(files):
                img_stack.append(load_img(f))
                self.progress.emit(int((i/N) * 30)) # 30% for loading

            img_stack = np.array(img_stack)

            # Calculate dM/dH
            # We can calculate forward difference or central difference. Forward: I_{n+1} - I_n
            # Let's do adjacent subtraction: I_{n} - I_{n-1}
            # For smooth_frames > 1, we can average adjacent diffs or subtract I_{n+k} - I_n

            diff_stack = np.zeros_like(img_stack)
            dM_dH_fields = np.zeros_like(fields)

            step = max(1, self.smooth_frames)

            for i in range(step, N):
                # Simple difference between current and previous `step` frame
                dI = img_stack[i] - img_stack[i - step]
                dH = fields[i] - fields[i - step]

                # To handle noise and arbitrary H directions, we track absolute change
                # or keep the sign depending on branch. For pinning landscape,
                # we usually care about the magnitude of the switching event.
                # Let's just store the raw difference.
                diff_stack[i] = dI

                self.progress.emit(int(30 + (i/N) * 40)) # Up to 70% for diff calculation

            # Iso-switching map: For each pixel, find the field at which max |dI| occurred
            # We split into ascending and descending branches if possible

            i_min = np.argmin(fields)
            i_max = np.argmax(fields)

            # We'll just find the max absolute difference over the whole loop for simplicity,
            # and map it to the corresponding field value.
            # A more advanced method separates asc/desc branches.

            abs_diff = np.abs(diff_stack)
            max_idx = np.argmax(abs_diff, axis=0) # shape (H, W)

            # Map indices to fields
            iso_map = np.take(fields, max_idx)

            # Also calculate max switching amplitude for thresholding/alpha mapping
            max_amp = np.max(abs_diff, axis=0)

            self.progress.emit(100)

            result = {
                'diff_stack': diff_stack,
                'fields': fields,
                'iso_map': iso_map,
                'max_amp': max_amp
            }
            self.finished.emit(result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

class DmdhGUI(QWidget):
    def __init__(self, theme="dark", parent=None):
        super().__init__(parent)
        self.theme = theme
        self.setObjectName("MainBg")
        self.setWindowTitle("Differential Magnetization & Pinning Landscape Mapper")

        self.img_dir = None
        self.txt_data = None

        self.diff_stack = None
        self.fields = None
        self.iso_map = None
        self.max_amp = None

        self.init_ui()

    def change_theme(self, theme):
        self.theme = theme
        apply_theme(self, theme)
        self.update_diff_plot()
        self.update_iso_plot()

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

        group_params = QGroupBox("Analysis Parameters")
        form_params = QFormLayout()

        self.spin_smooth = QSpinBox()
        self.spin_smooth.setRange(1, 10)
        self.spin_smooth.setValue(1)
        self.spin_smooth.setToolTip("Frame step for subtraction (I_n - I_{n-step})")
        form_params.addRow("Frame Step:", self.spin_smooth)

        group_params.setLayout(form_params)
        left_layout.addWidget(group_params)

        self.btn_calc = QPushButton("Calculate dM/dH Maps")
        self.btn_calc.clicked.connect(self.calculate_maps)
        self.btn_calc.setEnabled(False)
        left_layout.addWidget(self.btn_calc)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Viewer Controls
        group_view = QGroupBox("dM/dH Viewer")
        view_layout = QVBoxLayout()

        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setRange(0, 0)
        self.slider_frame.setEnabled(False)
        self.slider_frame.valueChanged.connect(self.on_frame_changed)
        view_layout.addWidget(QLabel("Frame:"))
        view_layout.addWidget(self.slider_frame)

        self.lbl_frame_info = QLabel("Field: N/A")
        view_layout.addWidget(self.lbl_frame_info)

        # Contrast slider for dM/dH
        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(1, 100)
        self.slider_contrast.setValue(50)
        self.slider_contrast.valueChanged.connect(self.update_diff_plot)
        view_layout.addWidget(QLabel("Contrast:"))
        view_layout.addWidget(self.slider_contrast)

        group_view.setLayout(view_layout)
        left_layout.addWidget(group_view)

        # Iso-Map Controls
        group_iso = QGroupBox("Iso-Switching Map")
        form_iso = QFormLayout()

        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["jet", "seismic", "plasma", "viridis", "inferno"])
        self.cmb_cmap.currentTextChanged.connect(self.update_iso_plot)
        form_iso.addRow("Colormap:", self.cmb_cmap)

        self.slider_thresh = QSlider(Qt.Horizontal)
        self.slider_thresh.setRange(0, 100)
        self.slider_thresh.setValue(10)
        self.slider_thresh.setToolTip("Filter out noise by requiring a minimum switching amplitude")
        self.slider_thresh.valueChanged.connect(self.update_iso_plot)
        form_iso.addRow("Noise Threshold:", self.slider_thresh)

        group_iso.setLayout(form_iso)
        left_layout.addWidget(group_iso)

        left_layout.addStretch()

        # Right Panel: Visualization Tabs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.tabs = QTabWidget()

        # Tab 1: dM/dH Movie
        self.tab_diff = QWidget()
        tab_diff_layout = QVBoxLayout(self.tab_diff)

        from gui_styles import get_theme_colors
        colors = get_theme_colors(self.theme)

        self.fig_diff = Figure(facecolor=colors["card"])
        self.ax_diff = self.fig_diff.add_subplot(111)
        self.ax_diff.set_facecolor(colors["bg"])
        self.canvas_diff = FigureCanvas(self.fig_diff)
        self.toolbar_diff = NavigationToolbar(self.canvas_diff, self)

        tab_diff_layout.addWidget(self.toolbar_diff)
        tab_diff_layout.addWidget(self.canvas_diff)

        # Tab 2: Iso-Switching Map
        self.tab_iso = QWidget()
        tab_iso_layout = QVBoxLayout(self.tab_iso)

        self.fig_iso = Figure(facecolor=colors["card"])
        self.ax_iso = self.fig_iso.add_subplot(111)
        self.ax_iso.set_facecolor(colors["bg"])
        self.canvas_iso = FigureCanvas(self.fig_iso)
        self.toolbar_iso = NavigationToolbar(self.canvas_iso, self)

        tab_iso_layout.addWidget(self.toolbar_iso)
        tab_iso_layout.addWidget(self.canvas_iso)

        self.tabs.addTab(self.tab_diff, "dM/dH Frame Viewer")
        self.tabs.addTab(self.tab_iso, "Iso-Switching Map")

        right_layout.addWidget(self.tabs)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 800])

        mlayout = QVBoxLayout(self)
        mlayout.addWidget(splitter)

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

            self.lbl_dir_status.setText(f"Loaded {len(self.txt_data)} images.")
            self.btn_calc.setEnabled(True)

            self.diff_stack = None
            self.slider_frame.setEnabled(False)
            self.ax_diff.clear()
            self.ax_iso.clear()
            self.canvas_diff.draw()
            self.canvas_iso.draw()

        except Exception as e:
            self.lbl_dir_status.setText(f"Error loading data: {e}")
            self.btn_calc.setEnabled(False)

    def calculate_maps(self):
        if self.txt_data is None:
            return

        self.btn_calc.setEnabled(False)
        self.btn_dir.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        step = self.spin_smooth.value()

        self.calc_thread = DmdhCalcThread(self.img_dir, self.txt_data, step)
        self.calc_thread.finished.connect(self.on_calc_finished)
        self.calc_thread.error.connect(self.on_calc_error)
        self.calc_thread.progress.connect(self.progress_bar.setValue)
        self.calc_thread.start()

    def on_calc_finished(self, result):
        self.diff_stack = result['diff_stack']
        self.fields = result['fields']
        self.iso_map = result['iso_map']
        self.max_amp = result['max_amp']

        self.slider_frame.setRange(0, len(self.fields) - 1)
        self.slider_frame.setValue(max(0, self.spin_smooth.value()))
        self.slider_frame.setEnabled(True)

        self.update_diff_plot()
        self.update_iso_plot()

        self._reset_calc_ui()

    def on_calc_error(self, err_msg):
        QMessageBox.critical(self, "Calculation Error", err_msg)
        self._reset_calc_ui()

    def _reset_calc_ui(self):
        self.btn_calc.setEnabled(True)
        self.btn_dir.setEnabled(True)
        self.progress_bar.setVisible(False)

    def on_frame_changed(self):
        idx = self.slider_frame.value()
        if self.fields is not None:
            self.lbl_frame_info.setText(f"Field: {self.fields[idx]:.2f} mT")
        self.update_diff_plot()

    def update_diff_plot(self):
        if self.diff_stack is None:
            return

        idx = self.slider_frame.value()
        img = self.diff_stack[idx]

        self.ax_diff.clear()

        from gui_styles import get_theme_colors
        colors = get_theme_colors(self.theme)
        self.fig_diff.patch.set_facecolor(colors["card"])
        self.ax_diff.set_facecolor(colors["bg"])
        self.ax_diff.tick_params(colors=colors["text"])

        # Contrast adjustment
        c_val = self.slider_contrast.value() / 50.0 # 0.02 to 2.0
        vmax = np.max(np.abs(self.diff_stack)) * (2.0 - c_val + 0.1) # Hacky contrast scaling
        if vmax == 0: vmax = 1

        # Diverging colormap for dM/dH (shows positive and negative changes)
        im = self.ax_diff.imshow(img, cmap='seismic', vmin=-vmax, vmax=vmax)
        self.ax_diff.set_title(f"dM/dH Map (Frame {idx})", color=colors["text"])

        self.fig_diff.tight_layout()
        self.canvas_diff.draw()

    def update_iso_plot(self):
        if self.iso_map is None:
            return

        self.ax_iso.clear()

        from gui_styles import get_theme_colors
        colors = get_theme_colors(self.theme)
        self.fig_iso.patch.set_facecolor(colors["card"])
        self.ax_iso.set_facecolor(colors["bg"])
        self.ax_iso.tick_params(colors=colors["text"])
        self.ax_iso.xaxis.label.set_color(colors["text"])
        self.ax_iso.yaxis.label.set_color(colors["text"])

        cmap_name = self.cmb_cmap.currentText()

        # Thresholding
        thresh_pct = self.slider_thresh.value() / 100.0
        global_max_amp = np.max(self.max_amp)
        mask = self.max_amp < (global_max_amp * thresh_pct)

        masked_iso = np.ma.masked_where(mask, self.iso_map)

        im = self.ax_iso.imshow(masked_iso, cmap=cmap_name)

        if not hasattr(self, 'cbar_iso') or self.cbar_iso is None:
            self.cbar_iso = self.fig_iso.colorbar(im, ax=self.ax_iso)
            self.cbar_iso.set_label("Switching Field (mT)", color=colors["text"])
        else:
            self.cbar_iso.update_normal(im)
            self.cbar_iso.set_label("Switching Field (mT)", color=colors["text"])

        try:
            self.cbar_iso.ax.yaxis.set_tick_params(color=colors["text"])
            self.cbar_iso.ax.yaxis.set_ticklabels(self.cbar_iso.ax.yaxis.get_ticklabels(), color=colors["text"])
        except Exception:
            pass

        self.ax_iso.set_title("Iso-Switching Map (Pinning Landscape)", color=colors["text"])

        self.fig_iso.tight_layout()
        self.canvas_iso.draw()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", type=str, default="dark")
    args, unknown = parser.parse_known_args()

    app = QApplication(sys.argv)
    window = DmdhGUI(theme=args.theme)
    window.resize(1000, 700)
    window.show()
    sys.exit(app.exec_())
