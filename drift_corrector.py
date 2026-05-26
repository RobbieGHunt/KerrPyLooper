# -*- coding: utf-8 -*-
"""
Kerr MOKE In-Plane Image Drift Corrector
=========================================
A GUI tool for correcting X/Y in-plane drift in Kerr microscopy image series.

When a magnetic field is applied, the sample can physically move in-plane,
causing defects to shift position between images. This tool:
  1. Loads a hysteresis image series in the same way as kerr_looper_AG.py.
  2. Lets the user select a "Defect ROI" - a region containing a static defect
     (dust, scratch, edge) that should remain at the same pixel position.
  3. Uses Sobel-gradient-based Normalised Cross-Correlation (NCC) within a
     user-defined search window to find the integer + sub-pixel shift of each
     target image relative to a chosen reference image.
  4. Applies scipy.ndimage.shift (spline interpolation) to correct each image.
  5. Displays a diagnostic drift plot (dx, dy vs. field) and saves the
     corrected images + mapping file to a new directory.

Created 2026.
"""

import sys
import os
import numpy as np
import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QSlider, QComboBox, QCheckBox,
    QSplitter, QProgressBar, QFileDialog, QMessageBox, QSizePolicy,
    QTextEdit, QScrollArea, QTabWidget
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush, QFont
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal, QSize, QRectF, QPointF
import scipy.ndimage as ndimage
from scipy.signal import fftconvolve
from PIL import Image
import pandas as pd
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from shared_utils.image_processing import crop600



# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers (mirrors kerr_looper_AG.py utilities)
# ──────────────────────────────────────────────────────────────────────────────

def crop600(arr):
    """Crop image to at most 600 rows × 900 cols (centred)."""
    h, w = arr.shape[0], arr.shape[1]
    h_crop = min(h, 600)
    w_crop = min(w, 900)
    w_start = (w - w_crop) // 2
    if arr.ndim == 3:
        return arr[:h_crop, w_start:w_start + w_crop, :]
    return arr[:h_crop, w_start:w_start + w_crop]


def to_gray(arr):
    """Convert an image array to float64 grayscale."""
    if arr.ndim == 3:
        r, g, b = arr[:, :, 0].astype(np.float64), arr[:, :, 1].astype(np.float64), arr[:, :, 2].astype(np.float64)
        return 0.2989 * r + 0.5870 * g + 0.1140 * b
    return arr.astype(np.float64)


def sobel_gradient(img):
    """Return Sobel gradient magnitude image."""
    dx = ndimage.sobel(img, axis=0)
    dy = ndimage.sobel(img, axis=1)
    return np.sqrt(dx ** 2 + dy ** 2)


def normalized_for_display(arr, contrast=1.0):
    """Normalise a float array into [0,255] uint8 for display."""
    arr = arr.astype(np.float32)
    scale = np.std(arr)
    if scale == 0:
        scale = 1.0
    arr_d = np.arcsinh(arr / scale)
    arr_d -= arr_d.min()
    ptp = arr_d.ptp()
    if ptp == 0:
        arr_d[:] = 0
    else:
        arr_d /= ptp
    arr_d = arr_d * 255.0
    arr_d = 127.5 + contrast * (arr_d - 127.5)
    return np.clip(arr_d, 0, 255).astype(np.uint8)


def robust_normalize_raw(arr):
    """Percentile-based contrast stretch for raw images."""
    arr = arr.astype(np.float32)
    mask = arr > 0
    if np.any(mask):
        active = arr[mask]
        low = np.percentile(active, 1)
        high = np.percentile(active, 99)
        if high > low:
            out = np.clip((arr - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
            out[~mask] = 0
            return out
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        return ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)
    # If image is totally flat, return a mid-gray image to avoid blacking out.
    return np.full_like(arr, 128, dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Core drift estimation algorithm
# ──────────────────────────────────────────────────────────────────────────────

def estimate_shift_sqdiff(ref_patch_grad, target_img_grad, roi_r, roi_c, patch_h, patch_w, search_width):
    """
    Estimate (dy, dx) shift of target_img relative to ref using Sum of Squared
    Differences (SQDIFF).

    Parameters
    ----------
    ref_patch_grad : 2-D float array
        Patch of the reference defect (patch_h × patch_w).
    target_img_grad : 2-D float array
        The full target image to search within.
    roi_r, roi_c : int
        Top-left corner of the ROI in the reference image.
    patch_h, patch_w : int
        Height and width of the ROI patch.
    search_width : int
        Maximum pixel search range in each direction.

    Returns
    -------
    (dy_sub, dx_sub) : float
        Estimated sub-pixel shift.  Positive dy means target shifted downward,
        positive dx means shifted rightward.
    ncc_map : 2-D float array
        SQDIFF scores map over the search grid (for diagnostics).
    """
    sw = search_width
    img_h, img_w = target_img_grad.shape

    # Build a grid of candidate shifts
    dy_vals = np.arange(-sw, sw + 1)
    dx_vals = np.arange(-sw, sw + 1)

    ncc_map = np.full((len(dy_vals), len(dx_vals)), float('inf'), dtype=np.float32)

    # ⚡ Bolt: Fast exact template matching via FFT convolution
    # Calculate bounding box for valid shifts to avoid out-of-bounds sampling
    dy_min = max(-sw, -roi_r)
    dy_max = min(sw, img_h - roi_r - patch_h)
    dx_min = max(-sw, -roi_c)
    dx_max = min(sw, img_w - roi_c - patch_w)

    if dy_min > dy_max or dx_min > dx_max:
        return 0.0, 0.0, ncc_map

    # Extract the full valid search region containing all overlapping patches
    r_start = roi_r + dy_min
    r_end = roi_r + dy_max + patch_h
    c_start = roi_c + dx_min
    c_end = roi_c + dx_max + patch_w

    search_region = target_img_grad[r_start:r_end, c_start:c_end].astype(np.float32)
    ref = ref_patch_grad.astype(np.float32)

    # Expand SQDIFF (A-B)^2 = A^2 - 2AB + B^2 for O(N log N) frequency-domain matching
    ref_sq = np.sum(ref ** 2)
    search_sq = search_region ** 2
    tgt_sq = fftconvolve(search_sq, np.ones_like(ref), mode='valid')
    cross_corr = fftconvolve(search_region, ref[::-1, ::-1], mode='valid')

    # Recombine to exact SQDIFF scores and clip near-zero precision errors
    sqdiff = tgt_sq - 2 * cross_corr + ref_sq
    sqdiff = np.maximum(sqdiff, 0)

    # Map the valid region back to the (-sw, +sw) array indices
    i_min = dy_min + sw
    i_max = dy_max + sw + 1
    j_min = dx_min + sw
    j_max = dx_max + sw + 1

    ncc_map[i_min:i_max, j_min:j_max] = sqdiff

    # Find integer peak (minimum diff)
    peak_i, peak_j = np.unravel_index(np.argmin(ncc_map), ncc_map.shape)

    # Sub-pixel refinement: fit quadratic along each axis through the peak
    def sub_pixel_1d(vals, scores, peak_idx):
        if peak_idx == 0 or peak_idx == len(scores) - 1:
            return float(vals[peak_idx])
        f0 = scores[peak_idx - 1]
        f1 = scores[peak_idx]
        f2 = scores[peak_idx + 1]
        denom = 2.0 * (f0 - 2.0 * f1 + f2)
        if abs(denom) < 1e-10:
            return float(vals[peak_idx])
        offset = (f0 - f2) / denom
        return float(vals[peak_idx]) + offset

    dy_sub = sub_pixel_1d(dy_vals, ncc_map[:, peak_j], peak_i)
    dx_sub = sub_pixel_1d(dx_vals, ncc_map[peak_i, :], peak_j)

    return dy_sub, dx_sub, ncc_map


def apply_shift(img_arr, dy, dx):
    """
    Shift a (possibly multi-channel) image array by (dy, dx) pixels using
    spline interpolation.  dy>0 shifts downward, dx>0 shifts rightward.
    """
    if img_arr.ndim == 3:
        shifted = np.zeros_like(img_arr)
        for c in range(img_arr.shape[2]):
            shifted[:, :, c] = ndimage.shift(img_arr[:, :, c].astype(np.float64),
                                              shift=(dy, dx), mode='nearest')
        # Preserve original dtype
        if np.issubdtype(img_arr.dtype, np.integer):
            max_v = np.iinfo(img_arr.dtype).max
            shifted = np.clip(shifted, 0, max_v).astype(img_arr.dtype)
        else:
            shifted = shifted.astype(img_arr.dtype)
    else:
        shifted = ndimage.shift(img_arr.astype(np.float64), shift=(dy, dx), mode='nearest')
        if np.issubdtype(img_arr.dtype, np.integer):
            max_v = np.iinfo(img_arr.dtype).max
            shifted = np.clip(shifted, 0, max_v).astype(img_arr.dtype)
        else:
            shifted = shifted.astype(img_arr.dtype)
    return shifted


# ──────────────────────────────────────────────────────────────────────────────
# ROI Selector Label (draw a rectangle on a QLabel-displayed image)
# ──────────────────────────────────────────────────────────────────────────────

class DefectROILabel(QLabel):
    """
    QLabel subclass that lets the user draw a single rectangular defect ROI
    by click-dragging.  ROI coordinates are in *image* pixel space.
    Also draws the search-width boundary around the ROI in a distinct colour.
    """
    roi_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self.allow_roi = True
        # ROI: (r, c, h, w) in image pixel space; None when not set
        self.roi = None          # (row_top, col_left, height, width)
        self.search_width = 20   # pixels — kept in sync by parent window
        self._dragging = False
        self._drag_start = None  # label coords
        self._drag_cur = None

    # ── coordinate transforms ──────────────────────────────────────────────

    def _scale(self):
        """Return scale factor and offsets from label → image coords."""
        if not self.pixmap() or self.pixmap().isNull():
            return None
        lw, lh = self.width(), self.height()
        pw, ph = self.pixmap().width(), self.pixmap().height()
        s = min(lw / pw, lh / ph)
        dw = int(pw * s)
        dh = int(ph * s)
        ox = (lw - dw) // 2
        oy = (lh - dh) // 2
        return s, ox, oy, pw, ph

    def label_to_image(self, lx, ly):
        info = self._scale()
        if info is None:
            return None
        s, ox, oy, pw, ph = info
        ix = (lx - ox) / s
        iy = (ly - oy) / s
        return max(0.0, min(pw - 1.0, ix)), max(0.0, min(ph - 1.0, iy))

    def image_to_label(self, ix, iy):
        info = self._scale()
        if info is None:
            return None
        s, ox, oy, pw, ph = info
        return ix * s + ox, iy * s + oy

    # ── mouse events ───────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if not self.allow_roi:
            return
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_start = (event.x(), event.y())
            self._drag_cur = self._drag_start
            self.update()

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._drag_cur = (event.x(), event.y())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._drag_cur = (event.x(), event.y())
            self._dragging = False
            self._commit_roi()
            self.update()
            self.roi_changed.emit()

    def _commit_roi(self):
        c1 = self.label_to_image(*self._drag_start)
        c2 = self.label_to_image(*self._drag_cur)
        if c1 is None or c2 is None:
            return
        ix1, iy1 = c1
        ix2, iy2 = c2
        col = int(min(ix1, ix2))
        row = int(min(iy1, iy2))
        w = max(4, int(abs(ix2 - ix1)))
        h = max(4, int(abs(iy2 - iy1)))
        self.roi = (row, col, h, w)

    def clear_roi(self):
        self.roi = None
        self._dragging = False
        self._drag_start = None
        self._drag_cur = None
        self.update()
        self.roi_changed.emit()

    # ── painting ──────────────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Background is now handled by QSS/QTabWidget::pane

        if not self.pixmap() or self.pixmap().isNull():
            painter.end()
            return

        info = self._scale()
        if info is None:
            painter.end()
            return
        s, ox, oy, pw, ph = info
        painter.drawPixmap(ox, oy, int(pw * s), int(ph * s), self.pixmap())

        # Draw live drag rectangle
        if self._dragging and self._drag_start and self._drag_cur and self.allow_roi:
            x1, y1 = self._drag_start
            x2, y2 = self._drag_cur
            rx, ry = min(x1, x2), min(y1, y2)
            rw, rh = abs(x2 - x1), abs(y2 - y1)
            painter.setPen(QPen(QColor(255, 200, 0), 2, Qt.DashLine))
            painter.setBrush(QBrush(QColor(255, 200, 0, 40)))
            painter.drawRect(rx, ry, rw, rh)

        # Draw committed ROI + search boundary
        elif self.roi is not None and self.allow_roi:
            row, col, h, w = self.roi
            sw = self.search_width

            # 1. Draw search boundary (orange dashed) — expanded by search_width
            sb_row = row - sw
            sb_col = col - sw
            sb_h   = h + 2 * sw
            sb_w   = w + 2 * sw
            sb_lx = sb_col * s + ox
            sb_ly = sb_row * s + oy
            sb_lw = sb_w * s
            sb_lh = sb_h * s
            painter.setPen(QPen(QColor(255, 140, 0), 2, Qt.DashLine))   # orange
            painter.setBrush(QBrush(QColor(255, 140, 0, 25)))
            painter.drawRect(QRectF(sb_lx, sb_ly, sb_lw, sb_lh))

            # Search-boundary label
            painter.setPen(QPen(QColor(255, 140, 0)))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(int(sb_lx) + 4, int(sb_ly) - 4,
                             f"Search ±{sw} px")

            # 2. Draw defect ROI (green solid)
            lx1 = col * s + ox
            ly1 = row * s + oy
            lw = w * s
            lh = h * s
            painter.setPen(QPen(QColor(50, 205, 50), 2, Qt.SolidLine))
            painter.setBrush(QBrush(QColor(50, 205, 50, 45)))
            painter.drawRect(QRectF(lx1, ly1, lw, lh))

            # ROI label
            painter.setPen(QPen(QColor(50, 205, 50)))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.drawText(int(lx1) + 4, int(ly1) - 6, "Defect ROI")

        painter.end()


# (No background thread — estimation runs synchronously with processEvents
#  for GUI responsiveness.  This avoids QThread lifecycle / GC crashes.)


# ──────────────────────────────────────────────────────────────────────────────
# Main GUI
# ──────────────────────────────────────────────────────────────────────────────

class DriftCorrectorWindow(QMainWindow):
    def __init__(self, theme="charcoal"):
        super().__init__()
        self.theme = theme
        self.setWindowTitle("Kerr In-Plane Drift Corrector")
        self.setMinimumSize(1100, 720)

        # ── State ──
        self.img_dir = None
        self.image_files = []
        self.txt_data = None
        self.loop_field = None
        self.ref_file = None
        self.ref_arr = None       # full uint array, cropped
        self.ref_gray = None      # float64 grayscale, cropped

        self.contrast = 1.0
        self.colormap_name = "gray"

        self.shifts = None        # list of (dy, dx) floats after estimation
        self._abort_flag = False
        self.crop_top = 0
        self.crop_bottom = 0
        self.crop_left = 0
        self.crop_right = 0

        self._build_ui()
        self._apply_theme()

    # ────────────────────────────── UI ──────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("MainBg")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(6)

        # ── LEFT PANEL ────────────────────────────────────────────────────
        left_widget = QWidget()
        left_widget.setMinimumWidth(280)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(8)

        # Directory selector
        self.btn_dir = QPushButton("📂  Select Image Directory")
        self.btn_dir.setObjectName("LaunchButton")
        self.btn_dir.clicked.connect(self._choose_directory)
        left_layout.addWidget(self.btn_dir)

        self.chk_unprocessed = QCheckBox("Use *_unprocessed files")
        self.chk_unprocessed.setChecked(True)
        self.chk_unprocessed.stateChanged.connect(self._toggle_unprocessed)
        left_layout.addWidget(self.chk_unprocessed)

        # Image list
        lbl_img_list = QLabel("Images  (click to preview)")
        lbl_img_list.setObjectName("SectionTitle")
        left_layout.addWidget(lbl_img_list)

        self.list_images = QListWidget()
        self.list_images.currentRowChanged.connect(self._on_image_selected)
        left_layout.addWidget(self.list_images, stretch=1)

        # Reference image selector
        self.btn_set_ref = QPushButton("⚓  Set as Reference Image")
        self.btn_set_ref.setObjectName("LaunchButton")
        self.btn_set_ref.clicked.connect(self._set_reference)
        self.btn_set_ref.setEnabled(False)
        left_layout.addWidget(self.btn_set_ref)

        self.lbl_ref_info = QLabel("Reference: (not set)")
        self.lbl_ref_info.setObjectName("InfoLabel")
        self.lbl_ref_info.setWordWrap(True)
        left_layout.addWidget(self.lbl_ref_info)

        # ROI group
        roi_group = QGroupBox("Defect ROI  (drag on image)")
        roi_layout = QFormLayout(roi_group)

        self.lbl_roi_info = QLabel("No ROI selected")
        self.lbl_roi_info.setWordWrap(True)
        roi_layout.addRow(self.lbl_roi_info)

        self.btn_clear_roi = QPushButton("✕  Clear ROI")
        self.btn_clear_roi.setObjectName("StopButton")
        self.btn_clear_roi.clicked.connect(self._clear_roi)
        roi_layout.addRow(self.btn_clear_roi)

        left_layout.addWidget(roi_group)

        # Search width
        search_group = QGroupBox("Drift Search Settings")
        search_layout = QFormLayout(search_group)

        sw_hbox = QHBoxLayout()
        self.sld_search = QSlider(Qt.Horizontal)
        self.sld_search.setMinimum(1)
        self.sld_search.setMaximum(100)
        self.sld_search.setValue(20)
        self.sld_search.valueChanged.connect(self._sync_search_spin)
        self.spin_search = QSpinBox()
        self.spin_search.setRange(1, 100)
        self.spin_search.setValue(20)
        self.spin_search.setSuffix(" px")
        self.spin_search.valueChanged.connect(self._sync_search_slider)
        # Keep the ROI label's search_width in sync for the overlay
        self.sld_search.valueChanged.connect(self._update_search_overlay)
        self.spin_search.valueChanged.connect(self._update_search_overlay)
        sw_hbox.addWidget(self.sld_search)
        sw_hbox.addWidget(self.spin_search)
        search_layout.addRow("Search Width:", sw_hbox)

        self.chk_use_sobel = QCheckBox("Use Edge Detection (Sobel)")
        self.chk_use_sobel.setChecked(True)
        search_layout.addRow(self.chk_use_sobel)

        left_layout.addWidget(search_group)

        # Contrast controls
        contrast_group = QGroupBox("Image Display")
        contrast_layout = QFormLayout(contrast_group)

        c_hbox = QHBoxLayout()
        self.sld_contrast = QSlider(Qt.Horizontal)
        self.sld_contrast.setMinimum(10)
        self.sld_contrast.setMaximum(400)
        self.sld_contrast.setValue(100)
        self.sld_contrast.valueChanged.connect(self._contrast_slider_changed)
        self.spin_contrast = QDoubleSpinBox()
        self.spin_contrast.setRange(0.1, 4.0)
        self.spin_contrast.setValue(1.0)
        self.spin_contrast.setSingleStep(0.05)
        self.spin_contrast.valueChanged.connect(self._contrast_spin_changed)
        c_hbox.addWidget(self.sld_contrast)
        c_hbox.addWidget(self.spin_contrast)
        contrast_layout.addRow("Contrast:", c_hbox)

        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["gray", "plasma", "seismic", "viridis", "magma"])
        self.cmb_cmap.currentTextChanged.connect(self._refresh_display)
        contrast_layout.addRow("Colormap:", self.cmb_cmap)

        left_layout.addWidget(contrast_group)

        # Action buttons
        self.btn_estimate = QPushButton("🔍  Estimate Drift")
        self.btn_estimate.setObjectName("LaunchButton")
        self.btn_estimate.clicked.connect(self._run_estimation)
        self.btn_estimate.setEnabled(False)
        left_layout.addWidget(self.btn_estimate)

        self.btn_abort = QPushButton("⏹  Abort")
        self.btn_abort.setObjectName("StopButton")
        self.btn_abort.clicked.connect(self._abort_estimation)
        self.btn_abort.setEnabled(False)
        left_layout.addWidget(self.btn_abort)

        self.btn_save = QPushButton("💾  Save Corrected Images")
        self.btn_save.setObjectName("LaunchButton")
        self.btn_save.clicked.connect(self._save_corrected)
        self.btn_save.setEnabled(False)
        left_layout.addWidget(self.btn_save)



        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        # Status log
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMaximumHeight(100)
        self.status_log.setObjectName("ConsoleOutput")
        left_layout.addWidget(self.status_log)

        splitter.addWidget(left_widget)

        # ── CENTRE PANEL: Image viewer ────────────────────────────────────
        centre_widget = QWidget()
        centre_layout = QVBoxLayout(centre_widget)
        centre_layout.setSpacing(6)

        lbl_viewer_title = QLabel("Image Preview")
        lbl_viewer_title.setObjectName("SectionTitle")
        centre_layout.addWidget(lbl_viewer_title)

        slider_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedWidth(30)
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self._step_prev)
        
        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedWidth(30)
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._step_next)

        self.slider_images = QSlider(Qt.Horizontal)
        self.slider_images.setEnabled(False)
        self.slider_images.valueChanged.connect(self._sync_list_to_slider)

        slider_layout.addWidget(self.btn_prev)
        slider_layout.addWidget(self.slider_images)
        slider_layout.addWidget(self.btn_next)
        
        centre_layout.addLayout(slider_layout)

        self.preview_tabs = QTabWidget()
        self.preview_tabs.currentChanged.connect(self._refresh_display)

        # Raw Image tab
        self.raw_tab = QWidget()
        raw_layout = QVBoxLayout(self.raw_tab)
        raw_layout.setContentsMargins(0, 0, 0, 0)
        self.img_label = DefectROILabel(self)
        self.img_label.setMinimumSize(400, 400)
        self.img_label.roi_changed.connect(self._on_roi_changed)
        raw_layout.addWidget(self.img_label, stretch=1)
        self.preview_tabs.addTab(self.raw_tab, "Raw Image (Draw ROI)")

        # Corrected Image tab
        self.corrected_tab = QWidget()
        corr_layout = QVBoxLayout(self.corrected_tab)
        corr_layout.setContentsMargins(0, 0, 0, 0)
        self.corrected_label = DefectROILabel(self)
        self.corrected_label.allow_roi = False
        self.corrected_label.setMinimumSize(400, 400)
        corr_layout.addWidget(self.corrected_label, stretch=1)
        self.preview_tabs.addTab(self.corrected_tab, "Drift Corrected")

        centre_layout.addWidget(self.preview_tabs, stretch=1)

        # Manual Shift Override
        manual_group = QGroupBox("Manual Shift Override")
        manual_layout = QFormLayout(manual_group)
        
        hx = QHBoxLayout()
        self.sld_man_x = QSlider(Qt.Horizontal)
        self.sld_man_x.setMinimum(-40)
        self.sld_man_x.setMaximum(40)
        self.sld_man_x.setValue(0)
        self.spin_man_x = QSpinBox()
        self.spin_man_x.setRange(-40, 40)
        self.spin_man_x.setValue(0)
        self.spin_man_x.setSuffix(" px")
        self.sld_man_x.valueChanged.connect(self._sync_man_x_spin)
        self.spin_man_x.valueChanged.connect(self._sync_man_x_slider)
        hx.addWidget(self.sld_man_x)
        hx.addWidget(self.spin_man_x)
        manual_layout.addRow("dx (Horiz):", hx)
        
        hy = QHBoxLayout()
        self.sld_man_y = QSlider(Qt.Horizontal)
        self.sld_man_y.setMinimum(-40)
        self.sld_man_y.setMaximum(40)
        self.sld_man_y.setValue(0)
        self.spin_man_y = QSpinBox()
        self.spin_man_y.setRange(-40, 40)
        self.spin_man_y.setValue(0)
        self.spin_man_y.setSuffix(" px")
        self.sld_man_y.valueChanged.connect(self._sync_man_y_spin)
        self.spin_man_y.valueChanged.connect(self._sync_man_y_slider)
        hy.addWidget(self.sld_man_y)
        hy.addWidget(self.spin_man_y)
        manual_layout.addRow("dy (Vert):", hy)
        
        centre_layout.addWidget(manual_group)

        self.lbl_field = QLabel("Field: —")
        self.lbl_field.setAlignment(Qt.AlignCenter)
        centre_layout.addWidget(self.lbl_field)

        splitter.addWidget(centre_widget)

        # ── RIGHT PANEL: Drift plot ──────────────────────────────────────
        right_widget = QWidget()
        right_widget.setMinimumWidth(280)
        right_layout = QVBoxLayout(right_widget)

        lbl_plot_title = QLabel("Drift Curves")
        lbl_plot_title.setObjectName("SectionTitle")
        right_layout.addWidget(lbl_plot_title)

        self._build_drift_plot(right_layout)

        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)

        root.addWidget(splitter)

    def _build_drift_plot(self, layout):
        """Create the matplotlib figure for the dx/dy drift curves."""
        from gui_styles import get_theme_colors
        colors = get_theme_colors(self.theme)
        fig_bg = colors["card"]
        ax_bg  = colors["bg"]
        self.drift_fig = Figure(facecolor=fig_bg)
        self.drift_ax  = self.drift_fig.add_subplot(111)
        self.drift_ax.set_facecolor(ax_bg)
        self.drift_canvas = FigureCanvas(self.drift_fig)
        self.drift_canvas.setMinimumHeight(200)
        layout.addWidget(self.drift_canvas, stretch=1)
        self._plot_drift_placeholder()

    def _plot_drift_placeholder(self):
        from gui_styles import get_theme_colors
        colors = get_theme_colors(self.theme)
        self.drift_ax.clear()
        self.drift_ax.set_facecolor(colors["bg"])
        self.drift_ax.text(0.5, 0.5, "Run drift estimation\nto see curves",
                           transform=self.drift_ax.transAxes,
                           ha='center', va='center',
                           color=colors["text_muted"], fontsize=11)
        self.drift_ax.tick_params(colors=colors["text_muted"])
        for sp in self.drift_ax.spines.values():
            sp.set_color(colors["border"])
        self.drift_fig.tight_layout()
        self.drift_canvas.draw_idle()

    def _plot_drift_curves(self):
        from gui_styles import get_theme_colors
        colors = get_theme_colors(self.theme)
        self.drift_ax.clear()
        self.drift_ax.set_facecolor(colors["bg"])
        self.drift_fig.patch.set_facecolor(colors["card"])

        fields = self.loop_field if self.loop_field is not None else np.arange(len(self.shifts))
        dys = np.array([s[0] for s in self.shifts])
        dxs = np.array([s[1] for s in self.shifts])

        tc = colors["text"]
        self.drift_ax.plot(fields, dxs, color="#38bdf8", lw=1.5, marker='o',
                           markersize=3, label="dx (horizontal)")
        self.drift_ax.plot(fields, dys, color="#f472b6", lw=1.5, marker='s',
                           markersize=3, label="dy (vertical)")
        self.drift_ax.axhline(0, color=colors["border"], lw=1, ls='--')
        self.drift_ax.set_xlabel("Field (mT)" if self.loop_field is not None else "Image Index",
                                 color=tc, fontsize=10)
        self.drift_ax.set_ylabel("Shift (pixels)", color=tc, fontsize=10)
        self.drift_ax.set_title("Measured In-Plane Drift", color=tc, fontsize=11)
        self.drift_ax.legend(facecolor=colors["card"], edgecolor=colors["border"],
                             labelcolor=tc, fontsize=9)
        self.drift_ax.tick_params(colors=tc, labelsize=8)
        for sp in self.drift_ax.spines.values():
            sp.set_color(colors["spine"])
        self.drift_ax.grid(True, color=colors["border"], ls='--', lw=0.6, alpha=0.5)
        self.drift_fig.tight_layout()
        self.drift_canvas.draw_idle()

    # ────────────────────────── Theming ─────────────────────────────────────

    def _apply_theme(self):
        from gui_styles import apply_theme
        apply_theme(self, self.theme)

    def change_theme(self, theme):
        self.theme = theme
        self._apply_theme()
        # Rebuild plot colours
        if self.shifts is not None:
            self._plot_drift_curves()
        else:
            self._plot_drift_placeholder()

    # ────────────────────────── Slots & Helpers ──────────────────────────────

    def _log(self, msg):
        self.status_log.append(msg)
        self.status_log.verticalScrollBar().setValue(
            self.status_log.verticalScrollBar().maximum()
        )

    def _sync_search_spin(self, val):
        self.spin_search.blockSignals(True)
        self.spin_search.setValue(val)
        self.spin_search.blockSignals(False)

    def _sync_search_slider(self, val):
        self.sld_search.blockSignals(True)
        self.sld_search.setValue(val)
        self.sld_search.blockSignals(False)

    def _update_search_overlay(self, _=None):
        """Push current search-width value into the image label for painting."""
        self.img_label.search_width = self.spin_search.value()
        self.img_label.update()

    def _sync_man_x_spin(self, val):
        self.spin_man_x.blockSignals(True)
        self.spin_man_x.setValue(val)
        self.spin_man_x.blockSignals(False)
        self._on_manual_shift_changed()

    def _sync_man_x_slider(self, val):
        self.sld_man_x.blockSignals(True)
        self.sld_man_x.setValue(val)
        self.sld_man_x.blockSignals(False)
        self._on_manual_shift_changed()

    def _sync_man_y_spin(self, val):
        self.spin_man_y.blockSignals(True)
        self.spin_man_y.setValue(val)
        self.spin_man_y.blockSignals(False)
        self._on_manual_shift_changed()

    def _sync_man_y_slider(self, val):
        self.sld_man_y.blockSignals(True)
        self.sld_man_y.setValue(val)
        self.sld_man_y.blockSignals(False)
        self._on_manual_shift_changed()

    def _sync_list_to_slider(self, val):
        self.list_images.setCurrentRow(val)

    def _step_prev(self):
        if not self.image_files: return
        r = self.list_images.currentRow() - 1
        if r < 0: r = len(self.image_files) - 1
        self.list_images.setCurrentRow(r)

    def _step_next(self):
        if not self.image_files: return
        r = self.list_images.currentRow() + 1
        if r >= len(self.image_files): r = 0
        self.list_images.setCurrentRow(r)

    def _on_manual_shift_changed(self):
        row = self.list_images.currentRow()
        if row < 0 or row >= len(self.image_files) or self.txt_data is None:
            return
            
        fname = self.image_files[row]
        match = self.txt_data[self.txt_data['File'] == fname.strip()]
        if match.empty:
            return
        idx = match.index[0]
        
        # Initialize shifts if they don't exist so we can do pure manual correction
        if self.shifts is None:
            self.shifts = [(0.0, 0.0)] * len(self.txt_data)
            self.btn_save.setEnabled(True)
            
        dx = float(self.spin_man_x.value())
        dy = float(self.spin_man_y.value())
        self.shifts[idx] = (dy, dx)
        
        self._recompute_crop_bounds()
        self._plot_drift_curves()
        self._refresh_display()

    def _recompute_crop_bounds(self):
        if not self.shifts: return
        dys = [s[0] for s in self.shifts]
        dxs = [s[1] for s in self.shifts]
        self.crop_top = max(0, int(np.ceil(max([-dy for dy in dys]))))
        self.crop_bottom = max(0, int(np.ceil(max(dys))))
        self.crop_left = max(0, int(np.ceil(max([-dx for dx in dxs]))))
        self.crop_right = max(0, int(np.ceil(max(dxs))))

    def _contrast_slider_changed(self, val):
        self.contrast = val / 100.0
        self.spin_contrast.blockSignals(True)
        self.spin_contrast.setValue(self.contrast)
        self.spin_contrast.blockSignals(False)
        self._refresh_display()

    def _contrast_spin_changed(self, val):
        self.contrast = val
        self.sld_contrast.blockSignals(True)
        self.sld_contrast.setValue(int(val * 100))
        self.sld_contrast.blockSignals(False)
        self._refresh_display()

    def _choose_directory(self):
        d = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if not d:
            return
        self.img_dir = d
        self.ref_file = None
        self.ref_arr = None
        self.ref_gray = None
        self.shifts = None
        self.lbl_ref_info.setText("Reference: (not set)")
        self.btn_set_ref.setEnabled(False)
        self.btn_estimate.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.slider_images.setEnabled(False)
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.img_label.clear_roi()
        self._plot_drift_placeholder()
        self._load_directory()

    def _load_directory(self):
        # Load mapping txt first — we only care about images with field data
        self.txt_data = None
        self.loop_field = None
        self.image_files = []
        for fn in os.listdir(self.img_dir):
            if fn.lower().endswith('.txt'):
                try:
                    df = pd.read_csv(os.path.join(self.img_dir, fn),
                                     sep=None, engine='python',
                                     comment='#', skip_blank_lines=True)
                    df.columns = [c.strip() for c in df.columns]
                    if len(df.columns) >= 3:
                        df_filtered = df[df[df.columns[2]].str.lower().str.endswith('.png', na=False)]
                        if len(df_filtered) >= 3:
                            df = df_filtered.rename(columns={df.columns[0]: 'Field',
                                                             df.columns[1]: 'Intensity',
                                                             df.columns[2]: 'File'})
                            df['File'] = df['File'].str.strip()
                            df['File_Original'] = df['File']
                            self.txt_data = df.reset_index(drop=True)
                            self.loop_field = df['Field'].to_numpy(dtype=np.float32)
                            self._log(f"Loaded mapping: {fn}  ({len(df)} entries)")
                            break
                except Exception as e:
                    self._log(f"Could not parse {fn}: {e}")

        self._populate_images_from_mapping()

    def _toggle_unprocessed(self):
        if getattr(self, 'img_dir', None) is None or self.txt_data is None:
            return
        
        # Reset current state when switching file sets
        self.ref_file = None
        self.ref_arr = None
        self.ref_gray = None
        self.shifts = None
        self.lbl_ref_info.setText("Reference: (not set)")
        self.btn_estimate.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.img_label.clear_roi()
        self._plot_drift_placeholder()

        self._populate_images_from_mapping()

    def _populate_images_from_mapping(self):
        if self.txt_data is None or len(self.txt_data) == 0:
            self._log("No mapping .txt file found — cannot list images.")
            return

        use_unproc = self.chk_unprocessed.isChecked()
        mapped_files = list(self.txt_data['File_Original'])
        processed_files = []
        for f in mapped_files:
            if use_unproc:
                # Cleanly extract base prefix by stripping any existing suffixes
                base, ext = os.path.splitext(f)
                suffixes = ["_unprocessed", "_unproccessed", "_background", "_drift_corrected"]
                lower_base = base.lower()
                for s in suffixes:
                    if lower_base.endswith(s):
                        base = base[:-len(s)]
                        lower_base = base.lower()

                # Check which spelling variant exists on disk
                f_unproc = base + "_unprocessed.png"
                f_unproc_alt = base + "_unproccessed.png"
                if os.path.isfile(os.path.join(self.img_dir, f_unproc)):
                    processed_files.append(f_unproc)
                elif os.path.isfile(os.path.join(self.img_dir, f_unproc_alt)):
                    processed_files.append(f_unproc_alt)
                else:
                    # Fallback to the default one if neither is found on disk
                    processed_files.append(f_unproc)
            else:
                processed_files.append(f)
        
        self.txt_data['File'] = processed_files
        
        self.image_files = [
            f for f in processed_files
            if os.path.isfile(os.path.join(self.img_dir, f))
        ]

        n_missing = len(processed_files) - len(self.image_files)
        if n_missing > 0:
            self._log(f"Warning: {n_missing} file(s) in mapping not found on disk.")

        # Populate list widget
        self.list_images.blockSignals(True)
        self.list_images.clear()
        for f in self.image_files:
            field = self._field_for_file(f)
            if field is not None:
                self.list_images.addItem(f"{f}  ({field:.2f} mT)")
            else:
                self.list_images.addItem(f)
        self.list_images.blockSignals(False)

        if self.image_files:
            self.btn_set_ref.setEnabled(True)
            self.slider_images.setMinimum(0)
            self.slider_images.setMaximum(len(self.image_files) - 1)
            self.slider_images.setValue(0)
            self.slider_images.setEnabled(True)
            self.btn_prev.setEnabled(True)
            self.btn_next.setEnabled(True)
            self.list_images.setCurrentRow(0)
            self._log(f"Directory loaded: {self.img_dir}  ({len(self.image_files)} field-mapped images)")
        else:
            self.btn_set_ref.setEnabled(False)
            self.slider_images.setEnabled(False)
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            self._log("No valid images found in the selected directory.")

    def _field_for_file(self, filename):
        if self.txt_data is not None:
            match = self.txt_data[self.txt_data['File'] == filename.strip()]
            if not match.empty:
                return float(match.iloc[0]['Field'])
        return None

    def _on_image_selected(self, row):
        if row < 0 or row >= len(self.image_files):
            return
            
        if self.shifts is not None and self.txt_data is not None:
            fname = self.image_files[row]
            match = self.txt_data[self.txt_data['File'] == fname.strip()]
            if not match.empty:
                idx = match.index[0]
                dy, dx = self.shifts[idx]
                self.sld_man_x.blockSignals(True)
                self.spin_man_x.blockSignals(True)
                self.sld_man_y.blockSignals(True)
                self.spin_man_y.blockSignals(True)
                self.sld_man_x.setValue(int(dx))
                self.spin_man_x.setValue(int(dx))
                self.sld_man_y.setValue(int(dy))
                self.spin_man_y.setValue(int(dy))
                self.sld_man_x.blockSignals(False)
                self.spin_man_x.blockSignals(False)
                self.sld_man_y.blockSignals(False)
                self.spin_man_y.blockSignals(False)
        else:
            self.sld_man_x.blockSignals(True)
            self.spin_man_x.blockSignals(True)
            self.sld_man_y.blockSignals(True)
            self.spin_man_y.blockSignals(True)
            self.sld_man_x.setValue(0)
            self.spin_man_x.setValue(0)
            self.sld_man_y.setValue(0)
            self.spin_man_y.setValue(0)
            self.sld_man_x.blockSignals(False)
            self.spin_man_x.blockSignals(False)
            self.sld_man_y.blockSignals(False)
            self.spin_man_y.blockSignals(False)

        self.slider_images.blockSignals(True)
        self.slider_images.setValue(row)
        self.slider_images.blockSignals(False)

        self._refresh_display()
        fname = self.image_files[row]
        fpath = os.path.join(self.img_dir, fname)
        field = self._field_for_file(fname)
        if field is not None:
            self.lbl_field.setText(f"Field: {field:.2f} mT  |  {fname}")
        else:
            self.lbl_field.setText(fname)
        self._display_image(fpath)

    def _display_image(self, fpath):
        """Display a raw image."""
        try:
            arr = np.array(Image.open(fpath))
            arr = crop600(arr)

            # Raw image display
            if arr.ndim == 3:
                gray = to_gray(arr)
                disp = robust_normalize_raw(gray)
            else:
                disp = robust_normalize_raw(arr)

            self._set_label_array(disp, target_label=self.img_label)
        except Exception as e:
            self.img_label.setText(f"Error: {e}")

    def _display_corrected_image(self, fpath, fname):
        """Display the drift-corrected image, cropped to the common valid region."""
        if self.shifts is None or self.txt_data is None:
            self.corrected_label.setText("Run drift estimation first to see corrected images.")
            self.corrected_label.setPixmap(QPixmap())
            return

        match = self.txt_data[self.txt_data['File'] == fname.strip()]
        if match.empty:
            return
        idx = match.index[0]
        dy, dx = self.shifts[idx]

        try:
            arr = np.array(Image.open(fpath))
            arr = crop600(arr)
            
            # Apply correction
            corrected = apply_shift(arr, -dy, -dx)
            
            # Apply common valid region cropping
            h, w = corrected.shape[:2]
            r1 = self.crop_top
            r2 = h - self.crop_bottom
            c1 = self.crop_left
            c2 = w - self.crop_right
            
            if r2 <= r1 or c2 <= c1:
                self.corrected_label.setText("Drift too large, no valid overlap.")
                return

            if corrected.ndim == 3:
                cropped = corrected[r1:r2, c1:c2, :]
            else:
                cropped = corrected[r1:r2, c1:c2]

            # Display
            if cropped.ndim == 3:
                gray = to_gray(cropped)
                disp = robust_normalize_raw(gray)
            else:
                disp = robust_normalize_raw(cropped)
                
            self._set_label_array(disp, target_label=self.corrected_label)
        except Exception as e:
            self.corrected_label.setText(f"Error: {e}")

    def _set_label_array(self, disp_uint8, target_label=None):
        """Push a uint8 grayscale (H×W) array through the chosen colormap to QLabel."""
        if target_label is None:
            target_label = self.img_label

        cmap_name = self.cmb_cmap.currentText()
        if cmap_name == "gray":
            h, w = disp_uint8.shape
            qimg = QImage(disp_uint8.tobytes(), w, h, w, QImage.Format_Grayscale8)
        else:
            norm = disp_uint8.astype(np.float32) / 255.0
            try:
                cmap = mpl.colormaps.get_cmap(cmap_name)
            except AttributeError:
                cmap = mpl.cm.get_cmap(cmap_name)
            rgba = (cmap(norm) * 255).astype(np.uint8)
            h, w, _ = rgba.shape
            qimg = QImage(rgba.tobytes(), w, h, w * 4, QImage.Format_RGBA8888)

        target_label.setPixmap(QPixmap.fromImage(qimg))
        target_label.setText("")

    def _refresh_display(self):
        """Re-display the currently selected image with updated settings."""
        row = self.list_images.currentRow()
        if row >= 0 and row < len(self.image_files):
            fpath = os.path.join(self.img_dir, self.image_files[row])
            if self.preview_tabs.currentIndex() == 0:
                self._display_image(fpath)
            else:
                self._display_corrected_image(fpath, self.image_files[row])

    def _set_reference(self):
        row = self.list_images.currentRow()
        if row < 0 or row >= len(self.image_files):
            return
        fname = self.image_files[row]
        fpath = os.path.join(self.img_dir, fname)
        try:
            arr = np.array(Image.open(fpath))
            arr = crop600(arr)
            self.ref_arr = arr
            self.ref_gray = to_gray(arr)
            self.ref_file = fname
            field = self._field_for_file(fname)
            if field is not None:
                self.lbl_ref_info.setText(f"Reference: {fname}\n({field:.2f} mT)")
            else:
                self.lbl_ref_info.setText(f"Reference: {fname}")
            self._log(f"Reference image set: {fname}")
            self._check_ready()
            self._refresh_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load reference image:\n{e}")

    def _on_roi_changed(self):
        roi = self.img_label.roi
        if roi is not None:
            r, c, h, w = roi
            self.lbl_roi_info.setText(
                f"Row {r}–{r+h},  Col {c}–{c+w}\n({w}×{h} px)"
            )
        else:
            self.lbl_roi_info.setText("No ROI selected")
        self._check_ready()

    def _clear_roi(self):
        self.img_label.clear_roi()

    def _check_ready(self):
        """Enable estimation button when all prerequisites are satisfied."""
        ready = (self.ref_gray is not None and self.img_label.roi is not None)
        self.btn_estimate.setEnabled(ready)

    # ────────────────────────── Drift Estimation ─────────────────────────────

    def _run_estimation(self):
        """Run drift estimation synchronously with processEvents for GUI updates."""
        if self.txt_data is None or len(self.txt_data) == 0:
            QMessageBox.warning(self, "No Data", "No mapping .txt file found in the selected directory.")
            return
        roi = self.img_label.roi
        if roi is None:
            QMessageBox.warning(self, "No ROI", "Please draw a Defect ROI on the image first.")
            return
        if self.ref_gray is None:
            QMessageBox.warning(self, "No Reference", "Please set a reference image first.")
            return

        search_width = self.spin_search.value()
        # Clamp ROI to reference image bounds
        img_h, img_w = self.ref_gray.shape
        r, c, h, w = roi
        r = max(0, r);  c = max(0, c)
        h = min(h, img_h - r);  w = min(w, img_w - c)
        if h < 4 or w < 4:
            QMessageBox.warning(self, "ROI too small",
                                "The selected ROI is too small or outside image bounds.")
            return

        # Verify ROI + search_width fits in image
        if (r - search_width < 0 or c - search_width < 0 or
                r + h + search_width > img_h or c + w + search_width > img_w):
            QMessageBox.warning(self, "Search out of bounds",
                                f"The ROI + search width ({search_width} px) extends "
                                f"beyond the image edges.\n"
                                f"Please shrink the search width or move the ROI "
                                f"further from the edge.")
            return

        self.btn_estimate.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.btn_dir.setEnabled(False)
        self.progress_bar.setValue(0)
        self._abort_flag = False
        self._log("Starting drift estimation …")
        QApplication.setOverrideCursor(Qt.WaitCursor)

        use_sobel = self.chk_use_sobel.isChecked()

        try:
            if use_sobel:
                ref_grad = sobel_gradient(self.ref_gray)
            else:
                ref_grad = self.ref_gray
                
            ref_patch = ref_grad[r:r + h, c:c + w]

            shifts = []
            n = len(self.txt_data)
            # ⚡ Bolt: Use itertuples instead of iterrows for much faster iteration
            for i, row_data in enumerate(self.txt_data.itertuples(index=False)):
                QApplication.processEvents()   # keep GUI alive
                if self._abort_flag:
                    self._log("Estimation aborted by user.")
                    break

                fname = row_data.File.strip()
                fpath = os.path.join(self.img_dir, fname)
                self._log(f"  [{i+1}/{n}] {fname}")

                if not os.path.isfile(fpath):
                    self._log(f"    File not found — skipping.")
                    shifts.append((0.0, 0.0))
                    continue

                try:
                    arr = np.array(Image.open(fpath))
                    arr = crop600(arr)
                    gray = to_gray(arr)
                    
                    if use_sobel:
                        grad = sobel_gradient(gray)
                    else:
                        grad = gray

                    dy, dx, _ = estimate_shift_sqdiff(
                        ref_patch, grad, r, c, h, w, search_width
                    )
                    shifts.append((dy, dx))
                except Exception as e:
                    self._log(f"    Warning: {e}")
                    shifts.append((0.0, 0.0))

                self.progress_bar.setValue(int((i + 1) / n * 100))

            if not self._abort_flag:
                self.shifts = shifts
                self._recompute_crop_bounds()

                self.btn_save.setEnabled(True)
                self.progress_bar.setValue(100)
                self._log(f"Drift estimation complete. {len(shifts)} shifts computed.")
                self._plot_drift_curves()
                
                # Auto-switch to corrected tab
                self.preview_tabs.setCurrentIndex(1)
                self._refresh_display()
            else:
                self.shifts = None
                self.progress_bar.setValue(0)

        except Exception as e:
            self._log(f"Error during estimation: {e}")
            QMessageBox.critical(self, "Drift Estimation Error", str(e))
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_estimate.setEnabled(True)
            self.btn_abort.setEnabled(False)
            self.btn_dir.setEnabled(True)

    def _abort_estimation(self):
        self._abort_flag = True
        self.btn_abort.setEnabled(False)
        self._log("Abort requested …")

    # ────────────────────────── Saving ──────────────────────────────────────

    def _save_corrected(self):
        if self.shifts is None:
            return

        # Suggest an output directory
        default_out = self.img_dir.rstrip('/\\') + "_drift_corrected"
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select / Create Output Directory", default_out
        )
        if not out_dir:
            return

        os.makedirs(out_dir, exist_ok=True)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        errors = []
        try:
            n = len(self.txt_data)
            # ⚡ Bolt: Use itertuples instead of iterrows for much faster iteration
            for i, row in enumerate(self.txt_data.itertuples(index=False)):
                input_fname = row.File.strip()
                output_fname = row.File_Original.strip()
                fpath = os.path.join(self.img_dir, input_fname)
                dy, dx = self.shifts[i]
                try:
                    arr = np.array(Image.open(fpath))
                    arr_cropped = crop600(arr)
                    corrected = apply_shift(arr_cropped, -dy, -dx)
                    
                    # Apply common valid region cropping
                    h, w = corrected.shape[:2]
                    r1 = self.crop_top
                    r2 = h - self.crop_bottom
                    c1 = self.crop_left
                    c2 = w - self.crop_right
                    if corrected.ndim == 3:
                        final_arr = corrected[r1:r2, c1:c2, :]
                    else:
                        final_arr = corrected[r1:r2, c1:c2]
                        
                    Image.fromarray(final_arr).save(os.path.join(out_dir, output_fname))
                except Exception as e:
                    errors.append(f"{output_fname}: {e}")

                self.progress_bar.setValue(int((i + 1) / n * 100))
                QApplication.processEvents()

            # Copy / rewrite the mapping txt
            if self.txt_data is not None:
                src_txt = None
                for fn in os.listdir(self.img_dir):
                    if fn.lower().endswith('.txt'):
                        src_txt = fn
                        break
                if src_txt:
                    import shutil
                    shutil.copy2(os.path.join(self.img_dir, src_txt),
                                 os.path.join(out_dir, src_txt))

            # Save diagnostic drift plot
            self._save_drift_plot(out_dir)

            if errors:
                QMessageBox.warning(self, "Some errors",
                                    f"Saved with {len(errors)} error(s):\n" + "\n".join(errors[:5]))
            else:
                QMessageBox.information(self, "Done",
                    f"Corrected images saved to:\n{out_dir}")
        finally:
            QApplication.restoreOverrideCursor()

    def _save_drift_plot(self, out_dir):
        """Save the drift analysis plot as a PNG."""
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            fields = self.loop_field if self.loop_field is not None else np.arange(len(self.shifts))
            dys = np.array([s[0] for s in self.shifts])
            dxs = np.array([s[1] for s in self.shifts])
            ax.plot(fields, dxs, 'b-o', ms=4, label="dx (horizontal shift)")
            ax.plot(fields, dys, 'r-s', ms=4, label="dy (vertical shift)")
            ax.axhline(0, color='gray', lw=1, ls='--')
            ax.set_xlabel("Field (mT)" if self.loop_field is not None else "Image index")
            ax.set_ylabel("Shift (pixels)")
            ax.set_title("In-Plane Drift Correction")
            ax.legend()
            ax.grid(True, ls='--', alpha=0.5)
            plot_path = os.path.join(out_dir, "drift_correction_analysis.png")
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            self._log(f"Drift plot saved: {plot_path}")
        except Exception as e:
            self._log(f"Could not save drift plot: {e}")

    # ────────────────────────── Close ───────────────────────────────────────

    def closeEvent(self, event):
        self._abort_flag = True
        event.accept()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point (standalone)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", default="charcoal",
                        choices=["dark", "charcoal", "light"])
    args, _ = parser.parse_known_args()
    w = DriftCorrectorWindow(theme=args.theme)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
