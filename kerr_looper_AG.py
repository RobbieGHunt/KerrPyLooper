# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:36:05 2026

@author: robhu413
"""

import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QFileDialog, QVBoxLayout, QPushButton,
    QListWidget, QLabel, QMessageBox, QHBoxLayout, QSlider,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSplitter, QTextEdit,
    QCheckBox, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import scipy.ndimage as ndimage


def crop600(arr):
    h, w = arr.shape[0], arr.shape[1]
    h_crop = min(h, 600)
    w_crop = min(w, 600)
    w_start = (w - w_crop) // 2
    if arr.ndim == 3:
        return arr[:h_crop, w_start:w_start+w_crop, :]
    else:
        return arr[:h_crop, w_start:w_start+w_crop]

def normalized_for_display(arr, scale=None, contrast=1.0):
    arr = arr.astype(np.float32)
    arr = crop600(arr)
    if scale is None or scale == 0:
        scale = np.std(arr) if np.std(arr) > 0 else 1.0
    arr_disp = np.arcsinh(arr / scale)
    arr_disp -= arr_disp.min()
    if arr_disp.ptp() == 0:
        arr_disp[:] = 0
    else:
        arr_disp /= arr_disp.ptp()
    arr_disp = (arr_disp * 255)
    arr_disp = 127.5 + contrast * (arr_disp - 127.5)
    arr_disp = np.clip(arr_disp, 0, 255).astype(np.uint8)
    return arr_disp

def normalize_image(img):
    m = np.mean(img)
    s = np.std(img)
    if s == 0:
        return img - m
    return (img - m) / s

def get_gradient_magnitude(img):
    dx = ndimage.sobel(img, axis=0)
    dy = ndimage.sobel(img, axis=1)
    return np.sqrt(dx**2 + dy**2)

def find_defect_roi(img, patch_size=128):
    """
    Find coordinates of the patch_size x patch_size region in the image with
    the highest Sobel gradient energy (indicative of static defects/scratches).
    """
    grad = get_gradient_magnitude(img)
    h, w = img.shape[0], img.shape[1]
    best_score = -1
    best_coords = (150, 150) # default fallback center area
    
    step = 10
    margin = 20
    for r in range(margin, h - patch_size - margin, step):
        for c in range(margin, w - patch_size - margin, step):
            patch_grad = grad[r:r+patch_size, c:c+patch_size]
            score = np.sum(patch_grad)
            if score > best_score:
                best_score = score
                best_coords = (r, c)
                
    return best_coords

def estimate_defocus(ref_img_norm, target_img_norm):
    target_grad = get_gradient_magnitude(target_img_norm)
    def loss(sigma):
        if sigma <= 0.01:
            blurred_ref = ref_img_norm
        else:
            blurred_ref = ndimage.gaussian_filter(ref_img_norm, sigma=sigma)
        blurred_grad = get_gradient_magnitude(blurred_ref)
        return np.mean((target_grad - blurred_grad) ** 2)
    
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(loss, bounds=(0.0, 3.0), method='bounded')
    return res.x

def wiener_deconvolve(image, sigma, balance=0.02):
    if sigma <= 0.05:
        return image
    h, w = image.shape
    u = np.fft.fftfreq(h)
    v = np.fft.fftfreq(w)
    uu, vv = np.meshgrid(u, v, indexing='ij')
    otf = np.exp(-2 * np.pi**2 * sigma**2 * (uu**2 + vv**2))
    img_fft = np.fft.fft2(image)
    otf_conj = np.conj(otf)
    wiener_filter = otf_conj / (np.abs(otf)**2 + balance)
    deblurred_fft = img_fft * wiener_filter
    deblurred = np.real(np.fft.ifft2(deblurred_fft))
    return deblurred

class LoopCorrectionPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.coeffs = dict(drift=0.0, linear=0.0, quad=0.0, quad_offset=0.0)
        self.z_coeff = 0.0
        self.normalize = False
        self.contrast = 1.0
        self.parent_widget = parent
        self.hc_hr_marks = None  # Stores latest Hc/Hr marks for highlighting

    def init_ui(self):
        main_layout = QVBoxLayout()
        splitter = QSplitter(Qt.Vertical)

        # Top widget: Plot canvas
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(100)
        splitter.addWidget(self.canvas)

        # Bottom widget: Controls container
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_widget.setLayout(controls_layout)

        self.btn_norm = QPushButton("Normalize (Intensity: -1 .. 1)")
        self.btn_norm.setCheckable(True)
        self.btn_norm.clicked.connect(self.toggle_normalize)
        controls_layout.addWidget(self.btn_norm)

        group = QGroupBox("Corrections")
        glayout = QFormLayout()

        # Drift
        hbox_drift = QHBoxLayout()
        self.sld_drift = QSlider(Qt.Horizontal)
        self.sld_drift.setMinimum(-10)
        self.sld_drift.setMaximum(10)
        self.sld_drift.setValue(0)
        self.sld_drift.valueChanged.connect(self.slider_changed)
        self.spin_drift = QDoubleSpinBox()
        self.spin_drift.setDecimals(3)
        self.spin_drift.setRange(-0.5, 0.5)
        self.spin_drift.setSingleStep(0.001)
        self.spin_drift.setValue(0.0)
        self.spin_drift.valueChanged.connect(self.spinbox_changed)
        hbox_drift.addWidget(self.sld_drift)
        hbox_drift.addWidget(self.spin_drift)
        glayout.addRow("Drift", hbox_drift)

        # Linear Faraday
        hbox_linear = QHBoxLayout()
        self.sld_linear = QSlider(Qt.Horizontal)
        self.sld_linear.setMinimum(-100)
        self.sld_linear.setMaximum(100)
        self.sld_linear.setValue(0)
        self.sld_linear.valueChanged.connect(self.slider_changed)
        self.spin_linear = QDoubleSpinBox()
        self.spin_linear.setDecimals(3)
        self.spin_linear.setRange(-1.0, 1.0)
        self.spin_linear.setSingleStep(0.001)
        self.spin_linear.setValue(0.0)
        self.spin_linear.valueChanged.connect(self.spinbox_changed)
        hbox_linear.addWidget(self.sld_linear)
        hbox_linear.addWidget(self.spin_linear)
        glayout.addRow("Linear Faraday", hbox_linear)

        # Quadratic Faraday
        hbox_quad = QHBoxLayout()
        self.sld_quad = QSlider(Qt.Horizontal)
        self.sld_quad.setMinimum(-10)
        self.sld_quad.setMaximum(10)
        self.sld_quad.setValue(0)
        self.sld_quad.valueChanged.connect(self.slider_changed)
        self.spin_quad = QDoubleSpinBox()
        self.spin_quad.setDecimals(4)
        self.spin_quad.setRange(-0.01, 0.01)
        self.spin_quad.setSingleStep(0.001)
        self.spin_quad.setValue(0.0)
        self.spin_quad.valueChanged.connect(self.spinbox_changed)
        hbox_quad.addWidget(self.sld_quad)
        hbox_quad.addWidget(self.spin_quad)
        glayout.addRow("Quadratic Faraday", hbox_quad)

        # Quadratic Offset
        hbox_quad_offset = QHBoxLayout()
        self.sld_quad_offset = QSlider(Qt.Horizontal)
        self.sld_quad_offset.setMinimum(-2000)
        self.sld_quad_offset.setMaximum(2000)
        self.sld_quad_offset.setValue(0)
        self.sld_quad_offset.valueChanged.connect(self.slider_changed)
        self.spin_quad_offset = QDoubleSpinBox()
        self.spin_quad_offset.setDecimals(2)
        self.spin_quad_offset.setRange(-2000.0, 2000.0)
        self.spin_quad_offset.setSingleStep(0.1)
        self.spin_quad_offset.setValue(0.0)
        self.spin_quad_offset.valueChanged.connect(self.spinbox_changed)
        hbox_quad_offset.addWidget(self.sld_quad_offset)
        hbox_quad_offset.addWidget(self.spin_quad_offset)
        glayout.addRow("Quadratic Field Offset", hbox_quad_offset)

        # Contrast control
        hbox_contrast = QHBoxLayout()
        self.sld_contrast = QSlider(Qt.Horizontal)
        self.sld_contrast.setMinimum(10)
        self.sld_contrast.setMaximum(400)
        self.sld_contrast.setValue(100)
        self.sld_contrast.valueChanged.connect(self.contrast_slider_changed)
        self.spin_contrast = QDoubleSpinBox()
        self.spin_contrast.setDecimals(2)
        self.spin_contrast.setRange(0.1, 4.0)
        self.spin_contrast.setSingleStep(0.01)
        self.spin_contrast.setValue(1.0)
        self.spin_contrast.valueChanged.connect(self.contrast_spinbox_changed)
        hbox_contrast.addWidget(self.sld_contrast)
        hbox_contrast.addWidget(self.spin_contrast)
        glayout.addRow("Contrast Stretch", hbox_contrast)

        # Auto, Zero and Hc/Hr buttons
        hbox_auto = QHBoxLayout()
        self.btn_auto = QPushButton("Auto Correct")
        self.btn_auto.clicked.connect(self.auto_correct)
        hbox_auto.addWidget(self.btn_auto)
        
        self.btn_zero = QPushButton("Zero All")
        self.btn_zero.clicked.connect(self.zero_all)
        hbox_auto.addWidget(self.btn_zero)
        
        self.btn_hc = QPushButton("Calc Hc/Hr")
        self.btn_hc.clicked.connect(self.calc_hc_hr)
        hbox_auto.addWidget(self.btn_hc)
        glayout.addRow(hbox_auto)

        group.setLayout(glayout)
        controls_layout.addWidget(group)

        # Out-of-Plane Focus Correction
        group_z = QGroupBox("Out-of-Plane Focus Correction")
        zlayout = QFormLayout()

        self.chk_z_drift = QCheckBox("Enable Z-Drift Correction")
        self.chk_z_drift.stateChanged.connect(self.z_drift_toggled)
        zlayout.addRow(self.chk_z_drift)

        self.cmb_z_method = QComboBox()
        self.cmb_z_method.addItems(["Blur Reference (Noise-free)", "Deblur Target (Wiener Filter)"])
        self.cmb_z_method.currentIndexChanged.connect(self.z_method_changed)
        zlayout.addRow("Method", self.cmb_z_method)

        hbox_z_quad = QHBoxLayout()
        self.sld_z_quad = QSlider(Qt.Horizontal)
        self.sld_z_quad.setMinimum(0)
        self.sld_z_quad.setMaximum(200)
        self.sld_z_quad.setValue(0)
        self.sld_z_quad.valueChanged.connect(self.z_slider_changed)

        self.spin_z_quad = QDoubleSpinBox()
        self.spin_z_quad.setDecimals(2)
        self.spin_z_quad.setRange(0.0, 20.0)
        self.spin_z_quad.setSingleStep(0.1)
        self.spin_z_quad.setValue(0.0)
        self.spin_z_quad.valueChanged.connect(self.z_spinbox_changed)

        hbox_z_quad.addWidget(self.sld_z_quad)
        hbox_z_quad.addWidget(self.spin_z_quad)
        zlayout.addRow("Focus Coeff (10^-6/mT^2)", hbox_z_quad)

        self.btn_z_auto = QPushButton("Auto Estimate Focus Drift")
        self.btn_z_auto.clicked.connect(self.auto_estimate_z_drift)
        zlayout.addRow(self.btn_z_auto)

        group_z.setLayout(zlayout)
        controls_layout.addWidget(group_z)

        self.hc_hr_output = QTextEdit()
        self.hc_hr_output.setReadOnly(True)
        self.hc_hr_output.setMinimumHeight(20)
        controls_layout.addWidget(self.hc_hr_output)

        splitter.addWidget(controls_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([300, 400])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def update_correction_ranges(self, ptp_val):
        if np.isnan(ptp_val) or ptp_val <= 0:
            ptp_val = 1.0
        
        # Get physical parameters from parent widget
        max_field = 220.0
        n_points = 100.0
        if self.parent_widget and self.parent_widget.loop_field is not None:
            max_field = float(np.max(np.abs(self.parent_widget.loop_field)))
            n_points = float(len(self.parent_widget.loop_field))
            
        # Calculate ranges based on physical impact on loop height (ptp_val)
        max_drift = max(0.5, 0.1 * ptp_val / n_points)
        max_linear = max(1.0, 0.2 * ptp_val / max_field)
        max_quad = max(0.01, 0.2 * ptp_val / (max_field**2))
        
        self.spin_drift.blockSignals(True)
        self.sld_drift.blockSignals(True)
        self.spin_drift.setRange(-max_drift, max_drift)
        self.sld_drift.setMinimum(int(-max_drift * 100))
        self.sld_drift.setMaximum(int(max_drift * 100))
        self.spin_drift.blockSignals(False)
        self.sld_drift.blockSignals(False)
        
        self.spin_linear.blockSignals(True)
        self.sld_linear.blockSignals(True)
        self.spin_linear.setRange(-max_linear, max_linear)
        self.sld_linear.setMinimum(int(-max_linear * 100))
        self.sld_linear.setMaximum(int(max_linear * 100))
        self.spin_linear.blockSignals(False)
        self.sld_linear.blockSignals(False)
        
        self.spin_quad.blockSignals(True)
        self.sld_quad.blockSignals(True)
        self.spin_quad.setRange(-max_quad, max_quad)
        self.sld_quad.setMinimum(int(-max_quad * 1000))
        self.sld_quad.setMaximum(int(max_quad * 1000))
        self.spin_quad.blockSignals(False)
        self.sld_quad.blockSignals(False)
        
        # Scale quad_offset range
        max_field_val = 500.0
        if self.parent_widget and self.parent_widget.loop_field is not None:
            max_field_val = float(np.max(np.abs(self.parent_widget.loop_field)))
        max_qo = max(2000.0, 10.0 * max_field_val)
        self.spin_quad_offset.blockSignals(True)
        self.sld_quad_offset.blockSignals(True)
        self.spin_quad_offset.setRange(-max_qo, max_qo)
        self.sld_quad_offset.setMinimum(int(-max_qo * 10))
        self.sld_quad_offset.setMaximum(int(max_qo * 10))
        self.spin_quad_offset.blockSignals(False)
        self.sld_quad_offset.blockSignals(False)

    def toggle_normalize(self, checked=None):
        self.normalize = self.btn_norm.isChecked() if checked is None else checked
        if self.parent_widget:
            self.parent_widget.request_loop_update()

    def zero_all(self):
        self.set_slider_spinbox('drift', 0.0)
        self.set_slider_spinbox('linear', 0.0)
        self.set_slider_spinbox('quad', 0.0)
        self.set_slider_spinbox('quad_offset', 0.0)
        self.coeffs['drift'] = 0.0
        self.coeffs['linear'] = 0.0
        self.coeffs['quad'] = 0.0
        self.coeffs['quad_offset'] = 0.0
        self.normalize = False
        self.btn_norm.setChecked(False)
        
        # Reset Z-drift focus parameters
        self.set_z_coeff(0.0)
        self.chk_z_drift.setChecked(False)
        
        self.hc_hr_marks = None
        if self.parent_widget:
            self.parent_widget.request_loop_update()
            self.parent_widget.show_current_subtracted_image_contrast_only()

    def z_drift_toggled(self, state):
        if self.parent_widget:
            self.parent_widget.request_loop_update()
            self.parent_widget.show_current_subtracted_image_contrast_only()

    def z_method_changed(self, idx):
        if self.parent_widget:
            self.parent_widget.request_loop_update()
            self.parent_widget.show_current_subtracted_image_contrast_only()

    def z_slider_changed(self, val):
        coeff_val = val / 10.0
        self.spin_z_quad.blockSignals(True)
        self.spin_z_quad.setValue(coeff_val)
        self.spin_z_quad.blockSignals(False)
        if self.parent_widget:
            self.parent_widget.request_loop_update()
            self.parent_widget.show_current_subtracted_image_contrast_only()

    def z_spinbox_changed(self, val):
        self.sld_z_quad.blockSignals(True)
        self.sld_z_quad.setValue(int(val * 10))
        self.sld_z_quad.blockSignals(False)
        if self.parent_widget:
            self.parent_widget.request_loop_update()
            self.parent_widget.show_current_subtracted_image_contrast_only()

    def set_z_coeff(self, val):
        self.sld_z_quad.blockSignals(True)
        self.spin_z_quad.blockSignals(True)
        self.sld_z_quad.setValue(int(val * 10))
        self.spin_z_quad.setValue(val)
        self.sld_z_quad.blockSignals(False)
        self.spin_z_quad.blockSignals(False)

    def auto_estimate_z_drift(self):
        if self.parent_widget is None or self.parent_widget.txt_data is None:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            df = self.parent_widget.txt_data
            fields = self.parent_widget.loop_field
            if fields is None or len(df) == 0:
                return

            zero_idx = np.argmin(np.abs(fields))
            ref_file = df.iloc[zero_idx]['File'].strip()
            ref_path = os.path.join(self.parent_widget.img_dir, ref_file)

            ref_img = np.array(Image.open(ref_path)).astype(np.float64)
            ref_cropped = crop600(ref_img)
            
            # Find defect ROI coordinates in 600x600 image
            patch_size = 128
            r, c = find_defect_roi(ref_cropped, patch_size=patch_size)
            ref_patch = ref_cropped[r:r+patch_size, c:c+patch_size]
            ref_patch_norm = normalize_image(ref_patch)

            # Select 5 test images from the txt mapping (ends, center, and intermediates)
            n_imgs = len(df)
            test_indices = sorted(list(set([0, n_imgs//4, zero_idx, 3*n_imgs//4, n_imgs-1])))

            sigmas = []
            test_fields = []
            for idx in test_indices:
                fname = df.iloc[idx]['File'].strip()
                fpath = os.path.join(self.parent_widget.img_dir, fname)
                target_img = np.array(Image.open(fpath)).astype(np.float64)
                target_cropped = crop600(target_img)
                target_patch = target_cropped[r:r+patch_size, c:c+patch_size]
                target_patch_norm = normalize_image(target_patch)
                
                sigma_est = estimate_defocus(ref_patch_norm, target_patch_norm)
                sigmas.append(sigma_est)
                test_fields.append(fields[idx])

            # Fit model: s = a * H^2 + c using linear regression Y = a * X + c
            X = np.array(test_fields) ** 2
            Y = np.array(sigmas)
            X_mean = np.mean(X)
            Y_mean = np.mean(Y)
            num = np.sum((X - X_mean) * (Y - Y_mean))
            den = np.sum((X - X_mean) ** 2)
            a_est = num / den if den != 0 else 0.0
            a_est = max(0.0, a_est)
            a_units = a_est * 1e6

            # Clip to valid range [0, 20]
            a_units = min(20.0, max(0.0, a_units))

            self.chk_z_drift.setChecked(True)
            self.set_z_coeff(a_units)
            self.parent_widget.request_loop_update()
            self.parent_widget.show_current_subtracted_image_contrast_only()
            
            # Inform user about selected defect ROI
            QMessageBox.information(
                self, "Z-Drift Auto Estimate",
                f"Selected defect region at: row {r}, col {c} (size {patch_size}x{patch_size}).\n"
                f"Fitted quadratic coefficient: {a_units:.2f} * 10^-6 / mT^2."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Auto estimation failed:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

    def set_slider_spinbox(self, key, value):
        mapping = {'drift': (self.sld_drift, self.spin_drift, 100),
                   'linear': (self.sld_linear, self.spin_linear, 100),
                   'quad': (self.sld_quad, self.spin_quad, 1000),
                   'quad_offset': (self.sld_quad_offset, self.spin_quad_offset, 10)}
        slider, spin, mult = mapping[key]
        
        # Dynamically expand range if the value exceeds current limits
        min_val = spin.minimum()
        max_val = spin.maximum()
        if value > max_val or value < min_val:
            new_max = max(abs(value) * 1.2, max_val)
            slider.blockSignals(True)
            spin.blockSignals(True)
            spin.setRange(-new_max, new_max)
            slider.setMinimum(int(-new_max * mult))
            slider.setMaximum(int(new_max * mult))
            slider.blockSignals(False)
            spin.blockSignals(False)

        slider.blockSignals(True)
        spin.blockSignals(True)
        slider.setValue(int(round(value * mult)))
        spin.setValue(value)
        slider.blockSignals(False)
        spin.blockSignals(False)

    def slider_changed(self, _):
        self.coeffs['drift'] = self.sld_drift.value() / 100.0
        self.spin_drift.blockSignals(True)
        self.spin_drift.setValue(self.coeffs['drift'])
        self.spin_drift.blockSignals(False)

        self.coeffs['linear'] = self.sld_linear.value() / 100.0
        self.spin_linear.blockSignals(True)
        self.spin_linear.setValue(self.coeffs['linear'])
        self.spin_linear.blockSignals(False)

        self.coeffs['quad'] = self.sld_quad.value() / 1000.0
        self.spin_quad.blockSignals(True)
        self.spin_quad.setValue(self.coeffs['quad'])
        self.spin_quad.blockSignals(False)

        self.coeffs['quad_offset'] = self.sld_quad_offset.value() / 10.0
        self.spin_quad_offset.blockSignals(True)
        self.spin_quad_offset.setValue(self.coeffs['quad_offset'])
        self.spin_quad_offset.blockSignals(False)

        self.hc_hr_marks = None  # clear markers
        if self.parent_widget:
            self.parent_widget.request_loop_update()
            self.parent_widget.show_current_subtracted_image_contrast_only()

    def spinbox_changed(self, _):
        self.coeffs['drift'] = self.spin_drift.value()
        self.sld_drift.blockSignals(True)
        self.sld_drift.setValue(int(round(self.coeffs['drift'] * 100)))
        self.sld_drift.blockSignals(False)

        self.coeffs['linear'] = self.spin_linear.value()
        self.sld_linear.blockSignals(True)
        self.sld_linear.setValue(int(round(self.coeffs['linear'] * 100)))
        self.sld_linear.blockSignals(False)

        self.coeffs['quad'] = self.spin_quad.value()
        self.sld_quad.blockSignals(True)
        self.sld_quad.setValue(int(round(self.coeffs['quad'] * 1000)))
        self.sld_quad.blockSignals(False)

        self.coeffs['quad_offset'] = self.spin_quad_offset.value()
        self.sld_quad_offset.blockSignals(True)
        self.sld_quad_offset.setValue(int(round(self.coeffs['quad_offset'] * 10)))
        self.sld_quad_offset.blockSignals(False)

        self.hc_hr_marks = None  # clear markers
        if self.parent_widget:
            self.parent_widget.request_loop_update()
            self.parent_widget.show_current_subtracted_image_contrast_only()

    def contrast_slider_changed(self, val):
        self.contrast = val / 100.0
        self.spin_contrast.blockSignals(True)
        self.spin_contrast.setValue(self.contrast)
        self.spin_contrast.blockSignals(False)
        if self.parent_widget:
            if hasattr(self.parent_widget, 'sld_img_contrast'):
                self.parent_widget.sld_img_contrast.blockSignals(True)
                self.parent_widget.sld_img_contrast.setValue(val)
                self.parent_widget.sld_img_contrast.blockSignals(False)
            self.parent_widget.show_current_subtracted_image_contrast_only()

    def contrast_spinbox_changed(self, val):
        self.contrast = val
        self.sld_contrast.blockSignals(True)
        self.sld_contrast.setValue(int(val * 100))
        self.sld_contrast.blockSignals(False)
        if self.parent_widget:
            if hasattr(self.parent_widget, 'sld_img_contrast'):
                self.parent_widget.sld_img_contrast.blockSignals(True)
                self.parent_widget.sld_img_contrast.setValue(int(val * 100))
                self.parent_widget.sld_img_contrast.blockSignals(False)
            self.parent_widget.show_current_subtracted_image_contrast_only()

    def plot_loop(self, field, intensity, show_title="Hysteresis Loop"):
        self.ax.clear()
        self.ax.plot(field, intensity, 'o-', lw=1.5, markersize=4)
        self.ax.set_xlabel("Field (mT)")
        self.ax.set_ylabel("Intensity")
        self.ax.set_title(show_title)
        self.ax.grid()
        # Highlight Hc/Hr if they exist
        if self.hc_hr_marks is not None:
            hc_pos, hc_neg, rem_fields, rem_values = self.hc_hr_marks
            if hc_pos is not None:
                self.ax.plot(hc_pos[0], hc_pos[1], 'o', ms=12, mec='black', mfc='lime', label='Hc+')
                self.ax.annotate(f"Hc+\n{hc_pos[0]:.2f}", xy=hc_pos, xytext=(hc_pos[0], hc_pos[1]+0.1*(max(intensity)-min(intensity))), 
                                 ha="center", fontsize=9, color='green',
                                 arrowprops=dict(arrowstyle='->', color='green'))
            if hc_neg is not None:
                self.ax.plot(hc_neg[0], hc_neg[1], 'o', ms=12, mec='black', mfc='magenta', label='Hc-')
                self.ax.annotate(f"Hc-\n{hc_neg[0]:.2f}", xy=hc_neg, xytext=(hc_neg[0], hc_neg[1]+0.1*(max(intensity)-min(intensity))),
                                 ha="center", fontsize=9, color='magenta',
                                 arrowprops=dict(arrowstyle='->', color='magenta'))
            if rem_fields is not None and rem_values is not None:
                for rf, rv in zip(rem_fields, rem_values):
                    self.ax.plot(rf, rv, 's', ms=10, mec='black', mfc='orange', label='Hr')
                self.ax.annotate(f"Hr\n{np.mean(rem_values):.3f}", 
                                 xy=(np.mean(rem_fields), np.mean(rem_values)), 
                                 xytext=(np.mean(rem_fields), np.mean(rem_values) + 0.15*(max(intensity)-min(intensity))), 
                                 ha="center", fontsize=9, color='orange',
                                 arrowprops=dict(arrowstyle='->', color='orange'))
        self.ax.legend(loc='best', fontsize=9, frameon=False)
        self.figure.tight_layout()
        self.canvas.draw()

    def correct_intensity(self, field, raw_intens, coeffoverride=None):
        arr = np.asarray(raw_intens, dtype=np.float32)
        idx = np.arange(len(arr))
        coeffs = (coeffoverride if coeffoverride is not None else self.coeffs)
        qo = coeffs.get('quad_offset', 0.0)
        arr_corr = (arr.copy()
            + coeffs['drift'] * (idx - idx.mean())
            + coeffs['linear'] * (field - np.mean(field))
            + coeffs['quad'] * ((field - np.mean(field) - qo)**2)
        )
        if self.normalize:
            arr_corr -= arr_corr.min()
            ptp = arr_corr.ptp()
            if ptp == 0:
                arr_corr[:] = 0
            else:
                arr_corr = (arr_corr / ptp) * 2 - 1
        return arr_corr

    def auto_correct(self):
        parent = self.parent_widget
        if parent is None or parent.loop_field is None:
            return
        field = parent.loop_field
        intensity0 = parent.loop_intens_subtracted if parent.loop_intens_subtracted is not None else parent.loop_intens_txt
        idx = np.arange(len(field))
        idx_off = idx - idx.mean()
        field_off = field - np.mean(field)
        drift1 = (intensity0[0] - intensity0[-1]) / len(intensity0)
        intensity1 = intensity0 + drift1 * idx_off
        high_percent = 0.8
        field_abs = np.abs(field_off)
        field_abs_max = np.max(field_abs)
        fit_mask = field_abs > (high_percent * field_abs_max)
        if np.sum(fit_mask) < 3:
            fit_mask = np.ones_like(field, dtype=bool)
        
        # Fit 2nd order polynomial
        polyfit1 = np.polyfit(field_off[fit_mask], intensity1[fit_mask], 2)
        a, b, c = polyfit1[0], polyfit1[1], polyfit1[2]
        quad1 = -a
        quad_offset_val = -b / (2.0 * a) if a != 0 else 0.0
        
        # Extra refinement step to correct remaining linear slope in saturated regions
        temp_corr = (intensity0 
            + drift1 * idx_off
            + quad1 * ((field_off - quad_offset_val)**2)
        )
        mask_pos = field_off > (high_percent * field_abs_max)
        mask_neg = field_off < (-high_percent * field_abs_max)
        if np.sum(mask_pos) >= 2 and np.sum(mask_neg) >= 2:
            p_pos = np.polyfit(field_off[mask_pos], temp_corr[mask_pos], 1)
            p_neg = np.polyfit(field_off[mask_neg], temp_corr[mask_neg], 1)
            avg_slope = 0.5 * (p_pos[0] + p_neg[0])
            linear_val = -avg_slope
        else:
            linear_val = 0.0
        
        # Automatically apply the normalization
        self.normalize = True
        self.btn_norm.setChecked(True)
        
        # Update sliders & spinboxes (ranges already adjusted to prevent clipping)
        self.set_slider_spinbox('drift', drift1)
        self.set_slider_spinbox('quad', quad1)
        self.set_slider_spinbox('quad_offset', quad_offset_val)
        self.set_slider_spinbox('linear', linear_val)
        
        # Save exact coefficients from spinboxes
        self.coeffs['drift'] = self.spin_drift.value()
        self.coeffs['quad'] = self.spin_quad.value()
        self.coeffs['quad_offset'] = self.spin_quad_offset.value()
        self.coeffs['linear'] = self.spin_linear.value()

        self.hc_hr_marks = None  # clear markers
        
        # Instantly update loop plot and subtracted image display
        if self.parent_widget:
            self.parent_widget.request_loop_update()
            self.parent_widget.show_current_subtracted_image_contrast_only()

    def calc_hc_hr(self):
        parent = self.parent_widget
        if parent is None or parent.loop_field is None:
            return
        field = parent.loop_field
        ycorr = self.correct_intensity(field,
            parent.loop_intens_subtracted if parent.loop_intens_subtracted is not None else parent.loop_intens_txt)
        
        # Split into ascending and descending sweeps using extrema
        i_min = np.argmin(field)
        i_max = np.argmax(field)
        
        if i_min < i_max:
            # Ascending branch runs from i_min to i_max
            asc_idx = list(range(i_min, i_max + 1))
            # Descending branch runs from i_max to end, then wraps to i_min
            desc_idx = list(range(i_max, len(field))) + list(range(0, i_min + 1))
        else:
            # Descending branch runs from i_max to i_min
            desc_idx = list(range(i_max, i_min + 1))
            # Ascending branch runs from i_min to end, then wraps to i_max
            asc_idx = list(range(i_min, len(field))) + list(range(0, i_max + 1))
            
        f_asc = field[asc_idx]
        y_asc = ycorr[asc_idx]
        sort_asc = np.argsort(f_asc)
        f_asc = f_asc[sort_asc]
        y_asc = y_asc[sort_asc]
        
        f_desc = field[desc_idx]
        y_desc = ycorr[desc_idx]
        sort_desc = np.argsort(f_desc)
        f_desc = f_desc[sort_desc]
        y_desc = y_desc[sort_desc]
        
        # Saturation-based midpoint
        field_off = field - np.mean(field)
        field_abs_max = np.max(np.abs(field_off))
        sat_mask_pos = field_off > (0.8 * field_abs_max)
        sat_mask_neg = field_off < (-0.8 * field_abs_max)
        
        sat_pos = np.mean(ycorr[sat_mask_pos]) if np.sum(sat_mask_pos) > 0 else ycorr[i_max]
        sat_neg = np.mean(ycorr[sat_mask_neg]) if np.sum(sat_mask_neg) > 0 else ycorr[i_min]
        mid = 0.5 * (sat_pos + sat_neg)
        
        def find_crossings(f_branch, y_branch, mid_level):
            crossings = []
            for i in range(len(f_branch) - 1):
                y0, y1 = y_branch[i], y_branch[i+1]
                f0, f1 = f_branch[i], f_branch[i+1]
                if (y0 - mid_level) * (y1 - mid_level) <= 0 and y0 != y1:
                    frac = (mid_level - y0) / (y1 - y0)
                    f_cross = f0 + frac * (f1 - f0)
                    crossings.append((f_cross, abs(y1 - y0)))
            return crossings

        crossings_asc = find_crossings(f_asc, y_asc, mid)
        crossings_desc = find_crossings(f_desc, y_desc, mid)
        
        hc_pos = None
        if crossings_asc:
            crossings_asc.sort(key=lambda x: x[1], reverse=True)
            hc_pos = crossings_asc[0][0]
        elif len(f_asc) > 0:
            hc_pos = f_asc[np.argmin(np.abs(y_asc - mid))]
            
        hc_neg = None
        if crossings_desc:
            crossings_desc.sort(key=lambda x: x[1], reverse=True)
            hc_neg = crossings_desc[0][0]
        elif len(f_desc) > 0:
            hc_neg = f_desc[np.argmin(np.abs(y_desc - mid))]
            
        hcpos_coords = (hc_pos, mid) if hc_pos is not None else None
        hcneg_coords = (hc_neg, mid) if hc_neg is not None else None
        
        # Remanence: intensity nearest zero field
        zero_inds = np.where(np.abs(field)==np.min(np.abs(field)))[0]
        hr_fields = field[zero_inds]
        hr_vals = ycorr[zero_inds]
        
        self.hc_hr_marks = (hcpos_coords, hcneg_coords, hr_fields, hr_vals)
        def fmt_or_na(val, fmt=".2f"):
            if val is None:
                return "n/a"
            try:
                return format(val, fmt)
            except Exception:
                return "n/a"
        msg = (
            f"Coercivity Hc+ = {fmt_or_na(hc_pos, '.2f')}, "
            f"Hc- = {fmt_or_na(hc_neg, '.2f')}\nRemanence Hr = {fmt_or_na(np.mean(hr_vals), '.3f')}"
        )
        self.hc_hr_output.setText(msg)
        self.plot_loop(field, ycorr)
        QMessageBox.information(self, "Hc, Hr", msg)

class MOKEImageSubtractor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MOKE Image Subtraction Analysis")
        self.img_dir = None
        self.image_files = []
        self.background_image = None
        self.background_array = None
        self.txt_data = None
        self.current_difference_img = None
        self.current_result_filename = None
        self.current_difference_arr = None
        self.current_difference_arr_raw = None
        self.current_image_idx = None
        self.current_image_file = None
        self.loop_field = None
        self.loop_indices = None
        self.loop_intens_txt = None
        self.loop_intens_subtracted = None
        self.mean_index = None
        self.mean_field = None
        self.init_ui()

    def init_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        widget_left = QWidget()
        left_layout = QVBoxLayout()
        self.btn_dir = QPushButton("Select Hysteresis Image Directory")
        self.btn_dir.clicked.connect(self.choose_directory)
        left_layout.addWidget(self.btn_dir)
        self.list_images = QListWidget()
        self.list_images.currentRowChanged.connect(self.on_image_selected)
        left_layout.addWidget(QLabel("Select Background Image:"))
        left_layout.addWidget(self.list_images)
        self.btn_set_bg = QPushButton("Set as Background")
        self.btn_set_bg.clicked.connect(self.set_background)
        left_layout.addWidget(self.btn_set_bg)
        left_layout.addWidget(QLabel("Browse Images & View Subtraction:"))
        self.list_results = QListWidget()
        self.list_results.currentRowChanged.connect(self.show_subtracted_image)
        left_layout.addWidget(self.list_results)
        self.btn_save = QPushButton("Save This Subtraction Result")
        self.btn_save.clicked.connect(self.save_current_result)
        self.btn_save.setEnabled(False)
        left_layout.addWidget(self.btn_save)
        self.btn_make_loop = QPushButton("Make Loop (Plot Hysteresis)")
        self.btn_make_loop.clicked.connect(self.run_subtraction_loop)
        self.btn_make_loop.setEnabled(False)
        left_layout.addWidget(self.btn_make_loop)
        self.lbl_img = QLabel("Preview will appear here")
        self.lbl_img.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.lbl_img, stretch=1)

        # Contrast slider below image preview

        widget_left.setLayout(left_layout)

        self.loop_panel = LoopCorrectionPanel(parent=self)

        widget_left.setMinimumWidth(320)
        self.loop_panel.setMinimumWidth(320)
        splitter.addWidget(widget_left)
        splitter.addWidget(self.loop_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout = QVBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)

    def request_loop_update(self):
        if self.loop_field is not None and self.loop_intens_txt is not None:
            values = self.loop_intens_subtracted if self.loop_intens_subtracted is not None else self.loop_intens_txt
            self.loop_panel.plot_loop(self.loop_field, self.loop_panel.correct_intensity(self.loop_field, values))

    def choose_directory(self):
        d = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if d:
            self.img_dir = d
            self.update_image_list()
            self.list_results.clear()
            self.lbl_img.setText("Preview will appear here")
            self.background_image = None
            self.background_array = None
            self.txt_data = None
            self.btn_make_loop.setEnabled(False)
            self.loop_field = None
            self.loop_indices = None
            self.loop_intens_txt = None
            self.loop_intens_subtracted = None
            self.mean_index = None
            self.mean_field = None
            self.load_txt_data()

    def update_image_list(self):
        self.image_files = [
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
        ]
        self.image_files.sort()
        self.list_images.clear()
        self.list_images.addItems(self.image_files)
        self.list_results.clear()

    def load_txt_data(self):
        txt_file = None
        for fn in os.listdir(self.img_dir):
            if fn.lower().endswith('.txt'):
                txt_file = os.path.join(self.img_dir, fn)
                break
        if not txt_file:
            self.txt_data = None
            self.btn_make_loop.setEnabled(False)
            self.loop_field = self.loop_indices = self.loop_intens_txt = self.loop_intens_subtracted = None
            self.mean_index = self.mean_field = None
            self.loop_panel.ax.clear()
            self.loop_panel.canvas.draw()
            return
        try:
            df = pd.read_csv(txt_file, sep=None, engine='python', comment="#", skip_blank_lines=True)
            df.columns = [c.strip() for c in df.columns]
            if len(df.columns) < 3:
                QMessageBox.critical(self, "Error", f"Text file {os.path.basename(txt_file)} missing columns.")
                self.txt_data = None
                self.btn_make_loop.setEnabled(False)
                self.loop_field = self.loop_indices = self.loop_intens_txt = self.loop_intens_subtracted = None
                self.mean_index = self.mean_field = None
                self.loop_panel.ax.clear()
                self.loop_panel.canvas.draw()
                return
            df = df[df[df.columns[2]].str.lower().str.endswith(".png", na=False)]
            self.txt_data = df.rename(
                columns={df.columns[0]:"Field", df.columns[1]:"Intensity", df.columns[2]:"File"}
            ).reset_index(drop=True)
            self.loop_field = self.txt_data["Field"].to_numpy(dtype=np.float32)
            self.loop_indices = np.arange(len(self.txt_data))
            self.loop_intens_txt = self.txt_data["Intensity"].to_numpy(dtype=np.float32)
            self.loop_intens_subtracted = None
            self.mean_index = self.loop_indices.mean()
            self.mean_field = self.loop_field.mean()
            
            # Dynamically adjust correction ranges
            ptp_val = np.ptp(self.loop_intens_txt)
            self.loop_panel.update_correction_ranges(ptp_val)
            
            self.btn_make_loop.setEnabled(True)
            self.request_loop_update()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read .txt data file.\n{e}")
            self.txt_data = None
            self.loop_field = self.loop_indices = self.loop_intens_txt = self.loop_intens_subtracted = None
            self.mean_index = self.mean_field = None
            self.btn_make_loop.setEnabled(False)
            self.loop_panel.ax.clear()
            self.loop_panel.canvas.draw()

    def get_intensity_correction(self, idx_in_txt):
        a = self.loop_panel.coeffs['drift']
        b = self.loop_panel.coeffs['linear']
        c = self.loop_panel.coeffs['quad']
        index_offset = idx_in_txt - self.mean_index
        field_offset = self.loop_field[idx_in_txt] - self.mean_field
        return a * index_offset + b * field_offset + c * (field_offset ** 2)

    def on_image_selected(self, idx):
        if idx < 0 or idx >= len(self.image_files):
            return
        filepath = os.path.join(self.img_dir, self.image_files[idx])
        self.display_image(filepath)

    def img_contrast_changed(self, val):
        contrast_val = val / 100.0
        self.loop_panel.contrast = contrast_val
        self.loop_panel.sld_contrast.blockSignals(True)
        self.loop_panel.spin_contrast.blockSignals(True)
        self.loop_panel.sld_contrast.setValue(val)
        self.loop_panel.spin_contrast.setValue(contrast_val)
        self.loop_panel.sld_contrast.blockSignals(False)
        self.loop_panel.spin_contrast.blockSignals(False)
        self.show_current_subtracted_image_contrast_only()

    def display_image(self, filepath):
        img = Image.open(filepath)
        arr = np.array(img)
        arr = crop600(arr)
        if arr.ndim == 3 and arr.shape[2] in [3, 4]:
            if arr.shape[2] == 4:
                data = arr.tobytes()
                h, w, c = arr.shape
                qimg = QImage(data, w, h, QImage.Format_RGBA8888)
            else:
                data = arr.tobytes()
                h, w, c = arr.shape
                qimg = QImage(data, w, h, QImage.Format_RGB888)
        else:
            arr_disp = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
            h, w = arr_disp.shape
            data = arr_disp.tobytes()
            qimg = QImage(data, w, h, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        self.lbl_img.setPixmap(pix)

    def set_background(self):
        idx = self.list_images.currentRow()
        if idx < 0:
            QMessageBox.warning(self, "No selection", "Select a background image first.")
            return
        bg_file = self.image_files[idx]
        filepath = os.path.join(self.img_dir, bg_file)
        self.background_image = bg_file
        self.background_array = np.array(Image.open(filepath))
        QMessageBox.information(self, "Background Set", f"Background set to: {bg_file}")
        self.list_results.clear()
        self.list_results.addItems(self.image_files)
        self.btn_save.setEnabled(True)

    def show_subtracted_image(self, idx):
        if self.background_array is None or idx < 0 or idx >= len(self.image_files):
            return
        img_file = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        try:
            img_arr = np.array(Image.open(img_path))
            bg_arr = self.background_array.copy()
            
            # Crop target and background first to prevent metadata ringing & optimize speed
            img_arr = crop600(img_arr)
            bg_arr = crop600(bg_arr)
            
            # Apply focus drift correction if enabled
            if self.loop_panel.chk_z_drift.isChecked():
                coeff = self.loop_panel.spin_z_quad.value() * 1e-6
                field = 0.0
                if self.txt_data is not None:
                    match = self.txt_data[self.txt_data['File'] == img_file]
                    if not match.empty:
                        field = match.iloc[0]['Field']
                
                sigma = coeff * (field ** 2)
                if sigma > 0.05:
                    method_idx = self.loop_panel.cmb_z_method.currentIndex()
                    if method_idx == 0:
                        # Blur Reference
                        if bg_arr.ndim == 3:
                            for c in range(bg_arr.shape[2]):
                                bg_arr[:, :, c] = ndimage.gaussian_filter(bg_arr[:, :, c].astype(np.float64), sigma=sigma)
                        else:
                            bg_arr = ndimage.gaussian_filter(bg_arr.astype(np.float64), sigma=sigma)
                    else:
                        # Deblur Target
                        max_val = np.iinfo(img_arr.dtype).max if np.issubdtype(img_arr.dtype, np.integer) else 255
                        orig_dtype = img_arr.dtype
                        if img_arr.ndim == 3:
                            img_deblurred = np.zeros_like(img_arr, dtype=np.float64)
                            for c in range(img_arr.shape[2]):
                                img_deblurred[:, :, c] = wiener_deconvolve(img_arr[:, :, c].astype(np.float64), sigma=sigma)
                            img_arr = np.clip(img_deblurred, 0, max_val).astype(orig_dtype)
                        else:
                            img_deblurred = wiener_deconvolve(img_arr.astype(np.float64), sigma=sigma)
                            img_arr = np.clip(img_deblurred, 0, max_val).astype(orig_dtype)

            min_shape = tuple(min(sa, sb) for sa, sb in zip(img_arr.shape, bg_arr.shape))
            if img_arr.ndim == 3:
                img_c = img_arr[:min_shape[0], :min_shape[1], :min_shape[2]]
                bg_c = bg_arr[:min_shape[0], :min_shape[1], :min_shape[2]]
            else:
                img_c = img_arr[:min_shape[0], :min_shape[1]]
                bg_c = bg_arr[:min_shape[0], :min_shape[1]]
            arr = img_c.astype(np.float32) - bg_c.astype(np.float32)
            arr_cropped = crop600(arr)
            
            # Save raw cropped subtraction and selection info
            self.current_difference_arr_raw = arr_cropped.copy()
            self.current_image_idx = idx
            self.current_image_file = img_file
            
            self.show_current_subtracted_image_contrast_only()
            self.current_result_filename = f"{os.path.splitext(img_file)[0]}_contrast.png"
        except Exception as e:
            self.lbl_img.setText(f"Error: {e}")
            self.current_difference_img = None
            self.current_difference_arr = None
            self.current_difference_arr_raw = None

    def show_current_subtracted_image_contrast_only(self):
        if not hasattr(self, 'current_difference_arr_raw') or self.current_difference_arr_raw is None:
            if self.current_difference_arr is None:
                return
            arr_disp = normalized_for_display(self.current_difference_arr, contrast=self.loop_panel.contrast)
            show_img = Image.fromarray(arr_disp)
            h, w = arr_disp.shape
            data = arr_disp.tobytes()
            qimg = QImage(data, w, h, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg)
            self.lbl_img.setPixmap(pix)
            self.current_difference_img = show_img
            return
            
        # Get correction dynamically
        correction = 0.0
        index_in_txt = None
        if self.txt_data is not None:
            match = self.txt_data[self.txt_data['File'] == self.current_image_file]
            if not match.empty:
                index_in_txt = match.index[0]
            elif len(self.txt_data) > self.current_image_idx:
                index_in_txt = self.current_image_idx
        if index_in_txt is not None and self.mean_index is not None and self.mean_field is not None:
            correction = self.get_intensity_correction(index_in_txt)
            
        # Apply correction to the raw array
        self.current_difference_arr = self.current_difference_arr_raw + correction
        
        # Display with contrast stretch
        arr_disp = normalized_for_display(self.current_difference_arr, contrast=self.loop_panel.contrast)
        show_img = Image.fromarray(arr_disp)
        h, w = arr_disp.shape
        data = arr_disp.tobytes()
        qimg = QImage(data, w, h, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        self.lbl_img.setPixmap(pix)
        self.current_difference_img = show_img

    def save_current_result(self):
        if self.current_difference_img is not None:
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Contrast Image", self.current_result_filename, "PNG Files (*.png)")
            if save_path:
                arr = self.current_difference_arr
                if arr is None:
                    arr = np.array(self.current_difference_img)
                if arr.min() < 0 or arr.max() >= 256:
                    arr16 = arr - arr.min()
                    arr16 = arr16 / arr16.max() * 65535
                    arr16 = arr16.astype(np.uint16)
                    Image.fromarray(arr16).save(save_path)
                else:
                    arr8 = np.clip(arr, 0, 255).astype(np.uint8)
                    Image.fromarray(arr8).save(save_path)
                QMessageBox.information(self, "Saved", f"Contrast image saved to:\n{save_path}")
        else:
            QMessageBox.warning(self, "Nothing to save", "No subtracted image to save.")

    def run_subtraction_loop(self):
        if self.background_array is None:
            QMessageBox.warning(self, "Background not set", "Set a background image first.")
            return
        if self.txt_data is None:
            QMessageBox.warning(self, "No Data File", "No valid data file loaded.")
            return
        means = []
        
        # Pre-cache coefficients for focus correction
        enable_z = self.loop_panel.chk_z_drift.isChecked()
        coeff = self.loop_panel.spin_z_quad.value() * 1e-6
        method_idx = self.loop_panel.cmb_z_method.currentIndex()
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for idx, row in self.txt_data.iterrows():
                img_file = row['File'].strip()
                img_path = os.path.join(self.img_dir, img_file)
                if not os.path.exists(img_path):
                    means.append(np.nan)
                    continue
                try:
                    img_arr = np.array(Image.open(img_path))
                    bg_arr = self.background_array.copy()
                    
                    # Crop target and background first to prevent metadata ringing & optimize speed
                    img_arr = crop600(img_arr)
                    bg_arr = crop600(bg_arr)
                    
                    if enable_z:
                        field = row['Field']
                        sigma = coeff * (field ** 2)
                        if sigma > 0.05:
                            if method_idx == 0:
                                # Blur Reference
                                if bg_arr.ndim == 3:
                                    for c in range(bg_arr.shape[2]):
                                        bg_arr[:, :, c] = ndimage.gaussian_filter(bg_arr[:, :, c].astype(np.float64), sigma=sigma)
                                else:
                                    bg_arr = ndimage.gaussian_filter(bg_arr.astype(np.float64), sigma=sigma)
                            else:
                                # Deblur Target
                                max_val = np.iinfo(img_arr.dtype).max if np.issubdtype(img_arr.dtype, np.integer) else 255
                                orig_dtype = img_arr.dtype
                                if img_arr.ndim == 3:
                                    img_deblurred = np.zeros_like(img_arr, dtype=np.float64)
                                    for c in range(img_arr.shape[2]):
                                        img_deblurred[:, :, c] = wiener_deconvolve(img_arr[:, :, c].astype(np.float64), sigma=sigma)
                                    img_arr = np.clip(img_deblurred, 0, max_val).astype(orig_dtype)
                                else:
                                    img_deblurred = wiener_deconvolve(img_arr.astype(np.float64), sigma=sigma)
                                    img_arr = np.clip(img_deblurred, 0, max_val).astype(orig_dtype)

                    min_shape = tuple(min(sa, sb) for sa, sb in zip(img_arr.shape, bg_arr.shape))
                    if img_arr.ndim == 3:
                        img_c = img_arr[:min_shape[0], :min_shape[1], :min_shape[2]]
                        bg_c = bg_arr[:min_shape[0], :min_shape[1], :min_shape[2]]
                    else:
                        img_c = img_arr[:min_shape[0], :min_shape[1]]
                        bg_c = bg_arr[:min_shape[0], :min_shape[1]]
                    arr = img_c.astype(np.float32) - bg_c.astype(np.float32)
                    arr_cropped = crop600(arr)
                    
                    # Store raw subtraction (do not bake the correction in)
                    mean_val = np.mean(arr_cropped)
                    means.append(mean_val)
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    means.append(np.nan)
            self.loop_intens_subtracted = np.array(means, dtype=np.float32)
            
            # Dynamically adjust correction ranges based on subtracted intensity
            ptp_val = np.ptp(self.loop_intens_subtracted)
            self.loop_panel.update_correction_ranges(ptp_val)
            
            self.request_loop_update()
        finally:
            QApplication.restoreOverrideCursor()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MOKEImageSubtractor()
    window.show()
    sys.exit(app.exec_())
