# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:36:05 2026

@author: robhu413
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QFileDialog, QVBoxLayout, QPushButton,
    QListWidget, QLabel, QMessageBox, QHBoxLayout, QSlider,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSplitter, QTextEdit,
    QCheckBox, QComboBox, QSpinBox, QLineEdit, QSizePolicy, QScrollArea,
    QSplitterHandle
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal, QSize, QRectF, QPointF
from PIL import Image
import pandas as pd
import scipy.ndimage as ndimage
from shared_utils.image_processing import crop600, wiener_deconvolve, get_roi_mean, compute_subtracted_mean
from scipy.optimize import minimize_scalar
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class StyledSplitterHandle(QSplitterHandle):
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        
        # Try to find theme from parent widgets
        theme = "dark"
        p = self.splitter()
        while p:
            if hasattr(p, "theme"):
                theme = p.theme
                break
            p = p.parent()
            
        from gui_styles import get_theme_colors
        colors = get_theme_colors(theme)
        bg_color = QColor(colors["card"])
        border_color = QColor(colors["border"])
        grip_color = QColor(colors["accent"])
            
        # Fill background
        painter.fillRect(self.rect(), bg_color)
        
        w, h = self.width(), self.height()
        
        # Draw border lines
        pen_border = QPen(border_color, 1, Qt.SolidLine)
        painter.setPen(pen_border)
        if self.orientation() == Qt.Horizontal:
            painter.drawLine(0, 0, 0, h)
            painter.drawLine(w - 1, 0, w - 1, h)
        else:
            painter.drawLine(0, 0, w, 0)
            painter.drawLine(0, h - 1, w, h - 1)
            
        # Draw 3 neat grip indicators in the center
        pen_grip = QPen(grip_color, 2, Qt.SolidLine)
        painter.setPen(pen_grip)
        
        if self.orientation() == Qt.Horizontal:
            cx = w // 2
            cy = h // 2
            painter.drawLine(cx - 2, cy - 8, cx + 2, cy - 8)
            painter.drawLine(cx - 2, cy, cx + 2, cy)
            painter.drawLine(cx - 2, cy + 8, cx + 2, cy + 8)
        else:
            cx = w // 2
            cy = h // 2
            painter.drawLine(cx - 8, cy - 2, cx - 8, cy + 2)
            painter.drawLine(cx, cy - 2, cx, cy + 2)
            painter.drawLine(cx + 8, cy - 2, cx + 8, cy + 2)


class StyledSplitter(QSplitter):
    def createHandle(self):
        return StyledSplitterHandle(self.orientation(), self)


def normalized_for_display(arr, scale=None, contrast=0.5):
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
    
    step = 10
    margin = 20

    if h <= 2 * margin + patch_size or w <= 2 * margin + patch_size:
        return (150, 150) # default fallback center area

    r_coords = np.arange(margin, h - patch_size - margin, step)
    c_coords = np.arange(margin, w - patch_size - margin, step)

    if r_coords.size == 0 or c_coords.size == 0:
        return (150, 150)

    means = ndimage.uniform_filter(grad, size=patch_size, mode='constant')

    R, C = np.meshgrid(r_coords, c_coords, indexing='ij')
    r_centers = R + patch_size // 2
    c_centers = C + patch_size // 2

    scores = means[r_centers, c_centers]

    if scores.size == 0:
        return (150, 150)

    max_idx = np.argmax(scores)
    max_r_idx, max_c_idx = np.unravel_index(max_idx, scores.shape)

    return (int(r_coords[max_r_idx]), int(c_coords[max_c_idx]))

def estimate_defocus(ref_img_norm, target_img_norm):
    target_grad = get_gradient_magnitude(target_img_norm)
    def loss(sigma):
        if sigma <= 0.01:
            blurred_ref = ref_img_norm
        else:
            blurred_ref = ndimage.gaussian_filter(ref_img_norm, sigma=sigma)
        blurred_grad = get_gradient_magnitude(blurred_ref)
        return np.mean((target_grad - blurred_grad) ** 2)
    
    res = minimize_scalar(loss, bounds=(0.0, 3.0), method='bounded')
    return res.x

class ROISelectLabel(QLabel):
    roi_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.roi_shape = "None"
        self.roi_data = None  # (cx, cy, w, h, angle) or (cx, cy, r)
        self.is_dragging = False
        self.start_pos = None
        self.current_pos = None
        self.drag_mode = None  # None, "draw", "move", "rotate"
        self.initial_roi_data = None
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def minimumSizeHint(self):
        return QSize(100, 100)
        
    def set_roi_shape(self, shape):
        self.roi_shape = shape
        self.roi_data = None
        self.update()
        
    def clear_roi(self):
        self.roi_data = None
        self.update()
        self.roi_changed.emit()

    def label_to_image_coords(self, pos):
        if not self.pixmap() or self.pixmap().isNull():
            return None
        
        lw, lh = self.width(), self.height()
        pw, ph = self.pixmap().width(), self.pixmap().height()
        
        s = min(lw / pw, lh / ph)
        dw = int(pw * s)
        dh = int(ph * s)
        x_offset = (lw - dw) // 2
        y_offset = (lh - dh) // 2
        
        if s <= 0:
            return None
            
        ix = (pos.x() - x_offset) / s
        iy = (pos.y() - y_offset) / s
        
        ix = max(0.0, min(float(pw - 1), ix))
        iy = max(0.0, min(float(ph - 1), iy))
        
        return ix, iy

    def get_rotation_handle_pos(self):
        if not self.pixmap() or self.pixmap().isNull() or self.roi_shape not in ["Rectangle", "Square"] or self.roi_data is None:
            return None
            
        cx, cy, w, h, angle = self.roi_data
        
        lw, lh = self.width(), self.height()
        pw, ph = self.pixmap().width(), self.pixmap().height()
        
        s = min(lw / pw, lh / ph)
        dw = int(pw * s)
        dh = int(ph * s)
        x_offset = (lw - dw) // 2
        y_offset = (lh - dh) // 2
        
        theta = np.radians(angle)
        dist_screen = (h * s) / 2.0 + 20.0
        
        scx = cx * s + x_offset
        scy = cy * s + y_offset
        
        hx = scx + dist_screen * np.sin(theta)
        hy = scy - dist_screen * np.cos(theta)
        return hx, hy

    def is_point_inside_roi(self, ix, iy):
        if self.roi_shape == "None" or self.roi_data is None:
            return False
            
        if self.roi_shape in ["Rectangle", "Square"]:
            cx, cy, w, h, angle = self.roi_data
            theta = np.radians(angle)
            cos_val = np.cos(theta)
            sin_val = np.sin(theta)
            
            dx = ix - cx
            dy = iy - cy
            
            x_local = dx * cos_val + dy * sin_val
            y_local = -dx * sin_val + dy * cos_val
            
            return (abs(x_local) <= w / 2.0) and (abs(y_local) <= h / 2.0)
            
        elif self.roi_shape == "Circle":
            cx, cy, r = self.roi_data
            dist_sq = (ix - cx)**2 + (iy - cy)**2
            return dist_sq <= r**2
            
        return False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.roi_shape != "None":
            coords = self.label_to_image_coords(event.pos())
            if not coords:
                super().mousePressEvent(event)
                return
                
            ix, iy = coords
            
            # Check rotation handle first
            handle_pos = self.get_rotation_handle_pos()
            if handle_pos:
                hx, hy = handle_pos
                click_x, click_y = event.pos().x(), event.pos().y()
                dist = np.sqrt((click_x - hx)**2 + (click_y - hy)**2)
                if dist < 12.0:
                    self.drag_mode = "rotate"
                    self.is_dragging = True
                    self.start_pos = coords
                    self.current_pos = coords
                    self.update()
                    return
            
            # Check inside ROI
            if self.is_point_inside_roi(ix, iy):
                self.drag_mode = "move"
                self.is_dragging = True
                self.start_pos = coords
                self.current_pos = coords
                self.initial_roi_data = self.roi_data
                self.update()
                return
                
            # Default to draw mode
            self.drag_mode = "draw"
            self.start_pos = coords
            self.current_pos = coords
            self.is_dragging = True
            self.update_roi_from_drag()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            coords = self.label_to_image_coords(event.pos())
            if coords:
                self.current_pos = coords
                if self.drag_mode == "draw":
                    self.update_roi_from_drag()
                elif self.drag_mode == "move":
                    self.update_roi_position()
                elif self.drag_mode == "rotate":
                    self.update_roi_rotation()
                self.roi_changed.emit()
        else:
            coords = self.label_to_image_coords(event.pos())
            if coords and self.roi_shape != "None" and self.roi_data is not None:
                ix, iy = coords
                handle_pos = self.get_rotation_handle_pos()
                is_near_handle = False
                if handle_pos:
                    hx, hy = handle_pos
                    dist = np.sqrt((event.pos().x() - hx)**2 + (event.pos().y() - hy)**2)
                    if dist < 12.0:
                        is_near_handle = True
                        
                if is_near_handle:
                    self.setCursor(Qt.PointingHandCursor)
                elif self.is_point_inside_roi(ix, iy):
                    self.setCursor(Qt.SizeAllCursor)
                else:
                    self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
                
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_dragging:
            coords = self.label_to_image_coords(event.pos())
            if coords:
                self.current_pos = coords
            self.is_dragging = False
            
            if self.drag_mode == "draw":
                self.update_roi_from_drag()
            elif self.drag_mode == "move":
                self.update_roi_position()
            elif self.drag_mode == "rotate":
                self.update_roi_rotation()
                
            self.drag_mode = None
            self.initial_roi_data = None
            self.roi_changed.emit()
        else:
            super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)

    def update_roi_from_drag(self):
        if not self.pixmap() or self.pixmap().isNull() or not self.start_pos or not self.current_pos:
            return
            
        pw, ph = self.pixmap().width(), self.pixmap().height()
        
        if self.roi_shape == "Rectangle":
            w = abs(self.start_pos[0] - self.current_pos[0])
            h = abs(self.start_pos[1] - self.current_pos[1])
            cx = (self.start_pos[0] + self.current_pos[0]) / 2.0
            cy = (self.start_pos[1] + self.current_pos[1]) / 2.0
            self.roi_data = (cx, cy, w, h, 0.0)
            
        elif self.roi_shape == "Square":
            dx = self.current_pos[0] - self.start_pos[0]
            dy = self.current_pos[1] - self.start_pos[1]
            side = max(abs(dx), abs(dy))
            sx = 1 if dx >= 0 else -1
            sy = 1 if dy >= 0 else -1
            
            if sx > 0:
                side = min(side, pw - 1 - self.start_pos[0])
            else:
                side = min(side, self.start_pos[0])
            if sy > 0:
                side = min(side, ph - 1 - self.start_pos[1])
            else:
                side = min(side, self.start_pos[1])
                
            x = self.start_pos[0] if sx > 0 else self.start_pos[0] - side
            y = self.start_pos[1] if sy > 0 else self.start_pos[1] - side
            cx = x + side / 2.0
            cy = y + side / 2.0
            self.roi_data = (cx, cy, side, side, 0.0)
            
        elif self.roi_shape == "Circle":
            cx, cy = self.start_pos
            r = int(np.sqrt((self.current_pos[0] - self.start_pos[0])**2 + (self.current_pos[1] - self.start_pos[1])**2))
            r = min(r, cx, pw - 1 - cx, cy, ph - 1 - cy)
            self.roi_data = (cx, cy, r)
            
        self.update()

    def update_roi_position(self):
        if not self.initial_roi_data or not self.start_pos or not self.current_pos:
            return
            
        dx = self.current_pos[0] - self.start_pos[0]
        dy = self.current_pos[1] - self.start_pos[1]
        
        pw, ph = self.pixmap().width(), self.pixmap().height()
        
        if self.roi_shape in ["Rectangle", "Square"]:
            init_cx, init_cy, w, h, angle = self.initial_roi_data
            new_cx = init_cx + dx
            new_cy = init_cy + dy
            
            new_cx = max(0, min(pw - 1, new_cx))
            new_cy = max(0, min(ph - 1, new_cy))
            self.roi_data = (new_cx, new_cy, w, h, angle)
            
        elif self.roi_shape == "Circle":
            init_cx, init_cy, r = self.initial_roi_data
            new_cx = init_cx + dx
            new_cy = init_cy + dy
            
            new_cx = max(r, min(pw - 1 - r, new_cx))
            new_cy = max(r, min(ph - 1 - r, new_cy))
            self.roi_data = (new_cx, new_cy, r)
            
        self.update()

    def update_roi_rotation(self):
        if not self.roi_data or not self.current_pos:
            return
            
        cx, cy, w, h, _ = self.roi_data
        mx, my = self.current_pos
        angle_rad = np.arctan2(my - cy, mx - cx)
        angle_deg = np.degrees(angle_rad) + 90.0
        angle_deg = (angle_deg + 180.0) % 360.0 - 180.0
        
        self.roi_data = (cx, cy, w, h, angle_deg)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Clear background (QLabel's default background)
        painter.fillRect(self.rect(), self.palette().window().color())
        
        if not self.pixmap() or self.pixmap().isNull():
            return
            
        lw, lh = self.width(), self.height()
        pw, ph = self.pixmap().width(), self.pixmap().height()
        
        s = min(lw / pw, lh / ph)
        dw = int(pw * s)
        dh = int(ph * s)
        x_offset = (lw - dw) // 2
        y_offset = (lh - dh) // 2
        
        if s > 0:
            painter.drawPixmap(x_offset, y_offset, dw, dh, self.pixmap())
        
        if self.roi_shape == "None" or self.roi_data is None:
            return
            
        pen = QPen(QColor(50, 205, 50), 2, Qt.SolidLine)
        brush = QBrush(QColor(50, 205, 50, 45))
        painter.setPen(pen)
        painter.setBrush(brush)
        
        if self.roi_shape in ["Rectangle", "Square"]:
            cx, cy, w, h, angle = self.roi_data
            painter.save()
            painter.translate(cx * s + x_offset, cy * s + y_offset)
            painter.rotate(angle)
            
            sw = w * s
            sh = h * s
            
            rect = QRectF(-sw / 2.0, -sh / 2.0, sw, sh)
            painter.drawRect(rect)
            
            pen_line = QPen(QColor(50, 205, 50), 1, Qt.DashLine)
            painter.setPen(pen_line)
            handle_dist = 20.0
            painter.drawLine(QPointF(0.0, -sh / 2.0), QPointF(0.0, -sh / 2.0 - handle_dist))
            
            pen_handle = QPen(QColor(50, 205, 50), 2, Qt.SolidLine)
            brush_handle = QBrush(QColor(0, 255, 0))
            painter.setPen(pen_handle)
            painter.setBrush(brush_handle)
            painter.drawEllipse(QPointF(0.0, -sh / 2.0 - handle_dist), 5, 5)
            
            painter.restore()
        elif self.roi_shape == "Circle":
            cx, cy, r = self.roi_data
            painter.drawEllipse(QPointF(cx * s + x_offset, cy * s + y_offset), r * s, r * s)


class LoopCorrectionPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_saving = False
        self.coeffs = dict(drift=0.0, linear=0.0, quad=0.0, quad_offset=0.0)
        self.z_coeff = 0.0
        self.normalize = False
        self.contrast = 1.0
        self.parent_widget = parent
        self.hc_hr_marks = None  # Stores latest Hc/Hr marks for highlighting
        self.figure = None
        self.ax = None
        self.canvas = None
        self.toolbar = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        splitter = StyledSplitter(Qt.Vertical)
        splitter.setHandleWidth(8)

        # Top widget: Plot canvas container
        plot_container = QWidget()
        self.plot_layout = QVBoxLayout()
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        
        theme = "dark"
        if self.parent_widget and hasattr(self.parent_widget, 'theme'):
            theme = self.parent_widget.theme
        from gui_styles import get_theme_colors
        colors = get_theme_colors(theme)
        ax_bg = colors["bg"]
        text_color = colors["text"]
        
        # Placeholder label
        self.placeholder_label = QLabel("No data loaded. Select a directory and click 'Make Loop' to plot.")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setObjectName("PlotPlaceholder")
        self.placeholder_label.setStyleSheet(
            f"background-color: {ax_bg}; color: {text_color}; border-radius: 6px; "
            f"font-size: 14px; font-weight: 500; border: 1px solid {colors['border']};"
        )
        self.placeholder_label.setWordWrap(True)
        self.placeholder_label.setContentsMargins(20, 20, 20, 20)
        
        self.plot_layout.addWidget(self.placeholder_label)
        plot_container.setLayout(self.plot_layout)
        splitter.addWidget(plot_container)

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
        self.sld_quad.setMinimum(-20)
        self.sld_quad.setMaximum(20)
        self.sld_quad.setValue(0)
        self.sld_quad.valueChanged.connect(self.slider_changed)
        self.spin_quad = QDoubleSpinBox()
        self.spin_quad.setDecimals(6)
        self.spin_quad.setRange(-0.001, 0.001)
        self.spin_quad.setSingleStep(0.00001)
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
        
        self.btn_save_loop = QPushButton("Save Loop")
        self.btn_save_loop.clicked.connect(self.save_loop)
        hbox_auto.addWidget(self.btn_save_loop)
        
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

        # Plot Style Settings GroupBox
        group_plot = QGroupBox("Plot Settings")
        plot_settings_layout = QFormLayout()
        
        self.color_map = {
            "Dark Navy": "#1F4E79",
            "Slate Blue": "#2E4057",
            "Crimson": "#C62828",
            "Teal": "#00796B",
            "Charcoal": "#263238",
            "Purple": "#6A1B9A"
        }
        self.line_style_map = {
            "Solid": "-",
            "Dashed": "--",
            "Dotted": ":",
            "Dash-dot": "-.",
            "None": "None"
        }
        self.marker_style_map = {
            "Circle": "o",
            "Square": "s",
            "Diamond": "d",
            "Triangle": "^",
            "None": "None"
        }

        # Row 1: Grid
        self.chk_grid = QCheckBox("Show Grid")
        self.chk_grid.setChecked(True)
        self.chk_grid.stateChanged.connect(self.replot_current_data)
        
        self.cmb_grid_style = QComboBox()
        self.cmb_grid_style.addItems(["Dashed", "Solid", "Dotted", "Dash-dot"])
        self.cmb_grid_style.currentIndexChanged.connect(self.replot_current_data)
        
        hbox_grid = QHBoxLayout()
        hbox_grid.addWidget(self.chk_grid)
        hbox_grid.addWidget(QLabel("Style:"))
        hbox_grid.addWidget(self.cmb_grid_style)
        plot_settings_layout.addRow("Grid", hbox_grid)

        # Row 2: Line/Marker Format
        self.cmb_line_color = QComboBox()
        self.cmb_line_color.addItems(list(self.color_map.keys()))
        self.cmb_line_color.currentIndexChanged.connect(self.replot_current_data)

        self.cmb_line_style = QComboBox()
        self.cmb_line_style.addItems(list(self.line_style_map.keys()))
        self.cmb_line_style.currentIndexChanged.connect(self.replot_current_data)

        self.cmb_marker_style = QComboBox()
        self.cmb_marker_style.addItems(list(self.marker_style_map.keys()))
        self.cmb_marker_style.currentIndexChanged.connect(self.replot_current_data)

        hbox_format = QHBoxLayout()
        hbox_format.addWidget(QLabel("Color:"))
        hbox_format.addWidget(self.cmb_line_color)
        hbox_format.addWidget(QLabel("Line:"))
        hbox_format.addWidget(self.cmb_line_style)
        hbox_format.addWidget(QLabel("Marker:"))
        hbox_format.addWidget(self.cmb_marker_style)
        plot_settings_layout.addRow("Format", hbox_format)

        # Row 3: Title
        self.chk_auto_title = QCheckBox("Auto")
        self.chk_auto_title.setChecked(True)
        self.chk_auto_title.stateChanged.connect(self.toggle_auto_title)

        self.txt_title = QLineEdit()
        self.txt_title.setText("Hysteresis Loop")
        self.txt_title.setEnabled(False)
        self.txt_title.textChanged.connect(self.replot_current_data)

        hbox_title = QHBoxLayout()
        hbox_title.addWidget(self.chk_auto_title)
        hbox_title.addWidget(self.txt_title)
        plot_settings_layout.addRow("Title", hbox_title)

        # Row 4: Labels
        self.chk_auto_labels = QCheckBox("Auto")
        self.chk_auto_labels.setChecked(True)
        self.chk_auto_labels.stateChanged.connect(self.toggle_auto_labels)

        self.txt_xlabel = QLineEdit()
        self.txt_xlabel.setText("Field (mT)")
        self.txt_xlabel.setEnabled(False)
        self.txt_xlabel.textChanged.connect(self.replot_current_data)

        self.txt_ylabel = QLineEdit()
        self.txt_ylabel.setText("MOKE Intensity")
        self.txt_ylabel.setEnabled(False)
        self.txt_ylabel.textChanged.connect(self.replot_current_data)

        hbox_labels = QHBoxLayout()
        hbox_labels.addWidget(self.chk_auto_labels)
        hbox_labels.addWidget(QLabel("X:"))
        hbox_labels.addWidget(self.txt_xlabel)
        hbox_labels.addWidget(QLabel("Y:"))
        hbox_labels.addWidget(self.txt_ylabel)
        plot_settings_layout.addRow("Labels", hbox_labels)

        # Row 5: Visibility checkboxes
        self.chk_highlights = QCheckBox("Show Highlights")
        self.chk_highlights.setChecked(True)
        self.chk_highlights.stateChanged.connect(self.replot_current_data)

        self.chk_legend = QCheckBox("Show Legend")
        self.chk_legend.setChecked(False)
        self.chk_legend.stateChanged.connect(self.replot_current_data)

        self.chk_pub_ticks = QCheckBox("Pub Ticks")
        self.chk_pub_ticks.setChecked(False)
        self.chk_pub_ticks.stateChanged.connect(self.replot_current_data)

        hbox_checks = QHBoxLayout()
        hbox_checks.addWidget(self.chk_highlights)
        hbox_checks.addWidget(self.chk_legend)
        hbox_checks.addWidget(self.chk_pub_ticks)
        plot_settings_layout.addRow(hbox_checks)

        group_plot.setLayout(plot_settings_layout)
        controls_layout.addWidget(group_plot)

        self.hc_hr_output = QTextEdit()
        self.hc_hr_output.setReadOnly(True)
        self.hc_hr_output.setMinimumHeight(20)
        controls_layout.addWidget(self.hc_hr_output)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        scroll_area.setWidget(controls_widget)

        splitter.addWidget(scroll_area)
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


    def clear_plot(self):
        if self.canvas is not None:
            self.ax.clear()
            self.canvas.draw()

    def replot_current_data(self, *args):
        if self.canvas is None:
            return
        if self.parent_widget:
            self.parent_widget.request_loop_update()

    def toggle_auto_title(self, state):
        self.txt_title.setEnabled(not self.chk_auto_title.isChecked())
        self.replot_current_data()

    def toggle_auto_labels(self, state):
        enabled = not self.chk_auto_labels.isChecked()
        self.txt_xlabel.setEnabled(enabled)
        self.txt_ylabel.setEnabled(enabled)
        self.replot_current_data()

    def on_canvas_resize(self, event):
        if self.canvas is None:
            return
        self.canvas.original_resizeEvent(event)
        if self.parent_widget and self.parent_widget.loop_field is not None:
            # Save limits to be restored inside plot_loop
            self._resize_limits = (self.ax.get_xlim(), self.ax.get_ylim())
            self.replot_current_data()
        else:
            self.replot_current_data()

    def plot_loop(self, field, intensity, show_title="Hysteresis Loop"):
        if self.canvas is None:
            theme = "dark"
            if self.parent_widget and hasattr(self.parent_widget, 'theme'):
                theme = self.parent_widget.theme
            from gui_styles import get_theme_colors
            colors = get_theme_colors(theme)
            fig_bg = colors["card"]
            ax_bg = colors["bg"]
            
            self.figure = Figure(figsize=(4, 6), facecolor=fig_bg)
            self.ax = self.figure.add_subplot(111)
            self.ax.set_facecolor(ax_bg)
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setMinimumHeight(100)
            
            # Override the resize event to scale plot elements proportionally
            self.canvas.original_resizeEvent = self.canvas.resizeEvent
            self.canvas.resizeEvent = self.on_canvas_resize
            
            self.toolbar = NavigationToolbar(self.canvas, self)
            
            if hasattr(self, 'placeholder_label') and self.placeholder_label is not None:
                self.plot_layout.removeWidget(self.placeholder_label)
                self.placeholder_label.deleteLater()
                self.placeholder_label = None
                
            self.plot_layout.addWidget(self.canvas)
            self.plot_layout.addWidget(self.toolbar)

        self.ax.clear()
        
        # Calculate dynamic scale factor based on canvas widget size relative to baseline (500x400)
        # Baseline diagonal is sqrt(500^2 + 400^2) = ~640.31 pixels
        if getattr(self, 'is_saving', False):
            scale = 1.0
        else:
            w = max(50, self.canvas.width())
            h = max(50, self.canvas.height())
            diag = np.sqrt(w**2 + h**2)
            scale = max(0.5, min(4.0, diag / 640.3))
        
        # Read style choices from controls
        grid_style_map = {"Solid": "-", "Dashed": "--", "Dotted": ":", "Dash-dot": "-."}
        grid_style = grid_style_map.get(self.cmb_grid_style.currentText(), "--")
        show_grid = self.chk_grid.isChecked()
        
        color_name = self.cmb_line_color.currentText()
        line_color = self.color_map.get(color_name, "#1F4E79")
        
        ls_name = self.cmb_line_style.currentText()
        line_style = self.line_style_map.get(ls_name, "-")
        
        marker_name = self.cmb_marker_style.currentText()
        marker_style = self.marker_style_map.get(marker_name, "o")
        
        show_highlights = self.chk_highlights.isChecked()
        show_legend = self.chk_legend.isChecked()
        pub_ticks = self.chk_pub_ticks.isChecked()
        
        # Font settings for publication look
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
        
        # Auto-configure title if requested
        if self.chk_auto_title.isChecked():
            self.txt_title.blockSignals(True)
            self.txt_title.setText(show_title)
            self.txt_title.blockSignals(False)
            display_title = show_title
        else:
            display_title = self.txt_title.text()
            
        # Auto-configure labels if requested
        if self.chk_auto_labels.isChecked():
            xlabel_text = "Field (mT)"
            ylabel_text = "MOKE Intensity (Normalized)" if self.normalize else "MOKE Intensity (a.u.)"
            
            self.txt_xlabel.blockSignals(True)
            self.txt_xlabel.setText(xlabel_text)
            self.txt_xlabel.blockSignals(False)
            
            self.txt_ylabel.blockSignals(True)
            self.txt_ylabel.setText(ylabel_text)
            self.txt_ylabel.blockSignals(False)
        else:
            xlabel_text = self.txt_xlabel.text()
            ylabel_text = self.txt_ylabel.text()
            
        # Draw the main data loop with clean styling
        if line_style == "None" and marker_style == "None":
            # Fallback to visible markers if both are None
            marker_style = "o"
            
        self.ax.plot(field, intensity, color=line_color, linestyle=line_style, 
                     marker=marker_style, lw=1.5 * scale, markersize=4 * scale, label='MOKE intensity')
        
        # Determine theme dynamically
        theme = "dark"
        if self.parent_widget and hasattr(self.parent_widget, 'theme'):
            theme = self.parent_widget.theme
            
        from gui_styles import get_theme_colors
        colors = get_theme_colors(theme)
        fig_bg = colors["card"]
        ax_bg = colors["bg"]
        text_color = colors["text"]
        grid_color = colors["border"]
        spine_color = colors["spine"]
            
        self.figure.patch.set_facecolor(fig_bg)
        self.ax.set_facecolor(ax_bg)
        self.ax.xaxis.label.set_color(text_color)
        self.ax.yaxis.label.set_color(text_color)
        if self.ax.title:
            self.ax.title.set_color(text_color)
        
        self.ax.set_xlabel(xlabel_text, fontsize=18 * scale, fontweight='normal', color=text_color)
        self.ax.set_ylabel(ylabel_text, fontsize=18 * scale, fontweight='normal', color=text_color)
        if display_title:
            self.ax.set_title(display_title, fontsize=11 * scale, fontweight='normal', pad=10 * scale, color=text_color)
        
        # Configure Grid
        if show_grid:
            self.ax.grid(True, which='both', color=grid_color, linestyle=grid_style, linewidth=1 * scale)
        else:
            self.ax.grid(False)
            
        # Configure Tick Style
        if pub_ticks:
            self.ax.tick_params(axis='both', which='major', direction='in', 
                                top=True, right=True, bottom=True, left=True,
                                labelsize=9.5 * scale, width=1.0 * scale, length=5.0 * scale, colors=text_color)
            self.ax.tick_params(axis='both', which='minor', direction='in',
                                top=True, right=True, bottom=True, left=True,
                                width=0.75 * scale, length=2.5 * scale, colors=text_color)
            self.ax.minorticks_on()
        else:
            self.ax.tick_params(axis='both', which='both', direction='out',
                                top=False, right=False, bottom=True, left=True,
                                labelsize=9.5 * scale, width=1.0 * scale, length=5.0 * scale, colors=text_color)
            self.ax.minorticks_off()
            
        # Style Spines (Borders)
        for spine in self.ax.spines.values():
            spine.set_linewidth(1.0 * scale)
            spine.set_color(spine_color)
            
            
        # Highlight Hc/Hr if they exist and checkbox is checked
        if show_highlights and self.hc_hr_marks is not None:
            hc_pos, hc_neg, rem_fields, rem_values = self.hc_hr_marks
            
            hc_pos_color = '#2E7D32'  # Forest green
            hc_neg_color = '#C62828'  # Crimson/dark red
            rem_color = '#E65100'     # Dark orange
            
            if len(intensity) > 0:
                yrange = max(intensity) - min(intensity)
            else:
                yrange = 1.0
                
            if hc_pos is not None:
                self.ax.plot(hc_pos[0], hc_pos[1], 'o', ms=6 * scale, mec=hc_pos_color, mfc='white', mew=1.5 * scale, label='Hc+')
                self.ax.axvline(x=hc_pos[0], color=hc_pos_color, linestyle=':', alpha=0.7, lw=1.2 * scale)
                self.ax.annotate(f"Hc+\n{hc_pos[0]:.2f} mT", xy=hc_pos, 
                                 xytext=(hc_pos[0], hc_pos[1] + 0.15 * yrange),
                                 ha="center", va="bottom", fontsize=8.5 * scale, color=hc_pos_color,
                                 bbox=dict(boxstyle='round,pad=0.2', fc='#f1f8e9', ec=hc_pos_color, lw=0.5 * scale, alpha=0.9),
                                 arrowprops=dict(arrowstyle='->', color=hc_pos_color, lw=0.8 * scale))
                                 
            if hc_neg is not None:
                self.ax.plot(hc_neg[0], hc_neg[1], 'o', ms=6 * scale, mec=hc_neg_color, mfc='white', mew=1.5 * scale, label='Hc-')
                self.ax.axvline(x=hc_neg[0], color=hc_neg_color, linestyle=':', alpha=0.7, lw=1.2 * scale)
                self.ax.annotate(f"Hc-\n{hc_neg[0]:.2f} mT", xy=hc_neg,
                                 xytext=(hc_neg[0], hc_neg[1] - 0.15 * yrange),
                                 ha="center", va="top", fontsize=8.5 * scale, color=hc_neg_color,
                                 bbox=dict(boxstyle='round,pad=0.2', fc='#ffebee', ec=hc_neg_color, lw=0.5 * scale, alpha=0.9),
                                 arrowprops=dict(arrowstyle='->', color=hc_neg_color, lw=0.8 * scale))
                                 
            if rem_fields is not None and rem_values is not None and len(rem_fields) > 0:
                mean_rf = np.mean(rem_fields)
                mean_rv = np.mean(rem_values)
                self.ax.plot(mean_rf, mean_rv, 's', ms=6 * scale, mec=rem_color, mfc='white', mew=1.5 * scale, label='Hr')
                self.ax.axhline(y=mean_rv, color=rem_color, linestyle=':', alpha=0.6, lw=1.0 * scale)
                
                # Make sure annotation arrow starts properly
                x_offset = 0.15 * (max(field) - min(field)) if len(field) > 0 else 10.0
                self.ax.annotate(f"Hr\n{mean_rv:.3f}", 
                                 xy=(mean_rf, mean_rv), 
                                 xytext=(mean_rf + x_offset, mean_rv), 
                                 ha="left", va="center", fontsize=8.5 * scale, color=rem_color,
                                 bbox=dict(boxstyle='round,pad=0.2', fc='#fff3e0', ec=rem_color, lw=0.5 * scale, alpha=0.9),
                                 arrowprops=dict(arrowstyle='->', color=rem_color, lw=0.8 * scale))

        if show_legend:
            self.ax.legend(loc='best', fontsize=8.5 * scale, frameon=True, facecolor='#ffffff', edgecolor='#e0e0e0', framealpha=0.9)
            
        # Restore limits if we are resizing
        if hasattr(self, '_resize_limits') and self._resize_limits is not None:
            xlim, ylim = self._resize_limits
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self._resize_limits = None
            
        self.figure.tight_layout()
        self.canvas.draw_idle()

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
        field      = parent.loop_field
        intensity0 = (parent.loop_intens_subtracted
                      if parent.loop_intens_subtracted is not None
                      else parent.loop_intens_txt)

        idx           = np.arange(len(field))
        idx_off       = idx - idx.mean()
        field_off     = field - np.mean(field)
        field_abs_max = float(np.max(np.abs(field_off)))

        # ------------------------------------------------------------------
        # Pass 1 – endpoint drift alignment
        # ------------------------------------------------------------------
        drift1     = float((intensity0[0] - intensity0[-1]) / len(intensity0))
        intensity1 = intensity0 + drift1 * idx_off

        # ------------------------------------------------------------------
        # Branch separation (ascending vs descending field sweep).
        # Fitting both branches together at the same field values distorts the
        # Faraday estimate because the two branches carry a hysteresis offset.
        # ------------------------------------------------------------------
        i_min = int(np.argmin(field))
        i_max = int(np.argmax(field))
        if i_min < i_max:
            asc_idx  = np.arange(i_min, i_max + 1)
            desc_idx = np.concatenate([np.arange(i_max, len(field)),
                                       np.arange(0, i_min + 1)])
        else:
            desc_idx = np.arange(i_max, i_min + 1)
            asc_idx  = np.concatenate([np.arange(i_min, len(field)),
                                       np.arange(0, i_max + 1)])

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Linear Faraday: fit positive and negative saturation shelves
        # independently on each branch, then average. This prevents the
        # hysteresis step from biasing the Faraday slope.
        # ------------------------------------------------------------------
        sat_threshold = 0.80 * field_abs_max
        slopes = []
        for branch_idx in (asc_idx, desc_idx):
            f_b = field_off[branch_idx]
            y_b = intensity1[branch_idx]
            sat_pos = f_b > sat_threshold
            sat_neg = f_b < -sat_threshold
            branch_slopes = []
            if np.sum(sat_pos) >= 2:
                p_pos = np.polyfit(f_b[sat_pos], y_b[sat_pos], 1)
                branch_slopes.append(p_pos[0])
            if np.sum(sat_neg) >= 2:
                p_neg = np.polyfit(f_b[sat_neg], y_b[sat_neg], 1)
                branch_slopes.append(p_neg[0])
            if branch_slopes:
                slopes.append(np.mean(branch_slopes))

        linear_val = -float(np.mean(slopes)) if slopes else 0.0
        intensity2 = intensity1 + linear_val * field_off

        # ------------------------------------------------------------------
        # Residual quadratic (Cotton–Mouton): fit on the step-subtracted
        # background to prevent the MOKE step height from biasing the parameters.
        # ------------------------------------------------------------------
        sat_pos = field_off > sat_threshold
        sat_neg = field_off < -sat_threshold
        sat_all = sat_pos | sat_neg

        quad1           = 0.0
        quad_offset_val = 0.0

        if np.sum(sat_pos) >= 2 and np.sum(sat_neg) >= 2:
            # Subtract shelf means to get background-only signal at saturation
            M_pos = np.mean(intensity2[sat_pos])
            M_neg = np.mean(intensity2[sat_neg])
            y_bg = intensity2.copy()
            y_bg[sat_pos] -= M_pos
            y_bg[sat_neg] -= M_neg

            p2 = np.polyfit(field_off[sat_all], y_bg[sat_all], 2)
            a2, b2 = float(p2[0]), float(p2[1])
            if abs(a2) > 0:
                candidate_offset = -b2 / (2.0 * a2)
                if abs(candidate_offset) <= field_abs_max:
                    # Physically plausible vertex position – use full quadratic
                    quad1           = -a2
                    quad_offset_val = candidate_offset
                    linear_val      -= b2
                else:
                    # Vertex far outside field range – absorb only linear part
                    linear_val -= b2

        # ------------------------------------------------------------------
        # Pass 2 – second drift correction after shape corrections
        # ------------------------------------------------------------------
        intensity3 = intensity1 + linear_val * field_off + quad1 * (field_off - quad_offset_val) ** 2
        drift2     = float((intensity3[0] - intensity3[-1]) / len(intensity3))

        # Automatically apply the normalization
        self.normalize = True
        self.btn_norm.setChecked(True)

        # Update sliders & spinboxes
        self.set_slider_spinbox('drift', drift1 + drift2)
        self.set_slider_spinbox('quad', quad1)
        self.set_slider_spinbox('quad_offset', quad_offset_val)
        self.set_slider_spinbox('linear', linear_val)

        # Save exact coefficients from spinboxes
        self.coeffs['drift']       = self.spin_drift.value()
        self.coeffs['quad']        = self.spin_quad.value()
        self.coeffs['quad_offset'] = self.spin_quad_offset.value()
        self.coeffs['linear']      = self.spin_linear.value()

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
        
        # Choose title based on whether ROI is used
        title = "Hysteresis Loop"
        if parent and parent.loop_intens_subtracted is not None and getattr(parent, 'last_loop_was_roi', False):
            roi_shape = parent.lbl_img.roi_shape
            roi_data = parent.lbl_img.roi_data
            if roi_shape in ["Rectangle", "Square"] and roi_data is not None:
                cx, cy, w, h, angle = roi_data
                title = f"Hysteresis Loop (ROI: {roi_shape} at CX:{int(cx)} CY:{int(cy)} W:{int(w)} H:{int(h)} A:{int(angle)}°)"
            elif roi_shape == "Circle" and roi_data is not None:
                cx, cy, r = roi_data
                title = f"Hysteresis Loop (ROI: Circle at CX:{int(cx)} CY:{int(cy)} R:{int(r)})"
            else:
                title = f"Hysteresis Loop (ROI: {roi_shape})"
                
        self.plot_loop(field, ycorr, show_title=title)
        QMessageBox.information(self, "Hc, Hr", msg)

    def save_loop(self):
        import datetime
        parent = self.parent_widget
        if parent is None or parent.loop_field is None:
            QMessageBox.warning(self, "No Loop Data", "No hysteresis loop has been generated yet. Please load a dataset first.")
            return

        raw_intens = parent.loop_intens_subtracted if parent.loop_intens_subtracted is not None else parent.loop_intens_txt
        if raw_intens is None:
            QMessageBox.warning(self, "No Loop Data", "No hysteresis loop intensity data available.")
            return

        field = parent.loop_field
        corrected_intens = self.correct_intensity(field, raw_intens)
        
        save_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Hysteresis Loop",
            "",
            "Text Files (*.txt);;CSV Files (*.csv);;PNG Image (*.png);;All Files (*)"
        )
        
        if not save_path:
            return
            
        ext = os.path.splitext(save_path)[1].lower()
        
        if ext == '.png' or "PNG Image" in selected_filter:
            if not save_path.lower().endswith('.png'):
                save_path += '.png'
            try:
                # Set saving flag to force scale = 1.0 for output
                self.is_saving = True
                
                # Cache interactive zoom/pan limits
                old_xlim = self.ax.get_xlim()
                old_ylim = self.ax.get_ylim()
                
                # Replot current data at scale = 1.0
                self.replot_current_data()
                
                # Reset axis limits to show full data
                self.ax.autoscale(True)
                self.ax.relim()
                self.ax.autoscale_view()
                self.figure.canvas.draw()
                
                self.figure.savefig(save_path, dpi=300)
                
                # Restore state
                self.is_saving = False
                
                # Replot current data back at GUI scale
                self.replot_current_data()
                
                # Restore interactive zoom/pan limits
                self.ax.set_xlim(old_xlim)
                self.ax.set_ylim(old_ylim)
                self.canvas.draw()
                
                QMessageBox.information(self, "Success", f"Plot image saved to:\n{save_path}")
            except Exception as e:
                self.is_saving = False
                try:
                    self.replot_current_data()
                    self.ax.set_xlim(old_xlim)
                    self.ax.set_ylim(old_ylim)
                    self.canvas.draw()
                except Exception:
                    pass
                QMessageBox.critical(self, "Error", f"Failed to save plot image:\n{e}")
        else:
            if not ext:
                save_path += '.txt'
                ext = '.txt'
                
            sep = ',' if ext == '.csv' else '\t'
            
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(f"# KerrPyLooper Hysteresis Loop Export\n")
                    f.write(f"# Export Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"#\n")
                    
                    if parent.lbl_img.roi_shape != "None" and parent.lbl_img.roi_data is not None:
                        roi_shape = parent.lbl_img.roi_shape
                        f.write(f"# ROI Type: {roi_shape}\n")
                        if roi_shape in ["Rectangle", "Square"]:
                            cx, cy, w, h, angle = parent.lbl_img.roi_data
                            f.write(f"# ROI Center X: {cx:.2f}\n")
                            f.write(f"# ROI Center Y: {cy:.2f}\n")
                            f.write(f"# ROI Width: {w:.2f}\n")
                            f.write(f"# ROI Height: {h:.2f}\n")
                            f.write(f"# ROI Angle: {angle:.2f} deg\n")
                        elif roi_shape == "Circle":
                            cx, cy, r = parent.lbl_img.roi_data
                            f.write(f"# ROI Center X: {cx:.2f}\n")
                            f.write(f"# ROI Center Y: {cy:.2f}\n")
                            f.write(f"# ROI Radius: {r:.2f}\n")
                    else:
                        f.write(f"# ROI Type: None (Full Frame Mean)\n")
                        
                    f.write(f"#\n")
                    f.write(f"# Drift Coefficient: {self.coeffs['drift']:.6e}\n")
                    f.write(f"# Linear Faraday: {self.coeffs['linear']:.6e}\n")
                    f.write(f"# Quadratic Faraday: {self.coeffs['quad']:.6e}\n")
                    f.write(f"# Quadratic Field Offset: {self.coeffs['quad_offset']:.6f} mT\n")
                    f.write(f"# Intensity Normalized: {self.normalize}\n")
                    
                    z_enabled = self.chk_z_drift.isChecked()
                    f.write(f"# Z-Drift Correction Enabled: {z_enabled}\n")
                    if z_enabled:
                        f.write(f"# Z-Drift Focus Coeff: {self.spin_z_quad.value():.6e}\n")
                        f.write(f"# Z-Drift Method: {self.cmb_z_method.currentText()}\n")
                    
                    if self.hc_hr_marks is not None:
                        hc_pos, hc_neg, rem_fields, rem_values = self.hc_hr_marks
                        f.write(f"# Coercivity Hc+: {hc_pos[0]:.2f} mT\n" if hc_pos is not None else "# Coercivity Hc+: n/a\n")
                        f.write(f"# Coercivity Hc-: {hc_neg[0]:.2f} mT\n" if hc_neg is not None else "# Coercivity Hc-: n/a\n")
                        f.write(f"# Remanence Hr: {np.mean(rem_values):.3f}\n" if rem_values is not None else "# Remanence Hr: n/a\n")
                        
                    f.write(f"#\n")
                    f.write(f"Field_mT{sep}Raw_Intensity{sep}Corrected_Intensity\n")
                    for fd, ri, ci in zip(field, raw_intens, corrected_intens):
                        f.write(f"{fd:.6f}{sep}{ri:.6f}{sep}{ci:.6f}\n")
                
                img_path = os.path.splitext(save_path)[0] + '.png'
                try:
                    self.figure.savefig(img_path, dpi=300)
                    QMessageBox.information(self, "Saved", f"Data exported successfully to:\n{save_path}\n\nPlot image saved successfully to:\n{img_path}")
                except Exception as e_img:
                    QMessageBox.warning(self, "Warning", f"Data saved to:\n{save_path}\n\nBut failed to save plot image:\n{e_img}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save data file:\n{e}")

class MOKEImageSubtractor(QWidget):
    def __init__(self, theme="dark"):
        super().__init__()
        self.theme = theme
        self.setObjectName("MainBg")
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
        self.last_loop_was_roi = False
        self.init_ui()

    def init_ui(self):
        splitter = StyledSplitter(Qt.Horizontal)
        splitter.setHandleWidth(8)
        widget_left = QWidget()
        left_layout = QVBoxLayout()
        self.btn_dir = QPushButton("Select Hysteresis Image Directory")
        self.btn_dir.clicked.connect(self.choose_directory)
        left_layout.addWidget(self.btn_dir)
        self.list_images = QListWidget()
        self.list_images.currentRowChanged.connect(self.on_image_selected)
        self.lbl_select_bg = QLabel("Select Background Image:")
        left_layout.addWidget(self.lbl_select_bg)
        left_layout.addWidget(self.list_images)
        self.btn_set_bg = QPushButton("Set as Background")
        self.btn_set_bg.clicked.connect(self.set_background)
        left_layout.addWidget(self.btn_set_bg)
        self.lbl_browse_images = QLabel("Browse Images & View Subtraction:")
        left_layout.addWidget(self.lbl_browse_images)
        self.list_results = QListWidget()
        self.list_results.currentRowChanged.connect(self.show_subtracted_image)
        left_layout.addWidget(self.list_results)
        self.btn_save = QPushButton("Save This Subtraction Result")
        self.btn_save.clicked.connect(self.save_current_result)
        self.btn_save.setEnabled(False)
        self.btn_save.setToolTip("Set a background image and select an image to subtract first")
        left_layout.addWidget(self.btn_save)
        self.btn_make_loop = QPushButton("Make Loop (Plot Hysteresis)")
        self.btn_make_loop.clicked.connect(self.run_subtraction_loop)
        self.btn_make_loop.setEnabled(False)
        self.btn_make_loop.setToolTip("Load a directory with a mapping .txt file first to make a loop")
        left_layout.addWidget(self.btn_make_loop)
        # Horizontal layout for preview and ROI tools
        hbox_img_roi = QHBoxLayout()
        
        # Wrap image label and its contrast slider in a vertical layout
        img_preview_layout = QVBoxLayout()
        
        # Contrast slider directly above the image
        hbox_contrast = QHBoxLayout()
        hbox_contrast.addWidget(QLabel("Contrast Stretch:"))
        self.sld_img_contrast = QSlider(Qt.Horizontal)
        self.sld_img_contrast.setMinimum(00)
        self.sld_img_contrast.setMaximum(200)
        self.sld_img_contrast.setValue(100)
        self.sld_img_contrast.valueChanged.connect(self.img_contrast_changed)
        
        self.spin_img_contrast = QDoubleSpinBox()
        self.spin_img_contrast.setDecimals(2)
        self.spin_img_contrast.setRange(0.1, 4.0)
        self.spin_img_contrast.setSingleStep(0.01)
        self.spin_img_contrast.setValue(1.0)
        self.spin_img_contrast.valueChanged.connect(self.img_contrast_spinbox_changed)
        
        hbox_contrast.addWidget(self.sld_img_contrast)
        hbox_contrast.addWidget(self.spin_img_contrast)
        
        hbox_contrast.addWidget(QLabel("Colormap:"))
        self.cmb_colormap = QComboBox()
        self.cmb_colormap.addItems(["gray", "plasma", "seismic", "viridis", "magma"])
        self.cmb_colormap.currentTextChanged.connect(self.show_current_subtracted_image_contrast_only)
        hbox_contrast.addWidget(self.cmb_colormap)
        
        img_preview_layout.addLayout(hbox_contrast)
        
        self.lbl_img = ROISelectLabel(self)
        self.lbl_img.setAlignment(Qt.AlignCenter)
        self.lbl_img.roi_changed.connect(self.on_roi_changed)
        img_preview_layout.addWidget(self.lbl_img, stretch=1)
        
        hbox_img_roi.addLayout(img_preview_layout, stretch=1)
        
        # ROI Tools panel
        roi_group = QGroupBox("ROI Settings")
        roi_layout = QFormLayout()
        
        self.cmb_roi_shape = QComboBox()
        self.cmb_roi_shape.addItems(["None", "Rectangle", "Square", "Circle"])
        self.cmb_roi_shape.currentIndexChanged.connect(self.on_roi_shape_changed)
        roi_layout.addRow("Shape:", self.cmb_roi_shape)
        
        self.spin_roi_x = QSpinBox()
        self.spin_roi_x.setRange(0, 2000)
        self.spin_roi_x.setEnabled(False)
        self.spin_roi_x.valueChanged.connect(self.on_roi_spinbox_changed)
        roi_layout.addRow("X / Center X:", self.spin_roi_x)
        
        self.spin_roi_y = QSpinBox()
        self.spin_roi_y.setRange(0, 2000)
        self.spin_roi_y.setEnabled(False)
        self.spin_roi_y.valueChanged.connect(self.on_roi_spinbox_changed)
        roi_layout.addRow("Y / Center Y:", self.spin_roi_y)
        
        self.spin_roi_w = QSpinBox()
        self.spin_roi_w.setRange(1, 2000)
        self.spin_roi_w.setEnabled(False)
        self.spin_roi_w.valueChanged.connect(self.on_roi_spinbox_changed)
        roi_layout.addRow("Width / Radius:", self.spin_roi_w)
        
        self.spin_roi_h = QSpinBox()
        self.spin_roi_h.setRange(1, 2000)
        self.spin_roi_h.setEnabled(False)
        self.spin_roi_h.valueChanged.connect(self.on_roi_spinbox_changed)
        roi_layout.addRow("Height:", self.spin_roi_h)
        
        self.spin_roi_angle = QSpinBox()
        self.spin_roi_angle.setRange(-180, 180)
        self.spin_roi_angle.setEnabled(False)
        self.spin_roi_angle.valueChanged.connect(self.on_roi_spinbox_changed)
        roi_layout.addRow("Angle (deg):", self.spin_roi_angle)
        
        self.btn_clear_roi = QPushButton("Clear Selection")
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        roi_layout.addRow(self.btn_clear_roi)
        
        roi_group.setLayout(roi_layout)
        roi_group.setFixedWidth(200)
        hbox_img_roi.addWidget(roi_group)
        
        left_layout.addLayout(hbox_img_roi, stretch=1)

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
        
        from gui_styles import apply_theme
        apply_theme(self, self.theme)

    def change_theme(self, theme):
        self.theme = theme
        from gui_styles import apply_theme
        apply_theme(self, theme)
        if hasattr(self, 'loop_panel') and self.loop_panel is not None:
            # Update placeholder label style if it exists
            if hasattr(self.loop_panel, 'placeholder_label') and self.loop_panel.placeholder_label is not None:
                from gui_styles import get_theme_colors
                colors = get_theme_colors(theme)
                ax_bg = colors["bg"]
                text_color = colors["text"]
                self.loop_panel.placeholder_label.setStyleSheet(
                    f"background-color: {ax_bg}; color: {text_color}; border-radius: 6px; "
                    f"font-size: 14px; font-weight: 500; border: 1px solid {colors['border']};"
                )
            if self.loop_field is not None:
                self.loop_panel.replot_current_data()

    def on_roi_shape_changed(self, idx):
        shape = self.cmb_roi_shape.currentText()
        self.lbl_img.set_roi_shape(shape)
        
        if shape == "None":
            self.spin_roi_x.setEnabled(False)
            self.spin_roi_y.setEnabled(False)
            self.spin_roi_w.setEnabled(False)
            self.spin_roi_h.setEnabled(False)
            self.spin_roi_angle.setEnabled(False)
            self.lbl_img.clear_roi()
        else:
            self.spin_roi_x.setEnabled(True)
            self.spin_roi_y.setEnabled(True)
            self.spin_roi_w.setEnabled(True)
            
            if shape == "Rectangle":
                self.spin_roi_h.setEnabled(True)
                self.spin_roi_angle.setEnabled(True)
            elif shape == "Square":
                self.spin_roi_h.setEnabled(False)
                self.spin_roi_angle.setEnabled(True)
            else: # Circle
                self.spin_roi_h.setEnabled(False)
                self.spin_roi_angle.setEnabled(False)
                
            if self.lbl_img.pixmap() and not self.lbl_img.pixmap().isNull():
                pw, ph = self.lbl_img.pixmap().width(), self.lbl_img.pixmap().height()
                self.update_roi_spinbox_ranges(pw, ph)
                
                if shape in ["Rectangle", "Square"]:
                    size = 100
                    cx = pw // 2
                    cy = ph // 2
                    self.lbl_img.roi_data = (cx, cy, size, size, 0.0)
                elif shape == "Circle":
                    cx = pw // 2
                    cy = ph // 2
                    r = 50
                    self.lbl_img.roi_data = (cx, cy, r)
                    
                self.update_spinboxes_from_roi()
                self.lbl_img.update()

    def update_roi_spinbox_ranges(self, pw, ph):
        self.spin_roi_x.setRange(0, pw - 1)
        self.spin_roi_y.setRange(0, ph - 1)
        self.spin_roi_w.setRange(1, pw)
        self.spin_roi_h.setRange(1, ph)

    def update_spinboxes_from_roi(self):
        roi_data = self.lbl_img.roi_data
        shape = self.lbl_img.roi_shape
        if roi_data is None:
            return
            
        self.spin_roi_x.blockSignals(True)
        self.spin_roi_y.blockSignals(True)
        self.spin_roi_w.blockSignals(True)
        self.spin_roi_h.blockSignals(True)
        self.spin_roi_angle.blockSignals(True)
        
        if shape in ["Rectangle", "Square"]:
            cx, cy, w, h, angle = roi_data
            self.spin_roi_x.setValue(int(cx))
            self.spin_roi_y.setValue(int(cy))
            self.spin_roi_w.setValue(int(w))
            self.spin_roi_h.setValue(int(h))
            self.spin_roi_angle.setValue(int(angle))
        elif shape == "Circle":
            cx, cy, r = roi_data
            self.spin_roi_x.setValue(int(cx))
            self.spin_roi_y.setValue(int(cy))
            self.spin_roi_w.setValue(int(r))
            self.spin_roi_h.setValue(int(r))
            self.spin_roi_angle.setValue(0)
            
        self.spin_roi_x.blockSignals(False)
        self.spin_roi_y.blockSignals(False)
        self.spin_roi_w.blockSignals(False)
        self.spin_roi_h.blockSignals(False)
        self.spin_roi_angle.blockSignals(False)

    def on_roi_spinbox_changed(self, _):
        shape = self.lbl_img.roi_shape
        if shape == "None":
            return
            
        cx = self.spin_roi_x.value()
        cy = self.spin_roi_y.value()
        w = self.spin_roi_w.value()
        h = self.spin_roi_h.value()
        angle = self.spin_roi_angle.value()
        
        if shape == "Rectangle":
            self.lbl_img.roi_data = (cx, cy, w, h, angle)
        elif shape == "Square":
            self.lbl_img.roi_data = (cx, cy, w, w, angle)
            self.spin_roi_h.blockSignals(True)
            self.spin_roi_h.setValue(w)
            self.spin_roi_h.blockSignals(False)
        elif shape == "Circle":
            self.lbl_img.roi_data = (cx, cy, w)
            self.spin_roi_h.blockSignals(True)
            self.spin_roi_h.setValue(w)
            self.spin_roi_h.blockSignals(False)
            
        self.lbl_img.update()

    def on_roi_changed(self):
        self.update_spinboxes_from_roi()

    def clear_roi(self):
        self.cmb_roi_shape.setCurrentIndex(0)

    def request_loop_update(self):
        if self.loop_field is not None and self.loop_intens_txt is not None:
            values = self.loop_intens_subtracted if self.loop_intens_subtracted is not None else self.loop_intens_txt
            
            # Choose title based on whether ROI is used
            title = "Hysteresis Loop"
            if self.loop_intens_subtracted is not None and getattr(self, 'last_loop_was_roi', False):
                roi_shape = self.lbl_img.roi_shape
                roi_data = self.lbl_img.roi_data
                if roi_shape in ["Rectangle", "Square"] and roi_data is not None:
                    cx, cy, w, h, angle = roi_data
                    title = f"Hysteresis Loop (ROI: {roi_shape} at CX:{int(cx)} CY:{int(cy)} W:{int(w)} H:{int(h)} A:{int(angle)}°)"
                elif roi_shape == "Circle" and roi_data is not None:
                    cx, cy, r = roi_data
                    title = f"Hysteresis Loop (ROI: Circle at CX:{int(cx)} CY:{int(cy)} R:{int(r)})"
                else:
                    title = f"Hysteresis Loop (ROI: {roi_shape})"
                    
            self.loop_panel.plot_loop(self.loop_field, self.loop_panel.correct_intensity(self.loop_field, values), show_title=title)

    def choose_directory(self):
        d = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if d:
            self.img_dir = d
            self.clear_roi()
            self.background_image = None
            self.background_array = None
            self.txt_data = None
            self.btn_make_loop.setEnabled(False)
            self.btn_make_loop.setToolTip("Load a directory with a mapping .txt file first to make a loop")
            self.btn_save.setEnabled(False)
            self.btn_save.setToolTip("Set a background image and select an image to subtract first")
            self.loop_field = None
            self.loop_indices = None
            self.loop_intens_txt = None
            self.loop_intens_subtracted = None
            self.mean_index = None
            self.mean_field = None
            self.lbl_img.setText("Preview will appear here")
            self.update_select_bg_label(None, is_set=False)
            self.update_browse_images_label(None)
            self.update_image_list()
            self.load_txt_data()

    def get_field_for_file(self, filename):
        if self.txt_data is not None:
            clean_fn = filename.strip()
            match = self.txt_data[self.txt_data['File'] == clean_fn]
            if not match.empty:
                return float(match.iloc[0]['Field'])
        return None

    def update_list_widget_items(self, list_widget, files):
        list_widget.blockSignals(True)
        curr_row = list_widget.currentRow()
        list_widget.clear()
        display_items = []
        for f in files:
            field_val = self.get_field_for_file(f)
            if field_val is not None:
                display_items.append(f"{f}  ({field_val:.2f} mT)")
            else:
                display_items.append(f)
        list_widget.addItems(display_items)
        if 0 <= curr_row < len(files):
            list_widget.setCurrentRow(curr_row)
        list_widget.blockSignals(False)

    def update_image_list(self):
        self.image_files = [
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
        ]
        self.image_files.sort()
        self.update_list_widget_items(self.list_images, self.image_files)
        self.update_list_widget_items(self.list_results, self.image_files)

    def load_txt_data(self):
        txt_file = None
        for fn in os.listdir(self.img_dir):
            if fn.lower().endswith('.txt'):
                txt_file = os.path.join(self.img_dir, fn)
                break
        if not txt_file:
            self.txt_data = None
            self.btn_make_loop.setEnabled(False)
            self.btn_make_loop.setToolTip("Load a directory with a mapping .txt file first to make a loop")
            self.loop_field = self.loop_indices = self.loop_intens_txt = self.loop_intens_subtracted = None
            self.mean_index = self.mean_field = None
            self.loop_panel.clear_plot()
            return
        try:
            df = pd.read_csv(txt_file, sep=None, engine='python', comment="#", skip_blank_lines=True)
            df.columns = [c.strip() for c in df.columns]
            if len(df.columns) < 3:
                QMessageBox.critical(self, "Error", f"Text file {os.path.basename(txt_file)} missing columns.")
                self.txt_data = None
                self.btn_make_loop.setEnabled(False)
                self.btn_make_loop.setToolTip("Load a directory with a mapping .txt file first to make a loop")
                self.loop_field = self.loop_indices = self.loop_intens_txt = self.loop_intens_subtracted = None
                self.mean_index = self.mean_field = None
                self.loop_panel.clear_plot()
                return
            df = df[df[df.columns[2]].str.lower().str.endswith(".png", na=False)]
            self.txt_data = df.rename(
                columns={df.columns[0]:"Field", df.columns[1]:"Intensity", df.columns[2]:"File"}
            ).reset_index(drop=True)
            self.txt_data["File"] = self.txt_data["File"].str.strip()
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
            self.btn_make_loop.setToolTip("Calculate and plot the hysteresis loop")

            # Filter image_files to only those listed in the .txt file
            valid_files = set(self.txt_data["File"].str.strip().tolist())
            self.image_files = [f for f in self.image_files if f in valid_files]

            self.request_loop_update()
            
            # Update list widgets with the newly loaded field values
            self.update_list_widget_items(self.list_images, self.image_files)
            self.update_list_widget_items(self.list_results, self.image_files)
            
            # Re-sync select_bg and browse_images labels with current row/file details
            idx_bg = self.list_images.currentRow()
            if 0 <= idx_bg < len(self.image_files):
                self.update_select_bg_label(self.image_files[idx_bg], is_set=(self.background_image is not None))
            else:
                self.update_select_bg_label(None, is_set=(self.background_image is not None))
                
            idx_browse = self.list_results.currentRow()
            if 0 <= idx_browse < len(self.image_files):
                self.update_browse_images_label(self.image_files[idx_browse])
            else:
                self.update_browse_images_label(None)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read .txt data file.\n{e}")
            self.txt_data = None
            self.loop_field = self.loop_indices = self.loop_intens_txt = self.loop_intens_subtracted = None
            self.mean_index = self.mean_field = None
            self.btn_make_loop.setEnabled(False)
            self.btn_make_loop.setToolTip("Load a directory with a mapping .txt file first to make a loop")
            self.loop_panel.clear_plot()

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
        filename = self.image_files[idx]
        filepath = os.path.join(self.img_dir, filename)
        self.display_image(filepath)
        self.update_select_bg_label(filename, is_set=False)

    def img_contrast_changed(self, val):
        contrast_val = val / 100.0
        self.loop_panel.contrast = contrast_val
        self.spin_img_contrast.blockSignals(True)
        self.spin_img_contrast.setValue(contrast_val)
        self.spin_img_contrast.blockSignals(False)
        self.show_current_subtracted_image_contrast_only()

    def img_contrast_spinbox_changed(self, val):
        self.loop_panel.contrast = val
        self.sld_img_contrast.blockSignals(True)
        self.sld_img_contrast.setValue(int(val * 100))
        self.sld_img_contrast.blockSignals(False)
        self.show_current_subtracted_image_contrast_only()

    def robust_normalize_raw(self, arr):
        arr = arr.astype(np.float32)
        mask = arr > 0
        if np.any(mask):
            active = arr[mask]
            low = np.percentile(active, 1)
            high = np.percentile(active, 99)
            if high > low:
                arr_disp = np.clip((arr - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
                arr_disp[~mask] = 0
                return arr_disp
        min_val = arr.min()
        max_val = arr.max()
        ptp = max_val - min_val
        if ptp > 0:
            return ((arr - min_val) / ptp * 255.0).astype(np.uint8)
        return np.zeros_like(arr, dtype=np.uint8)

    def format_file_and_field(self, filename):
        if not filename:
            return ""
        field_val = self.get_field_for_file(filename)
        if field_val is not None:
            return f"{filename}  {field_val:.2f} mT"
        return filename

    def update_select_bg_label(self, current_file=None, is_set=False):
        if not hasattr(self, 'lbl_select_bg'):
            return
        if is_set:
            if self.background_image:
                info = self.format_file_and_field(self.background_image)
                self.lbl_select_bg.setText(f"Select Background Image (SET: {info}):")
            else:
                self.lbl_select_bg.setText("Select Background Image:")
        else:
            if current_file:
                info = self.format_file_and_field(current_file)
                self.lbl_select_bg.setText(f"Select Background Image (Current: {info}):")
            else:
                if self.background_image:
                    info = self.format_file_and_field(self.background_image)
                    self.lbl_select_bg.setText(f"Select Background Image (SET: {info}):")
                else:
                    self.lbl_select_bg.setText("Select Background Image:")

    def update_browse_images_label(self, current_file=None):
        if not hasattr(self, 'lbl_browse_images'):
            return
        if current_file:
            info = self.format_file_and_field(current_file)
            self.lbl_browse_images.setText(f"Browse Images & View Subtraction (Current: {info}):")
        else:
            self.lbl_browse_images.setText("Browse Images & View Subtraction:")

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
            if arr.dtype != np.uint8:
                arr_disp = self.robust_normalize_raw(arr)
            else:
                arr_disp = arr
            h, w = arr_disp.shape
            data = arr_disp.tobytes()
            qimg = QImage(data, w, h, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        self.lbl_img.setPixmap(pix)
        self.update_roi_spinbox_ranges(pix.width(), pix.height())

    def set_background(self):
        idx = self.list_images.currentRow()
        if idx < 0:
            QMessageBox.warning(self, "No selection", "Select a background image first.")
            return
        bg_file = self.image_files[idx]
        filepath = os.path.join(self.img_dir, bg_file)
        self.background_image = bg_file
        self.background_array = np.array(Image.open(filepath))
        #QMessageBox.information(self, "Background Set", f"Background set to: {bg_file}")
        
        self.update_select_bg_label(bg_file, is_set=True)
        self.update_list_widget_items(self.list_results, self.image_files)
        self.btn_save.setEnabled(True)
        self.btn_save.setToolTip("Save the currently displayed subtracted image")
        
        # If there is a current selection in list_results, refresh it!
        res_idx = self.list_results.currentRow()
        if res_idx >= 0:
            self.show_subtracted_image(res_idx)

    def show_subtracted_image(self, idx):
        if idx < 0 or idx >= len(self.image_files):
            return
        img_file = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        self.update_browse_images_label(img_file)
        
        # If no background is set, or if the selected image is the background itself,
        # display the original image instead of a flat/zero difference or static.
        if self.background_array is None or img_file == self.background_image:
            try:
                img_arr = np.array(Image.open(img_path))
                img_arr = crop600(img_arr)
                if img_arr.ndim == 3 and img_arr.shape[2] in [3, 4]:
                    if img_arr.shape[2] == 4:
                        data = img_arr.tobytes()
                        h, w, c = img_arr.shape
                        qimg = QImage(data, w, h, QImage.Format_RGBA8888)
                    else:
                        data = img_arr.tobytes()
                        h, w, c = img_arr.shape
                        qimg = QImage(data, w, h, QImage.Format_RGB888)
                else:
                    arr_disp = self.robust_normalize_raw(img_arr)
                    h, w = arr_disp.shape
                    data = arr_disp.tobytes()
                    qimg = QImage(data, w, h, QImage.Format_Grayscale8)
                pix = QPixmap.fromImage(qimg)
                self.lbl_img.setPixmap(pix)
                self.update_roi_spinbox_ranges(pix.width(), pix.height())
                
                # Update current image variables but clear subtraction arrays
                self.current_difference_arr_raw = None
                self.current_difference_arr = None
                self.current_difference_img = Image.fromarray(arr_disp) if (img_arr.ndim != 3 or img_arr.shape[2] not in [3, 4]) else Image.fromarray(img_arr)
                self.current_image_idx = idx
                self.current_image_file = img_file
                self.btn_save.setEnabled(False)
                self.btn_save.setToolTip("Set a background image and select an image to subtract first")  # Save is disabled for raw image preview
            except Exception as e:
                self.lbl_img.setText(f"Error: {e}")
                self.current_difference_img = None
                self.current_difference_arr = None
                self.current_difference_arr_raw = None
            return

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
            self.btn_save.setEnabled(True)
            self.btn_save.setToolTip("Save the currently displayed subtracted image")  # Enable save for subtracted image
        except Exception as e:
            self.lbl_img.setText(f"Error: {e}")
            self.current_difference_img = None
            self.current_difference_arr = None
            self.current_difference_arr_raw = None

    def apply_colormap_to_arr(self, arr_disp):
        colormap_name = self.cmb_colormap.currentText()
        if colormap_name == "gray":
            h, w = arr_disp.shape
            data = arr_disp.tobytes()
            qimg = QImage(data, w, h, QImage.Format_Grayscale8)
            show_img = Image.fromarray(arr_disp)
            return qimg, show_img
        else:
            normalized = arr_disp.astype(np.float32) / 255.0
            try:
                cmap = mpl.colormaps.get_cmap(colormap_name)
            except AttributeError:
                try:
                    cmap = mpl.cm.get_cmap(colormap_name)
                except AttributeError:
                    cmap = mpl.cm.gray
            rgba_arr = (cmap(normalized) * 255).astype(np.uint8)
            h, w, c = rgba_arr.shape
            data = rgba_arr.tobytes()
            qimg = QImage(data, w, h, w * 4, QImage.Format_RGBA8888)
            show_img = Image.fromarray(rgba_arr)
            return qimg, show_img

    def show_current_subtracted_image_contrast_only(self):
        if not hasattr(self, 'current_difference_arr_raw') or self.current_difference_arr_raw is None:
            if self.current_difference_arr is None:
                return
            arr_disp = normalized_for_display(self.current_difference_arr, contrast=self.loop_panel.contrast)
            qimg, show_img = self.apply_colormap_to_arr(arr_disp)
            pix = QPixmap.fromImage(qimg)
            self.lbl_img.setPixmap(pix)
            self.update_roi_spinbox_ranges(pix.width(), pix.height())
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
        qimg, show_img = self.apply_colormap_to_arr(arr_disp)
        pix = QPixmap.fromImage(qimg)
        self.lbl_img.setPixmap(pix)
        self.update_roi_spinbox_ranges(pix.width(), pix.height())
        self.current_difference_img = show_img

    def save_current_result(self):
        if self.current_difference_img is not None:
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Contrast Image", self.current_result_filename, "PNG Files (*.png)")
            if save_path:
                colormap_name = self.cmb_colormap.currentText()
                if colormap_name == "gray":
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
                else:
                    self.current_difference_img.save(save_path)
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
        
        # Check ROI selection
        roi_shape = self.lbl_img.roi_shape
        roi_data = self.lbl_img.roi_data
        enable_roi = (roi_shape != "None" and roi_data is not None)
        
        # ⚡ Bolt: Pre-crop and pre-cast the background array to float32
        # This avoids doing these expensive operations for every single image
        bg_base = crop600(self.background_array)

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
                    field = row['Field'] if enable_z else 0.0
                    mean_val = compute_subtracted_mean(
                        img_arr, bg_base,
                        enable_z=enable_z, coeff=coeff, method_idx=method_idx, field=field,
                        enable_roi=enable_roi, roi_shape=roi_shape, roi_data=roi_data
                    )
                    means.append(mean_val)
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    means.append(np.nan)
            self.last_loop_was_roi = enable_roi
            self.loop_intens_subtracted = np.array(means, dtype=np.float32)
            
            # Dynamically adjust correction ranges based on subtracted intensity
            ptp_val = np.ptp(self.loop_intens_subtracted)
            self.loop_panel.update_correction_ranges(ptp_val)
            
            self.request_loop_update()
        finally:
            QApplication.restoreOverrideCursor()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kerr MOKE Looper Analysis Tool")
    parser.add_argument("--theme", type=str, default="dark", choices=["dark", "charcoal", "light"], help="Theme to apply")
    args, unknown = parser.parse_known_args()

    app = QApplication(sys.argv)
    window = MOKEImageSubtractor(theme=args.theme)
    window.show()
    sys.exit(app.exec_())
