# -*- coding: utf-8 -*-
"""
KerrPyLooper Suite Launcher
==========================
Top-level dashboard GUI for launching and controlling analysis scripts 
in the Kerr MOKE image processing project.

Features:
- Extensible tool registry to add new GUI/CLI scripts.
- Premium Slate Dark Theme styling (QSS).
- Embedded monospace console terminal window showing stdout/stderr in real-time.
- Unbuffered output streams via Python environment configuration.
- Native folder dialog arguments generation.
- Full process control (start/stop/cleanup).

Created in 2026.
"""

import sys
import io
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

if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()
import os
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QSplitter, QTextEdit, QMessageBox,
    QFileDialog, QProgressBar, QGraphicsDropShadowEffect, QComboBox
)
from PyQt5.QtGui import QFont, QColor, QTextCursor, QPainter, QPainterPath, QPen, QBrush, QConicalGradient, QRadialGradient, QIcon
from PyQt5.QtCore import Qt, QProcess, QProcessEnvironment, QRectF, QPointF

from gui_styles import apply_theme, get_magnetic_icon

# ==============================================================================
# TOOL REGISTRY (Easily add new scripts here)
# ==============================================================================
TOOL_REGISTRY = [
    {
        "id": "kerr_looper",
        "name": "Kerr MOKE Looper",
        "subtitle": "Interactive Analysis Tool",
        "description": "Load MOKE image series, select background reference, perform out-of-plane focus drift & Faraday corrections, select regions of interest (ROIs), and plot/extract loop parameters (Hc, Hr).",
        "script": "kerr_looper_AG.py",
        "icon": "📊",
        "prompt_directory": False,
    },
    {
        "id": "batch_processor",
        "name": "Batch Loop Processor",
        "subtitle": "Automated Multi-Sweep Processing",
        "description": "Scan a parent directory for multiple sweep directories, apply Z-drift & Faraday corrections automatically, calculate Hc/Hr, and save individual plots, loop files, and tab-delimited summaries.",
        "script": "batch_processor.py",
        "icon": "⚙️",
        "prompt_directory": False,
    },
    {
        "id": "vector_analysis",
        "name": "Vector Maps",
        "subtitle": "Magnetization Vector Analysis",
        "description": "Generate 2D magnetization vector maps from X/Y MOKE image sweeps. Select cropping bounds, apply wavelet denoising, overlay quiver arrows with scale bars, and export vector map plots and loop curves.",
        "script": "vector_analysis.py",
        "icon": "🧭",
        "prompt_directory": False,
    },
    {
        "id": "drift_corrector",
        "name": "Drift Corrector",
        "subtitle": "In-Plane X/Y Drift Alignment",
        "description": "Correct field-induced in-plane (X/Y) image drift in Kerr hysteresis image series. Select a static defect ROI, set a search width, and apply sub-pixel NCC alignment to produce a drift-corrected image series and diagnostic drift curves.",
        "script": "drift_corrector.py",
        "icon": "⚓",
        "prompt_directory": False,
    },
    {
        "id": "hc_map",
        "name": "Local Hc Map",
        "subtitle": "Spatially Resolved Coercivity",
        "description": "Generate a 2D spatial map of local coercivity (Hc) across the field of view by fitting the hysteresis loop for every single pixel or binned region. Useful for analyzing defect pinning and material homogeneity.",
        "script": "hc_map_tool.py",
        "icon": "🗺️",
        "prompt_directory": False,
    },
    {
        "id": "dmdh_mapper",
        "name": "dM/dH Mapper",
        "subtitle": "Pinning Landscape Analysis",
        "description": "Calculate differential magnetization (dM/dH) to visualize domain wall pinning sites and create a complete 2D Iso-switching topographical map of the coercive landscape.",
        "script": "dmdh_tool.py",
        "icon": "📈",
        "prompt_directory": False,
    }
]

# ==============================================================================
# CUSTOM HYSTERESIS LOOP WIDGET
# ==============================================================================
class HysteresisLoopWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(80, 80)
        self.setMaximumSize(120, 120)
        self.theme = "dark"
        
    def set_theme(self, theme):
        self.theme = theme
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Determine colors based on theme
        from gui_styles import get_theme_colors
        colors = get_theme_colors(self.theme)
        line_color = QColor(colors["accent"])
        grid_color = QColor(colors["border"])
            
        w, h = self.width(), self.height()
        margin = 10
        cx, cy = w / 2.0, h / 2.0
        dw, dh = w - 2 * margin, h - 2 * margin
        
        # Draw background grids/axes
        pen_grid = QPen(grid_color, 1, Qt.DashLine)
        painter.setPen(pen_grid)
        painter.drawLine(margin, int(cy), w - margin, int(cy))  # Horizontal axis
        painter.drawLine(int(cx), margin, int(cx), h - margin)  # Vertical axis
        
        # Draw the loop
        path = QPainterPath()
        
        steps = 50
        hc = 0.2  # Coercivity offset in normalized units
        k = 4.0   # Steepness
        
        # Lower curve (going left to right):
        path.moveTo(float(cx - dw / 2.0), float(cy - math.tanh(k * (-1.0 - hc)) * (dh / 2.0)))
        for i in range(steps + 1):
            t = -1.0 + 2.0 * i / steps
            y = math.tanh(k * (t - hc))
            px = cx + t * (dw / 2.0)
            py = cy - y * (dh / 2.0)
            path.lineTo(float(px), float(py))
            
        # Upper curve (going right to left):
        for i in range(steps + 1):
            t = 1.0 - 2.0 * i / steps
            y = math.tanh(k * (t + hc))
            px = cx + t * (dw / 2.0)
            py = cy - y * (dh / 2.0)
            path.lineTo(float(px), float(py))
            
        path.closeSubpath()
        
        # Draw the curve
        pen_line = QPen(line_color, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen_line)
        painter.drawPath(path)
        painter.end()


# ==============================================================================
# CUSTOM MAGNETIC VORTEX WIDGET (ICON)
# ==============================================================================
class VortexWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(80, 80)
        self.setMaximumSize(120, 120)
        self.theme = "charcoal"
        
    def set_theme(self, theme):
        self.theme = theme
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Determine colors based on theme
        from gui_styles import get_theme_colors
        colors = get_theme_colors(self.theme)
        
        w, h = self.width(), self.height()
        margin = 10
        cx, cy = w / 2.0, h / 2.0
        r_max = min(w, h) / 2.0 - margin
        
        # Draw the permalloy dot circle filled with conical HSV gradient
        gradient = QConicalGradient(cx, cy, 90.0) # Start angle at 90 degrees
        for i in range(361):
            spin_angle = (i + 180) % 360
            color = QColor.fromHsv(spin_angle, 220, 240)
            gradient.setColorAt(i / 360.0, color)
            
        painter.setBrush(gradient)
        border_color = QColor(colors["border"])
        painter.setPen(QPen(border_color, 1.5, Qt.SolidLine))
        painter.drawEllipse(int(cx - r_max), int(cy - r_max), int(2 * r_max), int(2 * r_max))
        
        # Define helper for high-contrast vector arrows (visible on any HSV color background)
        def draw_vortex_arrow(radius, angle_deg):
            rad = math.radians(angle_deg)
            # The position of the arrow center
            px = cx + radius * math.cos(rad)
            py = cy - radius * math.sin(rad)
            
            # The direction of the arrow is tangent to the circle (CCW): (-sin(rad), -cos(rad)) in Qt
            tx = -math.sin(rad)
            ty = -math.cos(rad)
            
            # Draw a short line segment representing the vector
            length = 9
            start_x = px - (length / 2.0) * tx
            start_y = py - (length / 2.0) * ty
            end_x = px + (length / 2.0) * tx
            end_y = py + (length / 2.0) * ty
            
            # Arrowhead details
            nx = math.cos(rad)
            ny = -math.sin(rad)
            arrow_len = 4
            p1_x = end_x - arrow_len * tx + arrow_len * 0.45 * nx
            p1_y = end_y - arrow_len * ty + arrow_len * 0.45 * ny
            p2_x = end_x - arrow_len * tx - arrow_len * 0.45 * nx
            p2_y = end_y - arrow_len * ty - arrow_len * 0.45 * ny
            
            # 1. Draw black outline/shadow for readability
            pen_bg = QPen(QColor(0, 0, 0, 160), 2.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen_bg)
            painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
            painter.drawLine(int(end_x), int(end_y), int(p1_x), int(p1_y))
            painter.drawLine(int(end_x), int(end_y), int(p2_x), int(p2_y))
            
            # 2. Draw white foreground arrow line
            pen_fg = QPen(QColor(255, 255, 255, 240), 1.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen_fg)
            painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
            painter.drawLine(int(end_x), int(end_y), int(p1_x), int(p1_y))
            painter.drawLine(int(end_x), int(end_y), int(p2_x), int(p2_y))
            
        # Draw vector arrow distribution over concentric rings
        # Ring 1 (Outer, r = r_max * 0.8): 8 arrows
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            draw_vortex_arrow(r_max * 0.8, angle)
            
        # Ring 2 (Middle, r = r_max * 0.55): 6 arrows
        for angle in [30, 90, 150, 210, 270, 330]:
            draw_vortex_arrow(r_max * 0.55, angle)
            
        # Ring 3 (Inner, r = r_max * 0.3): 3 arrows
        for angle in [0, 120, 240]:
            draw_vortex_arrow(r_max * 0.3, angle)
            
        # Draw the out-of-plane core in the center using radial gradient (white core fading out)
        core_grad = QRadialGradient(cx, cy, 6)
        core_grad.setColorAt(0.0, QColor(255, 255, 255, 255))
        core_grad.setColorAt(0.3, QColor(255, 255, 255, 200))
        core_grad.setColorAt(1.0, QColor(255, 255, 255, 0))
        
        painter.setBrush(core_grad)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(int(cx - 6), int(cy - 6), 12, 12)
        
        # Add a tiny black core dot at the very center (core polarization direction)
        painter.setBrush(QBrush(QColor(0, 0, 0, 220)))
        painter.drawEllipse(int(cx - 1.5), int(cy - 1.5), 3, 3)
        
        painter.end()

# ==============================================================================================================
# CUSTOM DRIFT ALIGNMENT WIDGET (ICON)
# ==============================================================================
class DriftAlignmentWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(80, 80)
        self.setMaximumSize(120, 120)
        self.theme = "charcoal"
        
    def set_theme(self, theme):
        self.theme = theme
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Determine colors based on theme
        from gui_styles import get_theme_colors
        colors = get_theme_colors(self.theme)
        accent_color = QColor(colors["accent"])
        border_color = QColor(colors["border"])
        
        w, h = self.width(), self.height()
        margin = 10
        cx, cy = w / 2.0, h / 2.0
        r_max = min(w, h) / 2.0 - margin
        
        # 1. Outer circular dial base
        painter.setBrush(QBrush(QColor(colors["bg"])))
        painter.setPen(QPen(border_color, 1.5, Qt.SolidLine))
        painter.drawEllipse(int(cx - r_max), int(cy - r_max), int(2 * r_max), int(2 * r_max))
        
        # 2. Reference frame (centered, stable, solid accent color with soft translucent fill)
        frame_w = r_max * 1.1
        frame_h = r_max * 0.8
        ref_rect = QRectF(cx - frame_w/2, cy - frame_h/2, frame_w, frame_h)
        painter.setBrush(QBrush(QColor(accent_color.red(), accent_color.green(), accent_color.blue(), 15)))
        painter.setPen(QPen(accent_color, 2.0, Qt.SolidLine))
        painter.drawRoundedRect(ref_rect, 4.0, 4.0)
        
        # 3. Drifted frame (shifted, dotted, pinkish color representing uncorrected image)
        shift_x = 10
        shift_y = -8
        drift_rect = QRectF(cx - frame_w/2 + shift_x, cy - frame_h/2 + shift_y, frame_w, frame_h)
        painter.setBrush(Qt.NoBrush)
        drift_pen = QPen(QColor("#f472b6"), 1.2, Qt.DashLine)
        painter.setPen(drift_pen)
        painter.drawRoundedRect(drift_rect, 4.0, 4.0)
        
        # 4. Correction vector arrow (from center of drifted frame to center of ref frame)
        dcx = cx + shift_x
        dcy = cy + shift_y
        
        # Draw dotted path representing drift trajectory
        pen_path = QPen(QColor(255, 255, 255, 100), 1.0, Qt.DotLine)
        painter.setPen(pen_path)
        painter.drawLine(QPointF(dcx, dcy), QPointF(cx, cy))
        
        # Draw vector arrow head pointing to reference center (cx, cy)
        pen_vector = QPen(QColor("#f472b6"), 1.8, Qt.SolidLine)
        painter.setPen(pen_vector)
        
        vx = -shift_x
        vy = -shift_y
        length = math.sqrt(vx*vx + vy*vy)
        if length > 0.1:
            ux, uy = vx / length, vy / length
            # Draw line segment
            painter.drawLine(QPointF(dcx, dcy), QPointF(cx, cy))
            
            # Arrowhead details
            al = 6.0
            angle = math.radians(25)
            # Right wing
            rx = ux * math.cos(angle) - uy * math.sin(angle)
            ry = ux * math.sin(angle) + uy * math.cos(angle)
            painter.drawLine(QPointF(cx, cy), QPointF(cx - al * rx, cy - al * ry))
            # Left wing
            lx = ux * math.cos(-angle) - uy * math.sin(-angle)
            ly = ux * math.sin(-angle) + uy * math.cos(-angle)
            painter.drawLine(QPointF(cx, cy), QPointF(cx - al * lx, cy - al * ly))
            
        # 5. Corner brackets (Autofocus style) around the reference frame corners to indicate alignment active
        pen_brackets = QPen(accent_color, 1.5, Qt.SolidLine)
        painter.setPen(pen_brackets)
        bs = 6.0 # bracket size
        offset = 3.0
        
        # Top-Left
        painter.drawLine(QPointF(cx - frame_w/2 - offset, cy - frame_h/2 - offset), QPointF(cx - frame_w/2 - offset + bs, cy - frame_h/2 - offset))
        painter.drawLine(QPointF(cx - frame_w/2 - offset, cy - frame_h/2 - offset), QPointF(cx - frame_w/2 - offset, cy - frame_h/2 - offset + bs))
        # Top-Right
        painter.drawLine(QPointF(cx + frame_w/2 + offset, cy - frame_h/2 - offset), QPointF(cx + frame_w/2 + offset - bs, cy - frame_h/2 - offset))
        painter.drawLine(QPointF(cx + frame_w/2 + offset, cy - frame_h/2 - offset), QPointF(cx + frame_w/2 + offset, cy - frame_h/2 - offset + bs))
        # Bottom-Left
        painter.drawLine(QPointF(cx - frame_w/2 - offset, cy + frame_h/2 + offset), QPointF(cx - frame_w/2 - offset + bs, cy + frame_h/2 + offset))
        painter.drawLine(QPointF(cx - frame_w/2 - offset, cy + frame_h/2 + offset), QPointF(cx - frame_w/2 - offset, cy + frame_h/2 + offset - bs))
        # Bottom-Right
        painter.drawLine(QPointF(cx + frame_w/2 + offset, cy + frame_h/2 + offset), QPointF(cx + frame_w/2 + offset - bs, cy + frame_h/2 + offset))
        painter.drawLine(QPointF(cx + frame_w/2 + offset, cy + frame_h/2 + offset), QPointF(cx + frame_w/2 + offset, cy + frame_h/2 + offset - bs))
        
        painter.end()


# ==============================================================================
# CUSTOM BATCH LOOP STACK WIDGET (ICON)
# ==============================================================================
class BatchLoopStackWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(80, 80)
        self.setMaximumSize(120, 120)
        self.theme = "charcoal"
        
    def set_theme(self, theme):
        self.theme = theme
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        from gui_styles import get_theme_colors
        colors = get_theme_colors(self.theme)
        
        w, h = self.width(), self.height()
        margin = 10
        cx, cy = w / 2.0, h / 2.0
        dw, dh = w - 2 * margin, h - 2 * margin
        
        grid_color = QColor(colors["border"])
        pen_grid = QPen(QColor(grid_color.red(), grid_color.green(), grid_color.blue(), 100), 1, Qt.DashLine)
        painter.setPen(pen_grid)
        painter.drawLine(margin, int(cy), w - margin, int(cy))
        painter.drawLine(int(cx), margin, int(cx), h - margin)
        
        loop_colors = [
            QColor("#3b82f6"),
            QColor(colors["accent"]),
            QColor("#ec4899")
        ]
        
        steps = 40
        k = 4.5
        
        for idx, (hc, amp, color) in enumerate([
            (0.35, 0.7, loop_colors[0]),
            (0.20, 0.9, loop_colors[1]),
            (0.10, 0.8, loop_colors[2])
        ]):
            alpha = 100 + idx * 70
            line_color = QColor(color.red(), color.green(), color.blue(), alpha)
            
            path = QPainterPath()
            w_scale = 0.8 + idx * 0.1
            h_scale = 0.8 + idx * 0.1
            cur_dw = dw * w_scale
            cur_dh = dh * h_scale
            
            path.moveTo(float(cx - cur_dw / 2.0), float(cy - math.tanh(k * (-1.0 - hc)) * (cur_dh / 2.0)))
            for i in range(steps + 1):
                t = -1.0 + 2.0 * i / steps
                y = math.tanh(k * (t - hc))
                px = cx + t * (cur_dw / 2.0)
                py = cy - y * (cur_dh / 2.0)
                path.lineTo(float(px), float(py))
                
            for i in range(steps + 1):
                t = 1.0 - 2.0 * i / steps
                y = math.tanh(k * (t + hc))
                px = cx + t * (cur_dw / 2.0)
                py = cy - y * (cur_dh / 2.0)
                path.lineTo(float(px), float(py))
                
            path.closeSubpath()
            
            pen_line = QPen(line_color, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen_line)
            painter.drawPath(path)
            
        painter.end()


# ==============================================================================
# CUSTOM LOCAL HC MAP PREVIEW WIDGET (ICON)
# ==============================================================================
class HcMapPreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(80, 80)
        self.setMaximumSize(120, 120)
        self.theme = "charcoal"
        
    def set_theme(self, theme):
        self.theme = theme
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        from gui_styles import get_theme_colors
        colors = get_theme_colors(self.theme)
        
        w, h = self.width(), self.height()
        margin = 10
        cx, cy = w / 2.0, h / 2.0
        r_max = min(w, h) / 2.0 - margin
        
        border_color = QColor(colors["border"])
        painter.setPen(QPen(border_color, 1.5, Qt.SolidLine))
        painter.setBrush(QBrush(QColor(colors["bg"])))
        map_rect = QRectF(cx - r_max, cy - r_max, 2 * r_max, 2 * r_max)
        painter.drawRoundedRect(map_rect, 6.0, 6.0)
        
        grid_n = 6
        tile_w = (2 * r_max) / grid_n
        
        def get_plasma_color(val):
            r = int(255 * (val ** 0.8))
            g = int(200 * (val ** 2.0))
            b = int(255 * (1.0 - val ** 0.5) * 0.6 + 50)
            return QColor(r, g, b, 220)
            
        for r in range(grid_n):
            for c in range(grid_n):
                dist = math.sqrt((r - 2.5)**2 + (c - 2.5)**2)
                norm_dist = min(dist / 3.54, 1.0)
                val = 0.2 + 0.8 * (1.0 - norm_dist)
                
                # Pseudo-random jitter
                val = max(0.0, min(1.0, val - 0.15 + ((r * 13 + c * 37) % 10) / 30.0))
                
                tile_color = get_plasma_color(val)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(tile_color))
                
                tx = cx - r_max + c * tile_w
                ty = cy - r_max + r * tile_w
                painter.drawRect(QRectF(tx + 0.5, ty + 0.5, tile_w - 1.0, tile_w - 1.0))
                
        pen_contour = QPen(QColor(255, 255, 255, 180), 1.5, Qt.SolidLine)
        painter.setPen(pen_contour)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QRectF(cx - r_max * 0.6, cy - r_max * 0.6, r_max * 1.2, r_max * 1.2))
        
        painter.end()


# ==============================================================================
# CUSTOM SUSCEPTIBILITY PREVIEW WIDGET (ICON)
# ==============================================================================
class SusceptibilityWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(80, 80)
        self.setMaximumSize(120, 120)
        self.theme = "charcoal"
        
    def set_theme(self, theme):
        self.theme = theme
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Determine colors based on theme
        from gui_styles import get_theme_colors
        colors = get_theme_colors(self.theme)
        accent_color = QColor(colors["accent"])
        border_color = QColor(colors["border"])
        bg_color = QColor(colors["bg"])
            
        w, h = self.width(), self.height()
        margin = 12
        cx, cy = w / 2.0, h / 2.0
        dw, dh = w - 2 * margin, h - 2 * margin
        
        # 1. Draw outer boundary box (card style)
        painter.setBrush(QBrush(bg_color))
        painter.setPen(QPen(border_color, 1.5, Qt.SolidLine))
        painter.drawRoundedRect(QRectF(cx - dw/2 - 2, cy - dh/2 - 2, dw + 4, dh + 4), 6.0, 6.0)
        
        # 2. Draw background scientific grid
        pen_grid = QPen(QColor(border_color.red(), border_color.green(), border_color.blue(), 100), 1, Qt.DashLine)
        painter.setPen(pen_grid)
        
        # Vertical grid lines
        painter.drawLine(int(cx - dw / 4.0), int(cy - dh / 2.0), int(cx - dw / 4.0), int(cy + dh / 2.0))
        painter.drawLine(int(cx), int(cy - dh / 2.0), int(cx), int(cy + dh / 2.0))
        painter.drawLine(int(cx + dw / 4.0), int(cy - dh / 2.0), int(cx + dw / 4.0), int(cy + dh / 2.0))
        
        # Horizontal grid lines
        painter.drawLine(int(cx - dw / 2.0), int(cy - dh / 4.0), int(cx + dw / 2.0), int(cy - dh / 4.0))
        painter.drawLine(int(cx - dw / 2.0), int(cy), int(cx + dw / 2.0), int(cy))
        painter.drawLine(int(cx - dw / 2.0), int(cy + dh / 4.0), int(cx + dw / 2.0), int(cy + dh / 4.0))
        
        # Baseline
        baseline_y = cy + dh / 3.0
        pen_base = QPen(border_color, 2, Qt.SolidLine)
        painter.setPen(pen_base)
        painter.drawLine(int(cx - dw / 2.0), int(baseline_y), int(cx + dw / 2.0), int(baseline_y))
        
        # 3. Calculate paths for peaks (dM/dH)
        path_susc = QPainterPath()
        path_fill = QPainterPath()
        
        steps = 80
        hc = 0.35  # coercivity peak separation
        k_susc = 5.5
        
        # Map values
        def get_pt(t):
            val1 = 1.0 / (math.cosh(k_susc * (t - hc)) ** 2)
            val2 = 1.0 / (math.cosh(k_susc * (t + hc)) ** 2)
            y = val1 + val2  # combined peaks
            px = cx + t * (dw / 2.0)
            py = baseline_y - y * (dh * 0.65)
            return px, py
            
        # Build paths
        px0, py0 = get_pt(-1.0)
        path_susc.moveTo(float(px0), float(py0))
        path_fill.moveTo(float(cx - dw / 2.0), float(baseline_y))
        path_fill.lineTo(float(px0), float(py0))
        
        for i in range(1, steps + 1):
            t = -1.0 + 2.0 * i / steps
            px, py = get_pt(t)
            path_susc.lineTo(float(px), float(py))
            path_fill.lineTo(float(px), float(py))
            
        path_fill.lineTo(float(cx + dw / 2.0), float(baseline_y))
        path_fill.closeSubpath()
        
        # 4. Fill area under the peaks with a gradient
        from PyQt5.QtGui import QLinearGradient
        grad = QLinearGradient(cx, cy - dh/2, cx, baseline_y)
        grad.setColorAt(0.0, QColor(accent_color.red(), accent_color.green(), accent_color.blue(), 90))
        grad.setColorAt(1.0, QColor(accent_color.red(), accent_color.green(), accent_color.blue(), 10))
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.NoPen)
        painter.drawPath(path_fill)
        
        # 5. Draw bold susceptibility line
        pen_susc = QPen(accent_color, 3.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen_susc)
        painter.drawPath(path_susc)
        
        # 6. Draw glowing data nodes at the peak maxima
        node_radius = 4.0
        
        # Left peak
        n1_x, n1_y = get_pt(-hc)
        # Glow ring
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(accent_color.red(), accent_color.green(), accent_color.blue(), 80)))
        painter.drawEllipse(QPointF(n1_x, n1_y), node_radius + 3.0, node_radius + 3.0)
        # Inner white dot
        painter.setPen(QPen(accent_color, 2, Qt.SolidLine))
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawEllipse(QPointF(n1_x, n1_y), node_radius, node_radius)
        
        # Right peak
        n2_x, n2_y = get_pt(hc)
        # Glow ring
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(accent_color.red(), accent_color.green(), accent_color.blue(), 80)))
        painter.drawEllipse(QPointF(n2_x, n2_y), node_radius + 3.0, node_radius + 3.0)
        # Inner white dot
        painter.setPen(QPen(accent_color, 2, Qt.SolidLine))
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawEllipse(QPointF(n2_x, n2_y), node_radius, node_radius)
        
        painter.end()


# ==============================================================================
# TOOL CARD COMPONENT
# ==============================================================================
class ToolCardWidget(QFrame):
    def __init__(self, tool_config, on_launch_callback, parent=None):
        super().__init__(parent)
        self.setObjectName("ToolCard")
        self.config = tool_config
        self.loop_widget = None
        self.vortex_widget = None
        self.drift_widget = None
        self.batch_widget = None
        self.hc_map_widget = None
        self.dmdh_widget = None
        
        # Setup vertical layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # Emoji Icon or Custom Widget
        if tool_config["id"] == "kerr_looper":
            self.loop_widget = HysteresisLoopWidget(self)
            icon_layout = QHBoxLayout()
            icon_layout.addStretch()
            icon_layout.addWidget(self.loop_widget)
            icon_layout.addStretch()
            layout.addLayout(icon_layout)
        elif tool_config["id"] == "batch_processor":
            self.batch_widget = BatchLoopStackWidget(self)
            icon_layout = QHBoxLayout()
            icon_layout.addStretch()
            icon_layout.addWidget(self.batch_widget)
            icon_layout.addStretch()
            layout.addLayout(icon_layout)
        elif tool_config["id"] == "vector_analysis":
            self.vortex_widget = VortexWidget(self)
            icon_layout = QHBoxLayout()
            icon_layout.addStretch()
            icon_layout.addWidget(self.vortex_widget)
            icon_layout.addStretch()
            layout.addLayout(icon_layout)
        elif tool_config["id"] == "drift_corrector":
            self.drift_widget = DriftAlignmentWidget(self)
            icon_layout = QHBoxLayout()
            icon_layout.addStretch()
            icon_layout.addWidget(self.drift_widget)
            icon_layout.addStretch()
            layout.addLayout(icon_layout)
        elif tool_config["id"] == "hc_map":
            self.hc_map_widget = HcMapPreviewWidget(self)
            icon_layout = QHBoxLayout()
            icon_layout.addStretch()
            icon_layout.addWidget(self.hc_map_widget)
            icon_layout.addStretch()
            layout.addLayout(icon_layout)
        elif tool_config["id"] == "dmdh_mapper":
            self.dmdh_widget = SusceptibilityWidget(self)
            icon_layout = QHBoxLayout()
            icon_layout.addStretch()
            icon_layout.addWidget(self.dmdh_widget)
            icon_layout.addStretch()
            layout.addLayout(icon_layout)
        else:
            icon_lbl = QLabel(tool_config["icon"])
            icon_lbl.setObjectName("CardIcon")
            icon_lbl.setStyleSheet("font-size: 40px; font-family: 'Segoe UI Emoji', sans-serif; background-color: transparent; padding-bottom: 5px;")
            icon_lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(icon_lbl)
        
        # Header text (Title and Subtitle)
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        
        sub_lbl = QLabel(tool_config["subtitle"])
        sub_lbl.setObjectName("CardSubtitle")
        text_layout.addWidget(sub_lbl)
        
        title_lbl = QLabel(tool_config["name"])
        title_lbl.setObjectName("CardTitle")
        text_layout.addWidget(title_lbl)
        
        layout.addLayout(text_layout)
        
        # Description (Word-wrapped and left-aligned)
        desc_lbl = QLabel(tool_config["description"])
        desc_lbl.setObjectName("CardDescription")
        desc_lbl.setWordWrap(True)
        desc_lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(desc_lbl, stretch=1)
        
        # Action Launch Button
        self.btn_launch = QPushButton("Launch Tool")
        self.btn_launch.setObjectName("LaunchButton")
        self.btn_launch.setCursor(Qt.PointingHandCursor)
        self.btn_launch.clicked.connect(lambda: on_launch_callback(self.config))
        layout.addWidget(self.btn_launch)
        
        # Subtle premium drop shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(16)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 100))
        self.setGraphicsEffect(shadow)
 
    def set_theme(self, theme):
        if self.loop_widget:
            self.loop_widget.set_theme(theme)
        if self.vortex_widget:
            self.vortex_widget.set_theme(theme)
        if self.drift_widget:
            self.drift_widget.set_theme(theme)
        if self.batch_widget:
            self.batch_widget.set_theme(theme)
        if self.hc_map_widget:
            self.hc_map_widget.set_theme(theme)
        if self.dmdh_widget:
            self.dmdh_widget.set_theme(theme)


# ==============================================================================
# MAIN LAUNCHER WINDOW
# ==============================================================================
class SuiteLauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KerrPyLooper Analysis Suite")
        self.setMinimumSize(950, 700)
        self.current_process = None
        self.current_theme = "charcoal"
        self.cards = []
        self.active_looper_window = None
        self.active_batch_window = None
        self.active_vector_window = None
        self.active_drift_window = None
        self.active_hc_map_window = None
        self.active_dmdh_window = None
        self.init_ui()
        
    def init_ui(self):
        # Set central widget
        central_widget = QWidget()
        central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(20)
        
        # 1. Main Header Panel (Horizontal layout to fit theme toggle)
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        title_layout = QVBoxLayout()
        title_layout.setSpacing(4)
        
        title_lbl = QLabel("KerrPyLooper Suite")
        title_lbl.setObjectName("SuiteTitle")
        title_layout.addWidget(title_lbl)
        
        subtitle_lbl = QLabel("Unified control center for Kerr microscopy hysteresis loop analysis and batch processing")
        subtitle_lbl.setObjectName("SuiteSubtitle")
        title_layout.addWidget(subtitle_lbl)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        # Theme selector dropdown
        self.theme_selector = QComboBox()
        self.theme_selector.setObjectName("ThemeSelector")
        self.theme_selector.addItem("🌑 Charcoal Dark", "charcoal")
        self.theme_selector.addItem("🌙 Slate Dark", "dark")
        self.theme_selector.addItem("☀️ Slate Light", "light")
        self.theme_selector.setMinimumWidth(150)
        self.theme_selector.currentIndexChanged.connect(self.on_theme_changed)
        header_layout.addWidget(self.theme_selector, alignment=Qt.AlignVCenter)
        
        main_layout.addWidget(header_widget)
        
        # 2. Main Resizable Splitter (Cards on top, Console on bottom)
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(8)
        
        # Dashboard Panel (Cards)
        dashboard_widget = QWidget()
        dashboard_layout = QHBoxLayout(dashboard_widget)
        dashboard_layout.setContentsMargins(0, 0, 0, 0)
        dashboard_layout.setSpacing(20)
        
        self.cards = []
        for tool in TOOL_REGISTRY:
            card = ToolCardWidget(tool, self.on_launch_tool, self)
            card.set_theme(self.current_theme)
            dashboard_layout.addWidget(card)
            self.cards.append(card)
            
        splitter.addWidget(dashboard_widget)
        
        # Console Panel
        console_widget = QWidget()
        console_layout = QVBoxLayout(console_widget)
        console_layout.setContentsMargins(0, 5, 0, 0)
        console_layout.setSpacing(10)
        
        # Console Header Line
        console_header = QHBoxLayout()
        console_header.setSpacing(10)
        
        console_title = QLabel("Console Output Log")
        console_title.setObjectName("SectionTitle")
        console_header.addWidget(console_title)
        console_header.addStretch()
        
        self.btn_clear = QPushButton("Clear Logs")
        self.btn_clear.setObjectName("ConsoleControlButton")
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        self.btn_clear.clicked.connect(self.clear_console)
        console_header.addWidget(self.btn_clear)
        
        self.btn_stop = QPushButton("Stop Process")
        self.btn_stop.setObjectName("StopButton")
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("No process is currently running")
        self.btn_stop.clicked.connect(self.stop_process)
        console_header.addWidget(self.btn_stop)
        
        console_layout.addLayout(console_header)
        
        # Text Console Output
        self.console_output = QTextEdit()
        self.console_output.setObjectName("ConsoleOutput")
        self.console_output.setReadOnly(True)
        console_layout.addWidget(self.console_output)
        
        # Footer (Status Label & Indeterminate Progress Bar)
        footer_layout = QHBoxLayout()
        self.status_lbl = QLabel("Status: Idle")
        self.status_lbl.setObjectName("SuiteSubtitle")
        footer_layout.addWidget(self.status_lbl)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        footer_layout.addWidget(self.progress_bar, stretch=1)
        
        console_layout.addLayout(footer_layout)
        splitter.addWidget(console_widget)
        
        # Default distribution: 55% for dashboard cards, 45% for terminal log
        splitter.setStretchFactor(0, 11)
        splitter.setStretchFactor(1, 9)
        
        main_layout.addWidget(splitter)
        
        # Apply initial theme stylesheet
        apply_theme(self, self.current_theme)
        self.setWindowIcon(get_magnetic_icon(self.current_theme))
        
    def on_theme_changed(self, index):
        theme_name = self.theme_selector.itemData(index)
        self.current_theme = theme_name
        apply_theme(self, self.current_theme)
        self.setWindowIcon(get_magnetic_icon(self.current_theme))
        for card in self.cards:
            card.set_theme(self.current_theme)
        if self.active_looper_window is not None:
            self.active_looper_window.change_theme(self.current_theme)
            self.active_looper_window.setWindowIcon(get_magnetic_icon(self.current_theme))
        if self.active_batch_window is not None:
            self.active_batch_window.change_theme(self.current_theme)
            self.active_batch_window.setWindowIcon(get_magnetic_icon(self.current_theme))
        if self.active_vector_window is not None:
            self.active_vector_window.change_theme(self.current_theme)
            self.active_vector_window.setWindowIcon(get_magnetic_icon(self.current_theme))
        if self.active_drift_window is not None:
            self.active_drift_window.change_theme(self.current_theme)
            self.active_drift_window.setWindowIcon(get_magnetic_icon(self.current_theme))
        if self.active_hc_map_window is not None:
            self.active_hc_map_window.change_theme(self.current_theme)
            self.active_hc_map_window.setWindowIcon(get_magnetic_icon(self.current_theme))
        if self.active_dmdh_window is not None:
            self.active_dmdh_window.change_theme(self.current_theme)
            self.active_dmdh_window.setWindowIcon(get_magnetic_icon(self.current_theme))
            
    def log(self, text):
        """Append text to the console output text widget."""
        self.console_output.moveCursor(QTextCursor.End)
        self.console_output.insertPlainText(text + "\n")
        self.console_output.moveCursor(QTextCursor.End)
        
    def clear_console(self):
        """Clear all texts from the log panel."""
        self.console_output.clear()
        
    def stop_process(self):
        """Stop the currently executing subprocess."""
        if self.current_process is not None:
            self.log("\n[Info] Sending termination signal to process...")
            self.current_process.kill()
            
    def on_launch_tool(self, tool_config):
        """Callback when a card launch button is pressed."""
        if tool_config["id"] == "kerr_looper":
            if self.active_looper_window is not None:
                self.active_looper_window.show()
                self.active_looper_window.raise_()
                self.active_looper_window.activateWindow()
                self.log("[Info] Brought active Kerr MOKE Looper window to focus.")
            else:
                self.log("[Info] Launching Kerr MOKE Looper in same process...")
                try:
                    from kerr_looper_AG import MOKEImageSubtractor
                    window = MOKEImageSubtractor(theme=self.current_theme)
                    window.setWindowIcon(get_magnetic_icon(self.current_theme))
                    
                    # Intercept closeEvent to clear reference
                    orig_close = window.closeEvent
                    def custom_close(event):
                        orig_close(event)
                        if event.isAccepted():
                            self.active_looper_window = None
                            self.log("[Info] Kerr MOKE Looper window closed.")
                    window.closeEvent = custom_close
                    
                    self.active_looper_window = window
                    window.show()
                    self.log("[Info] Kerr MOKE Looper launched successfully.")
                except Exception as e:
                    self.log(f"[Error] Failed to launch Kerr MOKE Looper: {e}")
                    import traceback
                    self.log(traceback.format_exc())
                    QMessageBox.critical(self, "Error", f"Failed to launch Kerr MOKE Looper:\n{e}")
            return
        elif tool_config["id"] == "batch_processor":
            if self.active_batch_window is not None:
                self.active_batch_window.show()
                self.active_batch_window.raise_()
                self.active_batch_window.activateWindow()
                self.log("[Info] Brought active Batch Loop Processor window to focus.")
            else:
                self.log("[Info] Launching Batch Loop Processor in same process...")
                try:
                    from batch_processor import BatchProcessorGUI
                    window = BatchProcessorGUI(theme=self.current_theme, parent=self)
                    window.setWindowIcon(get_magnetic_icon(self.current_theme))
                    
                    # Intercept closeEvent to clear reference
                    orig_close = window.closeEvent
                    def custom_close(event):
                        orig_close(event)
                        if event.isAccepted():
                            self.active_batch_window = None
                            self.log("[Info] Batch Loop Processor window closed.")
                    window.closeEvent = custom_close
                    
                    self.active_batch_window = window
                    window.show()
                    self.log("[Info] Batch Loop Processor launched successfully.")
                except Exception as e:
                    self.log(f"[Error] Failed to launch Batch Loop Processor: {e}")
                    import traceback
                    self.log(traceback.format_exc())
                    QMessageBox.critical(self, "Error", f"Failed to launch Batch Loop Processor:\n{e}")
            return
        elif tool_config["id"] == "vector_analysis":
            if self.active_vector_window is not None:
                self.active_vector_window.show()
                self.active_vector_window.raise_()
                self.active_vector_window.activateWindow()
                self.log("[Info] Brought active Vector Maps window to focus.")
            else:
                self.log("[Info] Launching Vector Maps in same process...")
                try:
                    from vector_analysis import VectorAnalysisGUI
                    window = VectorAnalysisGUI(theme=self.current_theme, parent=None)
                    window.setWindowIcon(get_magnetic_icon(self.current_theme))
                    
                    # Intercept closeEvent to clear reference
                    orig_close = window.closeEvent
                    def custom_close(event):
                        orig_close(event)
                        if event.isAccepted():
                            self.active_vector_window = None
                            self.log("[Info] Vector Maps window closed.")
                    window.closeEvent = custom_close
                    
                    self.active_vector_window = window
                    window.show()
                    self.log("[Info] Vector Maps launched successfully.")
                except Exception as e:
                    self.log(f"[Error] Failed to launch Vector Maps: {e}")
                    import traceback
                    self.log(traceback.format_exc())
                    QMessageBox.critical(self, "Error", f"Failed to launch Vector Maps:\n{e}")
            return
        elif tool_config["id"] == "drift_corrector":
            if self.active_drift_window is not None:
                self.active_drift_window.show()
                self.active_drift_window.raise_()
                self.active_drift_window.activateWindow()
                self.log("[Info] Brought active Drift Corrector window to focus.")
            else:
                self.log("[Info] Launching Drift Corrector in same process...")
                try:
                    from drift_corrector import DriftCorrectorWindow
                    window = DriftCorrectorWindow(theme=self.current_theme)
                    window.setWindowIcon(get_magnetic_icon(self.current_theme))

                    # Intercept closeEvent to clear reference
                    orig_close = window.closeEvent
                    def custom_close(event):
                        orig_close(event)
                        if event.isAccepted():
                            self.active_drift_window = None
                            self.log("[Info] Drift Corrector window closed.")
                    window.closeEvent = custom_close

                    self.active_drift_window = window
                    window.show()
                    self.log("[Info] Drift Corrector launched successfully.")
                except Exception as e:
                    self.log(f"[Error] Failed to launch Drift Corrector: {e}")
                    import traceback
                    self.log(traceback.format_exc())
                    QMessageBox.critical(self, "Error", f"Failed to launch Drift Corrector:\n{e}")
            return
        elif tool_config["id"] == "hc_map":
            if self.active_hc_map_window is not None:
                self.active_hc_map_window.show()
                self.active_hc_map_window.raise_()
                self.active_hc_map_window.activateWindow()
                self.log("[Info] Brought active Local Hc Map window to focus.")
            else:
                self.log("[Info] Launching Local Hc Map in same process...")
                try:
                    from hc_map_tool import HcMapGUI
                    window = HcMapGUI(theme=self.current_theme)
                    window.setWindowIcon(get_magnetic_icon(self.current_theme))

                    # Intercept closeEvent to clear reference
                    orig_close = window.closeEvent
                    def custom_close(event):
                        orig_close(event)
                        if event.isAccepted():
                            self.active_hc_map_window = None
                            self.log("[Info] Local Hc Map window closed.")
                    window.closeEvent = custom_close

                    self.active_hc_map_window = window
                    window.show()
                    self.log("[Info] Local Hc Map launched successfully.")
                except Exception as e:
                    self.log(f"[Error] Failed to launch Local Hc Map: {e}")
                    import traceback
                    self.log(traceback.format_exc())
                    QMessageBox.critical(self, "Error", f"Failed to launch Local Hc Map:\n{e}")
            return
        elif tool_config["id"] == "dmdh_mapper":
            if self.active_dmdh_window is not None:
                self.active_dmdh_window.show()
                self.active_dmdh_window.raise_()
                self.active_dmdh_window.activateWindow()
                self.log("[Info] Brought active dM/dH Mapper window to focus.")
            else:
                self.log("[Info] Launching dM/dH Mapper in same process...")
                try:
                    from dmdh_tool import DmdhGUI
                    window = DmdhGUI(theme=self.current_theme)
                    window.setWindowIcon(get_magnetic_icon(self.current_theme))

                    # Intercept closeEvent to clear reference
                    orig_close = window.closeEvent
                    def custom_close(event):
                        orig_close(event)
                        if event.isAccepted():
                            self.active_dmdh_window = None
                            self.log("[Info] dM/dH Mapper window closed.")
                    window.closeEvent = custom_close

                    self.active_dmdh_window = window
                    window.show()
                    self.log("[Info] dM/dH Mapper launched successfully.")
                except Exception as e:
                    self.log(f"[Error] Failed to launch dM/dH Mapper: {e}")
                    import traceback
                    self.log(traceback.format_exc())
                    QMessageBox.critical(self, "Error", f"Failed to launch dM/dH Mapper:\n{e}")
            return

        if self.current_process is not None:
            QMessageBox.warning(
                self, "Process Running", 
                "A script is already running. Please terminate it or wait for it to complete."
            )
            return
            
        script_name = tool_config["script"]
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
        
        if not os.path.exists(script_path):
            QMessageBox.critical(
                self, "Error", 
                f"Could not locate the tool script:\n{script_path}"
            )
            return
            
        args = []
        # Prompt for parent directory if the tool takes it as argument
        if tool_config.get("prompt_directory"):
            selected_dir = QFileDialog.getExistingDirectory(
                self, f"Select Parent Directory for {tool_config['name']}",
                os.path.dirname(os.path.abspath(__file__))
            )
            if not selected_dir:
                self.log("[Info] Launch cancelled: no directory selected.")
                return
            args.append(selected_dir)
            
        # Propagate current theme setting
        args.extend(["--theme", self.current_theme])
        
        self.start_subprocess(tool_config, script_path, args)
        
    def start_subprocess(self, tool_config, script_path, args):
        """Initialize and launch the python script as a QProcess."""
        self.current_process = QProcess()
        
        # Merge stdout and stderr so we capture traceback lines as well
        self.current_process.setProcessChannelMode(QProcess.MergedChannels)
        
        # Set PYTHONUNBUFFERED=1 to ensure outputs flush in real-time
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.current_process.setProcessEnvironment(env)
        
        # Connect event handles
        self.current_process.readyReadStandardOutput.connect(self.read_process_output)
        self.current_process.finished.connect(self.process_finished)
        self.current_process.errorOccurred.connect(self.process_error)
        
        # Print status details
        self.console_output.clear()
        self.log(f"=== STARTING: {tool_config['name']} ===")
        self.log(f"Script: {tool_config['script']}")
        if len(args) > 2: # Has directory parameter (since theme adds 2 args)
            self.log(f"Target Directory: {args[0]}")
        self.log(f"Python: {sys.executable}")
        self.log("-" * 60 + "\n")
        
        # Launch using sys.executable to ensure matching virtual environment/dependencies
        cmd_args = [script_path] + args
        self.current_process.start(sys.executable, cmd_args)
        
        # Update UI state
        self.status_lbl.setText(f"Status: Running {tool_config['name']}...")
        self.btn_stop.setEnabled(True)
        self.btn_stop.setToolTip("Stop the currently running process")
        self.progress_bar.setRange(0, 0)  # Pulse style progress
        self.progress_bar.setVisible(True)
        
    def read_process_output(self):
        """Triggered when the process has new console stream data."""
        if self.current_process is None:
            return
        data = self.current_process.readAllStandardOutput()
        text = bytes(data).decode("utf-8", errors="replace")
        
        self.console_output.moveCursor(QTextCursor.End)
        self.console_output.insertPlainText(text)
        self.console_output.moveCursor(QTextCursor.End)
        
    def process_finished(self, exit_code, exit_status):
        """Triggered when the process terminates."""
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("No process is currently running")
        self.progress_bar.setVisible(False)
        
        if exit_status == QProcess.NormalExit and exit_code == 0:
            self.status_lbl.setText("Status: Completed successfully")
            self.log(f"\n" + "-" * 60)
            self.log(f"=== PROCESS COMPLETED SUCCESSFULLY ===")
        else:
            self.status_lbl.setText("Status: Process stopped or crashed")
            self.log(f"\n" + "-" * 60)
            self.log(f"=== PROCESS TERMINATED OR CRASHED (Exit Code: {exit_code}) ===")
            
        self.current_process = None
        
    def process_error(self, error):
        """Triggered if QProcess fails to execute the target script."""
        error_msgs = {
            QProcess.FailedToStart: "The process failed to start. Make sure Python is in your PATH.",
            QProcess.Crashed: "The process crashed or failed during execution.",
            QProcess.Timedout: "The process timed out.",
            QProcess.WriteError: "An error occurred when writing to the process.",
            QProcess.ReadError: "An error occurred when reading from the process.",
            QProcess.UnknownError: "An unknown process execution error occurred."
        }
        msg = error_msgs.get(error, f"Process error code: {error}")
        self.log(f"\n[ERROR] {msg}")
        
    def closeEvent(self, event):
        """Prompt user to stop any active process when closing launcher window."""
        if self.current_process is not None:
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "An analysis process is currently running.\n"
                "Are you sure you want to stop it and exit the suite?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.current_process.kill()
                self.current_process.waitForFinished(1000)
                if self.active_looper_window is not None:
                    self.active_looper_window.close()
                if self.active_batch_window is not None:
                    self.active_batch_window.close()
                if self.active_vector_window is not None:
                    self.active_vector_window.close()
                if self.active_drift_window is not None:
                    self.active_drift_window.close()
                if self.active_hc_map_window is not None:
                    self.active_hc_map_window.close()
                if self.active_dmdh_window is not None:
                    self.active_dmdh_window.close()
                event.accept()
            else:
                event.ignore()
        else:
            if self.active_looper_window is not None:
                self.active_looper_window.close()
            if self.active_batch_window is not None:
                self.active_batch_window.close()
            if self.active_vector_window is not None:
                self.active_vector_window.close()
            if self.active_drift_window is not None:
                self.active_drift_window.close()
            if self.active_hc_map_window is not None:
                self.active_hc_map_window.close()
            if self.active_dmdh_window is not None:
                self.active_dmdh_window.close()
            event.accept()


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main():
    import os
    if os.environ.get("DIAGNOSTIC_LAUNCH") == "1":
        import traceback
        log_path = os.path.abspath("launch_diagnosis.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Starting compiled launcher diagnostic...\n")
            f.flush()
            tools = [
                {"id": "kerr_looper", "name": "Kerr MOKE Looper", "module": "kerr_looper_AG"},
                {"id": "batch_processor", "name": "Batch Loop Processor", "module": "batch_processor"},
                {"id": "vector_analysis", "name": "Vector Maps", "module": "vector_analysis"},
                {"id": "drift_corrector", "name": "Drift Corrector", "module": "drift_corrector"},
                {"id": "hc_map", "name": "Local Hc Map", "module": "hc_map_tool"},
                {"id": "dmdh_mapper", "name": "dM/dH Mapper", "module": "dmdh_tool"}
            ]
            success_count = 0
            for t in tools:
                f.write(f"\nTesting import for: {t['name']} ({t['module']})...\n")
                f.flush()
                try:
                    # Import the module
                    mod = __import__(t['module'])
                    f.write(f"-> SUCCESS: Imported {t['module']}\n")
                    f.flush()
                    success_count += 1
                except Exception as e:
                    f.write(f"-> EXCEPTION importing {t['module']}: {e}\n")
                    f.write(traceback.format_exc() + "\n")
                    f.flush()
            f.write(f"\nResult: {success_count}/{len(tools)} sub-packages imported successfully.\n")
            f.flush()
        sys.exit(0 if success_count == len(tools) else 1)

    # Enable high DPI scaling if supported (must be set before QApplication creation)
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
    app = QApplication(sys.argv)
    launcher = SuiteLauncherWindow()
    launcher.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
