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
import os
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QSplitter, QTextEdit, QMessageBox,
    QFileDialog, QProgressBar, QGraphicsDropShadowEffect
)
from PyQt5.QtGui import QFont, QColor, QTextCursor, QPainter, QPainterPath, QPen
from PyQt5.QtCore import Qt, QProcess, QProcessEnvironment

from gui_styles import apply_theme

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
        "prompt_directory": True,
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
        if self.theme == "dark":
            line_color = QColor("#6366f1")  # Indigo accent
            grid_color = QColor("#334155")  # Slate 700
        else:
            line_color = QColor("#4f46e5")  # Indigo accent
            grid_color = QColor("#cbd5e1")  # Slate 300
            
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
# TOOL CARD COMPONENT
# ==============================================================================
class ToolCardWidget(QFrame):
    def __init__(self, tool_config, on_launch_callback, parent=None):
        super().__init__(parent)
        self.setObjectName("ToolCard")
        self.config = tool_config
        self.loop_widget = None
        
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


# ==============================================================================
# MAIN LAUNCHER WINDOW
# ==============================================================================
class SuiteLauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KerrPyLooper Analysis Suite")
        self.setMinimumSize(950, 700)
        self.current_process = None
        self.current_theme = "dark"
        self.cards = []
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
        
        # Theme toggle button
        self.btn_theme_toggle = QPushButton("☀️ Light Mode")
        self.btn_theme_toggle.setObjectName("ThemeToggleButton")
        self.btn_theme_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_theme_toggle.clicked.connect(self.toggle_theme)
        header_layout.addWidget(self.btn_theme_toggle, alignment=Qt.AlignVCenter)
        
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
        
    def toggle_theme(self):
        if self.current_theme == "dark":
            self.current_theme = "light"
            self.btn_theme_toggle.setText("🌙 Dark Mode")
        else:
            self.current_theme = "dark"
            self.btn_theme_toggle.setText("☀️ Light Mode")
            
        apply_theme(self, self.current_theme)
        for card in self.cards:
            card.set_theme(self.current_theme)
            
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
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main():
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
