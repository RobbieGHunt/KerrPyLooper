# -*- coding: utf-8 -*-
"""
Cross-platform Nuitka build script for KerrPyLooper.
Compiles `suite_launcher.py` and all 6 analysis sub-tools into a minimal
standalone or onefile executable for Windows (.exe), macOS (.app), and Linux.

Usage:
    python build_nuitka.py              # default: --onefile
    python build_nuitka.py --onefile
    python build_nuitka.py --standalone
"""

import argparse
import os
import subprocess
import sys


INCLUDED_MODULES = [
    "kerr_looper_AG",
    "batch_processor",
    "vector_analysis",
    "drift_corrector",
    "hc_map_tool",
    "dmdh_tool",
    "focus_corrector",
    "gui_styles",
    "shared_utils.app_paths",
    "shared_utils.image_processing",
]

EXCLUDED_IMPORTS = [
    "pytest",
    "tkinter",
    "_tkinter",
    "unittest",
    "pydoc",
    "doctest",
    "IPython",
    "ipykernel",
    "jupyter",
    "PyQt5.QtWebEngine",
    "PyQt5.QtWebEngineCore",
    "PyQt5.QtWebEngineWidgets",
    "PyQt5.QtQml",
    "PyQt5.QtQuick",
    "PyQt5.QtQuickWidgets",
    "PyQt5.QtMultimedia",
    "PyQt5.QtMultimediaWidgets",
    "PyQt5.QtSql",
    "PyQt5.QtTest",
    "PyQt5.QtBluetooth",
    "PyQt5.QtNfc",
    "PyQt5.QtPositioning",
    "PyQt5.QtLocation",
    "PyQt5.QtSensors",
    "PyQt5.QtSerialPort",
    "PyQt5.QtWebSockets",
    "PyQt5.QtXmlPatterns",
]


def build_command(mode: str = "onefile") -> list[str]:
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    entry_script = os.path.join(repo_dir, "suite_launcher.py")
    lens_path = os.path.join(repo_dir, "lens.json")

    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        f"--{mode}",
        "--enable-plugin=pyqt5",
        "--assume-yes-for-downloads",
        "--output-dir=dist",
        "--output-filename=KerrPyLooper",
    ]

    if sys.platform.startswith("win"):
        cmd.append("--windows-console-mode=disable")
        icon_path = os.path.join(repo_dir, "domain_icon.ico")
        if os.path.exists(icon_path):
            cmd.append(f"--windows-icon-from-ico={icon_path}")
    elif sys.platform == "darwin":
        cmd.append("--macos-create-app-bundle")
        cmd.append("--macos-app-name=KerrPyLooper")

    for mod in INCLUDED_MODULES:
        cmd.append(f"--include-module={mod}")

    for exc in EXCLUDED_IMPORTS:
        cmd.append(f"--nofollow-import-to={exc}")

    if os.path.exists(lens_path):
        cmd.append(f"--include-data-files={lens_path}=lens.json")

    cmd.append(entry_script)
    return cmd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build KerrPyLooper with Nuitka")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--onefile",
        action="store_true",
         default=True,
        help="Build a single-file executable (default)",
    )
    group.add_argument(
        "--standalone",
        action="store_true",
        help="Build a standalone folder distribution",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Nuitka command without executing it",
    )
    args = parser.parse_args(argv)

    mode = "standalone" if args.standalone else "onefile"
    cmd = build_command(mode=mode)

    print("Nuitka build command:")
    print("  " + " \\\n    ".join(cmd))

    if args.dry_run:
        return 0

    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
