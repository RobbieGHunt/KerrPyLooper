# -*- coding: utf-8 -*-
"""
Application path and frozen-environment helpers for KerrPyLooper.
Supports Nuitka (__compiled__, sys.frozen) and PyInstaller (sys.frozen, sys._MEIPASS).
"""

import os
import sys


def is_frozen() -> bool:
    """Return True when running inside a compiled/frozen binary (Nuitka or PyInstaller)."""
    if getattr(sys, "frozen", False):
        return True
    if hasattr(sys, "_MEIPASS"):
        return True
    if "__compiled__" in globals():
        return True
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and hasattr(main_mod, "__compiled__"):
        return True
    return False


def get_user_config_dir(app_name: str = "KerrPyLooper") -> str:
    """Return the platform-specific per-user configuration directory."""
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, app_name)
    elif sys.platform == "darwin":
        return os.path.join(
            os.path.expanduser("~"), "Library", "Application Support", app_name
        )
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
            os.path.expanduser("~"), ".config"
        )
        return os.path.join(base, app_name)


def get_lens_config_path() -> str:
    """
    Resolve the path to `lens.json`.
    - If `./lens.json` already exists in the current working directory, use it.
    - If not frozen and the current working directory is writable, use `./lens.json`.
    - Otherwise (e.g. frozen binary or read-only working directory), fall back to
      the per-user config directory (`%APPDATA%/KerrPyLooper/lens.json` on Windows,
      `~/Library/Application Support/KerrPyLooper/lens.json` on macOS,
      `~/.config/KerrPyLooper/lens.json` on Linux).
    """
    local_path = os.path.abspath("lens.json")
    if os.path.exists(local_path):
        return local_path

    if not is_frozen() and os.access(os.getcwd(), os.W_OK):
        return local_path

    cfg_dir = get_user_config_dir("KerrPyLooper")
    try:
        os.makedirs(cfg_dir, exist_ok=True)
    except OSError:
        pass
    return os.path.join(cfg_dir, "lens.json")
