# -*- coding: utf-8 -*-
"""
Tests for cross-platform Nuitka packaging readiness:
- shared_utils.app_paths.is_frozen() and get_lens_config_path()
- suite_launcher.dispatch_tool() routing for all 6 tools in TOOL_REGISTRY
- SuiteLauncherWindow.start_subprocess() command construction (unfrozen & frozen)
- Subprocess CLI smoke tests (--tool <script> --help and DIAGNOSTIC_LAUNCH=1)
"""

import os
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
from PyQt5.QtWidgets import QApplication

import shared_utils.app_paths as app_paths
import suite_launcher


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_is_frozen_detection(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.delitem(app_paths.__dict__, "__compiled__", raising=False)
    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        monkeypatch.delattr(main_mod, "__compiled__", raising=False)

    assert app_paths.is_frozen() is False

    # 1. sys.frozen = True
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert app_paths.is_frozen() is True
    monkeypatch.delattr(sys, "frozen", raising=False)

    # 2. sys._MEIPASS
    monkeypatch.setattr(sys, "_MEIPASS", "/tmp/meipass", raising=False)
    assert app_paths.is_frozen() is True
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    # 3. __compiled__ in globals()
    monkeypatch.setitem(app_paths.__dict__, "__compiled__", True)
    assert app_paths.is_frozen() is True


def test_get_lens_config_path_existing_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    local_lens = tmp_path / "lens.json"
    local_lens.write_text('{"20": [100, 50]}', encoding="utf-8")

    # Even when frozen, an existing ./lens.json in CWD takes precedence
    monkeypatch.setattr(app_paths, "is_frozen", lambda: True)
    resolved = app_paths.get_lens_config_path()
    assert os.path.abspath(resolved) == os.path.abspath(str(local_lens))


def test_get_lens_config_path_frozen_fallback(tmp_path, monkeypatch):
    empty_cwd = tmp_path / "cwd"
    empty_cwd.mkdir()
    fake_appdata = tmp_path / "appdata"
    fake_appdata.mkdir()

    monkeypatch.chdir(empty_cwd)
    monkeypatch.setattr(app_paths, "is_frozen", lambda: True)
    monkeypatch.setattr(
        app_paths,
        "get_user_config_dir",
        lambda app_name="KerrPyLooper": str(fake_appdata / app_name),
    )

    resolved = app_paths.get_lens_config_path()
    expected = fake_appdata / "KerrPyLooper" / "lens.json"
    assert os.path.abspath(resolved) == os.path.abspath(str(expected))
    assert (fake_appdata / "KerrPyLooper").is_dir()


def test_dispatch_tool_noop_and_unknown(monkeypatch):
    orig_argv = list(sys.argv)
    try:
        assert suite_launcher.dispatch_tool(["suite_launcher.py", "--theme", "dark"]) is False

        with pytest.raises(ValueError, match="Unknown tool script"):
            suite_launcher.dispatch_tool(["suite_launcher.py", "--tool", "nonexistent_tool.py"])

        with pytest.raises(ValueError, match="Missing script name"):
            suite_launcher.dispatch_tool(["suite_launcher.py", "--tool"])
    finally:
        sys.argv = orig_argv


@pytest.mark.parametrize(
    "tool_entry",
    suite_launcher.TOOL_REGISTRY,
    ids=[t["id"] for t in suite_launcher.TOOL_REGISTRY],
)
def test_dispatch_tool_routes_all_registered_tools(tool_entry, monkeypatch):
    script_name = tool_entry["script"]
    mod_name = os.path.splitext(script_name)[0]
    mod = __import__(mod_name)

    called = {"count": 0, "argv": None}

    def fake_main():
        called["count"] += 1
        called["argv"] = list(sys.argv)

    monkeypatch.setattr(mod, "main", fake_main)
    orig_argv = list(sys.argv)
    try:
        dispatched = suite_launcher.dispatch_tool(
            ["KerrPyLooper", "--tool", script_name, "--theme", "charcoal"]
        )
        assert dispatched is True
        assert called["count"] == 1
        assert called["argv"] == ["KerrPyLooper", "--theme", "charcoal"]
    finally:
        sys.argv = orig_argv


def test_start_subprocess_unfrozen_and_frozen(qapp, monkeypatch):
    win = suite_launcher.SuiteLauncherWindow()
    try:
        tool_cfg = {
            "id": "custom_subproc",
            "name": "Batch Loop Processor",
            "script": "batch_processor.py",
        }
        script_path = os.path.abspath("batch_processor.py")
        started_calls = []

        class DummyProcess:
            MergedChannels = 1

            def __init__(self):
                self.readyReadStandardOutput = MagicMock()
                self.finished = MagicMock()
                self.errorOccurred = MagicMock()

            def setProcessChannelMode(self, mode):
                pass

            def setProcessEnvironment(self, env):
                pass

            def start(self, program, cmd_args):
                started_calls.append((program, list(cmd_args)))

        monkeypatch.setattr(suite_launcher, "QProcess", DummyProcess)

        # 1. Unfrozen mode: passes [script_path, ...]
        monkeypatch.setattr(suite_launcher, "is_frozen", lambda: False)
        win.start_subprocess(tool_cfg, script_path, ["--theme", "dark"])
        assert started_calls[-1] == (sys.executable, [script_path, "--theme", "dark"])

        # 2. Frozen mode: passes ["--tool", script_name, ...]
        monkeypatch.setattr(suite_launcher, "is_frozen", lambda: True)
        win.start_subprocess(tool_cfg, "/nonexistent/batch_processor.py", ["--theme", "charcoal"])
        assert started_calls[-1] == (
            sys.executable,
            ["--tool", "batch_processor.py", "--theme", "charcoal"],
        )
    finally:
        win.current_process = None
        win.close()


@pytest.mark.parametrize("script_name", ["batch_processor.py", "vector_analysis.py"])
def test_subprocess_cli_dispatch_help(script_name):
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    launcher_path = os.path.join(repo_dir, "suite_launcher.py")
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"

    proc = subprocess.run(
        [sys.executable, launcher_path, "--tool", script_name, "--help"],
        cwd=repo_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
    assert "usage:" in proc.stdout.lower()


def test_subprocess_diagnostic_launch(tmp_path):
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    launcher_path = os.path.join(repo_dir, "suite_launcher.py")
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["DIAGNOSTIC_LAUNCH"] = "1"

    proc = subprocess.run(
        [sys.executable, launcher_path],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
    diag_file = tmp_path / "launch_diagnosis.txt"
    assert diag_file.exists()
    content = diag_file.read_text(encoding="utf-8")
    assert "Result: 6/6 sub-packages imported successfully." in content
