"""Manage the detached Streamlit process.

Persists a tiny JSON ``ui.pid`` file in the data dir so the CLI can stop
or restart the UI from any terminal -- no PID guessing.

Windows-specific bits:
  * Spawn with ``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`` so the
    child survives parent shutdown and Ctrl+C in the terminal that
    started it.
  * Stop via ``CTRL_BREAK_EVENT`` (which the child *can* receive because
    it's its own process group), with a force-kill fallback after a
    grace period.

POSIX: simpler -- ``start_new_session=True`` and ``SIGTERM``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PID_FILENAME = "ui.pid"


@dataclass
class UiProcessInfo:
    pid: int
    port: int
    host: str
    started_at: float  # epoch seconds


def pid_file_path(data_dir: Path) -> Path:
    return Path(data_dir) / PID_FILENAME


def write_pid_file(data_dir: Path, info: UiProcessInfo) -> Path:
    p = pid_file_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(info), indent=2), encoding="utf-8")
    return p


def read_pid_file(data_dir: Path) -> UiProcessInfo | None:
    p = pid_file_path(data_dir)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return UiProcessInfo(**data)
    except Exception:
        return None


def clear_pid_file(data_dir: Path) -> None:
    p = pid_file_path(data_dir)
    if p.is_file():
        p.unlink()


def is_alive(pid: int) -> bool:
    """Return True iff a process with this pid exists."""
    try:
        import psutil
    except ImportError:
        # Best-effort fallback: try os.kill(pid, 0) -- raises on dead pid.
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    return psutil.pid_exists(pid) and psutil.Process(pid).is_running()


def spawn_detached(
    args: list[str],
    env: dict | None = None,
    log_path: Path | None = None,
) -> int:
    """Spawn `args` fully detached. Returns the new PID.

    On Windows we use DETACHED_PROCESS so the child has no console and
    won't die when its launcher's terminal closes. stdout/stderr go to
    `log_path` if provided.
    """
    is_windows = sys.platform == "win32"
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("a", buffering=1, encoding="utf-8")
    else:
        log_fh = subprocess.DEVNULL  # type: ignore[assignment]
    creationflags = 0
    if is_windows:
        creationflags = (
            subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
            | subprocess.CREATE_NEW_PROCESS_GROUP
        )
        proc = subprocess.Popen(
            args,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            close_fds=True,
        )
    else:
        proc = subprocess.Popen(
            args,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    return proc.pid


def graceful_stop(pid: int, timeout: float = 5.0) -> bool:
    """Try a clean shutdown, then force-kill. Returns True iff process is dead."""
    if not is_alive(pid):
        return True
    is_windows = sys.platform == "win32"
    try:
        if is_windows:
            os.kill(pid, signal.CTRL_BREAK_EVENT)
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError:
        pass

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_alive(pid):
            return True
        time.sleep(0.1)

    # Force-kill.
    try:
        if is_windows:
            # taskkill /T kills the whole tree (streamlit may spawn helpers).
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        else:
            os.kill(pid, signal.SIGKILL)
    except OSError:
        pass

    deadline = time.time() + 3.0
    while time.time() < deadline:
        if not is_alive(pid):
            return True
        time.sleep(0.1)
    return not is_alive(pid)
