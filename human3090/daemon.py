"""Start/stop the bench_runner daemon (background queue watcher)."""

import os
import signal
import subprocess
import sys
from pathlib import Path

PIDFILE = Path("jobs/.daemon.pid")
LOGFILE = Path("jobs/daemon.log")


def _read_pid() -> int | None:
    if not PIDFILE.exists():
        return None
    try:
        pid = int(PIDFILE.read_text().strip())
        os.kill(pid, 0)  # check if alive
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        PIDFILE.unlink(missing_ok=True)
        return None


def start():
    """Start the bench_runner daemon in watch mode."""
    existing = _read_pid()
    if existing:
        print(f"Daemon already running (PID {existing})")
        sys.exit(1)

    # Ensure jobs directory exists
    Path("jobs/queued").mkdir(parents=True, exist_ok=True)

    log = open(LOGFILE, "a")
    proc = subprocess.Popen(
        [sys.executable, "-m", "human3090.bench_runner", "--queue", "--watch"],
        stdout=log, stderr=log,
        start_new_session=True,
    )
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(proc.pid))
    print(f"Daemon started (PID {proc.pid})")
    print(f"  Log: {LOGFILE.resolve()}")
    print(f"  Drop job files into jobs/queued/ to run benchmarks")


def stop():
    """Stop the running bench_runner daemon."""
    pid = _read_pid()
    if not pid:
        print("No daemon running")
        sys.exit(1)

    print(f"Stopping daemon (PID {pid})...")
    os.kill(pid, signal.SIGTERM)

    # Wait for graceful shutdown
    import time
    for _ in range(30):
        try:
            os.kill(pid, 0)
            time.sleep(1)
        except ProcessLookupError:
            break
    else:
        print("Force killing...")
        os.kill(pid, signal.SIGKILL)

    PIDFILE.unlink(missing_ok=True)
    print("Daemon stopped")
