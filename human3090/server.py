"""Manages llama.cpp server lifecycle with smart restart logic."""

import json
import os
import signal
import subprocess
import time
import urllib.request
from urllib.error import HTTPError, URLError


class ServerManager:
    """Manages the llama.cpp server, restarting only when config changes."""

    def __init__(
        self,
        server_path: str = "/code/llama.cpp/build/bin/llama-server",
        port: int = 8083,
        startup_timeout: int = 600,
    ):
        self.server_path = server_path
        self.port = port
        self.startup_timeout = startup_timeout
        self._process: subprocess.Popen | None = None
        self._current_key: tuple | None = None

    def ensure_server(self, job) -> None:
        """Start or restart the server if the job needs different settings."""
        key = job.server_key()
        if self._process and self._current_key == key:
            if self._health_check():
                return
            print("  Server died, restarting...")
            self._stop()

        if self._process:
            print("  Model/config changed, restarting server...")
            self._stop()

        self._start(job)
        self._current_key = key

    def _start(self, job) -> None:
        """Start llama-server with the given config."""
        if not os.path.exists(self.server_path):
            raise FileNotFoundError(f"Server not found at {self.server_path}")

        # Kill anything already on our port (stale server, manual run, etc.)
        self._kill_port_owner()

        cmd = [
            self.server_path,
            "-m", job.model,
            "--port", str(self.port),
            "-ngl", str(job.gpu_layers),
            "-c", str(job.context_size),
        ]
        print(f"  Starting server: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self._wait_until_ready()

    def _kill_port_owner(self) -> None:
        """Kill any process listening on our port."""
        try:
            result = subprocess.run(
                ["fuser", f"{self.port}/tcp"],
                capture_output=True, text=True
            )
            pids = result.stdout.strip().split()
            if pids:
                for pid in pids:
                    pid = int(pid)
                    print(f"  Killing existing process on port {self.port}: PID {pid}")
                    os.kill(pid, signal.SIGTERM)
                # Wait for port to be freed
                time.sleep(3)
        except (FileNotFoundError, ValueError, ProcessLookupError):
            pass

    def _stop(self) -> None:
        """Gracefully stop the running server."""
        if not self._process:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process = None
        self._current_key = None
        # Give GPU memory time to free
        time.sleep(2)

    def _health_check(self) -> bool:
        """Return True if server is healthy and model is loaded."""
        try:
            with urllib.request.urlopen(
                f"http://localhost:{self.port}/health", timeout=5
            ) as resp:
                body = json.loads(resp.read())
                return body.get("status") == "ok"
        except (URLError, HTTPError, OSError, json.JSONDecodeError):
            return False

    def _wait_until_ready(self) -> None:
        """Block until the server's model is fully loaded."""
        start_time = time.time()
        while time.time() - start_time < self.startup_timeout:
            # Check if process died during startup
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"Server exited during startup (exit code {self._process.returncode})"
                )

            if self._health_check():
                elapsed = time.time() - start_time
                print(f"  Server ready ({elapsed:.0f}s)")
                return

            time.sleep(2)

        raise TimeoutError(f"Server failed to start within {self.startup_timeout}s")

    def shutdown(self) -> None:
        """Stop the server (call at end of queue)."""
        self._stop()
