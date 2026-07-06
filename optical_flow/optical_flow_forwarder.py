import sys
import threading
import subprocess
from pathlib import Path
import re
import time


OPTICAL_FLOW_LINE_PATTERN = re.compile(r"^(?P<key>[a-zA-Z_]+)=(?P<value>[^\s]+)$")

class OpticalFlowForwarder:
    """Launch optical_flow_estimator.py for image/flow visualization only."""

    def __init__(
        self,
        altitude_m: float,
        topic: str,
        hfov: float,
        fps: float,
        display: bool,
        report_every: int,
        sensor_id: int,
    ) -> None:
        self._altitude_m = float(altitude_m)
        self._topic = topic
        self._hfov = float(hfov)
        self._fps = float(fps)
        self._display = bool(display)
        self._report_every = max(1, int(report_every))
        self._sensor_id = max(0, min(255, int(sensor_id)))

        self._process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._last_flow_wall_time = 0.0
        self._last_flow_quality = 0
        self._last_flow_comp_m_x = 0.0
        self._last_flow_comp_m_y = 0.0
        self._valid_flow_count = 0

    def start(self) -> None:
        package_dir = Path(__file__).resolve().parent
        workspace_root = package_dir.parent

        cmd = [
            sys.executable,
            "-m",
            "optical_flow.optical_flow_estimator",
            "--ros-image-topic",
            self._topic,
            "--altitude",
            f"{self._altitude_m}",
            "--hfov",
            f"{self._hfov}",
            "--report-every",
            str(self._report_every),
            "--sensor-id",
            str(self._sensor_id),
        ]
        if self._fps > 0:
            cmd.extend(["--fps", f"{self._fps}"])
        if self._display:
            cmd.append("--display")

        print("[*] Starting optical flow process:")
        print("    " + " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            cwd=str(workspace_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)

    def _reader_loop(self) -> None:
        if self._process is None or self._process.stdout is None:
            return

        for raw_line in self._process.stdout:
            if self._stop_event.is_set():
                break

            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("OPTICAL_FLOW(100) "):
                payload = self._parse_optical_flow_line(line)
                if payload is not None:
                    with self._state_lock:
                        self._last_flow_wall_time = time.time()
                        self._last_flow_quality = int(payload["quality"])
                        self._last_flow_comp_m_x = float(payload["flow_comp_m_x"])
                        self._last_flow_comp_m_y = float(payload["flow_comp_m_y"])
                        self._valid_flow_count += 1
                continue

            # print(f"[optical-flow] {line}")

        if self._process.poll() not in (None, 0):
            print(f"[optical-flow] Process exited with code {self._process.returncode}")

    def flow_health(self, max_stale_s: float, min_quality: int) -> tuple[bool, float, int, int]:
        with self._state_lock:
            count = self._valid_flow_count
            quality = self._last_flow_quality
            last_time = self._last_flow_wall_time
        if count <= 0 or last_time <= 0.0:
            return False, 9999.0, quality, count
        age_s = time.time() - last_time
        healthy = age_s <= max_stale_s and quality >= min_quality
        return healthy, age_s, quality, count

    def latest_flow_velocity_mps(self) -> tuple[float, float] | None:
        with self._state_lock:
            count = self._valid_flow_count
            comp_x = self._last_flow_comp_m_x
            comp_y = self._last_flow_comp_m_y
        if count <= 0:
            return None
        return comp_x * self._fps, comp_y * self._fps

    @staticmethod
    def _parse_optical_flow_line(line: str) -> dict[str, float | int] | None:
        result: dict[str, float | int] = {}
        fields = line.split()[1:]
        for field in fields:
            match = OPTICAL_FLOW_LINE_PATTERN.match(field)
            if not match:
                return None
            key = match.group("key")
            value = match.group("value")
            if key in {"time_usec", "sensor_id", "flow_x", "flow_y", "quality"}:
                result[key] = int(value)
            elif key in {"flow_comp_m_x", "flow_comp_m_y", "ground_distance"}:
                result[key] = float(value)
            else:
                return None

        required = {
            "time_usec",
            "sensor_id",
            "flow_x",
            "flow_y",
            "flow_comp_m_x",
            "flow_comp_m_y",
            "quality",
            "ground_distance",
        }
        if not required.issubset(result):
            return None
        return result