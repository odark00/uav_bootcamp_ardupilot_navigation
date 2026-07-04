"""
MAVLink GUIDED circle mission + AirSim camera stream (non-blocking, high FPS).
Works with ArduPilot SITL, AirSim SITL, and real drones.
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys
import threading
from pymavlink import mavutil
import time


OPTICAL_FLOW_LINE_PATTERN = re.compile(r"^(?P<key>[a-zA-Z_]+)=(?P<value>[^\s]+)$")


def init_connections():
    print("[+] Connecting MAVLink...")
    mav = mavutil.mavlink_connection("udpin:0.0.0.0:14550")
    mav.wait_heartbeat()
    print("[+] MAVLink connected")

    return mav


# MAVLINK HELPERS
def get_altitude(mav):
    msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2)
    if msg:
        return msg.relative_alt / 1000.0
    return 0.0


def is_armed(mav):
    # Drain buffered messages so we get the latest heartbeat
    msg = None
    for _ in range(20):
        m = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=0.5)
        if m:
            msg = m
            break
    if msg:
        return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    return False


def wait_for_ekf(mav, timeout=60):
    """Wait for GPS fix and EKF home position before arming."""
    print("[*] Waiting for GPS + EKF convergence...")
    deadline = time.time() + timeout
    gps_ok = False
    home_ok = False
    ekf_ok = False
    while time.time() < deadline:
        msg = mav.recv_match(
            type=['GPS_RAW_INT', 'HOME_POSITION', 'STATUSTEXT', 'EKF_STATUS_REPORT'],
            blocking=True, timeout=1
        )
        if msg is None:
            continue
        t = msg.get_type()
        if t == 'GPS_RAW_INT' and msg.fix_type >= 3:
            if not gps_ok:
                print(f"[+] GPS fix ({msg.fix_type}D, sats={msg.satellites_visible})")
            gps_ok = True
        elif t == 'HOME_POSITION':
            home_ok = True
            print("[+] Home position received")
        elif t == 'EKF_STATUS_REPORT':
            # Horizontal absolute position means EKF is fused and navigating.
            # This is often the most reliable readiness signal in SITL.
            if msg.flags & mavutil.mavlink.EKF_POS_HORIZ_ABS:
                if not ekf_ok:
                    print("[+] EKF_STATUS_REPORT: horizontal position is valid")
                ekf_ok = True
        elif t == 'STATUSTEXT' and 'origin set' in msg.text.lower():
            home_ok = True
            print(f"[+] EKF: {msg.text.strip()}")
        elif t == 'STATUSTEXT' and 'using gps' in msg.text.lower():
            if not ekf_ok:
                print(f"[+] EKF: {msg.text.strip()}")
            ekf_ok = True

        if gps_ok and (home_ok or ekf_ok):
            print("[+] EKF ready")
            return True
    print("[!] EKF wait timeout - arming anyway (ARMING_CHECK=0 will override)")
    return False


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

    def start(self) -> None:
        root = Path(__file__).resolve().parent
        script_path = root / "optical_flow" / "optical_flow_estimator.py"

        cmd = [
            sys.executable,
            str(script_path),
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
            cwd=str(root),
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
                # Never forward OPTICAL_FLOW into SITL; this process is display-only.
                continue

            print(f"[flow] {line}")

        if self._process.poll() not in (None, 0):
            print(f"[flow] Process exited with code {self._process.returncode}")

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

    def _send_optical_flow(self, payload: dict[str, float | int]) -> None:
        try:
            with self._send_lock:
                print(f"[flow] Forwarding to MAVLink: {payload}")
                self._mav.mav.optical_flow_send(
                    int(payload["time_usec"]),
                    int(payload["sensor_id"]),
                    int(payload["flow_x"]),
                    int(payload["flow_y"]),
                    float(payload["flow_comp_m_x"]),
                    float(payload["flow_comp_m_y"]),
                    int(payload["quality"]),
                    float(payload["ground_distance"]),
                )
        except Exception as exc:
            print(f"[flow] MAVLink forward error: {exc}")

    def wait_until_ready(self, min_messages: int = 5, timeout_s: float = 10.0) -> bool:
        """Wait until enough recent optical-flow packets were forwarded."""
        threshold = max(1, int(min_messages))
        deadline = time.time() + max(0.5, float(timeout_s))

        while time.time() < deadline:
            with self._send_lock:
                count = self._valid_flow_count
                age = time.time() - self._last_flow_time if self._last_flow_time > 0 else 9999.0
            if count >= threshold and age < 2.0:
                print(f"[+] Optical flow ready ({count} messages forwarded)")
                return True
            time.sleep(0.1)

        with self._send_lock:
            count = self._valid_flow_count
        print(f"[!] Optical flow readiness timeout ({count}/{threshold} messages)")
        return False


def set_mode(mav, mode_name, timeout=5):
    mode_mapping = mav.mode_mapping()
    if mode_name not in mode_mapping:
        print(f"[!] Unknown mode: {mode_name}")
        return False

    mode_id = mode_mapping[mode_name]

    mav.mav.set_mode_send(
        mav.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )

    start = time.time()
    while time.time() - start < timeout:
        msg = mav.recv_match(type='HEARTBEAT', blocking=False)
        if msg and msg.custom_mode == mode_id:
            print(f"[+] Mode: {mode_name}")
            return True
        time.sleep(0.05)

    print(f"[~] Mode {mode_name} sent (no confirmation)")
    return True


def arm(mav, force=False, timeout=15):
    print("[*] Arming...")

    mav.mav.param_set_send(
        mav.target_system, mav.target_component,
        b'ARMING_CHECK', 0,
        mavutil.mavlink.MAV_PARAM_TYPE_INT32
    )
    time.sleep(2.0)  # give ArduPilot time to apply the param

    param2 = 21196 if force else 0
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, param2, 0, 0, 0, 0, 0
    )

    start = time.time()
    while time.time() - start < timeout:
        if is_armed(mav):
            print("[+] Armed")
            return True
        time.sleep(0.1)

    print("[!] Arming failed")
    return False



def takeoff(mav, altitude_m):
    print(f"[*] Takeoff to {altitude_m}m...")

    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude_m
    )

    while True:
        alt = get_altitude(mav)
        print(f"    Alt: {alt:.1f}m / {altitude_m}m", end="\r")
        if alt >= altitude_m * 0.92:
            print(f"\n[+] Reached {alt:.1f}m")
            break
        time.sleep(0.1)


def land(mav):
    print("[*] Landing...")
    set_mode(mav, "LAND")

    while True:
        alt = get_altitude(mav)
        print(f"    Alt: {alt:.2f}m", end="\r")
        if alt < 0.3:
            print("\n[+] Landed")
            break
        time.sleep(0.1)



def send_velocity(mav, vx=0.0, vy=0.0, vz=0.0, yaw=0.0, yaw_rate=0.0):
    """
    Send one velocity + yaw-rate command in body NED frame.
    vx: forward (m/s)   vy: right (m/s)   vz: down (m/s, negative = climb)
    yaw_rate: rad/s, positive = clockwise from above
    Call at ~20 Hz for continuous motion.
    """
    mav.mav.send(
        mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
            0,
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b010111000111,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            yaw,
            yaw_rate,
        )
    )

def mission_guided(mav, altitude_m=10, radius=25, speed=3):
    if not wait_for_ekf(mav, timeout=120):
        print("[~] Continuing despite EKF timeout (force arm path enabled)")

    set_mode(mav, "GUIDED")

    if not arm(mav, force=True):
        return

    takeoff(mav, altitude_m)

    yaw_rate=speed/radius
    duration = 60
    end = time.time() + duration
    while True: #time.time() < end:
        print("send_velocity")
        send_velocity(mav, vx=speed, vy=0, vz=0, yaw=0, yaw_rate=yaw_rate)
        time.sleep(0.05)

    land(mav)
    print("[+] Mission complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run guided mission with optional optical-flow display (no SITL forwarding)."
    )
    parser.add_argument("--altitude", type=float, default=10.0, help="Mission takeoff altitude in meters")
    parser.add_argument(
        "--flow-topic",
        default="/camera/image",
        help="ROS image topic consumed by optical flow estimator (default: camera/image)",
    )
    parser.add_argument("--flow-hfov", type=float, default=84.0, help="Camera HFOV used by estimator")
    parser.add_argument("--flow-fps", type=float, default=30.0, help="Estimator FPS")
    parser.add_argument(
        "--flow-report-every",
        type=int,
        default=1,
        help="Print one optical-flow report every N frames",
    )
    parser.add_argument(
        "--flow-sensor-id",
        type=int,
        default=0,
        help="MAVLink optical flow sensor ID (0..255)",
    )
    parser.add_argument(
        "--no-flow-display",
        action="store_true",
        help="Disable optical flow debug display window (display enabled by default)",
    )
    parser.add_argument(
        "--no-flow",
        action="store_true",
        help="Disable optical flow process launch/forwarding",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mav = init_connections()

    flow_forwarder: OpticalFlowForwarder | None = None
    if not args.no_flow:
        flow_forwarder = OpticalFlowForwarder(
            altitude_m=args.altitude,
            topic=args.flow_topic,
            hfov=args.flow_hfov,
            fps=args.flow_fps,
            display=not args.no_flow_display,
            report_every=args.flow_report_every,
            sensor_id=args.flow_sensor_id,
        )
        flow_forwarder.start()

    try:
        mission_guided(mav, altitude_m=args.altitude)
    finally:
        if flow_forwarder is not None:
            flow_forwarder.stop()

    print("[+] Mission finished. Press Q to close camera.")
