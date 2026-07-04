"""
MAVLink GUIDED circle mission + AirSim camera stream (non-blocking, high FPS).
Works with ArduPilot SITL, AirSim SITL, and real drones.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
import threading
from pymavlink import mavutil
import time


OPTICAL_FLOW_LINE_PATTERN = re.compile(r"^(?P<key>[a-zA-Z_]+)=(?P<value>[^\s]+)$")


@dataclass
class GpsHealth:
    last_message_time: float = 0.0
    last_good_fix_time: float = 0.0
    fix_type: int = 0
    satellites_visible: int = 0

    def update(self, msg) -> None:
        now = time.time()
        self.last_message_time = now
        self.fix_type = int(msg.fix_type)
        self.satellites_visible = int(msg.satellites_visible)
        if self.fix_type >= 3:
            self.last_good_fix_time = now

    def is_healthy(self, now: float, max_stale_s: float) -> bool:
        if self.last_message_time <= 0.0:
            return False
        if now - self.last_message_time > max_stale_s:
            return False
        return self.fix_type >= 3


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
        self._state_lock = threading.Lock()
        self._last_flow_wall_time = 0.0
        self._last_flow_quality = 0
        self._last_flow_comp_m_x = 0.0
        self._last_flow_comp_m_y = 0.0
        self._valid_flow_count = 0

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
                payload = self._parse_optical_flow_line(line)
                if payload is not None:
                    with self._state_lock:
                        self._last_flow_wall_time = time.time()
                        self._last_flow_quality = int(payload["quality"])
                        self._last_flow_comp_m_x = float(payload["flow_comp_m_x"])
                        self._last_flow_comp_m_y = float(payload["flow_comp_m_y"])
                        self._valid_flow_count += 1
                continue

            print(f"[flow] {line}")

        if self._process.poll() not in (None, 0):
            print(f"[flow] Process exited with code {self._process.returncode}")

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


def _normalize_param_id(raw_param_id) -> str:
    if isinstance(raw_param_id, bytes):
        return raw_param_id.decode("ascii", errors="ignore").rstrip("\x00")
    return str(raw_param_id).rstrip("\x00")


def _read_param_value(mav, param_name: str, timeout_s: float = 1.5) -> float | None:
    deadline = time.time() + max(0.2, float(timeout_s))
    mav.mav.param_request_read_send(
        mav.target_system,
        mav.target_component,
        param_name.encode("ascii"),
        -1,
    )
    while time.time() < deadline:
        msg = mav.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.2)
        if msg is None:
            continue
        if _normalize_param_id(msg.param_id) == param_name:
            return float(msg.param_value)
    return None


def _set_param_with_confirm(
    mav,
    param_name: str,
    value: float,
    param_type: int,
    timeout_s: float = 2.0,
) -> bool:
    mav.mav.param_set_send(
        mav.target_system,
        mav.target_component,
        param_name.encode("ascii"),
        float(value),
        param_type,
    )

    confirmed = _read_param_value(mav, param_name, timeout_s=timeout_s)
    if confirmed is None:
        return False
    return abs(float(confirmed) - float(value)) < 0.01


def trigger_disable_gps(mav) -> bool:
    # Try multiple ArduPilot SITL parameter names for cross-version compatibility.
    candidates: list[tuple[str, float, int]] = [
        ("SIM_GPS_DISABLE", 1.0, mavutil.mavlink.MAV_PARAM_TYPE_INT8),
        ("SIM_GPS1_DISABLE", 1.0, mavutil.mavlink.MAV_PARAM_TYPE_INT8),
        ("SIM_GPS1_ENABLE", 0.0, mavutil.mavlink.MAV_PARAM_TYPE_INT8),
        ("GPS_TYPE", 0.0, mavutil.mavlink.MAV_PARAM_TYPE_INT8),
        ("GPS1_TYPE", 0.0, mavutil.mavlink.MAV_PARAM_TYPE_INT8),
    ]

    for param_name, value, param_type in candidates:
        print(f"[*] Trying GPS disable via {param_name}={value:g}")
        if _set_param_with_confirm(mav, param_name, value, param_type):
            print(f"[+] GPS disable trigger armed with {param_name}={value:g}")
            return True

    print("[!] Failed to disable GPS using known SITL parameters")
    return False


def _poll_health_messages(mav, gps_health: GpsHealth) -> None:
    while True:
        msg = mav.recv_match(type=['GPS_RAW_INT'], blocking=False)
        if msg is None:
            break
        if msg.get_type() == 'GPS_RAW_INT':
            gps_health.update(msg)


def mission_guided(
    mav,
    altitude_m=10.0,
    radius=25.0,
    speed=3.0,
    duration_s=120.0,
    gps_loss_timeout_s=1.5,
    fallback_speed=2.0,
    flow_forwarder: OpticalFlowForwarder | None = None,
    flow_max_stale_s=1.5,
    flow_min_quality=30,
    fallback_lateral_gain=0.8,
    fallback_max_vy=1.5,
    disable_gps_after_takeoff_s=15.0,
):
    if not wait_for_ekf(mav, timeout=120):
        print("[~] Continuing despite EKF timeout (force arm path enabled)")

    set_mode(mav, "GUIDED")

    if not arm(mav, force=True):
        return

    takeoff(mav, altitude_m)

    yaw_rate = speed / max(0.5, radius)
    gps_health = GpsHealth()
    mode = "CIRCLE_GPS"
    mission_end = time.time() + max(1.0, float(duration_s))
    next_status_at = 0.0
    gps_disable_triggered = False
    gps_disable_at = (
        time.time() + float(disable_gps_after_takeoff_s)
        if float(disable_gps_after_takeoff_s) >= 0.0
        else None
    )

    print(
        "[*] Mission started: GPS circle mode. "
        "If GPS is lost, switching to straight-forward vision-assisted mode."
    )

    while time.time() < mission_end:
        _poll_health_messages(mav, gps_health)
        now = time.time()

        if gps_disable_at is not None and (not gps_disable_triggered) and now >= gps_disable_at:
            print(f"[trigger] Disabling GPS at +{disable_gps_after_takeoff_s:.1f}s after takeoff")
            trigger_disable_gps(mav)
            gps_disable_triggered = True

        gps_ok = gps_health.is_healthy(now, max_stale_s=max(0.2, float(gps_loss_timeout_s)))
        target_mode = "CIRCLE_GPS" if gps_ok else "STRAIGHT_VISION"

        if target_mode != mode:
            mode = target_mode
            print(f"[mode] Switched to {mode}")

        if mode == "CIRCLE_GPS":
            cmd_vx = float(speed)
            cmd_vy = 0.0
            cmd_yaw_rate = yaw_rate
            flow_note = ""
        else:
            cmd_vx = float(fallback_speed)
            cmd_vy = 0.0
            cmd_yaw_rate = 0.0
            flow_note = "flow=unavailable"

            if flow_forwarder is not None:
                flow_ok, flow_age, flow_quality, flow_count = flow_forwarder.flow_health(
                    max_stale_s=float(flow_max_stale_s),
                    min_quality=int(flow_min_quality),
                )
                if flow_ok:
                    velocity = flow_forwarder.latest_flow_velocity_mps()
                    if velocity is not None:
                        _, flow_vy = velocity
                        raw_correction = -float(fallback_lateral_gain) * float(flow_vy)
                        cmd_vy = max(-float(fallback_max_vy), min(float(fallback_max_vy), raw_correction))
                    flow_note = (
                        f"flow=ok q={flow_quality} age={flow_age:.2f}s "
                        f"msgs={flow_count} vy_cmd={cmd_vy:.2f}"
                    )
                else:
                    flow_note = f"flow=bad q={flow_quality} age={flow_age:.2f}s msgs={flow_count}"

        send_velocity(mav, vx=cmd_vx, vy=cmd_vy, vz=0.0, yaw=0.0, yaw_rate=cmd_yaw_rate)

        if now >= next_status_at:
            print(
                f"[status] mode={mode} gps_fix={gps_health.fix_type} sats={gps_health.satellites_visible} "
                f"vx={cmd_vx:.2f} vy={cmd_vy:.2f} yaw_rate={cmd_yaw_rate:.3f} {flow_note}"
            )
            next_status_at = now + 1.0

        time.sleep(0.05)

    land(mav)
    print("[+] Mission complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run guided mission with optional optical-flow display (no SITL forwarding)."
    )
    parser.add_argument("--altitude", type=float, default=10.0, help="Mission takeoff altitude in meters")
    parser.add_argument("--duration", type=float, default=120.0, help="Mission duration in seconds")
    parser.add_argument("--circle-radius", type=float, default=25.0, help="GPS-mode circle radius in meters")
    parser.add_argument("--circle-speed", type=float, default=3.0, help="GPS-mode circle forward speed in m/s")
    parser.add_argument(
        "--gps-loss-timeout",
        type=float,
        default=1.5,
        help="Switch to fallback if GPS_RAW_INT is stale or degraded longer than this (seconds)",
    )
    parser.add_argument(
        "--disable-gps-after-takeoff",
        type=float,
        default=10.0,
        help="Disable GPS this many seconds after takeoff (set <0 to disable trigger)",
    )
    parser.add_argument(
        "--fallback-speed",
        type=float,
        default=2.0,
        help="Straight-forward speed in fallback mode (m/s)",
    )
    parser.add_argument(
        "--flow-min-quality",
        type=int,
        default=30,
        help="Minimum optical-flow quality (0..255) for lateral wind compensation",
    )
    parser.add_argument(
        "--flow-max-stale",
        type=float,
        default=1.5,
        help="Max accepted age of last optical-flow message in seconds",
    )
    parser.add_argument(
        "--fallback-lateral-gain",
        type=float,
        default=1.0,
        help="Gain from flow lateral velocity estimate to fallback vy correction",
    )
    parser.add_argument(
        "--fallback-max-vy",
        type=float,
        default=1.5,
        help="Clamp for lateral correction in fallback mode (m/s)",
    )
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
        mission_guided(
            mav,
            altitude_m=args.altitude,
            radius=args.circle_radius,
            speed=args.circle_speed,
            duration_s=args.duration,
            gps_loss_timeout_s=args.gps_loss_timeout,
            fallback_speed=args.fallback_speed,
            flow_forwarder=flow_forwarder,
            flow_max_stale_s=args.flow_max_stale,
            flow_min_quality=args.flow_min_quality,
            fallback_lateral_gain=args.fallback_lateral_gain,
            fallback_max_vy=args.fallback_max_vy,
            disable_gps_after_takeoff_s=args.disable_gps_after_takeoff,
        )
    finally:
        if flow_forwarder is not None:
            flow_forwarder.stop()

    print("[+] Mission finished. Press Q to close camera.")
