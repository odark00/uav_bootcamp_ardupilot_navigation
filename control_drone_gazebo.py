"""
MAVLink GUIDED circle mission + AirSim camera stream (non-blocking, high FPS).
Works with ArduPilot SITL, AirSim SITL, and real drones.
"""

import argparse
from dataclasses import dataclass
import math
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


def request_message(mav, msg_id, hz):
    """Ask the autopilot to stream a message at the given rate (Hz)."""
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        msg_id, int(1e6 / hz), 0, 0, 0, 0, 0,
    )


# --- Barometric altitude (GPS-denied: reliable, unlike GLOBAL_POSITION_INT) -- #
def pressure_to_alt(press_hpa, ground_hpa):
    """International barometric formula -> metres above the ground reference."""
    return 44330.0 * (1.0 - (press_hpa / ground_hpa) ** (1.0 / 5.255))


def read_pressure(mav):
    """Newest absolute pressure (hPa) from SCALED_PRESSURE, or None if none buffered."""
    press = None
    while True:
        msg = mav.recv_match(type="SCALED_PRESSURE", blocking=False)
        if msg is None:
            break
        press = msg.press_abs
    return press


def capture_ground_pressure(mav, samples=10):
    """Average a few barometer readings to establish the ground reference (hPa)."""
    print("[*] Reading ground barometer reference...")
    readings = []
    deadline = time.time() + 10
    while len(readings) < samples and time.time() < deadline:
        press = read_pressure(mav)
        if press is not None:
            readings.append(press)
        time.sleep(0.1)
    if not readings:
        raise RuntimeError("No SCALED_PRESSURE received - is the baro streaming?")
    ground = sum(readings) / len(readings)
    print(f"[+] Ground pressure: {ground:.2f} hPa")
    return ground


# --- Magnetometer heading (compass-only while GPS is disabled) -------------- #
def read_heading_deg(mav):
    """Latest magnetometer heading in DEGREES (0=N, cw+) from VFR_HUD, or None.

    GPS is disabled, so VFR_HUD.heading is derived purely from the compass -- the
    vehicle's nose direction, matching the yaw convention of SET_ATTITUDE_TARGET.
    """
    heading = None
    while True:
        msg = mav.recv_match(type="VFR_HUD", blocking=False)
        if msg is None:
            break
        heading = msg.heading
    return None if heading is None else float(heading)


def wait_for_heading_deg(mav, timeout=10):
    """Block until the magnetometer heading is available; return it (degrees)."""
    print("[*] Reading magnetometer heading...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        heading = read_heading_deg(mav)
        if heading is not None:
            print(f"[+] Nose heading: {heading:.0f} deg (magnetometer)")
            return heading
        time.sleep(0.1)
    print("[!] No VFR_HUD heading - defaulting to 0 (North)")
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
    """Wait for the EKF to converge enough to arm.

    This is a GPS-DENIED sim (see config/gps_denied.parm: GPS_TYPE=0,
    EK3_SRC1_POSXY=0, EK3_SRC1_VELXY=0), so there is no GPS fix, no HOME_POSITION,
    and the EKF_POS_HORIZ_ABS flag is never set. Blocking on any of those would
    always time out. Instead we wait for EKF ATTITUDE convergence, which is the
    strongest readiness signal available without a horizontal position source.
    The EKF runs in constant-position mode (EKF_CONST_POS_MODE) and provides
    attitude + baro altitude only.
    """
    print("[*] Waiting for EKF attitude convergence (GPS-denied)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = mav.recv_match(
            type=['EKF_STATUS_REPORT', 'STATUSTEXT'],
            blocking=True, timeout=1
        )
        if msg is None:
            continue
        t = msg.get_type()
        if t == 'EKF_STATUS_REPORT':
            # Attitude estimate valid = usable for GUIDED_NOGPS / attitude flight.
            if msg.flags & mavutil.mavlink.EKF_ATTITUDE:
                print("[+] EKF ready (attitude valid, no GPS)")
                return True
        elif t == 'STATUSTEXT' and 'ekf3 imu0 is using' in msg.text.lower():
            print(f"[+] EKF: {msg.text.strip()}")
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


# GUIDED_NOGPS climb/altitude-hold tunables (attitude + thrust only).
TAKEOFF_THRUST = 0.62   # climb thrust while ascending (0-1, must exceed hover)
ALT_ACCURACY = 0.4      # takeoff "reached" when within +/- this of target (m)
ALT_BAND = 1.0          # allowed height wobble during forward flight (m)
HOVER_THRUST = 0.5      # baseline thrust; P-control trims around it
KP_ALT = 0.08           # thrust change per metre of altitude error
MIN_THRUST = 0.10       # never idle the motors in flight
MAX_THRUST = 0.75       # climb thrust ceiling
NOGPS_LOOP_HZ = 25.0    # attitude command rate (keep well above 2 Hz)


def alt_thrust(current_alt, target_alt):
    """Proportional thrust around HOVER_THRUST to drive altitude to target."""
    thrust = HOVER_THRUST + KP_ALT * (target_alt - current_alt)
    return max(MIN_THRUST, min(MAX_THRUST, thrust))


def takeoff_nogps(mav, altitude_m, ground_hpa, hold_yaw_deg):
    """Climb STRAIGHT UP holding the current heading to a baro altitude target.

    Ported from takeoff_to() in nogps_control_drone_gazebo.py (the known-good
    GPS-denied takeoff). Two things keep the ascent vertical:
      * yaw is held at the captured nose heading (hold_yaw_deg), NOT forced to 0
        (north) -- forcing 0 makes the drone yaw and tilt during the climb.
      * altitude is the BARO height above the captured ground reference, which is
        reliable while GPS-denied (GLOBAL_POSITION_INT.relative_alt is not).
    Climb at TAKEOFF_THRUST, then ease in with P-control for the final capture so
    we don't overshoot.
    """
    target = float(altitude_m)
    print(f"[*] Takeoff (GUIDED_NOGPS) to {target:.1f} m (+/- {ALT_ACCURACY:.1f} m), "
          f"holding {hold_yaw_deg:.0f} deg, at thrust {TAKEOFF_THRUST:.2f}...")
    dt = 1.0 / NOGPS_LOOP_HZ
    alt = 0.0
    while True:
        press = read_pressure(mav)
        if press is not None:
            alt = pressure_to_alt(press, ground_hpa)
        # Full takeoff thrust while climbing, then P-control for the final
        # capture so we ease in instead of overshooting the target height.
        if alt < target - ALT_ACCURACY:
            thrust = TAKEOFF_THRUST
        else:
            thrust = alt_thrust(alt, target)
        # Level, nose held: all thrust goes into a vertical climb.
        send_attitude(mav, roll_deg=0.0, pitch_deg=0.0, yaw_deg=hold_yaw_deg, thrust=thrust)
        print(f"    Baro alt: {alt:5.2f} / {target:.1f} m  thrust: {thrust:.2f}", end="\r")
        if abs(alt - target) <= ALT_ACCURACY:
            print(f"\n[+] Reached {alt:.2f} m, cruising")
            return
        time.sleep(dt)


def send_velocity(
    mav,
    vx=0.0,
    vy=0.0,
    vz=0.0,
    yaw=0.0,
    yaw_rate=0.0,
    frame=mavutil.mavlink.MAV_FRAME_BODY_NED,
    use_yaw=False,
):
    """
    Send one velocity setpoint. Call at ~20 Hz for continuous motion.

    frame=MAV_FRAME_BODY_NED  (default): vx forward, vy right, vz down (body-relative).
    frame=MAV_FRAME_LOCAL_NED:           vx north,   vy east,  vz down (world-fixed).
    vz: m/s, negative = climb.

    use_yaw=False (default): command yaw_rate (rad/s, +cw); heading is left free.
    use_yaw=True:            command an absolute yaw angle (rad); yaw_rate ignored.
    """
    if use_yaw:
        # use velocity + absolute yaw; ignore position, accel, yaw_rate
        type_mask = 0b100111000111
    else:
        # use velocity + yaw_rate; ignore position, accel, yaw angle
        type_mask = 0b010111000111
    mav.mav.send(
        mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
            0,
            mav.target_system, mav.target_component,
            frame,
            type_mask,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            yaw,
            yaw_rate,
        )
    )


def euler_to_quaternion(roll, pitch, yaw):
    """(roll, pitch, yaw) in radians -> [w, x, y, z]."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,  # w
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
    ]


def send_attitude(mav, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0, thrust=0.5):
    """One attitude + thrust setpoint for GUIDED_NOGPS. Call at >=10 Hz.

    GUIDED_NOGPS has no horizontal position/velocity estimate, so it accepts
    attitude, not velocity. With the default GUID_OPTIONS the `thrust` field is a
    CLIMB-RATE proxy vs baro altitude: 0.5 = hold, >0.5 climb, <0.5 descend.
    pitch < 0 = nose down = forward; yaw is absolute (0 = north).
    """
    q = euler_to_quaternion(
        math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    )
    # type_mask 0b00000111: ignore the three body-rate fields; use attitude + thrust.
    mav.mav.set_attitude_target_send(
        0,
        mav.target_system, mav.target_component,
        0b00000111,
        q,
        0.0, 0.0, 0.0,
        float(max(0.0, min(1.0, thrust))),
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


def set_gz_wind(
    vx: float,
    vy: float,
    vz: float = 0.0,
    world_name: str = "map",
    enable: bool = True,
    attempts: int = 3,
) -> bool:
    """Set the Gazebo world wind at runtime via the WindEffects system.

    The wind comes from the world <wind> element applied by the WindEffects system
    (gz-sim-wind-effects-system), not from ArduPilot SITL, so ArduPilot params can't
    change it. Instead we publish a gz.msgs.Wind to /world/<world>/wind, which the
    WindEffects plugin adopts as the new wind seed velocity (world frame, m/s). This
    tunes wind live, no world-SDF rebuild needed.

    Must run in the same container as the gz server so gz-transport can reach it. The
    publish is retried a few times because the first message can be dropped while
    transport discovery settles.

    enable=False also flips the WindEffects enable_wind flag off (used by --no-wind).
    Returns True once a publish succeeds.
    """
    topic = f"/world/{world_name}/wind"
    payload = (
        f"linear_velocity: {{x: {float(vx)}, y: {float(vy)}, z: {float(vz)}}}, "
        f"enable_wind: {'true' if enable else 'false'}"
    )
    cmd = ["gz", "topic", "-t", topic, "-m", "gz.msgs.Wind", "-p", payload]
    print(f"[*] Setting Gazebo wind on {topic}: ({vx:g}, {vy:g}, {vz:g}) m/s enable={enable}")
    for i in range(max(1, int(attempts))):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        except FileNotFoundError:
            print("[!] 'gz' CLI not found - cannot set wind from this process")
            return False
        except subprocess.TimeoutExpired:
            print("[!] Timed out publishing wind")
            return False
        if result.returncode == 0:
            print(f"[+] Gazebo wind set to ({vx:g}, {vy:g}, {vz:g}) m/s")
            return True
        if i == 0:
            print(f"[~] gz topic returned {result.returncode}: {result.stderr.strip()} (retrying)")
        time.sleep(0.3)
    print("[!] Failed to set Gazebo wind (is the WindEffects system running and world name correct?)")
    return False


def _poll_health_messages(mav, gps_health: GpsHealth) -> None:
    while True:
        msg = mav.recv_match(type=['GPS_RAW_INT'], blocking=False)
        if msg is None:
            break
        if msg.get_type() == 'GPS_RAW_INT':
            gps_health.update(msg)


# def mission_guided(
#     mav,
#     altitude_m=10.0,
#     speed=3.0,
#     duration_s=120.0,
#     flow_forwarder: OpticalFlowForwarder | None = None,
#     flow_max_stale_s=1.5,
#     flow_min_quality=30,
#     fallback_lateral_gain=0.8,
#     fallback_max_vy=1.5,
#     disable_gps_after_takeoff_s=15.0,
# ):
#     if not wait_for_ekf(mav, timeout=120):
#         print("[~] Continuing despite EKF timeout (force arm path enabled)")

#     set_mode(mav, "GUIDED")

#     if not arm(mav, force=True):
#         return

#     takeoff(mav, altitude_m)

#     forward_speed = float(speed)
#     gps_health = GpsHealth()
#     mission_end = time.time() + max(1.0, float(duration_s))
#     next_status_at = 0.0
#     gps_disable_triggered = False
#     gps_disable_at = (
#         time.time() + float(disable_gps_after_takeoff_s)
#         if float(disable_gps_after_takeoff_s) >= 0.0
#         else None
#     )

#     print(f"[*] Mission started: flying straight ahead at {forward_speed:.1f} m/s along current heading.")

#     while time.time() < mission_end:
#         _poll_health_messages(mav, gps_health)
#         now = time.time()

#         if gps_disable_at is not None and (not gps_disable_triggered) and now >= gps_disable_at:
#             print(f"[trigger] Disabling GPS at +{disable_gps_after_takeoff_s:.1f}s after takeoff")
#             trigger_disable_gps(mav)
#             gps_disable_triggered = True

#         # Single forward velocity in body frame: +vx = straight ahead along the current
#         # heading. No yaw-rate and no attitude/angle targets are commanded. Optical flow,
#         # when enabled, adds only a lateral (vy) correction to counter wind drift.
#         cmd_vy = 0.0
#         flow_note = ""
#         if flow_forwarder is not None:
#             flow_ok, flow_age, flow_quality, flow_count = flow_forwarder.flow_health(
#                 max_stale_s=float(flow_max_stale_s),
#                 min_quality=int(flow_min_quality),
#             )
#             if flow_ok:
#                 velocity = flow_forwarder.latest_flow_velocity_mps()
#                 if velocity is not None:
#                     _, flow_vy = velocity
#                     raw_correction = -float(fallback_lateral_gain) * float(flow_vy)
#                     cmd_vy = max(-float(fallback_max_vy), min(float(fallback_max_vy), raw_correction))
#                 flow_note = (
#                     f"flow=ok q={flow_quality} age={flow_age:.2f}s "
#                     f"msgs={flow_count} vy_cmd={cmd_vy:.2f}"
#                 )
#             else:
#                 flow_note = f"flow=bad q={flow_quality} age={flow_age:.2f}s msgs={flow_count}"

#         send_velocity(mav, vx=forward_speed, vy=cmd_vy, vz=0.0, yaw=0.0, yaw_rate=0.0)

#         if now >= next_status_at:
#             print(
#                 f"[status] gps_fix={gps_health.fix_type} sats={gps_health.satellites_visible} "
#                 f"vx={forward_speed:.2f} vy={cmd_vy:.2f} {flow_note}"
#             )
#             next_status_at = now + 1.0

#         time.sleep(0.05)

#     print("[+] Mission complete")


def mission_cruise(mav, speed, altitude_m=10.0, guided_nogps=False):
    """Arm, take off, then cruise straight forever.

    Same mission either way; only the control path differs:
      guided_nogps=False -> GUIDED, world-frame velocity setpoints due NORTH
          (yaw=0; needs a horizontal position estimate from the EKF).
      guided_nogps=True  -> GUIDED_NOGPS, attitude + thrust (works GPS-denied):
          climb straight up holding the captured compass heading, then cruise on
          an open-loop forward pitch along that heading; altitude is held on the
          thrust (climb-rate) channel against baro.
    Runs until interrupted (Ctrl-C / SIGTERM).
    """
    if not wait_for_ekf(mav, timeout=120):
        print("[~] Continuing despite EKF timeout (force arm path enabled)")

    # GPS-denied prep: stream baro + compass, then (still on the ground, before
    # arming) capture the ground pressure reference and the current nose heading
    # so takeoff/cruise can climb on baro altitude while holding that heading.
    ground_hpa = hold_yaw_deg = None
    if guided_nogps:
        request_message(mav, mavutil.mavlink.MAVLINK_MSG_ID_SCALED_PRESSURE, NOGPS_LOOP_HZ)
        request_message(mav, mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD, NOGPS_LOOP_HZ)
        time.sleep(1.0)
        ground_hpa = capture_ground_pressure(mav)
        hold_yaw_deg = wait_for_heading_deg(mav)

    # Mode + takeoff differ: NAV_TAKEOFF needs a position estimate, so the
    # GPS-denied path climbs on attitude instead.
    set_mode(mav, "GUIDED_NOGPS" if guided_nogps else "GUIDED")

    if not arm(mav, force=True):
        return

    if guided_nogps:
        takeoff_nogps(mav, altitude_m, ground_hpa, hold_yaw_deg)
        # Open-loop forward pitch (~1.7 deg per m/s, nose down); heading held.
        pitch_cmd = -max(2.0, min(15.0, float(speed) * 1.7))
        print(
            f"[*] Cruise (GUIDED_NOGPS): pitch={pitch_cmd:.1f} deg fwd, "
            f"yaw={hold_yaw_deg:.0f} deg held, baro alt-hold @ {float(altitude_m):.1f} m. "
            f"Ctrl-C to stop."
        )
    else:
        takeoff(mav, altitude_m)
        print(f"[*] Cruise control: holding {float(speed):.1f} m/s due NORTH, yaw=0. Ctrl-C to stop.")

    north = east = down = 0.0
    heading_deg = 0.0
    alt = float(altitude_m)
    next_status_at = 0.0
    while True:
        if guided_nogps:
            press = read_pressure(mav)
            if press is not None:
                alt = pressure_to_alt(press, ground_hpa)
            # Pitch forward only while altitude is in band; if we drift out, level
            # off so all thrust recovers height (matches nogps fly_forward). Yaw is
            # held at the captured heading, so the track stays straight.
            in_band = abs(alt - float(altitude_m)) <= ALT_BAND
            cmd_pitch = pitch_cmd if in_band else 0.0
            thrust = alt_thrust(alt, float(altitude_m))
            send_attitude(mav, pitch_deg=cmd_pitch, yaw_deg=hold_yaw_deg, thrust=thrust)
            tag = "fwd " if in_band else "recov"
            status = (f"[flying:{tag}] baro alt={alt:.1f} m  pitch={cmd_pitch:.1f}  "
                      f"thrust={thrust:.2f}  yaw={hold_yaw_deg:.0f}")
        else:
            # Drain the latest position + attitude telemetry (also keeps reading
            # the socket) without blocking the command rate.
            while True:
                msg = mav.recv_match(type=['LOCAL_POSITION_NED', 'ATTITUDE'], blocking=False)
                if msg is None:
                    break
                if msg.get_type() == 'LOCAL_POSITION_NED':
                    north, east, down = msg.x, msg.y, msg.z
                else:  # ATTITUDE
                    heading_deg = (math.degrees(msg.yaw) + 360.0) % 360.0
            # World-frame NED velocity due north (vx=north, vy=0), fixed yaw=0.
            send_velocity(
                mav,
                vx=float(speed), vy=0.0, vz=0.0,
                yaw=0.0, use_yaw=True,
                frame=mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            )
            status = (
                f"[flying] N={north:+.1f} E={east:+.1f} alt={-down:.1f} m | "
                f"yaw={heading_deg:.1f} deg | vx_cmd={float(speed):.1f} m/s"
            )

        now = time.time()
        if now >= next_status_at:
            print(status)
            next_status_at = now + 1.0
        time.sleep(0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run guided mission with optional optical-flow display (no SITL forwarding)."
    )
    parser.add_argument("--altitude", type=float, default=10.0, help="Mission takeoff altitude in meters")
    parser.add_argument("--speed", type=float, default=3.0, help="Forward flight speed in m/s (body-frame vx, along heading)")
    parser.add_argument(
        "--guided_nogps",
        action="store_true",
        help="Fly the GPS-denied GUIDED_NOGPS attitude cruise instead of GUIDED velocity cruise",
    )
    parser.add_argument(
        "--disable-gps-after-takeoff",
        type=float,
        default=10.0,
        help="Disable GPS this many seconds after takeoff (set <0 to disable trigger)",
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
        help="Gain from flow lateral velocity estimate to vy (lateral) correction",
    )
    parser.add_argument(
        "--fallback-max-vy",
        type=float,
        default=1.5,
        help="Clamp for lateral (vy) wind correction in m/s",
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
    parser.add_argument(
        "--no-wind",
        action="store_true",
        help="Zero the Gazebo world wind at startup (via WindEffects) for testing",
    )
    parser.add_argument(
        "--wind",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "VZ"),
        default=None,
        help="Set Gazebo world wind (m/s, world frame) at startup, e.g. --wind 8 8 0",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.no_wind:
        set_gz_wind(0.0, 0.0, 0.0, enable=False)
    elif args.wind is not None:
        set_gz_wind(args.wind[0], args.wind[1], args.wind[2])

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
        # Endless straight-line cruise. --guided_nogps flies it on
        # attitude (GPS-denied); otherwise on GUIDED velocity.
        mission_cruise(
            mav, speed=args.speed, altitude_m=args.altitude,
            guided_nogps=args.guided_nogps,
        )
    finally:
        if flow_forwarder is not None:
            flow_forwarder.stop()

    print("[+] Mission finished. Press Q to close camera.")
