"""
GPS-denied attitude control demo for ArduPilot Copter (SITL / Gazebo).

Flow:
  1. GUIDED_NOGPS mode  -> accepts SET_ATTITUDE_TARGET, needs no GPS/position.
  2. Arm, then take off by commanding a level attitude with climb thrust.
  3. Altitude is measured from the BAROMETER (SCALED_PRESSURE) relative to the
     pressure captured on the ground, converted to metres.
  4. Heading (nose direction) is read from the MAGNETOMETER (VFR_HUD.heading,
     which is compass-only while GPS is disabled) and held for the whole flight,
     so the drone flies straight nose-forward instead of drifting to North.
  5. Once within ALT_ACCURACY of TARGET_ALT, pitch nose forward (still attitude
     control) to fly forward, while a proportional thrust controller keeps the
     height inside +/- ALT_BAND of the target.
  6. Descend and disarm.

Everything is done with attitude + thrust only - no velocity/position commands.

Tunable from the CLI:
  --height N          target altitude (m)
  --takeoff-speed N   takeoff climb thrust 0-1 (higher = faster)
  --forward-speed N   forward pitch angle in deg (higher = faster)
  --forward-time N    forward flight duration (s)
"""

import math
import time

from pymavlink import mavutil


# --- Tunables (the starred ones are overridable on the CLI, see __main__) --- #
TARGET_ALT = 5.0        # * HEIGHT: desired altitude above launch point (m)
TAKEOFF_THRUST = 0.62   # * TAKEOFF SPEED: climb thrust while ascending
                        #   (0-1, must be > hover; higher = faster takeoff)
FORWARD_PITCH_DEG = 8.0  # * FORWARD SPEED: nose-down angle for forward flight
                         #   (deg; higher = faster forward)
FORWARD_TIME = 15.0      # * how long to fly forward (s)

ALT_ACCURACY = 0.4      # takeoff is "reached" when within +/- this of target (m)
ALT_BAND = 1.0          # allowed height wobble during forward flight (m)

HOVER_THRUST = 0.5      # baseline thrust (~mid-stick); P-control trims around it
KP_ALT = 0.08           # thrust change per metre of altitude error
MIN_THRUST = 0.10       # never let motors go fully idle in flight
MAX_THRUST = 0.75       # climb thrust ceiling

LOOP_HZ = 25.0           # attitude command rate (keep well above 2 Hz)


# --- Connection ------------------------------------------------------------ #
def init_connections():
    print("[+] Connecting MAVLink...")
    mav = mavutil.mavlink_connection("udpin:0.0.0.0:14550")
    mav.wait_heartbeat()
    print("[+] MAVLink connected")
    return mav


def request_message(mav, msg_id, hz):
    """Ask the autopilot to stream a message at the given rate."""
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        msg_id, int(1e6 / hz), 0, 0, 0, 0, 0,
    )


# --- Barometric altitude --------------------------------------------------- #
def pressure_to_alt(press_hpa, ground_hpa):
    """International barometric formula -> metres above the ground reference."""
    return 44330.0 * (1.0 - (press_hpa / ground_hpa) ** (1.0 / 5.255))


def read_pressure(mav):
    """Return the newest absolute pressure (hPa), or None if nothing buffered."""
    press = None
    while True:
        msg = mav.recv_match(type="SCALED_PRESSURE", blocking=False)
        if msg is None:
            break
        press = msg.press_abs
    return press


def capture_ground_pressure(mav, samples=10):
    """Average a few barometer readings to establish the ground reference."""
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


# --- Magnetometer heading -------------------------------------------------- #
def read_heading(mav):
    """
    Latest compass heading in radians (0 = North, clockwise positive), or None.
    GPS is disabled, so VFR_HUD.heading is derived purely from the MAGNETOMETER.
    This is the vehicle's nose direction and matches the yaw convention used by
    SET_ATTITUDE_TARGET, so we can command it directly to hold "nose forward".
    """
    heading = None
    while True:
        msg = mav.recv_match(type="VFR_HUD", blocking=False)
        if msg is None:
            break
        heading = msg.heading
    if heading is None:
        return None
    return math.radians(heading)


def wait_for_heading(mav, timeout=10):
    """Block until the magnetometer heading is available; return it (radians)."""
    print("[*] Reading magnetometer heading...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        heading = read_heading(mav)
        if heading is not None:
            print(f"[+] Nose heading: {math.degrees(heading):.0f} deg (magnetometer)")
            return heading
        time.sleep(0.1)
    print("[!] No VFR_HUD heading - defaulting to 0 (North)")
    return 0.0


# --- Attitude command ------------------------------------------------------ #
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


def send_attitude(mav, roll, pitch, yaw, thrust):
    """
    Command an absolute attitude + thrust (SET_ATTITUDE_TARGET).
    roll/pitch/yaw in radians, thrust in [0, 1] (0.5 ~ hover).
    Body rates are ignored (type_mask 0b00000111).
    """
    q = euler_to_quaternion(roll, pitch, yaw)
    mav.mav.set_attitude_target_send(
        0,
        mav.target_system, mav.target_component,
        0b00000111,
        q,
        0.0, 0.0, 0.0,
        thrust,
    )


def alt_thrust(current_alt, target_alt):
    """Proportional thrust around HOVER_THRUST to drive altitude to target."""
    thrust = HOVER_THRUST + KP_ALT * (target_alt - current_alt)
    return max(MIN_THRUST, min(MAX_THRUST, thrust))


# --- Mode / arm ------------------------------------------------------------ #
def set_mode(mav, mode_name, timeout=5):
    mode_mapping = mav.mode_mapping()
    if mode_name not in mode_mapping:
        print(f"[!] Unknown mode: {mode_name}")
        return False
    mode_id = mode_mapping[mode_name]
    mav.mav.set_mode_send(
        mav.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
    )
    start = time.time()
    while time.time() - start < timeout:
        msg = mav.recv_match(type="HEARTBEAT", blocking=False)
        if msg and msg.custom_mode == mode_id:
            print(f"[+] Mode: {mode_name}")
            return True
        time.sleep(0.05)
    print(f"[~] Mode {mode_name} sent (no confirmation)")
    return True


def is_armed(mav):
    for _ in range(20):
        m = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=0.5)
        if m:
            return bool(m.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    return False


def arm(mav, force=True, timeout=15):
    print("[*] Arming...")
    # GPS-denied sim: skip arming checks that expect a position estimate.
    mav.mav.param_set_send(
        mav.target_system, mav.target_component,
        b"ARMING_CHECK", 0,
        mavutil.mavlink.MAV_PARAM_TYPE_INT32,
    )
    time.sleep(2.0)
    # param2 = 21196 forces arming past ANY remaining check. A plain arm still
    # enforces some consistency checks (e.g. SITL "Accels inconsistent") even
    # with ARMING_CHECK=0; the force magic value bypasses those too. 0 = normal.
    param2 = 21196 if force else 0
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, param2, 0, 0, 0, 0, 0,
    )
    start = time.time()
    while time.time() - start < timeout:
        if is_armed(mav):
            print("[+] Armed")
            return True
        time.sleep(0.1)
    print("[!] Arming failed")
    return False


def disarm(mav):
    print("[*] Disarming...")
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        0, 0, 0, 0, 0, 0, 0,
    )


# --- Flight phases --------------------------------------------------------- #
def takeoff_to(mav, ground_hpa, hold_yaw):
    """Climb at TAKEOFF_THRUST holding heading, until inside ALT_ACCURACY."""
    print(f"[*] Takeoff to {TARGET_ALT:.1f} m (+/- {ALT_ACCURACY:.1f} m) "
          f"at thrust {TAKEOFF_THRUST:.2f}...")
    dt = 1.0 / LOOP_HZ
    alt = 0.0
    while True:
        press = read_pressure(mav)
        if press is not None:
            alt = pressure_to_alt(press, ground_hpa)
        # Climb at the chosen takeoff thrust, then ease in with P-control for the
        # final capture so we don't overshoot the target height.
        if alt < TARGET_ALT - ALT_ACCURACY:
            thrust = TAKEOFF_THRUST
        else:
            thrust = alt_thrust(alt, TARGET_ALT)
        send_attitude(mav, 0.0, 0.0, hold_yaw, thrust)  # level, nose held
        print(f"    Baro alt: {alt:5.2f} m  thrust: {thrust:.2f}", end="\r")
        if abs(alt - TARGET_ALT) <= ALT_ACCURACY:
            print(f"\n[+] Reached {alt:.2f} m")
            return
        time.sleep(dt)


def fly_forward(mav, ground_hpa, hold_yaw):
    """Pitch nose forward along the held heading, regulating height in ALT_BAND."""
    print(f"[*] Forward flight for {FORWARD_TIME:.0f} s at {FORWARD_PITCH_DEG:.0f} deg, "
          f"holding {TARGET_ALT:.1f} +/- {ALT_BAND:.1f} m "
          f"on heading {math.degrees(hold_yaw):.0f} deg...")
    pitch = -math.radians(FORWARD_PITCH_DEG)  # nose down = forward
    dt = 1.0 / LOOP_HZ
    alt = TARGET_ALT
    heading = hold_yaw
    end = time.time() + FORWARD_TIME
    while time.time() < end:
        press = read_pressure(mav)
        if press is not None:
            alt = pressure_to_alt(press, ground_hpa)
        live = read_heading(mav)  # magnetometer heading, for display / drift check
        if live is not None:
            heading = live
        # Pitch forward only while height is within band; if we drift out,
        # level off so all thrust goes into recovering altitude.
        in_band = abs(alt - TARGET_ALT) <= ALT_BAND
        cmd_pitch = pitch if in_band else 0.0
        thrust = alt_thrust(alt, TARGET_ALT)
        # Yaw command stays at hold_yaw so the nose keeps pointing forward.
        send_attitude(mav, 0.0, cmd_pitch, hold_yaw, thrust)
        tag = "fwd " if in_band else "recov"
        print(f"    [{tag}] baro alt: {alt:5.2f} m  thrust: {thrust:.2f}  "
              f"hdg: {math.degrees(heading):3.0f} deg", end="\r")
        time.sleep(dt)
    print("\n[+] Forward leg complete")


def descend_and_disarm(mav, ground_hpa, hold_yaw):
    """Level attitude, thrust below hover, until near ground, then disarm."""
    print("[*] Descending...")
    dt = 1.0 / LOOP_HZ
    alt = TARGET_ALT
    while True:
        press = read_pressure(mav)
        if press is not None:
            alt = pressure_to_alt(press, ground_hpa)
        # Aim thrust at a slowly sinking altitude target just under current.
        thrust = alt_thrust(alt, -1.0)  # commands descent (clamped to MIN)
        send_attitude(mav, 0.0, 0.0, hold_yaw, thrust)  # level, nose held
        print(f"    Baro alt: {alt:5.2f} m", end="\r")
        if alt < 0.3:
            break
        time.sleep(dt)
    send_attitude(mav, 0.0, 0.0, 0.0, 0.0)  # cut thrust
    time.sleep(0.5)
    disarm(mav)
    print("\n[+] Landed and disarmed")


def mission(mav):
    request_message(mav, mavutil.mavlink.MAVLINK_MSG_ID_SCALED_PRESSURE, LOOP_HZ)
    request_message(mav, mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD, LOOP_HZ)
    time.sleep(1.0)

    ground_hpa = capture_ground_pressure(mav)
    # Capture the current nose direction from the magnetometer and hold it for
    # the whole flight, so the drone flies straight nose-forward (no yaw drift
    # to North and no spinning).
    hold_yaw = wait_for_heading(mav)

    set_mode(mav, "GUIDED_NOGPS")
    if not arm(mav):
        return

    takeoff_to(mav, ground_hpa, hold_yaw)
    fly_forward(mav, ground_hpa, hold_yaw)
    descend_and_disarm(mav, ground_hpa, hold_yaw)
    print("[+] Mission complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GPS-denied attitude flight for ArduPilot "
                    "(barometer altitude + magnetometer heading)."
    )
    parser.add_argument(
        "--height", type=float, default=TARGET_ALT,
        help="target altitude above launch point, metres (default: %(default)s)",
    )
    parser.add_argument(
        "--takeoff-thrust", type=float, default=TAKEOFF_THRUST,
        help="takeoff climb thrust 0-1, higher = faster takeoff "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--forward-angle", type=float, default=FORWARD_PITCH_DEG,
        help="forward pitch angle in degrees, higher = faster forward "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--forward-time", type=float, default=FORWARD_TIME,
        help="forward flight duration in seconds (default: %(default)s)",
    )
    args = parser.parse_args()

    # Override the module-level tunables the flight phases read.
    TARGET_ALT = args.height
    TAKEOFF_THRUST = args.takeoff_thrust
    FORWARD_PITCH_DEG = args.forward_angle
    FORWARD_TIME = args.forward_time

    mav = init_connections()
    mission(mav)
