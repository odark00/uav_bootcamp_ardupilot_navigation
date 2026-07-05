#!/usr/bin/env python3
"""
Visual-odometry-assisted GPS-denied flight demo for ArduPilot (SITL / Gazebo).

Flow:
  1. GUIDED_NOGPS mode -> take off using attitude + thrust only (barometer
     altitude, magnetometer heading), exactly like nogps_control_drone_gazebo.
  2. Stream dummy VISION_POSITION_ESTIMATE (external nav) over MAVLink so the
     EKF gains a horizontal position source (needs VISO_TYPE=1, EK3_SRC1_POSXY=6).
  3. Switch to GUIDED and move forward with a velocity command, dead-reckoning
     the vision position from the commanded velocity so the estimate stays
     consistent.
  4. Land.
"""

import math
import time

from pymavlink import mavutil

# --- Tunables -------------------------------------------------------------- #
TARGET_ALT = 3.0        # desired altitude above launch point (m)
TAKEOFF_THRUST = 0.62   # climb thrust while ascending (0-1, must be > hover)
ALT_ACCURACY = 0.4      # takeoff "reached" when within +/- this of target (m)

HOVER_THRUST = 0.5      # baseline thrust (~mid-stick); P-control trims around it
KP_ALT = 0.08           # thrust change per metre of altitude error
MIN_THRUST = 0.10       # never let motors go fully idle in flight
MAX_THRUST = 0.75       # climb thrust ceiling
LOOP_HZ = 25.0          # attitude command rate (keep well above 2 Hz)

ORIGIN_LAT = -353632621
ORIGIN_LON = 1491652374
ORIGIN_ALT_M = 584.0

GROUND_HPA = None       # ground barometer reference (set in mission), for down-z
# Dead-reckoned vision position (NED, metres), advanced during forward flight.
VIS_N = 0.0
VIS_E = 0.0

# Connect to SITL (adjust port if needed)
master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')
print("Waiting for heartbeat...")
master.wait_heartbeat()
print("Connected to system:", master.target_system, master.target_component)


# --- Visual odometry publisher -------------------------------------------- #
def send_vision(north, east, down, yaw):
    """
    Publish a dummy VISION_POSITION_ESTIMATE (external nav). Position is NED in
    metres (north, east, down); yaw in radians (0 = North). This is the EKF's
    horizontal position source, so GUIDED is no longer rejected for "position".
    """
    master.mav.vision_position_estimate_send(
        int(time.time() * 1e6),  # usec
        float(north), float(east), float(down),
        0.0, 0.0, float(yaw),    # roll, pitch, yaw (rad)
    )


# --- Barometric altitude --------------------------------------------------- #
def request_message(msg_id, hz):
    """Ask the autopilot to stream a message at the given rate."""
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        msg_id, int(1e6 / hz), 0, 0, 0, 0, 0,
    )


def pressure_to_alt(press_hpa, ground_hpa):
    """International barometric formula -> metres above the ground reference."""
    return 44330.0 * (1.0 - (press_hpa / ground_hpa) ** (1.0 / 5.255))


def read_pressure():
    """Return the newest absolute pressure (hPa), or None if nothing buffered."""
    press = None
    while True:
        msg = master.recv_match(type="SCALED_PRESSURE", blocking=False)
        if msg is None:
            break
        press = msg.press_abs
    return press


def capture_ground_pressure(samples=10):
    """Average a few barometer readings to establish the ground reference."""
    print("[*] Reading ground barometer reference...")
    readings = []
    deadline = time.time() + 10
    while len(readings) < samples and time.time() < deadline:
        press = read_pressure()
        if press is not None:
            readings.append(press)
        time.sleep(0.1)
    if not readings:
        raise RuntimeError("No SCALED_PRESSURE received - is the baro streaming?")
    ground = sum(readings) / len(readings)
    print(f"[+] Ground pressure: {ground:.2f} hPa")
    return ground


def current_alt():
    """Height above launch from the barometer, or TARGET_ALT if unavailable."""
    press = read_pressure()
    if press is not None and GROUND_HPA:
        return pressure_to_alt(press, GROUND_HPA)
    return TARGET_ALT


# --- Magnetometer heading -------------------------------------------------- #
def read_heading():
    """Latest compass heading in radians (0 = North, clockwise +), or None."""
    heading = None
    while True:
        msg = master.recv_match(type="VFR_HUD", blocking=False)
        if msg is None:
            break
        heading = msg.heading
    if heading is None:
        return None
    return math.radians(heading)


def wait_for_heading(timeout=10):
    """Block until the magnetometer heading is available; return it (radians)."""
    print("[*] Reading magnetometer heading...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        heading = read_heading()
        if heading is not None:
            print(f"[+] Nose heading: {math.degrees(heading):.0f} deg")
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


def send_attitude(roll, pitch, yaw, thrust):
    """Command absolute attitude + thrust (SET_ATTITUDE_TARGET), body rates ignored."""
    q = euler_to_quaternion(roll, pitch, yaw)
    master.mav.set_attitude_target_send(
        0,
        master.target_system, master.target_component,
        0b00000111,
        q,
        0.0, 0.0, 0.0,
        thrust,
    )


def alt_thrust(current, target):
    """Proportional thrust around HOVER_THRUST to drive altitude to target."""
    thrust = HOVER_THRUST + KP_ALT * (target - current)
    return max(MIN_THRUST, min(MAX_THRUST, thrust))


# --- Fly forward using velocity command ------------------------------------ #
def send_velocity(vx, vy, vz, yaw, duration=5):
    """
    vx, vy, vz: velocity in m/s (NED); duration: seconds to hold.
    Dead-reckons the vision position from the commanded velocity so the external
    nav estimate stays consistent with the motion the EKF is being asked to fly.
    """
    global VIS_N, VIS_E
    dt = 0.1
    for _ in range(int(duration / dt)):  # 10 Hz
        VIS_N += vx * dt
        VIS_E += vy * dt
        send_vision(VIS_N, VIS_E, -current_alt(), yaw)  # keep external nav alive
        master.mav.set_position_target_local_ned_send(
            0,                       # time_boot_ms (uint32); 0 = "now" is fine
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111,      # velocity only (ignore pos/accel/yaw)
            0, 0, 0,                 # x, y, z positions (ignored)
            vx, vy, vz,              # velocities
            0, 0, 0,                 # accelerations (ignored)
            0, 0                     # yaw, yaw_rate (ignored)
        )
        time.sleep(dt)


# --- Origin / home --------------------------------------------------------- #
def set_gps_origin(lat=ORIGIN_LAT, lon=ORIGIN_LON, alt_m=ORIGIN_ALT_M):
    """Set the EKF origin (SET_GPS_GLOBAL_ORIGIN) so the local frame is referenced."""
    master.mav.set_gps_global_origin_send(master.target_system, lat, lon, int(alt_m * 1000))


def set_home(lat=ORIGIN_LAT, lon=ORIGIN_LON, alt_m=ORIGIN_ALT_M):
    """Set HOME explicitly (DO_SET_HOME) so GUIDED/GUIDED_NOGPS can arm/switch."""
    master.mav.command_int_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL,
        mavutil.mavlink.MAV_CMD_DO_SET_HOME,
        0, 0,
        0,                 # param1: 0 = use the specified location (not current)
        0, 0, 0,
        int(lat), int(lon), float(alt_m),
    )


# --- Takeoff phase (attitude + thrust, no velocity/position) --------------- #
def takeoff_to(hold_yaw):
    """Climb at TAKEOFF_THRUST holding heading, until inside ALT_ACCURACY."""
    print(f"[*] Takeoff to {TARGET_ALT:.1f} m (+/- {ALT_ACCURACY:.1f} m) "
          f"at thrust {TAKEOFF_THRUST:.2f}...")
    dt = 1.0 / LOOP_HZ
    while True:
        alt = current_alt()
        # Climb at chosen takeoff thrust, then ease in with P-control for capture.
        if alt < TARGET_ALT - ALT_ACCURACY:
            thrust = TAKEOFF_THRUST
        else:
            thrust = alt_thrust(alt, TARGET_ALT)
        send_attitude(0.0, 0.0, hold_yaw, thrust)  # level, nose held
        # Stream vision position during climb so the EKF converges a horizontal
        # position estimate by the time we reach altitude.
        send_vision(VIS_N, VIS_E, -alt, hold_yaw)
        print(f"    Baro alt: {alt:5.2f} m  thrust: {thrust:.2f}", end="\r")
        if abs(alt - TARGET_ALT) <= ALT_ACCURACY:
            print(f"\n[+] Reached {alt:.2f} m")
            return
        time.sleep(dt)


# --- Wait for a horizontal position estimate ------------------------------- #
def wait_for_position(hold_yaw, timeout=30):
    """
    Stream vision position + hold attitude until the EKF reports a horizontal
    position (EKF_STATUS_REPORT flag EKF_POS_HORIZ_REL). GUIDED rejects the mode
    switch ("requires position") until this is set.
    """
    print("[*] Streaming vision position, waiting for EKF horizontal position...")
    request_message(mavutil.mavlink.MAVLINK_MSG_ID_EKF_STATUS_REPORT, 5)
    deadline = time.time() + timeout
    last_report = 0.0
    while time.time() < deadline:
        alt = current_alt()
        send_vision(VIS_N, VIS_E, -alt, hold_yaw)
        send_attitude(0.0, 0.0, hold_yaw, alt_thrust(alt, TARGET_ALT))
        msg = master.recv_match(type="EKF_STATUS_REPORT", blocking=False)
        if msg:
            if msg.flags & mavutil.mavlink.EKF_POS_HORIZ_REL:
                print("\n[+] EKF has horizontal position (vision fused)")
                return True
            # Periodically decode the flags so a hang is diagnosable: if
            # CONST_POS_MODE stays set, the EKF is getting no horizontal aiding
            # (vision not being fused -> VisOdom backend likely not allocated;
            # reboot the autopilot after setting VISO_TYPE=1).
            if time.time() - last_report > 1.0:
                f = msg.flags
                const_pos = bool(f & mavutil.mavlink.EKF_CONST_POS_MODE)
                pred_rel = bool(f & mavutil.mavlink.EKF_PRED_POS_HORIZ_REL)
                print(f"    EKF flags={f:#06x} const_pos_mode={const_pos} "
                      f"pred_pos_rel={pred_rel}", end="\r")
                last_report = time.time()
        time.sleep(0.1)
    print("\n[!] Timed out waiting for EKF position - GUIDED will be rejected.")
    print("    If const_pos_mode stayed True, vision was not fused: reboot the")
    print("    autopilot after loading VISO_TYPE=1 so the VisOdom backend exists.")
    return False


# --- Mode / arm ------------------------------------------------------------ #
def arm(timeout=15):
    """
    Arm with pre-arm checks disabled (GPS-denied sim): ARMING_CHECK=0 clears
    "Gyros inconsistent" / "VisOdom" gates, and param2=21196 force-arms past any
    remaining consistency checks. Keeps streaming vision so external nav is live.
    """
    print("[*] Arming (checks disabled for GPS-denied sim)...")
    master.mav.param_set_send(
        master.target_system, master.target_component,
        b"ARMING_CHECK", 0,
        mavutil.mavlink.MAV_PARAM_TYPE_INT32,
    )
    # Let the param apply while keeping external nav alive.
    for _ in range(20):
        send_vision(VIS_N, VIS_E, -current_alt(), 0.0)
        time.sleep(0.1)
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 21196, 0, 0, 0, 0, 0,  # arm, force
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        send_vision(VIS_N, VIS_E, -current_alt(), 0.0)
        master.recv_match(type="HEARTBEAT", blocking=False)  # refresh armed state
        if master.motors_armed():
            print("Armed!")
            return True
        time.sleep(0.1)
    print("[!] Arming failed")
    return False


# ========================= Mission ========================================= #
request_message(mavutil.mavlink.MAVLINK_MSG_ID_SCALED_PRESSURE, LOOP_HZ)
request_message(mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD, LOOP_HZ)
time.sleep(1.0)

set_gps_origin()
set_home()

# Establish barometer/heading references for attitude-based takeoff.
ground_hpa = capture_ground_pressure()
GROUND_HPA = ground_hpa  # used to report the vision-position down component
hold_yaw = wait_for_heading()

# Prime the estimator with vision position while the origin/home settle.
for _ in range(20):
    send_vision(VIS_N, VIS_E, 0.0, hold_yaw)
    time.sleep(0.1)

# --- GUIDED_NOGPS takeoff with attitude + thrust --- #
master.set_mode_apm("GUIDED_NOGPS")
if not arm():
    raise SystemExit("Arming failed - aborting mission")
takeoff_to(hold_yaw)

# --- Wait until vision gives the EKF a horizontal position, then GUIDED. --- #
wait_for_position(hold_yaw)

master.set_mode_apm("GUIDED")
time.sleep(1.0)
print("Beginning forward velocity commands...")
send_velocity(1.0, 0.0, 0.0, hold_yaw, duration=20)
print("Forward motion complete")

# --- Land --- #
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_NAV_LAND,
    0, 0, 0, 0, 0, 0, 0, 0
)
print("Land command sent")
