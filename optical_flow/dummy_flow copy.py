#!/usr/bin/env python3
"""
Optical-flow-assisted GPS-denied flight demo for ArduPilot (SITL / Gazebo).

Flow:
  1. GUIDED_NOGPS mode -> take off using attitude + thrust only (barometer
     altitude, magnetometer heading), exactly like nogps_control_drone_gazebo.
  2. Once at altitude, stream a few dummy OPTICAL_FLOW packets over MAVLink so
     the EKF gains a horizontal velocity/position source.
  3. Switch to GUIDED and move forward with a velocity command (which needs the
     flow-provided position estimate), while keeping the flow stream alive.
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

GROUND_HPA = None       # ground barometer reference (set in mission), for HAGL

# Connect to SITL (adjust port if needed)
master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')
print("Waiting for heartbeat...")
master.wait_heartbeat()
print("Connected to system:", master.target_system, master.target_component)


# --- Optical flow dummy publisher ----------------------------------------- #
def send_dummy_flow(vx=0.0, vy=0.0, quality=255):
    master.mav.optical_flow_send(
        int(time.time() * 1e6),  # time_usec (uint64)
        0,                       # sensor_id (uint8)
        0, 0,                    # flow_x, flow_y (int16, rad/sec*1000)
        float(vx), float(vy),    # flow_comp_m_x, flow_comp_m_y (float)
        quality,                 # quality (uint8)
        1.0,                     # ground_distance (float, meters)
        0.0, 0.0                 # flow_rate_x, flow_rate_y (float)
    )


# --- Dummy downward rangefinder ------------------------------------------- #
def send_dummy_distance(dist_m):
    """
    Publish a downward DISTANCE_SENSOR so the EKF has height-above-ground to
    scale optical flow into velocity. Without this, flow is never fused and
    GUIDED stays "requires position". Backed by RNGFND1_TYPE=10 (MAVLink).
    orientation=PITCH_270 (25) = straight down, matching RNGFND1_ORIENT.
    """
    d_cm = int(max(0.0, dist_m) * 100)
    master.mav.distance_sensor_send(
        0,       # time_boot_ms
        0,       # min_distance (cm)
        4000,    # max_distance (cm)
        d_cm,    # current_distance (cm)
        mavutil.mavlink.MAV_DISTANCE_SENSOR_LASER,      # type
        0,       # id
        mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270,  # orientation = down
        0,       # covariance (0 = unknown)
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


def alt_thrust(current_alt, target_alt):
    """Proportional thrust around HOVER_THRUST to drive altitude to target."""
    thrust = HOVER_THRUST + KP_ALT * (target_alt - current_alt)
    return max(MIN_THRUST, min(MAX_THRUST, thrust))


# --- Fly forward using velocity command ------------------------------------ #
def send_velocity(vx, vy, vz, duration=5):
    """vx, vy, vz: velocity in m/s (NED frame); duration: seconds to hold."""
    for _ in range(int(duration * 10)):  # 10 Hz
        # Keep streaming flow + height so the EKF keeps a position estimate.
        send_dummy_flow()
        press = read_pressure()
        alt = pressure_to_alt(press, GROUND_HPA) if (press and GROUND_HPA) else TARGET_ALT
        send_dummy_distance(alt)
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
        time.sleep(0.1)


# --- Origin / home --------------------------------------------------------- #
def set_gps_origin(lat=ORIGIN_LAT, lon=ORIGIN_LON, alt_m=ORIGIN_ALT_M):
    """Set the EKF origin (SET_GPS_GLOBAL_ORIGIN) so flow position is referenced."""
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
def takeoff_to(ground_hpa, hold_yaw):
    """Climb at TAKEOFF_THRUST holding heading, until inside ALT_ACCURACY."""
    print(f"[*] Takeoff to {TARGET_ALT:.1f} m (+/- {ALT_ACCURACY:.1f} m) "
          f"at thrust {TAKEOFF_THRUST:.2f}...")
    dt = 1.0 / LOOP_HZ
    alt = 0.0
    while True:
        press = read_pressure()
        if press is not None:
            alt = pressure_to_alt(press, ground_hpa)
        # Climb at chosen takeoff thrust, then ease in with P-control for capture.
        if alt < TARGET_ALT - ALT_ACCURACY:
            thrust = TAKEOFF_THRUST
        else:
            thrust = alt_thrust(alt, TARGET_ALT)
        send_attitude(0.0, 0.0, hold_yaw, thrust)  # level, nose held
        # Stream flow + height during climb so the EKF converges a position
        # estimate by the time we reach altitude.
        send_dummy_flow()
        send_dummy_distance(alt)
        print(f"    Baro alt: {alt:5.2f} m  thrust: {thrust:.2f}", end="\r")
        if abs(alt - TARGET_ALT) <= ALT_ACCURACY:
            print(f"\n[+] Reached {alt:.2f} m")
            return
        time.sleep(dt)


# --- Wait for a flow-derived horizontal position estimate ------------------ #
def wait_for_position(ground_hpa, hold_yaw, timeout=5):
    """
    Stream optical flow + hold attitude until the EKF reports a *relative*
    horizontal position (EKF_STATUS_REPORT flag EKF_POS_HORIZ_REL). GUIDED
    rejects the mode switch ("requires position") until this is set.
    """
    print("[*] Streaming optical flow, waiting for EKF horizontal position...")
    request_message(mavutil.mavlink.MAVLINK_MSG_ID_EKF_STATUS_REPORT, 5)
    deadline = time.time() + timeout
    while time.time() < deadline:
        send_dummy_flow()
        press = read_pressure()
        alt = pressure_to_alt(press, ground_hpa) if press is not None else TARGET_ALT
        send_dummy_distance(alt)
        send_attitude(0.0, 0.0, hold_yaw, alt_thrust(alt, TARGET_ALT))
        msg = master.recv_match(type="EKF_STATUS_REPORT", blocking=False)
        if msg and (msg.flags & mavutil.mavlink.EKF_POS_HORIZ_REL):
            print("[+] EKF has horizontal position (flow fused)")
            return True
        time.sleep(0.1)
    print("[!] Timed out waiting for EKF position - GUIDED may be rejected")
    return False


# ========================= Mission ========================================= #
request_message(mavutil.mavlink.MAVLINK_MSG_ID_SCALED_PRESSURE, LOOP_HZ)
request_message(mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD, LOOP_HZ)
time.sleep(1.0)

set_gps_origin()
set_home()

# Establish barometer/heading references for attitude-based takeoff.
ground_hpa = capture_ground_pressure()
GROUND_HPA = ground_hpa  # used by send_velocity to report height-above-ground
hold_yaw = wait_for_heading()

# Provide EKF time to process the origin and home changes.
for _ in range(20):
    send_dummy_flow()
    time.sleep(0.1)

# --- GUIDED_NOGPS takeoff with attitude + thrust --- #
master.set_mode_apm("GUIDED_NOGPS")
master.arducopter_arm()
takeoff_to(ground_hpa, hold_yaw)

# --- Stream dummy optical flow until the EKF has a horizontal position,
#     so the GUIDED switch below isn't rejected with "requires position". --- #
wait_for_position(ground_hpa, hold_yaw)

# --- Switch to GUIDED and move forward with velocity commands --- #

send_velocity(1.0, 0.0, 0.0, duration=5)

master.set_mode_apm("GUIDED")
time.sleep(1.0)
print("Beginning forward velocity commands...")
send_velocity(1.0, 0.0, 0.0, duration=1000)
print("Forward motion complete")

# --- Land --- #
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_NAV_LAND,
    0, 0, 0, 0, 0, 0, 0, 0
)
print("Land command sent")
