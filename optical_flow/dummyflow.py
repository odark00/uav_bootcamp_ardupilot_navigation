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


while True:
    send_dummy_flow()
    time.sleep(0.001)
