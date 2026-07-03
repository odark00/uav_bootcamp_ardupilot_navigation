"""
MAVLink GUIDED circle mission + AirSim camera stream (non-blocking, high FPS).
Works with ArduPilot SITL, AirSim SITL, and real drones.
"""

import threading
#import airsim
import cv2
import numpy as np
from pymavlink import mavutil
import time
import math
from queue import Queue


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
    print("[!] EKF wait timeout — arming anyway (ARMING_CHECK=0 will override)")
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

def mission_guided(mav, altitude_m=10, radius=10, speed=3):
    if not wait_for_ekf(mav, timeout=120):
        print("[~] Continuing despite EKF timeout (force arm path enabled)")

    set_mode(mav, "GUIDED")

    if not arm(mav, force=True):
        return

    takeoff(mav, altitude_m)

    duration = 60
    end = time.time() + duration
    while time.time() < end:
        print("send_velocity")
        send_velocity(mav, vx=2, vy=0, vz=0, yaw=0, yaw_rate=0.3)
        time.sleep(0.05)

    land(mav)
    print("[+] Mission complete")


if __name__ == "__main__":
    mav = init_connections()

    mission_guided(mav)

    print("[+] Mission finished. Press Q to close camera.")
