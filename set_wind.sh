#!/usr/bin/env bash
# Set the Gazebo world wind at runtime (no rebuild) via the WindEffects topic.
# Wind is given meteorologically: a SPEED and the direction it blows *from*.
#
# Usage:
#   ./set_wind.sh [SPEED_MPS] [FROM_DIRECTION]
#     SPEED_MPS       wind speed in m/s                       (default: 15)
#     FROM_DIRECTION  compass point the wind comes from:      (default: E)
#                     N NE E SE S SW W NW, or a bearing in
#                     degrees (0=N, 90=E, clockwise)
#
# Examples:
#   ./set_wind.sh              # 15 m/s from the east (default)
#   ./set_wind.sh 15 E         # same, explicit
#   ./set_wind.sh 8 NW         # 8 m/s from the north-west
#   ./set_wind.sh 0 N          # calm (stops the wind)
#
# Gazebo's world frame is ENU (x=East, y=North, z=Up). "From east" (easterly) blows
# toward the west, so it is published as linear_velocity x=-SPEED.
set -euo pipefail

CONTAINER="uav_bootcamp_ardupilot_navigation-ardupilot-sitl-1"
WORLD="map"

SPEED="${1:-3}"
FROM="${2:-E}"

# Compass point -> bearing in degrees (the direction the wind comes FROM).
case "${FROM^^}" in
  N)  DEG=0   ;;
  NE) DEG=45  ;;
  E)  DEG=90  ;;
  SE) DEG=135 ;;
  S)  DEG=180 ;;
  SW) DEG=225 ;;
  W)  DEG=270 ;;
  NW) DEG=315 ;;
  *[!0-9.]*) echo "Unknown direction: '$FROM' (use N/NE/E/SE/S/SW/W/NW or degrees)" >&2; exit 1 ;;
  *)  DEG="$FROM" ;;   # numeric bearing
esac

# ENU velocity of the moving air: it travels TOWARD (bearing + 180deg), so
#   x(East)  = -SPEED*sin(DEG)
#   y(North) = -SPEED*cos(DEG)
read -r VX VY < <(awk -v s="$SPEED" -v d="$DEG" 'BEGIN{
  r = d * atan2(0,-1) / 180.0            # atan2(0,-1) = pi
  x = -s*sin(r); y = -s*cos(r)
  if ((x<0?-x:x) < 1e-9) x = 0           # snap tiny rounding to 0
  if ((y<0?-y:y) < 1e-9) y = 0
  printf "%.6g %.6g\n", x, y
}')

echo "[*] Wind ${SPEED} m/s from ${FROM} (bearing ${DEG}deg) -> ENU linear_velocity x=${VX} y=${VY} z=0"

docker exec -i "$CONTAINER" bash -lc "
  source /root/ardu_ws/install/setup.bash
  gz topic -t /world/${WORLD}/wind -m gz.msgs.Wind \
    -p 'linear_velocity: {x: ${VX}, y: ${VY}, z: 0}, enable_wind: true'
"

echo "[+] Wind published on /world/${WORLD}/wind"
