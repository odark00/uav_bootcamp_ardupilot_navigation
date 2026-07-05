# Simulation startup notes

## Recommended environment
Use WSL2 Ubuntu + Docker Desktop with WSL integration enabled.

Reason:
- repo startup is Linux-oriented
- `start.sh` uses `xhost`
- `docker-compose.yml` mounts `/tmp/.X11-unix`
- compose uses `network_mode: host`
- container expects ROS 2 + Gazebo + ArduPilot SITL on a Linux-style runtime

## Intended startup path
1. Open the repo inside WSL, not PowerShell
2. Ensure Docker Desktop is running with WSL integration
3. In WSL, install:
   - `x11-xserver-utils` for `xhost`
4. Run:
   - `chmod +x start.sh`
   - `./start.sh`

## What the repo does
- builds image from `Dockerfile`
- starts container from `docker-compose.yml`
- launches:
  - `ros2 launch ardupilot_gz_bringup iris_runway.launch.py`
- later runs:
  - `python control_drone_gazebo.py`

## Relevant files
- `start.sh`
- `docker-compose.yml`
- `Dockerfile`
- `ardupilot_gz_bringup/launch/iris_runway.launch.py`
- `ardupilot_gz_bringup/launch/robots/iris.launch.py`
- `control_drone_gazebo.py`

## Expected runtime details
- MAVLink script listens on UDP `14550`
- SITL/launch config references:
  - port `2019`
  - master `tcp:127.0.0.1:5760`
  - sitl `127.0.0.1:5501`

## Fallback if Docker path is problematic
Native WSL Ubuntu run:
1. install ROS 2 Humble + Gazebo + ArduPilot deps
2. build workspace with `colcon`
3. run:
   - `source install/setup.bash`
   - `ros2 launch ardupilot_gz_bringup iris_runway.launch.py`
4. in another terminal run:
   - `python3 control_drone_gazebo.py`

## Main recommendation
Try WSL2 + Docker first.
Do not expect native Windows Docker/Desktop GUI forwarding to work cleanly with this repo as-is.