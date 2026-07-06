# UAV Bootcamp ArduPilot Navigation
# Run

Terminal 1

```bash
./start.sh
```


Terminal 2

```bash
./enter.sh
ros2 launch ardupilot_gz_bringup iris_runway.launch.py
```


Terminal 3

```bash
./enter.sh
mavproxy.py --console
param load dsgps.param
```

Exit from Terminal 3, and kill and run again "ros2 launch ardupilot_gz_bringup iris_runway.launch.py"

Terminal 4

```bash
./run_control_drone.sh
```



### ----

Iris quadcopter in Gazebo + ArduPilot SITL, with a straight-line "cruise" flight,
optical-flow lateral wind compensation, a runtime-tunable Gazebo wind field, and a
GPS-denied (GUIDED_NOGPS) flight path.

---

## 1. Start the simulator (Docker)

Brings up Gazebo + ArduPilot SITL and streams the Gazebo GUI to your browser over
noVNC (no host X server needed — works on macOS Docker Desktop).

```bash
docker compose up --build
```

Then open:

- **Gazebo GUI** → http://localhost:8080/vnc.html
- **RViz** → http://localhost:8081/vnc.html  (only when `RVIZ=1` is set in `docker-compose.yml`)

This launches `ros2 launch ardupilot_gz_bringup iris_runway.launch.py` on an in-container
virtual display (see `gz_web.sh`). The world (`ardupilot_gz_gazebo/worlds/iris_runway.sdf`)
starts with **no wind** — set it at runtime (see [Wind](#wind)).

Container name: `uav_bootcamp_ardupilot_navigation-ardupilot-sitl-1`.
Open a shell inside it any time with `./enter.sh`.

### Re-running / rebuilding

```bash
docker compose up --build     # rebuild + start (after editing the world, launch files,
                              #   gz_web.sh, Dockerfile, or the copied Python)
docker compose up             # start without rebuilding (image already current)
docker compose down           # stop (or Ctrl-C in the `up` terminal)
docker compose build --no-cache && docker compose up   # clean rebuild if a layer is stale
```

Files baked into the image (world SDF, launch files, `gz_web.sh`, `control_drone_gazebo.py`)
only take effect after a rebuild; live sim state (e.g. wind over the topic) does not.

---

## 2. Run the flight (control script)

The mission is **cruise control**: arm → take off to `--altitude` → fly **due north**
at `--speed` (world-NED velocity) with **yaw held at 0**, **endlessly**. Fixed axis + fixed
heading means only the N coordinate changes (E drifts only under an east wind), which makes
drift easy to read. No landing — stop with `Ctrl-C`. Logs position (N/E/alt) + yaw at 1 Hz.

Run it inside the container (helper script provided):

```bash
./run_control_drone.sh
# equivalent to:
docker exec -it uav_bootcamp_ardupilot_navigation-ardupilot-sitl-1 \
  bash -lc "python control_drone_gazebo.py --no-flow --altitude 10"
```

### Key flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--altitude M` | `10` | Takeoff altitude (m) |
| `--speed M/S` | `3` | Forward cruise speed (m/s, body-frame vx along heading, yaw held at 0) |
| `--guided_nogps` | off | Fly the GPS-denied GUIDED_NOGPS attitude cruise instead of GUIDED velocity cruise (see [GPS-denied](#gps-denied-guided_nogps)) |
| `--disable-gps-after-takeoff S` | `10` | Disable GPS this many seconds after takeoff (set `<0` to never disable) |
| `--no-flow` | off | Don't launch the optical-flow estimator (pure straight flight) |
| `--no-wind` | off | Zero the Gazebo wind at startup (WindEffects) — baseline / no-wind test |
| `--wind VX VY VZ` | — | Set Gazebo wind (m/s, **world frame**) at startup, e.g. `--wind 8 8 0` |

Optical-flow tuning flags (`--flow-min-quality`, `--flow-max-stale`, `--fallback-lateral-gain`,
`--fallback-max-vy`, `--flow-topic`, `--flow-hfov`, `--flow-fps`, `--flow-sensor-id`,
`--no-flow-display`) are also available — see `python control_drone_gazebo.py --help`.

`--no-wind` and `--wind` tune the live sim over gz-transport — **no world rebuild needed**.
If both are given, `--no-wind` wins.

### Examples

```bash
# Default: no wind (calm world), no optical flow
python control_drone_gazebo.py --no-flow --altitude 10

# Baseline: wind OFF
python control_drone_gazebo.py --no-flow --no-wind --altitude 10

# Custom crosswind at 45°, ~11.3 m/s
python control_drone_gazebo.py --no-flow --wind 8 8 0 --altitude 10

# With optical-flow lateral wind compensation enabled (drop --no-flow)
python control_drone_gazebo.py --altitude 10
```

### GPS-denied (GUIDED_NOGPS)

The same mission runs two ways; only the control path differs:

- **default (GUIDED)** — world-frame velocity setpoints. Needs a horizontal position
  estimate from the EKF (GPS, or optical flow fused into EKF3).
- **`--guided_nogps` (GUIDED_NOGPS)** — attitude + thrust, works GPS-denied. Forward
  motion is an open-loop pitch; altitude is held on the thrust (climb-rate) channel
  against **baro**. `MAV_CMD_NAV_TAKEOFF` is unavailable here (it needs a horizontal
  position estimate that baro + IMU cannot provide), so takeoff climbs on a level
  attitude to a baro-altitude target instead.

By default GPS is switched off `--disable-gps-after-takeoff` seconds (10 s) after takeoff,
so you can watch the aircraft fly the rest of the mission GPS-denied. Pass a negative value
to keep GPS on for the whole flight.

```bash
# GPS-denied cruise; GPS auto-disabled 10 s after takeoff
python control_drone_gazebo.py --guided_nogps --altitude 10

# GPS-denied, keep GPS on the whole flight
python control_drone_gazebo.py --guided_nogps --disable-gps-after-takeoff -1 --altitude 10
```

---

## 3. Optical flow (run separately)

Run the estimator in its own terminal in the same container (helper: `./run_optical_flow.sh`):

```bash
python optical_flow/optical_flow_estimator.py \
  --ros-image-topic /camera/image --altitude 10 --fps 30 --display --estimator yaw_robust
```

When flow is running and the control script is started **without** `--no-flow`, the flow's
lateral velocity estimate is used to command a small `vy` correction that counters wind
drift while cruising straight.

### Other optical-flow providers

Run from the project root:

```bash
# Onboard camera (webcam); use index 0 for the default camera
python optical_flow/optical_flow_estimator.py --camera-index 0 --altitude 700 --fps 30 --display

# Process a folder of BMP images once
python optical_flow/optical_flow_estimator.py --bmp-folder D:/path/to/bmp_folder --altitude 700 --fps 30 --display

# Ping-pong loop over a folder (forward/backward/forward...)
python optical_flow/optical_flow_estimator.py --loop-bmp-folder D:/path/to/bmp_folder --altitude 700 --fps 30 --display
```

---

## Wind

Wind is a **Gazebo-side** effect (ArduPilot's `SIM_WIND_*` params do **not** apply under the
Gazebo FDM). It's set up in two places:

- `ardupilot_gz_gazebo/worlds/iris_runway.sdf` — the world `<wind>` element (present but
  **zeroed**, purely so the world wind component exists) + the `gz-sim-wind-effects-system`
  plugin that turns the wind vector into a force.
- `iris_with_gimbal/model.sdf` — model-level `<enable_wind>true</enable_wind>` so the drone's
  links respond to the wind.

There is **no static wind** — it is controlled **only at runtime via the topic**
`/world/map/wind` (`gz.msgs.Wind`), no rebuild needed:

- **Helper script** (easiest) — `./set_wind.sh SPEED FROM`, where `FROM` is the compass
  direction the wind blows *from* (`N NE E SE S SW W NW`, or a bearing in degrees):

  ```bash
  ./set_wind.sh          # 15 m/s from the east (default)
  ./set_wind.sh 10 NW    # 10 m/s from the north-west
  ./set_wind.sh 0 N      # calm
  ```

- **Via the control script** — `--wind VX VY VZ` / `--no-wind`, or `set_gz_wind(vx, vy, vz)`
  in `control_drone_gazebo.py`.
- **Directly** — publish to the topic:

  ```bash
  docker compose exec ardupilot-sitl \
    gz topic -t /world/map/wind -m gz.msgs.Wind \
      -p 'linear_velocity: {x: 20, y: 20, z: 0}, enable_wind: true'
  ```

All routes publish to the same topic, which WindEffects adopts as the current wind. Gazebo's
world frame is **ENU** (x=East, y=North, z=Up), so a wind *from the east* is `x = -SPEED`.

---

## Helper scripts

| Script | Purpose |
|--------|---------|
| `start.sh` | `docker compose up --build` (with `xhost` for Linux hosts) |
| `enter.sh` | Open a bash shell in the running container |
| `run_control_drone.sh` | Run the cruise flight (`--no-flow --altitude 10`) |
| `run_optical_flow.sh` | Run the optical-flow estimator |
| `set_wind.sh` | Set the Gazebo wind at runtime, e.g. `./set_wind.sh 15 E` (15 m/s from the east) |
| `gz_web.sh` / `cam_web.sh` | Container entrypoints: Gazebo-GUI-over-noVNC / camera MJPEG |
