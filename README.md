# UAV Bootcamp ArduPilot Navigation

## Run As Two Separate Scripts

Use two terminals in the same ROS-enabled environment.

### 1) Run control script with optical flow disabled

```bash
python control_drone_gazebo.py --no-flow --altitude 10
```

### 2) Run optical flow script separately

```bash
python optical_flow/optical_flow_estimator.py --ros-image-topic /camera/image --altitude 10 --fps 30 --display --estimator yaw_robust
```

## Optical Flow Provider Launch

Run from the project root:

### 1) Onboard camera stream (connected webcam/camera)

Use camera index `0` for the default onboard camera (change index if needed):

```bash
python optical_flow/optical_flow_estimator.py --camera-index 0 --altitude 700 --fps 30 --display
```

### 2) Read from image folder

Process a folder of BMP images once:

```bash
python optical_flow/optical_flow_estimator.py --bmp-folder D:/path/to/bmp_folder --altitude 700 --fps 30 --display
```

Optional: ping-pong looping folder provider (forward/backward/forward...):

```bash
python optical_flow/optical_flow_estimator.py --loop-bmp-folder D:/path/to/bmp_folder --altitude 700 --fps 30 --display
```
