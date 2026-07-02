# UAV Bootcamp ArduPilot Navigation

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
