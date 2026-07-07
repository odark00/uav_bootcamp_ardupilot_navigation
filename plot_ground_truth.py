"""
Live ground-truth plots (velocity + yaw) from Gazebo, to confirm the drone flies
with a stable speed and heading while under optical-flow / GPS-denied control.

Ground truth = Gazebo's /odometry (nav_msgs/Odometry), bridged from
/model/iris/odometry. This is the TRUE simulator state, independent of the EKF /
optical-flow estimate -- so it tells you whether the drone is actually stable, not
just whether the autopilot *thinks* it is.

  * pose.pose.orientation  -> yaw (deg)
  * twist.twist.linear     -> ground speed (m/s), body frame (child_frame_id)

Rendering uses matplotlib's headless Agg backend (already in the base image -- no
tornado, no WebAgg, no X display). A background thread re-renders the two stacked
subplots to a PNG a few times a second; a tiny stdlib http.server serves an
auto-refreshing page. The SITL container publishes the port (docker-compose.yml),
so on the Mac just open:

  http://localhost:8088/

It is started automatically by the docker-compose `command`. Config via env:
  PLOT_PORT (default 8088), PLOT_WINDOW seconds (default 30),
  PLOT_FULL_SPEED=1 to plot 3D speed magnitude instead of horizontal ground speed.
"""

import io
import math
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import matplotlib

matplotlib.use("Agg")  # headless raster: no X display, no tornado/WebAgg
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

PORT = int(os.environ.get("PLOT_PORT", "8088"))
WINDOW_S = float(os.environ.get("PLOT_WINDOW", "30"))
FULL_SPEED = os.environ.get("PLOT_FULL_SPEED", "").lower() in ("1", "true", "yes")

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Ground truth: velocity &amp; yaw</title>
<style>
  html,body{margin:0;height:100%;background:#0e0e10;color:#ccc;
            font-family:system-ui,sans-serif}
  #wrap{display:flex;flex-direction:column;align-items:center}
  img{max-width:100%;height:auto;display:block}
  .hint{font-size:12px;opacity:.6;padding:6px}
</style></head>
<body><div id="wrap">
  <img id="p" src="/plot.png" alt="ground-truth velocity and yaw">
  <div class="hint">live from Gazebo /odometry &middot; refreshes ~4 Hz</div>
</div>
<script>
  setInterval(function(){
    document.getElementById('p').src = '/plot.png?t=' + Date.now();
  }, 250);
</script></body></html>
"""


def yaw_from_quaternion(x, y, z, w):
    """Yaw (rotation about Z) in radians from a quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class GroundTruthCollector(Node):
    """Subscribes to /odometry and stores a rolling window of (t, speed, yaw)."""

    def __init__(self, window_s, horizontal_only):
        super().__init__("ground_truth_plotter")
        self.window_s = float(window_s)
        self.horizontal_only = bool(horizontal_only)
        self.lock = threading.Lock()
        self.t = deque()
        self.speed = deque()
        self.yaw = deque()
        self.t0 = None
        self.create_subscription(Odometry, "/odometry", self._cb, 10)

    def _cb(self, msg: Odometry):
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = t
        t -= self.t0

        v = msg.twist.twist.linear
        if self.horizontal_only:
            speed = math.hypot(v.x, v.y)
        else:
            speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

        q = msg.pose.pose.orientation
        yaw = math.degrees(yaw_from_quaternion(q.x, q.y, q.z, q.w))

        with self.lock:
            self.t.append(t)
            self.speed.append(speed)
            self.yaw.append(yaw)
            while self.t and (self.t[-1] - self.t[0]) > self.window_s:
                self.t.popleft()
                self.speed.popleft()
                self.yaw.popleft()

    def snapshot(self):
        with self.lock:
            return list(self.t), list(self.speed), list(self.yaw)


class Renderer:
    """Owns the matplotlib figure; re-renders to PNG bytes on a background thread.

    Only this one thread ever touches the figure (matplotlib is not thread-safe);
    HTTP handlers just read the latest bytes under a lock.
    """

    def __init__(self, collector, full_speed):
        self.collector = collector
        self.full_speed = full_speed
        self.lock = threading.Lock()
        self.png = self._blank_png()

        self.fig, (self.ax_v, self.ax_y) = plt.subplots(
            2, 1, figsize=(9, 7), sharex=True)
        self.fig.subplots_adjust(hspace=0.28, left=0.1, right=0.97, top=0.95)

        speed_label = "3D speed" if full_speed else "horizontal ground speed"
        (self.v_line,) = self.ax_v.plot([], [], color="tab:blue", lw=1.6)
        self.ax_v.set_ylabel("speed (m/s)")
        self.ax_v.set_title(f"Ground-truth velocity  ({speed_label})")
        self.ax_v.grid(True, alpha=0.3)
        self.v_txt = self.ax_v.text(0.99, 0.95, "", transform=self.ax_v.transAxes,
                                    ha="right", va="top", color="tab:blue")

        (self.y_line,) = self.ax_y.plot([], [], color="tab:orange", lw=1.6)
        self.ax_y.set_ylabel("yaw (deg)")
        self.ax_y.set_xlabel("time (s, sim clock)")
        self.ax_y.set_title("Ground-truth yaw (heading)")
        self.ax_y.set_ylim(-185, 185)
        self.ax_y.grid(True, alpha=0.3)
        self.y_txt = self.ax_y.text(0.99, 0.95, "", transform=self.ax_y.transAxes,
                                    ha="right", va="top", color="tab:orange")

    def _blank_png(self):
        fig = plt.figure(figsize=(9, 7))
        fig.text(0.5, 0.5, "waiting for /odometry ...", ha="center", va="center")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        return buf.getvalue()

    def _render_once(self):
        t, speed, yaw = self.collector.snapshot()
        if not t:
            return
        self.v_line.set_data(t, speed)
        self.y_line.set_data(t, yaw)
        self.ax_v.set_xlim(t[0], max(t[-1], t[0] + 1e-3))
        smax = max(speed)
        self.ax_v.set_ylim(0, max(1.0, smax * 1.2))
        self.v_txt.set_text(f"now {speed[-1]:5.2f}   max {smax:5.2f}   "
                            f"mean {sum(speed)/len(speed):5.2f} m/s")
        self.y_txt.set_text(f"now {yaw[-1]:6.1f}   range {max(yaw)-min(yaw):5.1f} deg")

        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=100)
        with self.lock:
            self.png = buf.getvalue()

    def run(self, period_s=0.25):
        while True:
            try:
                self._render_once()
            except Exception as exc:  # keep serving even if one render hiccups
                print(f"[ground-truth plots] render error: {exc}", flush=True)
            time.sleep(period_s)

    def latest(self):
        with self.lock:
            return self.png


def make_handler(renderer):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/plot.png"):
                png = renderer.latest()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(png)
            else:
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *args):  # silence per-request logging
            pass

    return Handler


def main():
    rclpy.init()
    node = GroundTruthCollector(window_s=WINDOW_S, horizontal_only=not FULL_SPEED)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    renderer = Renderer(node, full_speed=FULL_SPEED)
    threading.Thread(target=renderer.run, daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", PORT), make_handler(renderer))
    print(f"[ground-truth plots] serving live velocity + yaw at "
          f"http://localhost:{PORT}/  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
