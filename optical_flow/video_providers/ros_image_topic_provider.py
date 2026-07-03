from __future__ import annotations

from collections import deque
import time

import numpy as np

try:
	import rclpy
	from cv_bridge import CvBridge
	from rclpy.node import Node
	from sensor_msgs.msg import Image
except ImportError:
	rclpy = None
	CvBridge = None
	Node = None
	Image = None

try:
	from .base_video_provider import VideoProvider
except ImportError:
	from base_video_provider import VideoProvider


class _ImageSubscriberNode(Node):
	def __init__(self, topic: str) -> None:
		super().__init__("optical_flow_image_subscriber")
		self._bridge = CvBridge()
		self._frames: deque[np.ndarray] = deque(maxlen=1)
		self._width = 0
		self._height = 0
		self._first_frame_received = False
		self._last_stamp: float | None = None
		self._fps_estimate: float = 0.0

		self.create_subscription(Image, topic, self._on_image, 10)

	def _on_image(self, msg: Image) -> None:
		frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
		self._frames.append(frame)
		self._width = int(msg.width)
		self._height = int(msg.height)
		self._first_frame_received = True

		stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
		if self._last_stamp is not None:
			dt = stamp_sec - self._last_stamp
			if dt > 1e-6:
				instant_fps = 1.0 / dt
				if self._fps_estimate <= 0.0:
					self._fps_estimate = instant_fps
				else:
					# Smooth jittery image timestamp deltas.
					self._fps_estimate = 0.9 * self._fps_estimate + 0.1 * instant_fps
		self._last_stamp = stamp_sec

	def pop_frame(self) -> np.ndarray | None:
		if not self._frames:
			return None
		return self._frames.popleft()

	def has_frame(self) -> bool:
		return self._first_frame_received

	@property
	def width(self) -> int:
		return self._width

	@property
	def fps_estimate(self) -> float:
		return self._fps_estimate


class RosImageTopicVideoProvider(VideoProvider):
	"""VideoProvider implementation that consumes ROS2 sensor_msgs/Image."""

	def __init__(self, topic: str, frame_timeout_s: float = 1.0, fallback_fps: float = 30.0) -> None:
		if rclpy is None or CvBridge is None:
			raise RuntimeError(
				"ROS2 image provider requires 'rclpy', 'sensor_msgs', and 'cv_bridge'."
			)

		self._frame_timeout_s = max(0.05, float(frame_timeout_s))
		self._fallback_fps = max(0.1, float(fallback_fps))
		self._closed = False

		if not rclpy.ok():
			rclpy.init(args=None)

		self._node = _ImageSubscriberNode(topic)

		# Wait briefly for the first frame so is_opened/frame_width are meaningful.
		deadline = time.time() + self._frame_timeout_s
		while time.time() < deadline and not self._node.has_frame():
			rclpy.spin_once(self._node, timeout_sec=0.05)

	def is_opened(self) -> bool:
		return (not self._closed) and self._node.has_frame()

	def frame_width(self) -> int:
		return self._node.width

	def fps(self) -> float:
		estimate = self._node.fps_estimate
		return estimate if estimate > 0.0 else self._fallback_fps

	def read(self) -> tuple[bool, np.ndarray | None]:
		if self._closed:
			return False, None

		deadline = time.time() + self._frame_timeout_s
		while time.time() < deadline:
			rclpy.spin_once(self._node, timeout_sec=0.05)
			frame = self._node.pop_frame()
			if frame is not None:
				return True, frame

		return False, None

	def release(self) -> None:
		if self._closed:
			return
		self._closed = True
		self._node.destroy_node()