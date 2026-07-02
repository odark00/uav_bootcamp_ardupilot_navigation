import cv2
import numpy as np

try:
	from .base_video_provider import VideoProvider
except ImportError:
	from base_video_provider import VideoProvider


class OpenCvCameraVideoProvider(VideoProvider):
	"""Streams frames from a connected camera device."""

	def __init__(self, camera_index: int = 0) -> None:
		self._capture = cv2.VideoCapture(int(camera_index))

	def is_opened(self) -> bool:
		return bool(self._capture.isOpened())

	def frame_width(self) -> int:
		return int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))

	def fps(self) -> float:
		fps_value = float(self._capture.get(cv2.CAP_PROP_FPS))
		return fps_value if fps_value > 0 else 30.0

	def read(self) -> tuple[bool, np.ndarray | None]:
		return self._capture.read()

	def release(self) -> None:
		self._capture.release()
