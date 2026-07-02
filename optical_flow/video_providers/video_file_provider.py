import cv2
import numpy as np

try:
	from .base_video_provider import VideoProvider
except ImportError:
	from base_video_provider import VideoProvider

class OpenCvVideoProvider(VideoProvider):
	def __init__(self, video_path: str) -> None:
		self._capture = cv2.VideoCapture(video_path)

	def is_opened(self) -> bool:
		return bool(self._capture.isOpened())

	def frame_width(self) -> int:
		return int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))

	def fps(self) -> float:
		return float(self._capture.get(cv2.CAP_PROP_FPS))

	def read(self) -> tuple[bool, np.ndarray | None]:
		return self._capture.read()

	def release(self) -> None:
		self._capture.release()