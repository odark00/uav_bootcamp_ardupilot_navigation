from pathlib import Path
import re

import cv2
import numpy as np

try:
	from .base_video_provider import VideoProvider
except ImportError:
	from base_video_provider import VideoProvider


class LoopingBmpFolderVideoProvider(VideoProvider):
	"""Reads BMP images using ping-pong order: forward then backward."""

	_STEM_PATTERN: re.Pattern[str] = re.compile(r"^cam0_image(\d+)$")

	def __init__(self, folder: str) -> None:
		folder_path = Path(folder)

		def _sort_key(p: Path) -> int:
			m = self._STEM_PATTERN.match(p.stem)
			return int(m.group(1)) if m else -1

		candidates = sorted(folder_path.glob("*.bmp"), key=_sort_key)
		self._files: list[Path] = [p for p in candidates if self._STEM_PATTERN.match(p.stem)]
		self._index: int = 0
		self._direction: int = 1
		self._width: int = 0
		if self._files:
			probe = cv2.imread(str(self._files[0]))
			if probe is not None:
				self._width = probe.shape[1]

	def is_opened(self) -> bool:
		return bool(self._files) and self._width > 0

	def frame_width(self) -> int:
		return self._width

	def fps(self) -> float:
		return 0.0  # not derivable from still images; caller must supply --fps

	def read(self) -> tuple[bool, np.ndarray | None]:
		if not self._files:
			return False, None
		if len(self._files) == 1:
			frame = cv2.imread(str(self._files[0]))
			return (frame is not None), frame

		for _ in range(len(self._files)):
			frame = cv2.imread(str(self._files[self._index]))

			next_index = self._index + self._direction
			if next_index >= len(self._files):
				self._direction = -1
				next_index = len(self._files) - 2
			elif next_index < 0:
				self._direction = 1
				next_index = 1
			self._index = next_index

			if frame is not None:
				return True, frame

		return False, None

	def release(self) -> None:
		pass  # nothing to close
