from pathlib import Path
import re
import cv2
import numpy as np

try:
	from .base_video_provider import VideoProvider
except ImportError:
	from base_video_provider import VideoProvider


class BmpFolderVideoProvider(VideoProvider):
	"""Drop-in replacement for cv2.VideoCapture that reads a BMP image folder.

	Hardcoded filter criteria: only files whose stem matches ``cam0_image<digits>``
	(e.g. ``cam0_image02700.bmp``), sorted in ascending numeric order.
	"""

	_STEM_PATTERN: re.Pattern[str] = re.compile(r"^cam0_image(\d+)$")

	def __init__(self, folder: str) -> None:
		folder_path = Path(folder)

		def _sort_key(p: Path) -> int:
			m = self._STEM_PATTERN.match(p.stem)
			return int(m.group(1)) if m else -1

		candidates = sorted(folder_path.glob("*.bmp"), key=_sort_key)
		self._files: list[Path] = [p for p in candidates if self._STEM_PATTERN.match(p.stem)]
		self._index: int = 0
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
		if self._index >= len(self._files):
			return False, None
		frame = cv2.imread(str(self._files[self._index]))
		self._index += 1
		if frame is None:
			return False, None
		return True, frame

	def release(self) -> None:
		pass  # nothing to close
