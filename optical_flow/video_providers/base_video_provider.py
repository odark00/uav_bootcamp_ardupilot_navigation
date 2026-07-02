from abc import ABC, abstractmethod

import numpy as np


class VideoProvider(ABC):
	@abstractmethod
	def is_opened(self) -> bool:
		raise NotImplementedError

	@abstractmethod
	def frame_width(self) -> int:
		raise NotImplementedError

	@abstractmethod
	def fps(self) -> float:
		raise NotImplementedError

	@abstractmethod
	def read(self) -> tuple[bool, np.ndarray | None]:
		raise NotImplementedError

	@abstractmethod
	def release(self) -> None:
		raise NotImplementedError