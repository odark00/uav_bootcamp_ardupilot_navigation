from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class MotionEstimate:
	dx_px: float
	dy_px: float
	valid: bool
	tracked_points: int
	source_points: int
	confidence: float
	note: str = ""

class MotionEstimator(ABC):
	@abstractmethod
	def process(self, gray_frame: np.ndarray) -> MotionEstimate:
		raise NotImplementedError