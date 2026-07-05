import cv2

from estimators.motion_estimator import MotionEstimate, MotionEstimator
import numpy as np


class LucasKanadeEstimator(MotionEstimator):
	def __init__(self) -> None:
		self.prev_gray: np.ndarray | None = None
		self.prev_points: np.ndarray | None = None

		self.feature_params = dict(
			maxCorners=500,
			qualityLevel=0.01,
			minDistance=7,
			blockSize=7,
		)

		self.lk_params = dict(
			winSize=(21, 21),
			maxLevel=3,
			criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
		)

		self.min_track_points = 30
		self.refresh_threshold = 120

	def _detect_features(self, gray_frame: np.ndarray) -> np.ndarray | None:
		return cv2.goodFeaturesToTrack(gray_frame, mask=None, **self.feature_params)

	@staticmethod
	def _robust_displacement(displacements: np.ndarray) -> tuple[float, float]:
		if displacements.shape[0] == 1:
			return float(displacements[0, 0]), float(displacements[0, 1])

		median_vec = np.median(displacements, axis=0)
		residual = np.linalg.norm(displacements - median_vec, axis=1)
		threshold = np.percentile(residual, 80)
		inlier_mask = residual <= threshold
		inliers = displacements[inlier_mask]

		if inliers.shape[0] < 8:
			inliers = displacements

		robust = np.median(inliers, axis=0)
		return float(robust[0]), float(robust[1])

	def process(self, gray_frame: np.ndarray) -> MotionEstimate:
		if self.prev_gray is None:
			self.prev_gray = gray_frame
			self.prev_points = self._detect_features(gray_frame)
			count = 0 if self.prev_points is None else int(len(self.prev_points))
			return MotionEstimate(0.0, 0.0, False, 0, count, 0.0, "warmup")

		if self.prev_points is None or len(self.prev_points) < self.min_track_points:
			self.prev_points = self._detect_features(self.prev_gray)
			source = 0 if self.prev_points is None else int(len(self.prev_points))
			if source < self.min_track_points:
				self.prev_gray = gray_frame
				self.prev_points = self._detect_features(gray_frame)
				return MotionEstimate(0.0, 0.0, False, 0, source, 0.0, "insufficient_features")

		next_points, status, _err = cv2.calcOpticalFlowPyrLK(
			self.prev_gray,
			gray_frame,
			self.prev_points,
			None,
			**self.lk_params,
		)

		source_points = int(len(self.prev_points))
		if next_points is None or status is None:
			self.prev_gray = gray_frame
			self.prev_points = self._detect_features(gray_frame)
			return MotionEstimate(0.0, 0.0, False, 0, source_points, 0.0, "lk_failed")

		good_mask = status.reshape(-1) == 1
		good_new = next_points[good_mask]
		good_old = self.prev_points[good_mask]
		tracked_points = int(len(good_new))

		if tracked_points < self.min_track_points:
			self.prev_gray = gray_frame
			self.prev_points = self._detect_features(gray_frame)
			confidence = tracked_points / max(source_points, 1)
			return MotionEstimate(
				0.0,
				0.0,
				False,
				tracked_points,
				source_points,
				confidence,
				"low_track_quality",
			)

		displacements = (good_new - good_old).reshape(-1, 2)
		dx_px, dy_px = self._robust_displacement(displacements)
		confidence = tracked_points / max(source_points, 1)

		self.prev_gray = gray_frame
		if tracked_points < self.refresh_threshold:
			refreshed = self._detect_features(gray_frame)
			self.prev_points = refreshed if refreshed is not None else good_new.reshape(-1, 1, 2)
		else:
			self.prev_points = good_new.reshape(-1, 1, 2)

		return MotionEstimate(dx_px, dy_px, True, tracked_points, source_points, confidence)
