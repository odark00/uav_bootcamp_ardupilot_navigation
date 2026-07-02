import argparse
import math
import re
import sys
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
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


@dataclass
class MavlinkOpticalFlow:
	time_usec: int
	sensor_id: int
	flow_x: int
	flow_y: int
	flow_comp_m_x: float
	flow_comp_m_y: float
	quality: int
	ground_distance: float


class MotionEstimator(ABC):
	@abstractmethod
	def process(self, gray_frame: np.ndarray) -> MotionEstimate:
		raise NotImplementedError


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


class RollingVelocitySmoother:
	def __init__(self, window_size: int) -> None:
		self.window_size = max(1, window_size)
		self.vx_window: deque[float] = deque(maxlen=self.window_size)
		self.vy_window: deque[float] = deque(maxlen=self.window_size)
		self.v_window: deque[float] = deque(maxlen=self.window_size)

	def update(self, vx: float, vy: float, speed: float) -> tuple[float, float, float]:
		self.vx_window.append(vx)
		self.vy_window.append(vy)
		self.v_window.append(speed)
		return (
			float(np.mean(self.vx_window)),
			float(np.mean(self.vy_window)),
			float(np.mean(self.v_window)),
		)


class MavlinkOpticalFlowComposer:
	def __init__(self, sensor_id: int = 0) -> None:
		self.sensor_id = max(0, min(255, int(sensor_id)))

	@staticmethod
	def _clamp_int16(value: int) -> int:
		return max(-32768, min(32767, value))

	@staticmethod
	def _quality_from_confidence(confidence: float) -> int:
		clamped = max(0.0, min(1.0, float(confidence)))
		return int(round(clamped * 255.0))

	def compose(
		self,
		frame_idx: int,
		fps: float,
		estimate: MotionEstimate,
		meters_per_pixel_scale: float,
		ground_distance_m: float,
	) -> MavlinkOpticalFlow:
		time_usec = int((frame_idx / fps) * 1_000_000.0)

		# MAVLink flow_x/flow_y use dpix units (deci-pixels).
		flow_x = self._clamp_int16(int(round(estimate.dx_px * 10.0)))
		flow_y = self._clamp_int16(int(round(estimate.dy_px * 10.0)))

		# Compensated ground displacement over this frame interval (meters).
		flow_comp_m_x = - estimate.dx_px * meters_per_pixel_scale
		flow_comp_m_y = - estimate.dy_px * meters_per_pixel_scale

		quality = self._quality_from_confidence(estimate.confidence)

		return MavlinkOpticalFlow(
			time_usec=time_usec,
			sensor_id=self.sensor_id,
			flow_x=flow_x,
			flow_y=flow_y,
			flow_comp_m_x=flow_comp_m_x,
			flow_comp_m_y=flow_comp_m_y,
			quality=quality,
			ground_distance=ground_distance_m,
		)


class StreamingReporter:
	def __init__(self, every_n_frames: int) -> None:
		self.every_n_frames = max(1, every_n_frames)

	def maybe_print_optical_flow(self, frame_idx: int, flow: MavlinkOpticalFlow) -> None:
		if frame_idx % self.every_n_frames != 0:
			return
		print(
			"OPTICAL_FLOW(100) "
			f"time_usec={flow.time_usec} "
			f"sensor_id={flow.sensor_id} "
			f"flow_x={flow.flow_x} "
			f"flow_y={flow.flow_y} "
			f"flow_comp_m_x={flow.flow_comp_m_x:.5f} "
			f"flow_comp_m_y={flow.flow_comp_m_y:.5f} "
			f"quality={flow.quality} "
			f"ground_distance={flow.ground_distance:.3f}",
			flush=True,
		)

	@staticmethod
	def print_warning(frame_idx: int, reason: str) -> None:
		print(f"frame={frame_idx:06d} warning={reason}", flush=True)


class VideoMotionDisplay:
	def __init__(self, window_name: str = "Motion direction", arrow_scale_px_per_mps: float = 40.0) -> None:
		self.window_name = window_name
		self.arrow_scale_px_per_mps = max(1.0, arrow_scale_px_per_mps)
		self.is_open = False

	def show(
		self,
		frame: np.ndarray,
		vx: float | None,
		vy: float | None,
		speed: float | None,
		confidence: float | None,
		frame_idx: int,
		note: str = "",
	) -> bool:
		display = frame.copy()
		height, width = display.shape[:2]
		center = (width // 2, height // 2)

		if vx is not None and vy is not None and speed is not None and speed > 0:
			arrow_dx = int(vx * self.arrow_scale_px_per_mps)
			arrow_dy = int(vy * self.arrow_scale_px_per_mps)
			max_len = max(30, int(min(width, height) * 0.35))
			length = math.hypot(arrow_dx, arrow_dy)
			if length > max_len:
				scale = max_len / length
				arrow_dx = int(arrow_dx * scale)
				arrow_dy = int(arrow_dy * scale)

			end = (center[0] + arrow_dx, center[1] + arrow_dy)
			cv2.arrowedLine(display, center, end, (0, 255, 255), 4, tipLength=0.25)
			status = f"vx={vx:.2f} m/s vy={vy:.2f} m/s speed={speed:.2f} m/s"
		else:
			cv2.circle(display, center, 5, (0, 255, 255), -1)
			status = f"waiting for valid motion ({note or 'warmup'})"

		cv2.putText(
			display,
			f"frame={frame_idx}",
			(16, 32),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.75,
			(255, 255, 255),
			2,
			cv2.LINE_AA,
		)
		cv2.putText(
			display,
			status,
			(16, height - 22),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.65,
			(255, 255, 255),
			2,
			cv2.LINE_AA,
		)
		if confidence is not None:
			cv2.putText(
				display,
				f"conf={confidence:.2f}",
				(16, 62),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.65,
				(255, 255, 255),
				2,
				cv2.LINE_AA,
			)

		if not self.is_open:
			cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
			self.is_open = True
		cv2.imshow(self.window_name, display)

		key = cv2.waitKey(1) & 0xFF
		return key not in (ord("q"), 27)

	def close(self) -> None:
		if self.is_open:
			cv2.destroyWindow(self.window_name)
			self.is_open = False


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


def create_video_provider(args: argparse.Namespace) -> VideoProvider:
	if args.bmp_folder:
		provider = BmpFolderVideoProvider(args.bmp_folder)
		if not provider.is_opened():
			raise ValueError(f"cannot open BMP folder '{args.bmp_folder}'")
		return provider

	if args.video:
		provider = OpenCvVideoProvider(args.video)
		if not provider.is_opened():
			raise ValueError(f"cannot open video '{args.video}'")
		return provider

	raise ValueError("provide either a video file or --bmp-folder")


def create_estimator(name: str) -> MotionEstimator:
	estimators: dict[str, type[MotionEstimator]] = {
		"lk": LucasKanadeEstimator,
	}
	if name not in estimators:
		available = ", ".join(sorted(estimators.keys()))
		raise ValueError(f"Unknown estimator '{name}'. Available: {available}")
	return estimators[name]()


def meters_per_pixel(altitude_m: float, hfov_deg: float, frame_width_px: int) -> float:
	hfov_rad = math.radians(hfov_deg)
	ground_width_m = 2.0 * altitude_m * math.tan(hfov_rad / 2.0)
	return ground_width_m / frame_width_px


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Estimate drone image-plane speed using modular optical flow backends."
	)
	parser.add_argument("video", nargs="?", default=None, help="Path to input video file (omit when using --bmp-folder)")
	parser.add_argument("--altitude", type=float, required=True, help="Drone altitude above ground in meters")
	parser.add_argument("--hfov", type=float, default=84.0, help="Horizontal FOV in degrees (default: 84)")
	parser.add_argument("--fps", type=float, default=0.0, help="FPS override (default: use video metadata)")
	parser.add_argument("--estimator", default="lk", help="Motion estimator backend (default: lk)")
	parser.add_argument(
		"--smoothing-window",
		type=int,
		default=5,
		help="Rolling smoothing window in frames (default: 5)",
	)
	parser.add_argument(
		"--report-every",
		type=int,
		default=1,
		help="Print one update every N frames (default: 1 for immediate output)",
	)
	parser.add_argument(
		"--bmp-folder",
		default=None,
		metavar="DIR",
		help="Path to folder of BMP images to process instead of a video file",
	)
	parser.add_argument(
		"--display",
		action="store_true",
		help="Display processed video with a direction arrow",
	)
	parser.add_argument(
		"--arrow-scale",
		type=float,
		default=40.0,
		help="Arrow length scale in pixels per m/s when using --display (default: 40)",
	)
	parser.add_argument(
		"--sensor-id",
		type=int,
		default=0,
		help="MAVLink optical flow sensor ID (0..255, default: 0)",
	)
	return parser.parse_args()


def main() -> int:
	if hasattr(sys.stdout, "reconfigure"):
		sys.stdout.reconfigure(line_buffering=True)

	args = parse_args()

	if args.altitude <= 0:
		print("Error: --altitude must be greater than 0", file=sys.stderr)
		return 2
	if args.hfov <= 0 or args.hfov >= 179:
		print("Error: --hfov must be in (0, 179)", file=sys.stderr)
		return 2

	try:
		provider = create_video_provider(args)
	except ValueError as exc:
		print(f"Error: {exc}", file=sys.stderr)
		return 2

	width = provider.frame_width()
	if width <= 0:
		print("Error: invalid frame width in video metadata", file=sys.stderr)
		provider.release()
		return 2

	fps_meta = provider.fps()
	fps = args.fps if args.fps > 0 else fps_meta
	if fps <= 0:
		print("Error: FPS unavailable. Provide --fps explicitly", file=sys.stderr)
		provider.release()
		return 2

	try:
		estimator = create_estimator(args.estimator)
	except ValueError as exc:
		print(f"Error: {exc}", file=sys.stderr)
		provider.release()
		return 2

	mpp = meters_per_pixel(args.altitude, args.hfov, width)
	smoother = RollingVelocitySmoother(args.smoothing_window)
	reporter = StreamingReporter(args.report_every)
	flow_composer = MavlinkOpticalFlowComposer(sensor_id=args.sensor_id)
	display = VideoMotionDisplay(arrow_scale_px_per_mps=args.arrow_scale) if args.display else None

	frame_idx = 0
	valid_frames = 0
	invalid_frames = 0
	sum_vx = 0.0
	sum_vy = 0.0
	sum_speed = 0.0
	stop_requested = False

	print("Starting processing...", flush=True)

	while True:
		ok, frame = provider.read()
		if not ok:
			break

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		estimate = estimator.process(gray)

		if estimate.valid:
			vx = - estimate.dx_px * mpp * fps
			vy = - estimate.dy_px * mpp * fps
			speed = math.hypot(vx, vy)
			smooth_vx, smooth_vy, smooth_speed = smoother.update(vx, vy, speed)

			valid_frames += 1
			sum_vx += smooth_vx
			sum_vy += smooth_vy
			sum_speed += smooth_speed

			flow_msg = flow_composer.compose(
				frame_idx=frame_idx,
				fps=fps,
				estimate=estimate,
				meters_per_pixel_scale=mpp,
				ground_distance_m=args.altitude,
			)
			reporter.maybe_print_optical_flow(frame_idx, flow_msg)

			if display is not None:
				stop_requested = not display.show(
					frame,
					smooth_vx,
					smooth_vy,
					smooth_speed,
					estimate.confidence,
					frame_idx,
				)
		else:
			invalid_frames += 1
			if estimate.note and frame_idx % max(10, args.report_every) == 0:
				reporter.print_warning(frame_idx, estimate.note)

			if display is not None:
				stop_requested = not display.show(
					frame,
					None,
					None,
					None,
					estimate.confidence,
					frame_idx,
					estimate.note,
				)

		frame_idx += 1
		if stop_requested:
			print("Display closed by user.", flush=True)
			break

	provider.release()
	if display is not None:
		display.close()

	print("\nFinal summary:")
	print(f"  Processed frames: {frame_idx}")
	print(f"  Valid motion frames: {valid_frames}")
	print(f"  Invalid/warmup frames: {invalid_frames}")

	if valid_frames == 0:
		print("  No valid motion estimates were produced.")
		return 1

	avg_vx = sum_vx / valid_frames
	avg_vy = sum_vy / valid_frames
	avg_speed = sum_speed / valid_frames

	print(f"  Average X speed: {avg_vx:.3f} m/s")
	print(f"  Average Y speed: {avg_vy:.3f} m/s")
	print(f"  Average speed magnitude: {avg_speed:.3f} m/s")
	print(f"  Effective FPS used: {fps:.3f}")
	print(f"  Meters per pixel: {mpp:.6f}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
