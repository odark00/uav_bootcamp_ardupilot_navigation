import argparse
import math
import sys
from collections import deque
from dataclasses import dataclass
from time import time
from constants import OPTOCAL_FLOW_TOPIC
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import cv2
import numpy as np

from optical_flow.estimators.lk_estimator import LucasKanadeEstimator
from optical_flow.estimators.yaw_robust_estimator import YawRobustEstimator
from optical_flow.estimators.motion_estimator import MotionEstimate, MotionEstimator

try:
	from optical_flow.video_providers.base_video_provider import VideoProvider
	from optical_flow.video_providers.bmp_folder_video_provider import BmpFolderVideoProvider
	from optical_flow.video_providers.camera_video_provider import OpenCvCameraVideoProvider
	from optical_flow.video_providers.looping_bmp_folder_video_provider import LoopingBmpFolderVideoProvider
	from optical_flow.video_providers.video_file_provider import OpenCvVideoProvider
except ModuleNotFoundError:
	from video_providers.base_video_provider import VideoProvider
	from video_providers.bmp_folder_video_provider import BmpFolderVideoProvider
	from video_providers.camera_video_provider import OpenCvCameraVideoProvider
	from video_providers.looping_bmp_folder_video_provider import LoopingBmpFolderVideoProvider
	from video_providers.video_file_provider import OpenCvVideoProvider



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
		time_usec = int(time() * 1e6)

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


class MavLinkReporter(Node):
	def __init__(self) -> None:
		super().__init__("optical_flow_reporter")
		self.publisher = self.create_publisher(String, OPTOCAL_FLOW_TOPIC, 10)

	def maybe_print_optical_flow(self, frame_idx: int, flow: MavlinkOpticalFlow) -> None:
		# Invert dx and dy to match the expected coordinate system for optical flow
		res_dx_px = flow.flow_comp_m_y * 3
		res_dy_px = -flow.flow_comp_m_x * 3
		msg = String()
		msg.data = (
            f"time_usec={flow.time_usec} sensor_id={flow.sensor_id} "
            f"flow_x={flow.flow_x} flow_y={flow.flow_y} "
            f"flow_comp_m_x={res_dx_px:.5f} flow_comp_m_y={res_dy_px:.5f} "
            f"quality={flow.quality} ground_distance={flow.ground_distance:.3f}"
        )
		self.publisher.publish(msg)
		

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


class OpticalFlowFacade:
	def __init__(self, report_every = 1, smoothing_window = 5, ros_image_topic: str = "camera/image", hfov: float = 115.0, width: int = 640) -> None:
		rclpy.init()
		self.report_every = max(1, report_every)
		self.smoothing_window = max(1, smoothing_window)
		self.hfov = hfov
		self.width = width
		self.provider: VideoProvider | None = ModuleNotFoundError
		self.estimator: MotionEstimator | None = create_estimator('yaw_robust')
		self.smoother: RollingVelocitySmoother | None = RollingVelocitySmoother(self.smoothing_window)
		self.reporter: MavLinkReporter | None = MavLinkReporter()
		self.flow_composer: MavlinkOpticalFlowComposer | None = MavlinkOpticalFlowComposer(sensor_id=0)
		self.display: VideoMotionDisplay | None = VideoMotionDisplay(arrow_scale_px_per_mps=40.0)
		self.invalid_frames = 0

		try:
			from optical_flow.video_providers.ros_image_topic_provider import RosImageTopicVideoProvider
		except ModuleNotFoundError:
			from video_providers.ros_image_topic_provider import RosImageTopicVideoProvider

		self.provider = RosImageTopicVideoProvider(
			topic=ros_image_topic,
			frame_timeout_s=1.0,
			fallback_fps=30,
		)

	def process_frame(
			self,
			frame_idx: int,
			frame: np.ndarray,
			fps: float,
			altitude_m: float,
		) -> MavlinkOpticalFlow:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		estimate = self.estimator.process(gray)
		mpp = meters_per_pixel(altitude_m, self.hfov, self.width)

		if estimate.valid:
			vx = - estimate.dx_px * mpp * fps
			vy = - estimate.dy_px * mpp * fps
			speed = math.hypot(vx, vy)
			smooth_vx, smooth_vy, smooth_speed = self.smoother.update(vx, vy, speed)

			flow_msg = self.flow_composer.compose(
				frame_idx=frame_idx,
				fps=fps,
				estimate=estimate,
				meters_per_pixel_scale=mpp,
				ground_distance_m=altitude_m,
			)
			self.reporter.maybe_print_optical_flow(frame_idx, flow_msg)

			if self.display is not None:
				_ = not self.display.show(
					frame,
					smooth_vx,
					smooth_vy,
					smooth_speed,
					estimate.confidence,
					frame_idx,
				)
			return flow_msg
		else:
			if estimate.note and frame_idx % max(10, self.report_every) == 0:
				self.reporter.print_warning(frame_idx, estimate.note)

			if self.display is not None:
				_ = not self.display.show(
					frame,
					None,
					None,
					None,
					estimate.confidence,
					frame_idx,
					estimate.note,
				)
			return None

		# if stop_requested:
		# 	print("Display closed by user.", flush=True)


def create_video_provider(args: argparse.Namespace) -> VideoProvider:
	source_count = int(bool(args.video))
	source_count += int(bool(args.bmp_folder))
	source_count += int(bool(args.loop_bmp_folder))
	source_count += int(args.camera_index is not None)
	source_count += int(bool(args.ros_image_topic))

	if source_count > 1:
		raise ValueError(
			"provide exactly one source: video file, --bmp-folder, --loop-bmp-folder, "
			"--camera-index, or --ros-image-topic"
		)

	if args.ros_image_topic:
		try:
			try:
				from optical_flow.video_providers.ros_image_topic_provider import RosImageTopicVideoProvider
			except ModuleNotFoundError:
				from video_providers.ros_image_topic_provider import RosImageTopicVideoProvider

			provider = RosImageTopicVideoProvider(
				topic=args.ros_image_topic,
				frame_timeout_s=args.ros_frame_timeout,
				fallback_fps=args.ros_fps,
			)
		except (ModuleNotFoundError, RuntimeError, TypeError) as exc:
			raise ValueError(str(exc)) from exc
		if not provider.is_opened():
			raise ValueError(
				f"cannot open ROS image topic '{args.ros_image_topic}' "
				f"(timeout {args.ros_frame_timeout:.2f}s)"
			)
		return provider

	if args.camera_index is not None:
		provider = OpenCvCameraVideoProvider(args.camera_index)
		if not provider.is_opened():
			raise ValueError(f"cannot open camera index {args.camera_index}")
		return provider

	if args.loop_bmp_folder:
		provider = LoopingBmpFolderVideoProvider(args.loop_bmp_folder)
		if not provider.is_opened():
			raise ValueError(f"cannot open looping BMP folder '{args.loop_bmp_folder}'")
		return provider

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

	raise ValueError(
		"provide either a video file, --bmp-folder, --loop-bmp-folder, "
		"--camera-index, or --ros-image-topic"
	)


def create_estimator(name: str) -> MotionEstimator:
	estimators: dict[str, type[MotionEstimator]] = {
		"lk": LucasKanadeEstimator,
		"yaw_robust": YawRobustEstimator,
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
	parser.add_argument("--hfov", type=float, default=115.0, help="Horizontal FOV in degrees (default: 115)")
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
		"--loop-bmp-folder",
		default=None,
		metavar="DIR",
		help="Path to folder of BMP images that should loop from the first image after the last",
	)
	parser.add_argument(
		"--camera-index",
		type=int,
		default=None,
		help="Camera device index to stream from (e.g. 0 for default webcam)",
	)
	parser.add_argument(
		"--ros-image-topic",
		default=None,
		help="ROS2 sensor_msgs/Image topic (e.g. camera/image)",
	)
	parser.add_argument(
		"--ros-fps",
		type=float,
		default=30.0,
		help="Fallback FPS for ROS image topic when timestamps are unavailable (default: 30)",
	)
	parser.add_argument(
		"--ros-frame-timeout",
		type=float,
		default=1.0,
		help="ROS image topic frame timeout in seconds (default: 1.0)",
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


def create_optical_flow_facade():
	return OpticalFlowFacade(
		report_every=1,
		smoothing_window=5,
		ros_image_topic="camera/image",
		hfov=115.0,
		width=640
	)


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


	frame_idx = 0

	facade = create_optical_flow_facade()

	print("Starting processing...", flush=True)

	while True:
		frame_idx += 1
		ok, frame = facade.provider.read()
		if not ok:
			print(f"Could not read frame {frame_idx}")
			continue

		msg = facade.process_frame(
			frame_idx=frame_idx,
			frame=frame,
			fps=30.0,
			altitude_m=args.altitude,
		)
		print(f"[optical_flow_estimator] frame={frame_idx} msg={msg}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
