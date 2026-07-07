import time

from constants import OPTOCAL_FLOW_TOPIC
from optical_flow.optical_flow_estimator import MavlinkOpticalFlow
from rclpy.node import Node
from std_msgs.msg import String


class OpticalFlowRos2Subscriber(Node):
    def __init__(self, callback):
        print("Initializing ROS2 subscriber for optical flow...")
        super().__init__('optical_flow_subscriber')
        self.subscription = self.create_subscription(
            String,
            OPTOCAL_FLOW_TOPIC,
            callback,
            10)
        self.subscription  # prevent unused variable warning


class MavlinkPublisher:
    def __init__(self, mav, log: bool = False):
        self.log = bool(log)
        if self.log:
            print("Initializing Mavlink publisher for optical flow...")
        self.mav = mav.mav

    def ros_callback(self, msg: String):
        if self.log:
            print(f"Received ROS message: {msg.data}")
        # Parse the String message to extract the necessary fields
        fields = dict(item.split('=') for item in msg.data.split() if '=' in item)
        flow_comp_m_x = float(fields.get('flow_comp_m_x', 0.0))
        flow_comp_m_y = float(fields.get('flow_comp_m_y', 0.0))
        sensor_id = int(fields.get('sensor_id', 0))
        flow_x = int(fields.get('flow_x', 0))
        flow_y = int(fields.get('flow_y', 0))
        quality = int(fields.get('quality', 255))
        ground_distance = float(fields.get('ground_distance', 1.0))
        self.send_mavlink_flow(flow_comp_m_x, flow_comp_m_y, flow_x, flow_y, sensor_id, quality, ground_distance)

    def send_mavlink_flow(self, vx=0.0, vy=0.0, flow_x=0, flow_y=0, sensor_id=0, quality=255, ground_distance=1.0):
        ground_distance = max(0.1, float(ground_distance))

        # ArduPilot's AP_OpticalFlow_MAV backend consumes ONLY flow_x/flow_y (it sums
        # packet.flow_x/flow_y and ignores flow_comp_m_*, flow_rate_*, and
        # ground_distance). So the camera->body-frame direction correction MUST be
        # applied here on flow_x/flow_y -- transforms on flow_comp_m_* never reach nav,
        # which is why earlier "invert dy" fixes didn't change flight direction.
        # Camera is rotated 90deg vs body frame: body_x = -image_y, body_y = +image_x.
        # (If it still flies wrong, adjust these two lines -- direction is mount-dependent.)
        body_flow_x = -flow_y
        body_flow_y = flow_x

        message = (
            int(time.time() * 1e6),    # time_usec (uint64)
            sensor_id,                 # sensor_id (uint8)
            body_flow_x, body_flow_y,  # flow_x, flow_y (int16) -- ONLY fields ArduPilot uses
            float(vx), float(vy),      # flow_comp_m_x, flow_comp_m_y (float, ignored by AP)
            quality,                   # quality (uint8)
            ground_distance,           # ground_distance (float, meters, ignored by AP)
            0.0, 0.0                   # flow_rate_x, flow_rate_y (float, ignored by AP)
        )
        if self.log:
            print(f"Sending received ROS message to Mavlink: {message}")
        self.mav.optical_flow_send(*message)
