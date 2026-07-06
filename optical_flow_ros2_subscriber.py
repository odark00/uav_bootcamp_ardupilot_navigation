import time

from optical_flow.optical_flow_estimator import MavlinkOpticalFlow
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class OpticalFlowRos2Subscriber(Node):

    def __init__(self, callback):
        rclpy.init()
        super().__init__('optical_flowl_subscriber')
        self.subscription = self.create_subscription(
            String,
            '/optical_flow',
            callback,
            10)
        self.subscription  # prevent unused variable warning


class MavlinkPublisher:
    def __init__(self, mav):
        self.mav = mav
        pass

    def ros_callback(self, msg: String):
        print(f"Sending received ROS message to Mavlink: {msg.data}")
        # Parse the String message to extract the necessary fields
        fields = dict(item.split('=') for item in msg.data.split() if '=' in item)
        flow_comp_m_x = float(fields.get('flow_comp_m_x', 0.0))
        flow_comp_m_y = float(fields.get('flow_comp_m_y', 0.0))
        quality = int(fields.get('quality', 255))
        self.send_mavlink_flow(flow_comp_m_x, flow_comp_m_y, quality)

    def send_mavlink_flow(self, vx=0.0, vy=0.0, quality=255):
        self.mav.optical_flow_send(
            int(time.time() * 1e6),  # time_usec (uint64)
            0,                       # sensor_id (uint8)
            0, 0,                    # flow_x, flow_y (int16, rad/sec*1000)
            float(vx), float(vy),    # flow_comp_m_x, flow_comp_m_y (float)
            quality,                 # quality (uint8)
            1.0,                     # ground_distance (float, meters)
            0.0, 0.0                 # flow_rate_x, flow_rate_y (float)
        )
