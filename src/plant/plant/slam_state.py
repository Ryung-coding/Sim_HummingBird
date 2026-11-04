#!/usr/bin/env python3
"""Feed RTAB-Map position and velocity into the existing controller state format."""

import time

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rtabmap_msgs.msg import OdomInfo
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener

from multirotor_interfaces.msg import MultirotorState
from plant.plant import quat_to_rotmat, rotmat_to_quat, rpy_to_rotmat, to_zdown


class SlamState(Node):
    def __init__(self):
        super().__init__("slam_state")
        environment_file = self.declare_parameter(
            "environment_file", f"{get_package_share_directory('plant')}/environment_map.yaml").value
        with open(environment_file, encoding="utf-8") as stream:
            start = yaml.safe_load(stream)["start"]
        self.start_world = to_zdown([start[axis] for axis in ("x", "y", "z")])
        flip = np.diag([1.0, -1.0, -1.0])
        self.start_rotation = flip @ rpy_to_rotmat([start[axis] for axis in ("roll", "pitch", "yaw")]) @ flip
        self.R_world_odom = None
        self.t_world_odom = None
        self.propagated_pos = to_zdown(self.start_world)
        self.last_sim_time = None
        self.new_visual = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.estimate_pos = None
        self.last_odom_time = 0.0
        self.last_status_time = 0.0
        self.last_valid = False
        self.odom_info_good = False
        self.last_info_time = 0.0
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(OdomInfo, "/odom_info", self.on_odom_info, 10)
        self.create_subscription(MultirotorState, "/multirotor_state", self.on_plant_state, 10)
        alignment_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                   durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_alignment = self.create_publisher(TransformStamped, "/slam/map_alignment", alignment_qos)
        self.pub_state = self.create_publisher(MultirotorState, "/slam_state", 10)
        self.pub_valid = self.create_publisher(Bool, "/slam/position_valid", 10)

    @staticmethod
    def rotation(quaternion):
        return quat_to_rotmat([quaternion.w, quaternion.x, quaternion.y, quaternion.z])

    def on_odom_info(self, msg):
        self.odom_info_good = not msg.lost and msg.inliers >= 10
        self.last_info_time = time.monotonic()

    def on_odom(self, msg):
        if msg.header.frame_id != "odom" or msg.child_frame_id != "base_link":
            self.get_logger().warn("Expected /odom frames odom -> base_link", throttle_duration_sec=5.0)
            return
        try:
            transform = self.tf_buffer.lookup_transform("map", "odom", Time())
        except TransformException:
            return
        q = transform.transform.rotation
        R_map_odom = self.rotation(q)
        q_body = msg.pose.pose.orientation
        R_odom_body = self.rotation(q_body)
        p = msg.pose.pose.position
        t = transform.transform.translation
        p_odom = np.array([p.x, p.y, p.z])
        t_map_odom = np.array([t.x, t.y, t.z])
        if not np.all(np.isfinite(p_odom)):
            return
        if not self.odom_info_good or time.monotonic() - self.last_info_time >= 0.5:
            return
        if self.R_world_odom is None:
            self.R_world_odom = self.start_rotation @ R_odom_body.T
            self.t_world_odom = self.start_world - self.R_world_odom @ p_odom
        # Keep local visual odometry continuous when RTAB-Map corrects map -> odom.
        R_world_map = self.R_world_odom @ R_map_odom.T
        translation = self.t_world_odom - R_world_map @ t_map_odom
        aligned = TransformStamped()
        aligned.header.stamp = msg.header.stamp
        aligned.header.frame_id = "controller_world_mj"
        aligned.child_frame_id = "map"
        aligned.transform.translation.x = float(translation[0])
        aligned.transform.translation.y = float(translation[1])
        aligned.transform.translation.z = float(translation[2])
        aligned.transform.rotation.w, aligned.transform.rotation.x, aligned.transform.rotation.y, aligned.transform.rotation.z = (float(x) for x in rotmat_to_quat(R_world_map))
        self.pub_alignment.publish(aligned)
        self.estimate_pos = to_zdown(self.R_world_odom @ p_odom + self.t_world_odom)
        self.new_visual = True
        self.last_odom_time = time.monotonic()

    def on_plant_state(self, msg):
        now = time.monotonic()
        if self.last_sim_time is not None:
            dt = max(0.0, min(float(msg.sim_time - self.last_sim_time), 0.1))
            self.propagated_pos[:2] += np.asarray(msg.vel[:2]) * dt
        self.last_sim_time = msg.sim_time
        plant_pos = np.asarray(msg.pos)
        self.propagated_pos[2] = plant_pos[2]
        visual_consistent = (self.estimate_pos is not None and
                             np.linalg.norm(self.estimate_pos[:2] - plant_pos[:2]) <= 0.7)
        if self.new_visual:
            self.new_visual = False
            if visual_consistent:
                self.propagated_pos[:2] = self.estimate_pos[:2]
            else:
                self.get_logger().warn("SLAM horizontal position differs from plant by >0.7 m; holding trajectory",
                                       throttle_duration_sec=5.0)
        valid = (visual_consistent and now - self.last_odom_time < 1.0
                 and self.odom_info_good and now - self.last_info_time < 0.5)
        msg.pos = self.propagated_pos.tolist()
        self.pub_state.publish(msg)
        if valid != self.last_valid or now - self.last_status_time >= 0.1:
            status = Bool()
            status.data = bool(valid)
            self.pub_valid.publish(status)
            self.last_status_time = now
            self.last_valid = valid
        if valid:
            self.get_logger().info(
                f"SLAM horizontal pose: x={msg.pos[0]:.2f}, y={msg.pos[1]:.2f}; altitude={msg.pos[2]:.2f}",
                throttle_duration_sec=5.0,
            )


def main():
    rclpy.init()
    node = SlamState()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
