#!/usr/bin/env python3
"""Render a synchronized RGB-D stream from the plant pose in a separate GL process."""

import math
import time

import mujoco
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from multirotor_interfaces.msg import MultirotorState
from plant.plant import quat_to_rotmat, rotmat_to_quat, to_zdown
from plant.map_visuals import add_map_visuals


class MapCamera(Node):
    def __init__(self):
        super().__init__("map_camera")
        share = get_package_share_directory("plant")
        environment_file = self.declare_parameter("environment_file", f"{share}/environment_map.yaml").value
        with open(environment_file, encoding="utf-8") as stream:
            environment = yaml.safe_load(stream)
        spec = mujoco.MjSpec.from_file(f"{share}/xml/HB_scene.xml")
        flip = np.diag([1.0, -1.0, -1.0])
        for obstacle in environment["obstacles"]:
            R_mj = flip @ np.eye(3) @ flip
            spec.worldbody.add_geom(name=obstacle["name"], type=mujoco.mjtGeom.mjGEOM_BOX,
                                    pos=to_zdown(obstacle["center"]), quat=rotmat_to_quat(R_mj),
                                    size=0.5 * np.asarray(obstacle["size"], dtype=float),
                                    rgba=obstacle["rgba"], material=obstacle.get("material", ""),
                                    contype=1, conaffinity=2, group=2)
        add_map_visuals(spec, environment["obstacles"])
        for name, color in (("start", [0.10, 0.85, 0.20, 0.75]),
                            ("goal", [0.10, 0.40, 1.00, 0.75])):
            spec.worldbody.add_geom(name=f"{name}_waypoint", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                    pos=to_zdown([environment[name][axis] for axis in ("x", "y", "z")]),
                                    size=[0.12, 0.12, 0.12], rgba=color,
                                    contype=0, conaffinity=0)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_adr = self.model.jnt_qposadr[self.root_id]
        self.servo_adrs = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                           for name in ("joint_beta1", "joint_beta2", "joint_beta3", "joint_beta4",
                                        "joint_alpha1", "joint_alpha2", "joint_alpha3", "joint_alpha4")]
        self.has_state = False
        self.create_subscription(MultirotorState, "/multirotor_state", self.on_state, 1)
        self.pub_rgb = self.create_publisher(Image, "/camera/rgb/image_raw", 5)
        self.pub_depth = self.create_publisher(Image, "/camera/depth/image_raw", 5)
        self.pub_info = self.create_publisher(CameraInfo, "/camera/rgb/camera_info", 5)
        self.width, self.height = 320, 240
        f = self.height / (2.0 * math.tan(math.radians(70.0) / 2.0))
        self.info = CameraInfo()
        self.info.width, self.info.height = self.width, self.height
        self.info.distortion_model = "plumb_bob"
        self.info.d = [0.0] * 5
        self.info.k = [f, 0.0, (self.width - 1) / 2.0, 0.0, f, (self.height - 1) / 2.0, 0.0, 0.0, 1.0]
        self.info.p = [f, 0.0, (self.width - 1) / 2.0, 0.0, 0.0, f,
                       (self.height - 1) / 2.0, 0.0, 0.0, 1.0, 0.0]

    def on_state(self, msg):
        flip = np.diag([1.0, -1.0, -1.0])
        q = np.asarray(msg.quat, dtype=float)
        R_mj = flip @ quat_to_rotmat(q) @ flip
        self.data.qpos[self.root_adr:self.root_adr + 3] = to_zdown(msg.pos)
        self.data.qpos[self.root_adr + 3:self.root_adr + 7] = rotmat_to_quat(R_mj)
        beta = np.asarray(msg.beta, dtype=float)
        servo = [beta[0], beta[1], beta[1], beta[0], *msg.alpha]
        for adr, angle in zip(self.servo_adrs, servo):
            self.data.qpos[adr] = angle
        mujoco.mj_forward(self.model, self.data)
        self.has_state = True

    def publish_frame(self, renderer):
        renderer.update_scene(self.data, camera="front_camera")
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera="front_camera")
        depth = renderer.render().astype(np.float32, copy=True)
        renderer.disable_depth_rendering()
        stamp = self.get_clock().now().to_msg()
        rgb_msg = Image()
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = "camera_optical_frame"
        rgb_msg.height, rgb_msg.width = self.height, self.width
        rgb_msg.encoding = "rgb8"
        rgb_msg.step = self.width * 3
        rgb_msg.data = rgb.tobytes()
        depth_msg = Image()
        depth_msg.header = rgb_msg.header
        depth_msg.height, depth_msg.width = self.height, self.width
        depth_msg.encoding = "32FC1"
        depth_msg.step = self.width * 4
        depth_msg.data = depth.tobytes()
        self.info.header = rgb_msg.header
        self.pub_rgb.publish(rgb_msg)
        self.pub_depth.publish(depth_msg)
        self.pub_info.publish(self.info)


def main():
    rclpy.init()
    node = MapCamera()
    period = 1.0 / 8.0
    next_frame = time.perf_counter()
    try:
        with mujoco.Renderer(node.model, height=node.height, width=node.width) as renderer:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.001)
                if not node.has_state or time.perf_counter() < next_frame:
                    continue
                node.publish_frame(renderer)
                next_frame = time.perf_counter() + period
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
