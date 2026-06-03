#!/usr/bin/env python3
import os
os.environ.setdefault("GLFW_PLATFORM", "x11")

import time
import signal
import math
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import mujoco
import mujoco.viewer

from multirotor_interfaces.msg import Input, MultirotorState


PHYSICS_HZ = 400.0
PUB_HZ = 400.0

USE_NOISE = True

SIG_POS = 0.005
SIG_VEL = 0.03
SIG_GYRO = 0.01
SIG_ACC = 0.10
SIG_ENCODER = 0.002

N_THRUST = 4
N_THETA = 4
N_PHI = 4
N_CTRL = N_THRUST + N_THETA + N_PHI

USE_FIXED_CAMERA = False
VIEW_CAMERA_NAME = "front_camera"


def to_zdown(v):
    return np.array([v[0], -v[1], -v[2]], dtype=float)


def quat_to_rotmat(q):
    w, x, y, z = q

    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)]
    ], dtype=float)


def rotmat_to_rpy(R):
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])

    return np.array([roll, pitch, yaw], dtype=float)


def imu_quat_to_zdown_rpy(q_imu):
    S = np.diag([1.0, -1.0, -1.0])
    R_imu_mj = quat_to_rotmat(q_imu)
    R_zdown = S @ R_imu_mj

    return rotmat_to_rpy(R_zdown)


def add_noise(x, sigma):
    if not USE_NOISE or sigma <= 0.0:
        return x

    return x + np.random.normal(0.0, sigma, size=x.shape)


class PlantRosNode(Node):
    def __init__(self):
        super().__init__("multirotor_plant")

        pkg_share = get_package_share_directory("plant")
        xml_path = os.path.join(pkg_share, "xml", "scene.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 1.0 / PHYSICS_HZ

        if self.model.nu != N_CTRL:
            self.get_logger().warn(f"model.nu is {self.model.nu}, but N_CTRL is {N_CTRL}")

        self.sid_imu_quat = self.sensor_id("imu_quat")
        self.sid_imu_gyro = self.sensor_id("imu_gyro")
        self.sid_imu_acc = self.sensor_id("imu_acc")
        self.sid_body_pos = self.sensor_id("body_pos")
        self.sid_body_linvel = self.sensor_id("body_linvel")

        self.sid_encoder_theta1 = self.sensor_id("encoder_theta1")
        self.sid_encoder_theta2 = self.sensor_id("encoder_theta2")
        self.sid_encoder_theta3 = self.sensor_id("encoder_theta3")
        self.sid_encoder_theta4 = self.sensor_id("encoder_theta4")

        self.sid_encoder_phi1 = self.sensor_id("encoder_phi1")
        self.sid_encoder_phi2 = self.sensor_id("encoder_phi2")
        self.sid_encoder_phi3 = self.sensor_id("encoder_phi3")
        self.sid_encoder_phi4 = self.sensor_id("encoder_phi4")

        self.s_adr = self.model.sensor_adr
        self.s_dim = self.model.sensor_dim

        self.ctrl_recv = np.zeros(N_CTRL, dtype=float)
        self.ctrl = np.zeros(N_CTRL, dtype=float)

        self.prev_pub_t = None
        self.prev_linvel_zdown = None
        self.prev_gyro = None

        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.sub_input = self.create_subscription(Input, "/input", self.input_callback, 10)
        self.pub_state = self.create_publisher(MultirotorState, "/multirotor_state", 10)

        self.viewer_thread = threading.Thread(target=self.viewer_loop, daemon=True)
        self.sim_thread = threading.Thread(target=self.sim_loop, daemon=True)

        self.viewer_thread.start()
        self.sim_thread.start()

        self.get_logger().info("state convention: z-down, [x, y, z] = [x_mj, -y_mj, -z_mj]")
        self.get_logger().info("input order: f[4], theta[4], phi[4] -> ctrl[0:4], ctrl[4:8], ctrl[8:12]")

    def sensor_id(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)

        if sid < 0:
            raise RuntimeError(f"sensor not found: {name}")

        return sid

    def sensing_state(self, sid):
        adr = self.s_adr[sid]
        dim = self.s_dim[sid]

        return np.array(self.data.sensordata[adr:adr + dim], dtype=float)

    def input_callback(self, msg):
        f = np.asarray(msg.f, dtype=float)
        theta = np.asarray(msg.theta, dtype=float)
        phi = np.asarray(msg.phi, dtype=float)

        if f.shape[0] != N_THRUST:
            self.get_logger().warn(f"f size must be {N_THRUST}, but got {f.shape[0]}")
            return

        if theta.shape[0] != N_THETA:
            self.get_logger().warn(f"theta size must be {N_THETA}, but got {theta.shape[0]}")
            return

        if phi.shape[0] != N_PHI:
            self.get_logger().warn(f"phi size must be {N_PHI}, but got {phi.shape[0]}")
            return

        with self.lock:
            self.ctrl_recv[0:4] = f
            self.ctrl_recv[4:8] = theta
            self.ctrl_recv[8:12] = phi

    def apply_control(self):
        self.data.ctrl[:N_CTRL] = self.ctrl[:N_CTRL]

    def make_state_msg(self, now):
        imu_quat = self.sensing_state(self.sid_imu_quat)
        imu_gyro = add_noise(self.sensing_state(self.sid_imu_gyro), SIG_GYRO)
        imu_acc = add_noise(self.sensing_state(self.sid_imu_acc), SIG_ACC)
        pos_mj = add_noise(self.sensing_state(self.sid_body_pos), SIG_POS)
        vel_mj = add_noise(self.sensing_state(self.sid_body_linvel), SIG_VEL)

        pos = to_zdown(pos_mj)
        vel = to_zdown(vel_mj)

        theta = np.array([
            self.sensing_state(self.sid_encoder_theta1)[0],
            self.sensing_state(self.sid_encoder_theta2)[0],
            self.sensing_state(self.sid_encoder_theta3)[0],
            self.sensing_state(self.sid_encoder_theta4)[0],
        ], dtype=float)

        phi = np.array([
            self.sensing_state(self.sid_encoder_phi1)[0],
            self.sensing_state(self.sid_encoder_phi2)[0],
            self.sensing_state(self.sid_encoder_phi3)[0],
            self.sensing_state(self.sid_encoder_phi4)[0],
        ], dtype=float)

        theta = add_noise(theta, SIG_ENCODER)
        phi = add_noise(phi, SIG_ENCODER)

        rpy = imu_quat_to_zdown_rpy(imu_quat)

        if self.prev_pub_t is None:
            acc = np.zeros(3, dtype=float)
            a_rpy = np.zeros(3, dtype=float)
        else:
            dt = max(1e-6, now - self.prev_pub_t)
            acc = (vel - self.prev_linvel_zdown) / dt
            a_rpy = (imu_gyro - self.prev_gyro) / dt

        self.prev_pub_t = now
        self.prev_linvel_zdown = vel.copy()
        self.prev_gyro = imu_gyro.copy()

        msg = MultirotorState()

        msg.pos = pos.tolist()
        msg.vel = vel.tolist()
        msg.acc = acc.tolist()

        msg.rpy = rpy.tolist()
        msg.w_rpy = imu_gyro.tolist()
        msg.a_rpy = a_rpy.tolist()

        msg.imu_acc = imu_acc.tolist()

        msg.theta = theta.tolist()
        msg.phi = phi.tolist()

        return msg

    def sim_loop(self):
        next_step = time.perf_counter()
        next_pub = next_step

        dt_step = 1.0 / PHYSICS_HZ
        dt_pub = 1.0 / PUB_HZ

        while rclpy.ok() and not self.stop_event.is_set():
            now = time.perf_counter()

            with self.lock:
                self.ctrl = self.ctrl_recv.copy()
                self.apply_control()

                while now >= next_step:
                    mujoco.mj_step(self.model, self.data)
                    next_step += dt_step

                while now >= next_pub:
                    msg = self.make_state_msg(now)
                    self.pub_state.publish(msg)
                    next_pub += dt_pub

            sleep_t = next_step - time.perf_counter()

            if sleep_t > 0.0:
                time.sleep(sleep_t)

    def set_viewer_camera(self, viewer):
        if not USE_FIXED_CAMERA:
            self.get_logger().info("viewer camera: free camera")
            return

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, VIEW_CAMERA_NAME)

        if cam_id < 0:
            self.get_logger().warn(f"camera not found: {VIEW_CAMERA_NAME}")
            return

        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id

        self.get_logger().info(f"viewer camera: {VIEW_CAMERA_NAME}")

    def viewer_loop(self):
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self.set_viewer_camera(viewer)

                while viewer.is_running() and rclpy.ok() and not self.stop_event.is_set():
                    with self.lock:
                        viewer.sync()

                    time.sleep(0.002)

        except Exception as e:
            if not self.stop_event.is_set():
                self.get_logger().warn(f"viewer end: {e}")

    def close(self):
        self.stop_event.set()

        if self.sim_thread.is_alive():
            self.sim_thread.join(timeout=1.0)

        if self.viewer_thread.is_alive():
            self.viewer_thread.join(timeout=1.0)


def main():
    rclpy.init()
    node = PlantRosNode()

    def sigint_handler(signum, frame):
        node.get_logger().info("Ctrl+C received. Shutting down plant...")
        node.close()

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        while rclpy.ok() and not node.stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt. Shutting down plant...")

    finally:
        node.close()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()