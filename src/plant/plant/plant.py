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

USE_DELAY = False
USE_NOISE = True

DELAY_TIME = 0.01

SIG_POS = 0.005       # 5 mm
SIG_VEL = 0.03        # 3 cm/s
SIG_GYRO = 0.01       # 0.57 deg/s
SIG_ACC = 0.10        # 0.1 m/s^2
SIG_ENCODER = 0.002   # 0.11 deg

N_CTRL = 10

USE_FIXED_CAMERA = False
VIEW_CAMERA_NAME = "front_camera"


def to_zdown(v):
    return np.array([v[0], -v[1], -v[2]], dtype=float)


def quat_to_rpy(q):
    w, x, y, z = q

    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z)
    )

    s = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(s)

    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y)
    )

    return np.array([roll, pitch, yaw], dtype=float)

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
        self.apply_servo_parameters()
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 1.0 / PHYSICS_HZ

        self.sid_imu_quat = self.sensor_id("imu_quat")
        self.sid_imu_gyro = self.sensor_id("imu_gyro")
        self.sid_imu_acc = self.sensor_id("imu_acc")
        self.sid_body_pos = self.sensor_id("body_pos")
        self.sid_body_linvel = self.sensor_id("body_linvel")

        self.sid_encoder_theta1 = self.sensor_id("encoder_theta1")
        self.sid_encoder_theta2 = self.sensor_id("encoder_theta2")
        self.sid_encoder_phi1 = self.sensor_id("encoder_phi1")
        self.sid_encoder_phi2 = self.sensor_id("encoder_phi2")
        self.sid_encoder_phi3 = self.sensor_id("encoder_phi3")
        self.sid_encoder_phi4 = self.sensor_id("encoder_phi4")

        self.s_adr = self.model.sensor_adr
        self.s_dim = self.model.sensor_dim

        self.ctrl_recv = np.zeros(N_CTRL, dtype=float)
        self.ctrl = np.zeros(N_CTRL, dtype=float)

        self.delay_len = max(1, int(DELAY_TIME * PHYSICS_HZ))
        self.delay_buf = np.zeros((self.delay_len, N_CTRL), dtype=float)
        self.delay_idx = 0

        self.prev_pub_t = None
        self.prev_linvel_zdown = None
        self.prev_gyro = None

        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.sub_input = self.create_subscription(
            Input,
            "/input",
            self.input_callback,
            10
        )

        self.pub_state = self.create_publisher(
            MultirotorState,
            "/multirotor_state",
            10
        )

        self.viewer_thread = threading.Thread(
            target=self.viewer_loop,
            daemon=True
        )

        self.sim_thread = threading.Thread(
            target=self.sim_loop,
            daemon=True
        )

        self.viewer_thread.start()
        self.sim_thread.start()

        self.get_logger().info("state publish convention: z-down, [x, y, z] = [x_mj, -y_mj, -z_mj]")

    def sensor_id(self, name):
        sid = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            name
        )

        if sid < 0:
            raise RuntimeError(f"sensor not found: {name}")

        return sid

    def actuator_id(self, name):
        aid = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name
        )

        if aid < 0:
            raise RuntimeError(f"actuator not found: {name}")

        return aid

    def apply_position_servo_gains(self, actuator_names, kp, kv):
        for name in actuator_names:
            aid = self.actuator_id(name)
            self.model.actuator_gainprm[aid, 0] = kp
            self.model.actuator_biasprm[aid, 1] = -kp
            self.model.actuator_biasprm[aid, 2] = -kv

    def apply_servo_parameters(self):
        theta_kp = self.declare_parameter("theta_servo_kp", 12.0).value
        theta_kv = self.declare_parameter("theta_servo_kv", 4.0).value
        phi_kp = self.declare_parameter("phi_servo_kp", 12.0).value
        phi_kv = self.declare_parameter("phi_servo_kv", 4.0).value

        self.apply_position_servo_gains(
            ["theta1_servo", "theta2_servo"],
            theta_kp,
            theta_kv
        )
        self.apply_position_servo_gains(
            ["phi1_servo", "phi2_servo", "phi3_servo", "phi4_servo"],
            phi_kp,
            phi_kv
        )

        self.get_logger().info(
            "servo gains: "
            f"theta kp={theta_kp:.3f}, kv={theta_kv:.3f}, "
            f"phi kp={phi_kp:.3f}, kv={phi_kv:.3f}"
        )

    def sensing_state(self, sid):
        adr = self.s_adr[sid]
        dim = self.s_dim[sid]

        return np.array(
            self.data.sensordata[adr:adr + dim],
            dtype=float
        )

    def delay_step(self):
        if not USE_DELAY:
            return self.ctrl_recv.copy()

        self.delay_buf[self.delay_idx] = self.ctrl_recv
        self.delay_idx = (self.delay_idx + 1) % self.delay_len

        return self.delay_buf[self.delay_idx].copy()

    def input_callback(self, msg):
        u = np.asarray(msg.u, dtype=float)

        if u.shape[0] != N_CTRL:
            self.get_logger().warn(
                f"input size must be {N_CTRL}, but got {u.shape[0]}"
            )
            return

        with self.lock:
            self.ctrl_recv = u.copy()

    def apply_control(self):
        self.data.ctrl[0] = self.ctrl[0]
        self.data.ctrl[1] = self.ctrl[1]
        self.data.ctrl[2] = self.ctrl[2]
        self.data.ctrl[3] = self.ctrl[3]

        self.data.ctrl[4] = self.ctrl[4]
        self.data.ctrl[5] = self.ctrl[5]

        self.data.ctrl[6] = self.ctrl[6]
        self.data.ctrl[7] = self.ctrl[7]
        self.data.ctrl[8] = self.ctrl[8]
        self.data.ctrl[9] = self.ctrl[9]

    def make_state_msg(self, now):
        imu_quat = self.sensing_state(self.sid_imu_quat)

        imu_gyro = add_noise(
            self.sensing_state(self.sid_imu_gyro),
            SIG_GYRO
        )

        imu_acc = add_noise(
            self.sensing_state(self.sid_imu_acc),
            SIG_ACC
        )

        pos_mj = add_noise(
            self.sensing_state(self.sid_body_pos),
            SIG_POS
        )

        vel_mj = add_noise(
            self.sensing_state(self.sid_body_linvel),
            SIG_VEL
        )

        pos = to_zdown(pos_mj)
        vel = to_zdown(vel_mj)

        theta = np.array(
            [
                self.sensing_state(self.sid_encoder_theta1)[0],
                self.sensing_state(self.sid_encoder_theta2)[0],
            ],
            dtype=float
        )

        phi = np.array(
            [
                self.sensing_state(self.sid_encoder_phi1)[0],
                self.sensing_state(self.sid_encoder_phi2)[0],
                self.sensing_state(self.sid_encoder_phi3)[0],
                self.sensing_state(self.sid_encoder_phi4)[0],
            ],
            dtype=float
        )

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
                self.ctrl = self.delay_step()
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

        cam_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            VIEW_CAMERA_NAME
        )

        if cam_id < 0:
            self.get_logger().warn(
                f"camera not found: {VIEW_CAMERA_NAME}"
            )
            return

        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id

        self.get_logger().info(
            f"viewer camera: {VIEW_CAMERA_NAME}"
        )

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
