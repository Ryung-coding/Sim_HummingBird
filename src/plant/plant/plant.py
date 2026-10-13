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
from std_msgs.msg import Bool

from multirotor_interfaces.msg import Input, MultirotorState

# control rate and physics rate
PHYSICS_HZ = 400.0
PUB_HZ = 400.0
HB_SERVO_TAU_SEC = 0.0

# Servo start offset [deg]
# order: [alpha1, alpha2, alpha3, alpha4, beta1, beta2]
HB_SERVO_OFFSET_DEG = np.array([0.0, 0.0, 0.0, 0.0, -0.0, 0.0], dtype=float)

# Sensor noise standard deviations
USE_NOISE = False
SIG_POS = 0.005
SIG_VEL = 0.03
SIG_GYRO = 0.01
SIG_ACC = 0.10
SIG_ENCODER = 0.002

# Model dimensions
HB_N_THRUST = 4
HB_N_BETA = 2
HB_N_ALPHA = 4
HB_N_CTRL = HB_N_THRUST + 4 + HB_N_ALPHA

# Visualization parameters
USE_FIXED_CAMERA = False
SHOW_THRUST_ARROWS = True
USE_RANDOM_DISTURBANCE = False


VIEW_CAMERA_NAME = "front_camera"
    
THRUST_ARROW_SCALE = 0.025
THRUST_ARROW_MIN_LEN = 0.15
THRUST_ARROW_MAX_LEN = 1.20
THRUST_ARROW_WIDTH = 0.008
THRUST_ARROW_RGBA = np.array([1.00, 0.20, 0.05, 0.90], dtype=np.float32)
PROP_SPIN_MAX_RAD_PER_SEC = 80.0

RANDOM_DISTURBANCE_DIRECTION_ZDOWN = np.array([1.0, 0.0, 0.0], dtype=float)
RANDOM_DISTURBANCE_FORCE_MAX_N = 10.0
RANDOM_DISTURBANCE_TIME_CONSTANT = 0.2
RANDOM_DISTURBANCE_SEED = 20260916

USE_X_IMPULSE = False
USE_PITCH_MOMENT_IMPULSE = False

IMPULSE_DIRECTION_ZDOWN = np.array([1.0, 0.0, 0.0], dtype=float)
IMPULSE_FORCE_N = 10.0

IMPULSE_MOMENT_DIRECTION_BODY_ZDOWN = np.array([0.0, 1.0, 0.0], dtype=float)
IMPULSE_MOMENT_NM = 0.5

IMPULSE_DURATION_SEC = 0.1


class LPF:
    """First-order low-pass filter: y_dot = (x - y) / tau."""

    def __init__(self, tau_sec, dt_sec, initial):
        self.tau_sec = max(float(tau_sec), 0.0)
        self.dt_sec = max(float(dt_sec), 1.0e-9)
        self.alpha = (1.0 if self.tau_sec <= 1.0e-9 else 1.0 - math.exp(-self.dt_sec / self.tau_sec))
        self.y = np.asarray(initial, dtype=float).copy()

    def update(self, x):
        x = np.asarray(x, dtype=float)
        self.y += self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self, x):
        self.y = np.asarray(x, dtype=float).copy()


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

def rotmat_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=float)
    n = np.linalg.norm(q)

    if n < 1.0e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    return q / n

def imu_quat_to_zdown_rot(q_imu):
    S = np.diag([1.0, -1.0, -1.0])
    R_imu_mj = quat_to_rotmat(q_imu)

    return S @ R_imu_mj

def add_noise(x, sigma):
    if not USE_NOISE or sigma <= 0.0:
        return x

    return x + np.random.normal(0.0, sigma, size=x.shape)

class PlantRosNode(Node):
    def __init__(self):
        super().__init__("multirotor_plant")

        pkg_share = get_package_share_directory("plant")
        xml_path = os.path.join(pkg_share, "xml", "HB_scene.xml")
        self.n_ctrl = HB_N_CTRL
        self.body_name = "body"

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 1.0 / PHYSICS_HZ
        self.set_hb_servo_offset()
        mujoco.mj_forward(self.model, self.data)

        if self.model.nu != self.n_ctrl:
            self.get_logger().warn(f"model.nu is {self.model.nu}, but expected {self.n_ctrl}")

        self.sid_imu_quat = self.sensor_id("imu_quat")
        self.sid_imu_gyro = self.sensor_id("imu_gyro")
        self.sid_imu_acc = self.sensor_id("imu_acc")
        self.sid_body_pos = self.sensor_id("body_pos")
        self.sid_body_linvel = self.sensor_id("body_linvel")

        self.sid_encoder_beta1 = self.sensor_id("encoder_beta1")
        self.sid_encoder_beta2 = self.sensor_id("encoder_beta2")
        self.sid_encoder_beta3 = self.sensor_id("encoder_beta3")
        self.sid_encoder_beta4 = self.sensor_id("encoder_beta4")

        self.sid_encoder_alpha1 = self.sensor_id("encoder_alpha1")
        self.sid_encoder_alpha2 = self.sensor_id("encoder_alpha2")
        self.sid_encoder_alpha3 = self.sensor_id("encoder_alpha3")
        self.sid_encoder_alpha4 = self.sensor_id("encoder_alpha4")
        self.prop_site_ids = [self.site_id("prop1_site"), self.site_id("prop2_site"), self.site_id("prop3_site"), self.site_id("prop4_site"),]
        self.prop_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"prop{i}_geom") for i in range(1, 5)]
        if any(geom_id < 0 for geom_id in self.prop_geom_ids):
            raise RuntimeError("propeller visual geom not found")
        self.prop_base_quat = self.model.geom_quat[self.prop_geom_ids].copy()
        self.prop_phase = np.zeros(HB_N_THRUST, dtype=float)
        # Reaction torque acts opposite to rotor spin about the propeller local Z axis.
        self.prop_spin_sign = -np.sign(self.model.actuator_gear[:HB_N_THRUST, 5])

        self.bid_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        if self.bid_body < 0:
            raise RuntimeError(f"body not found: {self.body_name}")

        self.s_adr = self.model.sensor_adr
        self.s_dim = self.model.sensor_dim
        self.random_disturbance_state = 0.0
        self.random_disturbance_force_mj = np.zeros(3, dtype=float)
        self.impulse_end_time = None
        self.moment_impulse_end_time = None
        moment_direction_norm = np.linalg.norm(IMPULSE_MOMENT_DIRECTION_BODY_ZDOWN)

        if USE_PITCH_MOMENT_IMPULSE and moment_direction_norm <= 1.0e-9:
            raise ValueError("IMPULSE_MOMENT_DIRECTION_BODY_ZDOWN must be non-zero")

        self.impulse_moment_direction_body_mj = to_zdown(IMPULSE_MOMENT_DIRECTION_BODY_ZDOWN / max(moment_direction_norm, 1.0e-9))

        # Keep disturbance and sensor-noise realizations repeatable.
        np.random.seed(RANDOM_DISTURBANCE_SEED)

        direction_norm = np.linalg.norm(RANDOM_DISTURBANCE_DIRECTION_ZDOWN)
        if USE_RANDOM_DISTURBANCE and direction_norm <= 1.0e-9:
            raise ValueError("RANDOM_DISTURBANCE_DIRECTION_ZDOWN must be non-zero")
        self.random_disturbance_direction_mj = to_zdown(RANDOM_DISTURBANCE_DIRECTION_ZDOWN / max(direction_norm, 1.0e-9))

        impulse_direction_norm = np.linalg.norm(IMPULSE_DIRECTION_ZDOWN)
        if USE_X_IMPULSE and impulse_direction_norm <= 1.0e-9:
            raise ValueError("IMPULSE_DIRECTION_ZDOWN must be non-zero")
        self.impulse_direction_mj = to_zdown(IMPULSE_DIRECTION_ZDOWN / max(impulse_direction_norm, 1.0e-9))

        self.ctrl_recv = np.zeros(self.n_ctrl, dtype=float)
        self.ctrl = np.zeros(self.n_ctrl, dtype=float)

        servo_offset_rad = np.deg2rad(HB_SERVO_OFFSET_DEG)
        alpha_offset = servo_offset_rad[0:4]
        beta_offset = servo_offset_rad[4:6]
        beta_actuators = np.array([beta_offset[0], beta_offset[1], beta_offset[1], beta_offset[0]], dtype=float)

        self.ctrl_recv[HB_N_THRUST:HB_N_THRUST + 4] = beta_actuators
        self.ctrl_recv[HB_N_THRUST + 4:HB_N_CTRL] = alpha_offset

        self.ctrl[HB_N_THRUST:HB_N_THRUST + 4] = beta_actuators
        self.ctrl[HB_N_THRUST + 4:HB_N_CTRL] = alpha_offset
        self.data.ctrl[:HB_N_CTRL] = self.ctrl[:HB_N_CTRL]

        self.hb_servo_lpf = LPF(tau_sec=HB_SERVO_TAU_SEC, dt_sec=1.0 / PHYSICS_HZ, initial=self.ctrl[HB_N_THRUST:HB_N_CTRL])

        self.prev_pub_t = None
        self.prev_linvel_zdown = None
        self.prev_gyro = None

        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.sub_input = self.create_subscription(Input, "/input", self.input_callback, 10)
        self.sub_impulse_trigger = self.create_subscription(Bool, "/test/impulse_trigger", self.impulse_trigger_callback, 1)
        self.sub_moment_impulse_trigger = self.create_subscription(Bool, "/test/moment_impulse_trigger", self.moment_impulse_trigger_callback, 1)
        self.pub_state = self.create_publisher(MultirotorState, "/multirotor_state", 10)
        self.viewer_thread = threading.Thread(target=self.viewer_loop, daemon=True)
        self.sim_thread = threading.Thread(target=self.sim_loop, daemon=True)

        self.viewer_thread.start()
        self.sim_thread.start()

        self.get_logger().info("state convention: z-down, [x, y, z] = [x_mj, -y_mj, -z_mj]")
        self.get_logger().info("input order: f[4], beta[2] expanded to ctrl[4:8], alpha[4] -> ctrl[0:4], ctrl[4:8], ctrl[8:12]")

    def set_hb_servo_offset(self):
        if HB_SERVO_OFFSET_DEG.shape != (6,):
            raise ValueError("HB_SERVO_OFFSET_DEG must be " "[alpha1, alpha2, alpha3, alpha4, beta1, beta2]")

        servo_offset_rad = np.deg2rad(HB_SERVO_OFFSET_DEG)
        alpha_offset = servo_offset_rad[0:4]
        beta_offset = servo_offset_rad[4:6]

        joint_values = {
            "joint_alpha1": alpha_offset[0],
            "joint_alpha2": alpha_offset[1],
            "joint_alpha3": alpha_offset[2],
            "joint_alpha4": alpha_offset[3],
            "joint_beta1": beta_offset[0],
            "joint_beta2": beta_offset[1],
            "joint_beta3": beta_offset[1],
            "joint_beta4": beta_offset[0],
        }

        for joint_name, value in joint_values.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

            if joint_id < 0:
                raise RuntimeError(f"joint not found: {joint_name}")

            self.data.qpos[self.model.jnt_qposadr[joint_id]] = value

    def sensor_id(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)

        if sid < 0:
            raise RuntimeError(f"sensor not found: {name}")

        return sid

    def site_id(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)

        if sid < 0:
            raise RuntimeError(f"site not found: {name}")

        return sid

    def sensing_state(self, sid):
        adr = self.s_adr[sid]
        dim = self.s_dim[sid]

        return np.array(self.data.sensordata[adr:adr + dim], dtype=float)

    def apply_random_disturbance(self):
        self.random_disturbance_force_mj.fill(0.0)

        if not USE_RANDOM_DISTURBANCE:
            return

        decay = math.exp(-1.0 / (PHYSICS_HZ * RANDOM_DISTURBANCE_TIME_CONSTANT))
        self.random_disturbance_state = (decay * self.random_disturbance_state + math.sqrt(1.0 - decay * decay) * np.random.normal())

        random_scale = math.tanh(self.random_disturbance_state)
        disturbance_force = (RANDOM_DISTURBANCE_FORCE_MAX_N * random_scale * self.random_disturbance_direction_mj)
        self.random_disturbance_force_mj = disturbance_force
        self.data.xfrc_applied[self.bid_body, :3] += disturbance_force

    def impulse_trigger_callback(self, msg):
        if not USE_X_IMPULSE or not msg.data:
            return

        with self.lock:
            self.impulse_end_time = self.data.time + IMPULSE_DURATION_SEC
            self.get_logger().info(f"impulse triggered at sim t={self.data.time:.3f} s")

    def apply_x_impulse(self):
        if self.impulse_end_time is None or self.data.time >= self.impulse_end_time:
            return

        impulse_force = IMPULSE_FORCE_N * self.impulse_direction_mj
        self.random_disturbance_force_mj += impulse_force
        self.data.xfrc_applied[self.bid_body, :3] += impulse_force

    def moment_impulse_trigger_callback(self, msg):
        if not USE_PITCH_MOMENT_IMPULSE or not msg.data:
            return

        with self.lock:
            self.moment_impulse_end_time = self.data.time + IMPULSE_DURATION_SEC
            self.get_logger().info(f"pitch moment impulse triggered at sim t={self.data.time:.3f} s")


    def apply_pitch_moment_impulse(self):
        if self.moment_impulse_end_time is None or self.data.time >= self.moment_impulse_end_time:
            return

        R_body_world = np.asarray(self.data.xmat[self.bid_body], dtype=float).reshape(3, 3)

        moment_world_mj = IMPULSE_MOMENT_NM * (R_body_world @ self.impulse_moment_direction_body_mj)

        self.data.xfrc_applied[self.bid_body, 3:6] += moment_world_mj

    def input_callback(self, msg):
        f = np.asarray(msg.f, dtype=float)
        beta = np.asarray(msg.beta, dtype=float)
        alpha = np.asarray(msg.alpha, dtype=float)

        if f.shape[0] != HB_N_THRUST:
            self.get_logger().warn(f"f size must be {HB_N_THRUST}, but got {f.shape[0]}")
            return

        if beta.shape[0] != HB_N_BETA:
            self.get_logger().warn(f"beta size must be {HB_N_BETA}, but got {beta.shape[0]}")
            return

        if alpha.shape[0] != HB_N_ALPHA:
            self.get_logger().warn(f"alpha size must be {HB_N_ALPHA}, but got {alpha.shape[0]}")
            return

        with self.lock:
            self.ctrl_recv[0:4] = f
            self.ctrl_recv[4:8] = [beta[0], beta[1], beta[1], beta[0]]
            self.ctrl_recv[8:12] = alpha

    def apply_control(self):
        # BLDC thrust is applied directly.
        self.ctrl[0:HB_N_THRUST] = self.ctrl_recv[0:HB_N_THRUST]

        # Servo commands [beta1..4, alpha1..4] follow a first-order lag.
        self.ctrl[HB_N_THRUST:HB_N_CTRL] = self.hb_servo_lpf.update(self.ctrl_recv[HB_N_THRUST:HB_N_CTRL])

        self.data.ctrl[:self.n_ctrl] = self.ctrl[:self.n_ctrl]

    def update_prop_visuals(self):
        for i, geom_id in enumerate(self.prop_geom_ids):
            max_thrust = self.model.actuator_ctrlrange[i, 1]
            thrust_ratio = np.clip(self.ctrl[i] / max_thrust, 0.0, 1.0)
            spin_rate = self.prop_spin_sign[i] * PROP_SPIN_MAX_RAD_PER_SEC * math.sqrt(thrust_ratio)
            self.prop_phase[i] = (self.prop_phase[i] + spin_rate / PHYSICS_HZ) % (2.0 * math.pi)
            half_phase = 0.5 * self.prop_phase[i]
            spin_quat = np.array([math.cos(half_phase), 0.0, 0.0, math.sin(half_phase)])
            mujoco.mju_mulQuat(self.model.geom_quat[geom_id], spin_quat, self.prop_base_quat[i])

    def make_state_msg(self, now):
        imu_quat = self.sensing_state(self.sid_imu_quat)
        imu_gyro = add_noise(self.sensing_state(self.sid_imu_gyro), SIG_GYRO)
        imu_acc = add_noise(self.sensing_state(self.sid_imu_acc), SIG_ACC)
        pos_mj = add_noise(self.sensing_state(self.sid_body_pos), SIG_POS)
        vel_mj = add_noise(self.sensing_state(self.sid_body_linvel), SIG_VEL)

        pos = to_zdown(pos_mj)
        vel = to_zdown(vel_mj)
        disturbance_force = to_zdown(self.random_disturbance_force_mj)

        beta = np.array([self.sensing_state(self.sid_encoder_beta1)[0], self.sensing_state(self.sid_encoder_beta2)[0],], dtype=float)

        alpha = np.array([
            self.sensing_state(self.sid_encoder_alpha1)[0],
            self.sensing_state(self.sid_encoder_alpha2)[0],
            self.sensing_state(self.sid_encoder_alpha3)[0],
            self.sensing_state(self.sid_encoder_alpha4)[0],
        ], dtype=float)

        beta = add_noise(beta, SIG_ENCODER)
        alpha = add_noise(alpha, SIG_ENCODER)

        R_zdown = imu_quat_to_zdown_rot(imu_quat)
        rpy = rotmat_to_rpy(R_zdown)
        quat = rotmat_to_quat(R_zdown)

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

        msg.sim_time = float(self.data.time)
        msg.pos = pos.tolist()
        msg.vel = vel.tolist()
        msg.acc = acc.tolist()

        msg.rpy = rpy.tolist()
        msg.quat = quat.tolist()
        msg.w_rpy = imu_gyro.tolist()
        msg.a_rpy = a_rpy.tolist()

        msg.imu_acc = imu_acc.tolist()
        msg.disturbance_force = disturbance_force.tolist()

        msg.beta = beta.tolist()
        msg.alpha = alpha.tolist()

        return msg

    def sim_loop(self):
        next_step = time.perf_counter()
        next_pub = next_step

        dt_step = 1.0 / PHYSICS_HZ
        dt_pub = 1.0 / PUB_HZ

        while rclpy.ok() and not self.stop_event.is_set():
            now = time.perf_counter()

            with self.lock:
                while now >= next_step:
                    self.apply_control()
                    self.data.xfrc_applied[self.bid_body, :] = 0.0
                    self.update_prop_visuals()
                    self.apply_random_disturbance()
                    self.apply_x_impulse()
                    self.apply_pitch_moment_impulse()
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

    def update_thrust_arrows(self, viewer):
        if not SHOW_THRUST_ARROWS or not hasattr(viewer, "user_scn"):
            return

        scn = viewer.user_scn

        for i, site_id in enumerate(self.prop_site_ids):
            if scn.ngeom >= len(scn.geoms):
                return

            thrust = max(0.0, float(self.ctrl[i]))

            if thrust <= 1.0e-6:
                continue

            length = np.clip(THRUST_ARROW_SCALE * thrust, THRUST_ARROW_MIN_LEN, THRUST_ARROW_MAX_LEN)

            pos = np.array(self.data.site_xpos[site_id], dtype=np.float64)
            R = np.array(self.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
            thrust_dir = R @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
            tip = pos + length * thrust_dir

            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64).reshape(9), THRUST_ARROW_RGBA)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, THRUST_ARROW_WIDTH, pos, tip)
            scn.ngeom += 1

    def viewer_loop(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.set_viewer_camera(viewer)

            while viewer.is_running() and rclpy.ok() and not self.stop_event.is_set():
                with self.lock:
                    if hasattr(viewer, "user_scn"):
                        viewer.user_scn.ngeom = 0
                    self.update_thrust_arrows(viewer)
                    viewer.sync()

                time.sleep(0.002)

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