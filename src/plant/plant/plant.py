#!/usr/bin/env python3
import os
os.environ.setdefault("GLFW_PLATFORM", "x11")

import time
import signal
import math
import threading
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import mujoco
import mujoco.viewer
from std_msgs.msg import Bool
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

from multirotor_interfaces.msg import Cmd, Input, MultirotorState

# Simulation -------------------------------------------------------
PHYSICS_HZ = 400.0
PUB_HZ = 400.0
IMU_HZ = 100.0
HB_SERVO_TAU_SEC = 0.0

# HummingBird ------------------------------------------------------
# order: [alpha1, alpha2, alpha3, alpha4, beta1, beta2]
HB_SERVO_OFFSET_DEG = np.array([0.0, 0.0, 0.0, 0.0, -0.0, 0.0], dtype=float)
HB_N_THRUST = 4
HB_N_BETA = 2
HB_N_ALPHA = 4
HB_N_CTRL = HB_N_THRUST + 4 + HB_N_ALPHA

# Sensor noise -----------------------------------------------------
USE_NOISE = True
SIG_POS = 0.005
SIG_VEL = 0.03
SIG_GYRO = 0.01
SIG_ACC = 0.10
SIG_ENCODER = 0.002

# Aerodynamic disturbance -----------------------------------------
USE_TURBULENCE_DISTURBANCE = False
ROTOR_RADIUS_M = 0.1143  # HB_model.xml propeller mesh scale; OBJ radius is approximately 1.
TURBULENCE_NOISE_RATIO = 0.03  # heuristic axial force std / commanded thrust, not fitted by the cited papers
TURBULENCE_NOISE_TIME_CONSTANT_SEC = 0.2
TURBULENCE_SEED = 20261015
WALL_RAY_DIRECTIONS_MJ = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=float)
GROUND_RAY_DIRECTION_MJ = np.array([0.0, 0.0, -1.0], dtype=float)

# Test disturbances ------------------------------------------------
USE_RANDOM_DISTURBANCE = False
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

# Viewer -----------------------------------------------------------
USE_FIXED_CAMERA = False
SHOW_THRUST_ARROWS = True
SHOW_TURBULENCE_ARROWS = True
VIEW_CAMERA_NAME = "front_camera"
THRUST_ARROW_SCALE = 0.025
THRUST_ARROW_MIN_LEN = 0.15
THRUST_ARROW_MAX_LEN = 1.20
THRUST_ARROW_WIDTH = 0.008
THRUST_ARROW_RGBA = np.array([1.00, 0.20, 0.05, 0.90], dtype=np.float32)
TURBULENCE_ARROW_SCALE = 0.5
TURBULENCE_ARROW_RGBA = np.array([0.20, 0.85, 1.00, 0.95], dtype=np.float32)
POSE_AXIS_RGBA = np.array([[1.0, 0.15, 0.15, 1.0], [0.15, 1.0, 0.15, 1.0],
                           [0.15, 0.35, 1.0, 1.0]], dtype=np.float32)
POSE_AXIS_LENGTH = 0.42
POSE_TRACE_LENGTH = 0.25
POSE_AXIS_WIDTH = 0.012
PROP_SPIN_MAX_RAD_PER_SEC = 80.0

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

def expand_obstacles(obstacles):
    expanded = []
    for obstacle in obstacles:
        kind = obstacle.get("type", "wall")
        if kind != "tunnel":
            if kind not in ("wall", "cylinder"):
                raise ValueError(f"unsupported obstacle type: {kind}")
            expanded.append(obstacle)
            continue

        if "rpy" in obstacle:
            raise ValueError("tunnel rotation is not supported")
        center = np.asarray(obstacle["center"], dtype=float)
        size = np.asarray(obstacle["size"], dtype=float)
        offset = np.asarray(obstacle["opening_offset"], dtype=float)
        opening = np.asarray(obstacle["opening_size"], dtype=float)
        if center.shape != (3,) or size.shape != (3,) or offset.shape != (2,) or opening.shape != (2,) or np.any(size <= 0.0):
            raise ValueError("invalid tunnel center, size, or opening")
        x0, x1 = center[0] - size[0] / 2.0, center[0] + size[0] / 2.0
        z0, z1 = center[2] - size[2] / 2.0, center[2] + size[2] / 2.0
        gx0, gx1 = center[0] + offset[0] - opening[0] / 2.0, center[0] + offset[0] + opening[0] / 2.0
        gz0, gz1 = center[2] + offset[1] - opening[1] / 2.0, center[2] + offset[1] + opening[1] / 2.0
        if not (x0 < gx0 < gx1 < x1 and z0 < gz0 < gz1 < z1):
            raise ValueError("tunnel opening must lie inside the outer box")
        for suffix, xa, xb, za, zb in (("left", x0, gx0, z0, z1), ("right", gx1, x1, z0, z1),
                                       ("top", gx0, gx1, gz1, z1), ("bottom", gx0, gx1, z0, gz0)):
            expanded.append({"name": f"{obstacle['name']}_{suffix}",
                             "center": [(xa + xb) / 2.0, center[1], (za + zb) / 2.0],
                             "size": [xb - xa, size[1], zb - za], "rgba": obstacle["rgba"]})
    return expanded

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

def rpy_to_rotmat(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]], dtype=float)

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

        environment_path = self.declare_parameter(
            "environment_file", os.path.join(pkg_share, "environment.yaml")).value
        with open(environment_path, encoding="utf-8") as environment_file:
            environment = yaml.safe_load(environment_file)
        if environment["frame"] != "controller_z_down":
            raise ValueError("environment frame must be controller_z_down")
        obstacles = expand_obstacles(environment["obstacles"])
        self.publish_mapping_sensors = self.declare_parameter("publish_mapping_sensors", False).value
        spec = mujoco.MjSpec.from_file(xml_path)
        flip = np.diag([1.0, -1.0, -1.0])
        start_rpy = [environment["start"][axis] for axis in ("roll", "pitch", "yaw")]
        for obstacle in obstacles:
            center_mj = to_zdown(obstacle["center"])
            size = 0.5 * np.asarray(obstacle["size"], dtype=float)
            if obstacle.get("type") == "cylinder":
                spec.worldbody.add_geom(name=obstacle["name"], type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                        pos=center_mj, size=[size[0], size[2]], rgba=obstacle["rgba"],
                                        contype=1, conaffinity=2, group=2)
            else:
                R_mj = flip @ rpy_to_rotmat(obstacle.get("rpy", [0.0, 0.0, 0.0])) @ flip
                spec.worldbody.add_geom(name=obstacle["name"], type=mujoco.mjtGeom.mjGEOM_BOX,
                                        pos=center_mj, quat=rotmat_to_quat(R_mj), size=size, rgba=obstacle["rgba"],
                                        material=obstacle.get("material", ""),
                                        contype=1, conaffinity=2, group=2)
        if self.publish_mapping_sensors:
            from plant.map_visuals import add_map_visuals
            add_map_visuals(spec, obstacles)
        for name, color in (("start", [0.10, 0.85, 0.20, 0.75]),
                            ("goal", [0.10, 0.40, 1.00, 0.75])):
            spec.worldbody.add_geom(name=f"{name}_waypoint", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                    pos=to_zdown([environment[name][axis] for axis in ("x", "y", "z")]),
                                    size=[0.12, 0.12, 0.12], rgba=color,
                                    contype=0, conaffinity=0)
        self.model = spec.compile()
        self.obstacle_geom_ids = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                                    obstacle["name"]) for obstacle in obstacles}
        self.obstacle_collision = False
        self.data = mujoco.MjData(self.model)
        root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        root_qpos = self.model.jnt_qposadr[root_id]
        self.data.qpos[root_qpos:root_qpos + 3] = to_zdown([environment["start"][axis] for axis in ("x", "y", "z")])
        self.data.qpos[root_qpos + 3:root_qpos + 7] = rotmat_to_quat(
            flip @ rpy_to_rotmat(start_rpy) @ flip)
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
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id < 0 or ROTOR_RADIUS_M <= 0.0 or TURBULENCE_NOISE_TIME_CONSTANT_SEC <= 0.0:
            raise RuntimeError("invalid floor or turbulence parameters")
        self.floor_z_mj = float(self.data.geom_xpos[floor_id, 2])
        self.wall_geom_group = np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)
        self.turbulence_rng = np.random.default_rng(TURBULENCE_SEED)
        self.turbulence_noise_state = np.zeros(HB_N_THRUST, dtype=float)
        self.turbulence_force_mj = np.zeros((HB_N_THRUST, 3), dtype=float)
        self.turbulence_gain = np.ones((HB_N_THRUST, 2), dtype=float)
        self.turbulence_range = np.full((HB_N_THRUST, 2), np.inf, dtype=float)
        self.turbulence_delta_t = np.zeros(HB_N_THRUST, dtype=float)
        self.turbulence_noise_t = np.zeros(HB_N_THRUST, dtype=float)

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
        self.start_goal_frames = [
            (to_zdown([environment[name][axis] for axis in ("x", "y", "z")]),
             flip @ rpy_to_rotmat([environment[name][axis] for axis in ("roll", "pitch", "yaw")]))
            for name in ("start", "goal")
        ]
        self.planned_frames = []

        self.sub_input = self.create_subscription(Input, "/input", self.input_callback, 10)
        self.sub_planned_pose = self.create_subscription(Cmd, "/planning_poses", self.planned_pose_callback, 50)
        self.sub_impulse_trigger = self.create_subscription(Bool, "/test/impulse_trigger", self.impulse_trigger_callback, 1)
        self.sub_moment_impulse_trigger = self.create_subscription(Bool, "/test/moment_impulse_trigger", self.moment_impulse_trigger_callback, 1)
        self.pub_state = self.create_publisher(MultirotorState, "/multirotor_state", 10)
        if self.publish_mapping_sensors:
            self.pub_imu = self.create_publisher(Imu, "/imu/data", 20)
            self.static_tf = StaticTransformBroadcaster(self)
            self.publish_camera_tf()
            self.imu_timer = self.create_timer(1.0 / IMU_HZ, self.publish_imu)
        self.viewer_thread = threading.Thread(target=self.viewer_loop)
        self.sim_thread = threading.Thread(target=self.sim_loop, daemon=True)

        self.viewer_thread.start()
        self.sim_thread.start()

        self.get_logger().info("state convention: z-down, [x, y, z] = [x_mj, -y_mj, -z_mj]")
        self.get_logger().info("input order: f[4], beta[2] expanded to ctrl[4:8], alpha[4] -> ctrl[0:4], ctrl[4:8], ctrl[8:12]")

    def publish_camera_tf(self):
        # MuJoCo camera looks along local -Z; ROS optical frame uses +Z forward, +X right, +Y down.
        R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        q = rotmat_to_quat(R)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "base_link"
        tf.child_frame_id = "camera_optical_frame"
        tf.transform.translation.x = 0.195
        tf.transform.translation.z = 0.02
        tf.transform.rotation.x = float(q[1])
        tf.transform.rotation.y = float(q[2])
        tf.transform.rotation.z = float(q[3])
        tf.transform.rotation.w = float(q[0])
        self.static_tf.sendTransform(tf)

    def publish_imu(self):
        with self.lock:
            q_imu = self.data.sensordata[self.s_adr[self.sid_imu_quat]:self.s_adr[self.sid_imu_quat] + 4].copy()
            gyro_imu = self.data.sensordata[self.s_adr[self.sid_imu_gyro]:self.s_adr[self.sid_imu_gyro] + 3].copy()
            acc_imu = self.data.sensordata[self.s_adr[self.sid_imu_acc]:self.s_adr[self.sid_imu_acc] + 3].copy()
        # The MuJoCo IMU site is rotated 180 deg around body X.
        R_body_world = quat_to_rotmat(q_imu) @ np.diag([1.0, -1.0, -1.0])
        q = rotmat_to_quat(R_body_world)
        gyro = to_zdown(gyro_imu)
        acc = to_zdown(acc_imu)
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = map(float, q)
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = map(float, gyro)
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = map(float, acc)
        msg.orientation_covariance = [0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001]
        msg.angular_velocity_covariance = [0.0001, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0, 0.0, 0.0001]
        msg.linear_acceleration_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        self.pub_imu.publish(msg)

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

    def apply_turbulence_disturbance(self):
        self.turbulence_force_mj.fill(0.0)
        self.turbulence_gain.fill(1.0)
        self.turbulence_range.fill(np.inf)
        self.turbulence_delta_t.fill(0.0)
        self.turbulence_noise_t.fill(0.0)
        if not USE_TURBULENCE_DISTURBANCE:
            return

        decay = math.exp(-1.0 / (PHYSICS_HZ * TURBULENCE_NOISE_TIME_CONSTANT_SEC))
        self.turbulence_noise_state *= decay
        self.turbulence_noise_state += math.sqrt(1.0 - decay * decay) * self.turbulence_rng.standard_normal(HB_N_THRUST)
        geomid = np.array([-1], dtype=np.int32)

        for i, site_id in enumerate(self.prop_site_ids):
            T0 = float(np.clip(self.ctrl[i], *self.model.actuator_ctrlrange[i]))
            pos = np.asarray(self.data.site_xpos[site_id], dtype=float)
            R = np.asarray(self.data.site_xmat[site_id], dtype=float).reshape(3, 3)
            a = R @ self.model.actuator_gear[i, :3]
            a /= np.linalg.norm(a)
            x = np.inf
            for direction in WALL_RAY_DIRECTIONS_MJ:
                distance = mujoco.mj_ray(self.model, self.data, pos, direction, self.wall_geom_group, True, -1, geomid)
                if 0.0 <= distance < x:
                    x = distance
            below = mujoco.mj_ray(self.model, self.data, pos, GROUND_RAY_DIRECTION_MJ, self.wall_geom_group, True, -1, geomid)
            z = max(0.0, pos[2] - self.floor_z_mj)
            if below >= 0.0:
                z = min(z, below)

            # Garofano-Soldado et al. (2022), Eqs. (7), (13)-(14): ground effect for tilted rotors.
            theta = math.acos(float(np.clip(a[2], 0.0, 1.0)))
            fc = max(0.0, 0.415 - 0.712 * math.sin(theta) + 0.361 * math.cos(theta))
            KG = 1.0 / (1.0 - (ROTOR_RADIUS_M / (4.0 * max(z, 0.6 * ROTOR_RADIUS_M))) ** 2 * fc) if a[2] > 0.0 else 1.0
            # Sanchez-Cuevas et al. (2021), Fig. 11: approximately 1% thrust loss near a wall.
            wall_factor = float(np.clip((3.8 - x / ROTOR_RADIUS_M) / 2.5, 0.0, 1.0))
            KW = 1.0 - 0.01 * wall_factor ** 3
            delta_t = T0 * (KG + KW - 2.0)
            ground_factor = float(np.clip((4.0 - z / ROTOR_RADIUS_M) / 3.4, 0.0, 1.0))
            noise_t = T0 * TURBULENCE_NOISE_RATIO * max(ground_factor, wall_factor) * self.turbulence_noise_state[i]
            force = (delta_t + noise_t) * a
            r = pos - self.data.xipos[self.bid_body]
            self.data.xfrc_applied[self.bid_body, :3] += force
            self.data.xfrc_applied[self.bid_body, 3:6] += np.cross(r, force)
            self.turbulence_force_mj[i] = force
            self.turbulence_gain[i] = [KG, KW]
            self.turbulence_range[i] = [x, z]
            self.turbulence_delta_t[i] = delta_t
            self.turbulence_noise_t[i] = noise_t

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

    def planned_pose_callback(self, msg):
        if not msg.use_quaternion:
            return
        quat = np.asarray(msg.quat_cmd, dtype=float)
        if np.linalg.norm(quat) < 1.0e-9:
            return
        # The controller body axes are unchanged; only the world basis changes.
        R = np.diag([1.0, -1.0, -1.0]) @ quat_to_rotmat(quat / np.linalg.norm(quat))
        with self.lock:
            self.planned_frames.append((to_zdown(msg.pos_cmd), R))

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
        disturbance_force = to_zdown(self.data.xfrc_applied[self.bid_body, :3])
        disturbance_moment = to_zdown(self.data.xfrc_applied[self.bid_body, 3:6])

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
        msg.disturbance_moment = disturbance_moment.tolist()
        msg.rotor_wall_distance = self.turbulence_range[:, 0].tolist()
        msg.rotor_ground_distance = self.turbulence_range[:, 1].tolist()
        msg.rotor_ground_gain = self.turbulence_gain[:, 0].tolist()
        msg.rotor_wall_gain = self.turbulence_gain[:, 1].tolist()
        msg.rotor_thrust_delta = self.turbulence_delta_t.tolist()
        msg.rotor_thrust_noise = self.turbulence_noise_t.tolist()

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
                    self.apply_turbulence_disturbance()
                    self.apply_random_disturbance()
                    self.apply_x_impulse()
                    self.apply_pitch_moment_impulse()
                    mujoco.mj_step(self.model, self.data)
                    if not self.obstacle_collision:
                        for contact in self.data.contact:
                            if contact.geom1 in self.obstacle_geom_ids or contact.geom2 in self.obstacle_geom_ids:
                                self.obstacle_collision = True
                                self.get_logger().error(f"obstacle collision at sim t={self.data.time:.3f} s")
                                break
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
            viewer.cam.lookat[:] = 0.5 * (self.start_goal_frames[0][0] + self.start_goal_frames[1][0])
            viewer.cam.distance = 10.0
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -25.0
            self.get_logger().info("viewer camera: free camera")
            return

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, VIEW_CAMERA_NAME)

        if cam_id < 0:
            self.get_logger().warn(f"camera not found: {VIEW_CAMERA_NAME}")
            return

        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id

        self.get_logger().info(f"viewer camera: {VIEW_CAMERA_NAME}")

    def update_pose_axes(self, viewer):
        if not hasattr(viewer, "user_scn"):
            return
        scn = viewer.user_scn

        def draw(pos, R, length, alpha):
            for axis in range(3):
                if scn.ngeom >= len(scn.geoms):
                    return
                rgba = POSE_AXIS_RGBA[axis].copy()
                rgba[3] = alpha
                tip = pos + length * R[:, axis]
                geom = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                                    np.zeros(3), np.zeros(3), np.eye(3).reshape(9), rgba)
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                                     POSE_AXIS_WIDTH * length / POSE_AXIS_LENGTH, pos, tip)
                scn.ngeom += 1

        for pos, R in self.start_goal_frames:
            draw(pos, R, POSE_AXIS_LENGTH, 1.0)
        for pos, R in self.planned_frames:
            draw(pos, R, POSE_TRACE_LENGTH, 0.60)
        pos = np.array(self.data.xpos[self.bid_body], dtype=float)
        R = np.array(self.data.xmat[self.bid_body], dtype=float).reshape(3, 3)
        draw(pos, R, POSE_AXIS_LENGTH, 1.0)

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

    def update_turbulence_arrows(self, viewer):
        if not SHOW_TURBULENCE_ARROWS or not hasattr(viewer, "user_scn"):
            return
        scn = viewer.user_scn
        for i, site_id in enumerate(self.prop_site_ids):
            force = self.turbulence_force_mj[i]
            if np.linalg.norm(force) < 1.0e-3 or scn.ngeom >= len(scn.geoms):
                continue
            pos = np.array(self.data.site_xpos[site_id], dtype=np.float64)
            tip = pos + TURBULENCE_ARROW_SCALE * force
            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3, dtype=np.float64),
                                np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64).reshape(9), TURBULENCE_ARROW_RGBA)
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
                    self.update_turbulence_arrows(viewer)
                    self.update_pose_axes(viewer)
                    if USE_TURBULENCE_DISTURBANCE:
                        values = "\n".join(f"R{i + 1} x={self.turbulence_range[i, 0]:.2f} z={self.turbulence_range[i, 1]:.2f} "
                                           f"KG={self.turbulence_gain[i, 0]:.3f} KW={self.turbulence_gain[i, 1]:.3f} "
                                           f"dT={self.turbulence_delta_t[i]:+.3f} noise={self.turbulence_noise_t[i]:+.3f} N"
                                           for i in range(HB_N_THRUST))
                        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                                          "Rotor disturbance", values))
                    viewer.sync()

                time.sleep(0.01)

    def close(self):
        self.stop_event.set()

        if self.sim_thread.is_alive():
            self.sim_thread.join(timeout=1.0)

        if self.viewer_thread.is_alive():
            self.viewer_thread.join()


def main():
    rclpy.init()
    node = PlantRosNode()

    def sigint_handler(signum, frame):
        node.get_logger().info("Ctrl+C received. Shutting down plant...")
        node.stop_event.set()

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