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
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from ament_index_python.packages import get_package_share_directory

import mujoco
import mujoco.viewer
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool

from multirotor_interfaces.msg import HexaInput, Input, MultirotorState

# control rate and physics rate
PHYSICS_HZ = 400.0
PUB_HZ = 400.0
HB_SERVO_TAU_SEC = 0.2

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

HEXA_N_THRUST = 6
HEXA_N_ALPHA = 6
HEXA_N_CTRL = HEXA_N_ALPHA + 2 * HEXA_N_THRUST

# Visualization parameters
USE_FIXED_CAMERA = False
SHOW_THRUST_ARROWS = True
SHOW_PLANNING_PATH = True
USE_LIDAR = True
SHOW_LIDAR_RAYS = True
USE_WIND = False
USE_RANDOM_DISTURBANCE = False
USE_X_IMPULSE = True

VIEW_CAMERA_NAME = "front_camera"
    
THRUST_ARROW_SCALE = 0.025
THRUST_ARROW_MIN_LEN = 0.15
THRUST_ARROW_MAX_LEN = 1.20
THRUST_ARROW_WIDTH = 0.008
THRUST_ARROW_RGBA = np.array([1.00, 0.20, 0.05, 0.90], dtype=np.float32)

# Planning map is the controller-frame X-Z plane.
# Controller convention: x-forward, y-right, z-down.
PLANNING_MAP_RESOLUTION = 0.05
PLANNING_MAP_X_MIN = -0.30
PLANNING_MAP_X_MAX = 4.30
PLANNING_MAP_Z_MIN = -3.10
PLANNING_MAP_Z_MAX = -0.40

# Approximate 50 cm x 50 cm vehicle.
PLANNING_VEHICLE_RADIUS = 0.25
PLANNING_SAFETY_MARGIN = 0.05
PLANNING_PATH_WIDTH = 0.015
PLANNING_PATH_RGBA = np.array([0.05, 0.80, 1.00, 0.90], dtype=np.float32)

PLANNING_WAYPOINT_RADIUS = 0.06
PLANNING_WAYPOINT_RGBA = np.array([0.20, 1.00, 0.20, 0.95], dtype=np.float32)

# Physical duct interior is approximately 1.0 m wide/high.
PLANNING_DUCT_HALF_WIDTH = 0.50

# Available region for the vehicle center.
PLANNING_CENTER_HALF_WIDTH = (PLANNING_DUCT_HALF_WIDTH - PLANNING_VEHICLE_RADIUS - PLANNING_SAFETY_MARGIN)

LIDAR_MAX_RANGE = 2.0
LIDAR_RAY_WIDTH = 0.004
LIDAR_HIT_RGBA = np.array([0.05, 0.80, 1.00, 0.90], dtype=np.float32)
LIDAR_NO_HIT_RGBA = np.array([0.35, 0.55, 0.60, 0.20], dtype=np.float32)

AIRFLOW_RANGE = 0.50
AIRFLOW_FADE_RANGE = 0.10
AIRFLOW_TIME_CONSTANT = 0.45
AIR_DENSITY = 1.225
AIRFLOW_VELOCITY_STD = np.array([3.0, 3.0, 1.5], dtype=float)
AIRFLOW_DRAG_COEFF = np.array([1.10, 1.10, 1.00], dtype=float)
AIRFLOW_REFERENCE_AREA = np.array([0.20, 0.20, 0.08], dtype=float)
AIRFLOW_CP_OFFSET_BODY = np.array([0.0, 0.0, 0.01], dtype=float)
AIRFLOW_FORCE_LIMIT = 6.0
AIRFLOW_TORQUE_LIMIT = 0.15

RANDOM_DISTURBANCE_DIRECTION_ZDOWN = np.array([1.0, 0.0, 0.0], dtype=float)
RANDOM_DISTURBANCE_FORCE_MAX_N = 3.0
RANDOM_DISTURBANCE_TIME_CONSTANT = 0.2
RANDOM_DISTURBANCE_SEED = 20260916

IMPULSE_DIRECTION_ZDOWN = np.array([1.0, 0.0, 0.0], dtype=float)
IMPULSE_FORCE_N = 10.0
IMPULSE_DURATION_SEC = 0.1


class LPF:
    """First-order low-pass filter: y_dot = (x - y) / tau."""

    def __init__(self, tau_sec, dt_sec, initial):
        self.tau_sec = max(float(tau_sec), 0.0)
        self.dt_sec = max(float(dt_sec), 1.0e-9)
        self.alpha = (
            1.0
            if self.tau_sec <= 1.0e-9
            else 1.0 - math.exp(-self.dt_sec / self.tau_sec)
        )
        self.y = np.asarray(initial, dtype=float).copy()

    def update(self, x):
        x = np.asarray(x, dtype=float)
        self.y += self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self, x):
        self.y = np.asarray(x, dtype=float).copy()


def to_zdown(v):
    return np.array([v[0], -v[1], -v[2]], dtype=float)

def to_mj(v):
    """Convert controller world [x, y-right, z-down] back to MuJoCo world."""
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

        self.vehicle = self.declare_parameter("vehicle", "hummingbird").value
        self.mode = self.declare_parameter("mode", "position_cmd").value
        self.planning_enabled = self.mode == "planning"
        self.is_hexa = self.vehicle == "hexa"
        if self.vehicle not in ("hummingbird", "hexa"):
            raise ValueError(f"vehicle must be hummingbird or hexa, got {self.vehicle}")
        if self.mode not in ("position_cmd", "planning"):
            raise ValueError(f"mode must be position_cmd or planning, got {self.mode}")
        pkg_share = get_package_share_directory("plant")
        xml_path = (
            os.path.join(pkg_share, "xml", "HEXA_scene.xml")
            if self.is_hexa
            else os.path.join(pkg_share, "xml", "HB_scene.xml")
        )

        self.n_ctrl = HEXA_N_CTRL if self.is_hexa else HB_N_CTRL
        self.body_name = "base_link" if self.is_hexa else "body"
        self.use_lidar = USE_LIDAR and not self.is_hexa

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 1.0 / PHYSICS_HZ
        self.set_planning_obstacles_enabled()
        self.set_hb_servo_offset()
        mujoco.mj_forward(self.model, self.data)

        if self.model.nu != self.n_ctrl:
            self.get_logger().warn(f"model.nu is {self.model.nu}, but expected {self.n_ctrl}")

        self.sid_imu_quat = self.sensor_id("imu_quat")
        self.sid_imu_gyro = self.sensor_id("imu_gyro")
        self.sid_imu_acc = self.sensor_id("imu_acc")
        self.sid_body_pos = self.sensor_id("body_pos")
        self.sid_body_linvel = self.sensor_id("body_linvel")

        if self.is_hexa:
            self.sid_encoder_hexa_alpha = [
                self.sensor_id(f"encoder_hexa_alpha{i}") for i in range(HEXA_N_ALPHA)
            ]
            self.prop_site_ids = [
                self.site_id(f"rotor_{i}_thrust") for i in range(2 * HEXA_N_THRUST)
            ]
        else:
            self.sid_encoder_beta1 = self.sensor_id("encoder_beta1")
            self.sid_encoder_beta2 = self.sensor_id("encoder_beta2")
            self.sid_encoder_beta3 = self.sensor_id("encoder_beta3")
            self.sid_encoder_beta4 = self.sensor_id("encoder_beta4")

            self.sid_encoder_alpha1 = self.sensor_id("encoder_alpha1")
            self.sid_encoder_alpha2 = self.sensor_id("encoder_alpha2")
            self.sid_encoder_alpha3 = self.sensor_id("encoder_alpha3")
            self.sid_encoder_alpha4 = self.sensor_id("encoder_alpha4")
            self.prop_site_ids = [
                self.site_id("prop1_site"),
                self.site_id("prop2_site"),
                self.site_id("prop3_site"),
                self.site_id("prop4_site"),
            ]

        self.bid_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        if self.bid_body < 0:
            raise RuntimeError(f"body not found: {self.body_name}")

        self.sid_lidar = []
        self.lidar_site_ids = []
        if self.use_lidar:
            lidar_names = ("front", "back", "left", "right", "up", "down")
            self.sid_lidar = [
                self.sensor_id(f"lidar_{name}_range") for name in lidar_names
            ]
            self.lidar_site_ids = [
                self.site_id(f"lidar_{name}") for name in lidar_names
            ]

        self.s_adr = self.model.sensor_adr
        self.s_dim = self.model.sensor_dim
        self.lidar_ranges = np.full(6, -1.0, dtype=float)
        self.airflow_state = np.zeros(3, dtype=float)
        self.random_disturbance_state = 0.0
        self.random_disturbance_force_mj = np.zeros(3, dtype=float)
        self.impulse_end_time = None

        # Keep disturbance and sensor-noise realizations repeatable.
        np.random.seed(RANDOM_DISTURBANCE_SEED)

        direction_norm = np.linalg.norm(RANDOM_DISTURBANCE_DIRECTION_ZDOWN)
        if USE_RANDOM_DISTURBANCE and direction_norm <= 1.0e-9:
            raise ValueError("RANDOM_DISTURBANCE_DIRECTION_ZDOWN must be non-zero")
        self.random_disturbance_direction_mj = to_zdown(
            RANDOM_DISTURBANCE_DIRECTION_ZDOWN / max(direction_norm, 1.0e-9)
        )

        impulse_direction_norm = np.linalg.norm(IMPULSE_DIRECTION_ZDOWN)
        if USE_X_IMPULSE and impulse_direction_norm <= 1.0e-9:
            raise ValueError("IMPULSE_DIRECTION_ZDOWN must be non-zero")
        self.impulse_direction_mj = to_zdown(
            IMPULSE_DIRECTION_ZDOWN / max(impulse_direction_norm, 1.0e-9)
        )

        self.ctrl_recv = np.zeros(self.n_ctrl, dtype=float)
        self.ctrl = np.zeros(self.n_ctrl, dtype=float)

        if not self.is_hexa:
            servo_offset_rad = np.deg2rad(HB_SERVO_OFFSET_DEG)
            alpha_offset = servo_offset_rad[0:4]
            beta_offset = servo_offset_rad[4:6]
            beta_actuators = np.array(
                [beta_offset[0], beta_offset[1], beta_offset[1], beta_offset[0]],
                dtype=float
            )

            self.ctrl_recv[HB_N_THRUST:HB_N_THRUST + 4] = beta_actuators
            self.ctrl_recv[HB_N_THRUST + 4:HB_N_CTRL] = alpha_offset

            self.ctrl[HB_N_THRUST:HB_N_THRUST + 4] = beta_actuators
            self.ctrl[HB_N_THRUST + 4:HB_N_CTRL] = alpha_offset
            self.data.ctrl[:HB_N_CTRL] = self.ctrl[:HB_N_CTRL]

        self.hb_servo_lpf = None
        if not self.is_hexa:
            self.hb_servo_lpf = LPF(
                tau_sec=HB_SERVO_TAU_SEC,
                dt_sec=1.0 / PHYSICS_HZ,
                initial=self.ctrl[HB_N_THRUST:HB_N_CTRL],
            )

        self.prev_pub_t = None
        self.prev_linvel_zdown = None
        self.prev_gyro = None

        self.lock = threading.Lock()
        self.planning_path = np.empty((0, 3), dtype=float)
        self.planning_waypoints = np.empty((0, 3), dtype=float)
        self.stop_event = threading.Event()

        if self.is_hexa:
            self.sub_input = self.create_subscription(HexaInput, "/hexa_input", self.hexa_input_callback, 10)
        else:
            self.sub_input = self.create_subscription(Input, "/input", self.input_callback, 10)
        self.sub_impulse_trigger = self.create_subscription(
            Bool, "/test/impulse_trigger", self.impulse_trigger_callback, 1
        )
        self.pub_state = self.create_publisher(MultirotorState, "/multirotor_state", 10)
        self.pub_ogm = None
        if self.planning_enabled:
            map_qos = QoSProfile(
                depth=1,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                reliability=QoSReliabilityPolicy.RELIABLE
            )
            self.pub_ogm = self.create_publisher(OccupancyGrid, "/planning/ogm", map_qos)
            self.pub_ogm.publish(self.make_global_ogm())
            self.sub_planning_path = self.create_subscription(
                Path, "/planning/path", self.planning_path_callback, map_qos
            )
            self.sub_planning_waypoints = self.create_subscription(
                Path, "/planning/waypoints", self.planning_waypoints_callback, map_qos
            )

        self.viewer_thread = threading.Thread(target=self.viewer_loop, daemon=True)
        self.sim_thread = threading.Thread(target=self.sim_loop, daemon=True)

        self.viewer_thread.start()
        self.sim_thread.start()

        self.get_logger().info("state convention: z-down, [x, y, z] = [x_mj, -y_mj, -z_mj]")
        if self.is_hexa:
            self.get_logger().info("input order: alpha[6], then rotor_0..11 thrust; each f[i] is split across rotor_i and rotor_i+6")
        else:
            self.get_logger().info("input order: f[4], beta[2] expanded to ctrl[4:8], alpha[4] -> ctrl[0:4], ctrl[4:8], ctrl[8:12]")

    def set_hb_servo_offset(self):
        if self.is_hexa:
            return

        if HB_SERVO_OFFSET_DEG.shape != (6,):
            raise ValueError(
                "HB_SERVO_OFFSET_DEG must be "
                "[alpha1, alpha2, alpha3, alpha4, beta1, beta2]"
            )

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
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )

            if joint_id < 0:
                raise RuntimeError(f"joint not found: {joint_name}")

            self.data.qpos[self.model.jnt_qposadr[joint_id]] = value

    def set_planning_obstacles_enabled(self):
        """Enable physical planning obstacles only for mode:=planning."""
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name is None or not name.startswith("obstacle_"):
                continue

            if not self.planning_enabled:
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0
                self.model.geom_rgba[geom_id, 3] = 0.0

    def make_global_ogm(self):
        """Build a virtual X-Z occupancy grid for the duct.

        The second OccupancyGrid axis is controller z, not world y.

        All cells are occupied by default. Only a narrow tube around the
        duct centerline is carved as free space. Therefore A* cannot escape
        around the side of the duct.
        """

        width = int(round(
            (PLANNING_MAP_X_MAX - PLANNING_MAP_X_MIN)
            / PLANNING_MAP_RESOLUTION
        ))

        height = int(round(
            (PLANNING_MAP_Z_MAX - PLANNING_MAP_Z_MIN)
            / PLANNING_MAP_RESOLUTION
        ))

        x = (
            PLANNING_MAP_X_MIN
            + (np.arange(width) + 0.5)
            * PLANNING_MAP_RESOLUTION
        )

        z = (
            PLANNING_MAP_Z_MIN
            + (np.arange(height) + 0.5)
            * PLANNING_MAP_RESOLUTION
        )

        grid_x, grid_z = np.meshgrid(x, z)

        # Start with everything blocked.
        occupied = np.full(
            (height, width),
            100,
            dtype=np.int8
        )

        # Controller-frame centerline:
        #
        # lower:
        #   (0.0, -1.0) -> (1.5, -1.0)
        #
        # 45 deg slope:
        #   (1.5, -1.0) -> (3.0, -2.5)
        #
        # upper:
        #   (3.0, -2.5) -> (4.0, -2.5)
        #
        centerline = np.array([
            [-0.20, -1.00],
            [ 1.50, -1.00],
            [ 3.00, -2.50],
            [ 4.20, -2.50],
        ], dtype=float)

        min_dist_sq = np.full(
            grid_x.shape,
            np.inf,
            dtype=float
        )

        # Compute distance from every grid cell to the nearest
        # centerline segment.
        for k in range(centerline.shape[0] - 1):

            p0 = centerline[k]
            p1 = centerline[k + 1]

            vx = p1[0] - p0[0]
            vz = p1[1] - p0[1]

            seg_len_sq = vx * vx + vz * vz

            wx = grid_x - p0[0]
            wz = grid_z - p0[1]

            t = (
                wx * vx + wz * vz
            ) / max(seg_len_sq, 1.0e-12)

            t = np.clip(t, 0.0, 1.0)

            proj_x = p0[0] + t * vx
            proj_z = p0[1] + t * vz

            dist_sq = (
                (grid_x - proj_x) ** 2
                + (grid_z - proj_z) ** 2
            )

            min_dist_sq = np.minimum(
                min_dist_sq,
                dist_sq
            )

        free = (
            min_dist_sq
            <= PLANNING_CENTER_HALF_WIDTH ** 2
        )

        occupied[free] = 0

        msg = OccupancyGrid()

        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        # This topic is intentionally an X-Z planning grid.
        msg.header.frame_id = "planning_xz"

        msg.info.resolution = (
            PLANNING_MAP_RESOLUTION
        )

        msg.info.width = width
        msg.info.height = height

        msg.info.origin.position.x = (
            PLANNING_MAP_X_MIN
        )

        # OccupancyGrid's second coordinate is reused as z.
        msg.info.origin.position.y = (
            PLANNING_MAP_Z_MIN
        )

        msg.info.origin.orientation.w = 1.0
        msg.data = occupied.ravel().tolist()

        self.get_logger().info(
            f"planning X-Z OGM: "
            f"{width}x{height}, "
            f"free half width "
            f"{PLANNING_CENTER_HALF_WIDTH:.2f} m"
        )
        return msg
    
    def planning_path_callback(self, msg):
        path = np.asarray([
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
            for pose in msg.poses
        ], dtype=float).reshape((-1, 3))
        with self.lock:
            self.planning_path = path

    def planning_waypoints_callback(self, msg):
        waypoints = np.asarray([
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
            for pose in msg.poses
        ], dtype=float).reshape((-1, 3))
        with self.lock:
            self.planning_waypoints = waypoints

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

    def read_lidar_ranges(self):
        if not self.use_lidar:
            return np.full(6, -1.0, dtype=float)

        lidar = np.array([
            self.sensing_state(sid)[0] for sid in self.sid_lidar
        ], dtype=float)
        lidar[(lidar < 0.0) | (lidar > LIDAR_MAX_RANGE)] = -1.0
        return lidar

    def apply_wall_airflow(self):
        if not USE_WIND or not self.use_lidar:
            return

        decay = math.exp(
            -1.0 / (PHYSICS_HZ * AIRFLOW_TIME_CONSTANT)
        )
        self.airflow_state = (
            decay * self.airflow_state
            + math.sqrt(1.0 - decay * decay)
            * np.random.normal(size=self.airflow_state.shape)
        )

        nearby = self.lidar_ranges[
            (self.lidar_ranges >= 0.0)
            & (self.lidar_ranges < AIRFLOW_RANGE)
        ]

        activation = 0.0
        if nearby.size >= 2:
            activation = np.clip(
                (AIRFLOW_RANGE - np.min(nearby)) / AIRFLOW_FADE_RANGE,
                0.0,
                1.0
            )
            activation = activation * activation * (3.0 - 2.0 * activation)

        wind_velocity = AIRFLOW_VELOCITY_STD * self.airflow_state
        body_velocity = self.sensing_state(self.sid_body_linvel)
        relative_wind_world = wind_velocity - body_velocity

        R_body_world = np.asarray(
            self.data.xmat[self.bid_body], dtype=float
        ).reshape(3, 3)
        relative_wind_body = R_body_world.T @ relative_wind_world
        airflow_force_body = (
            0.5
            * AIR_DENSITY
            * AIRFLOW_DRAG_COEFF
            * AIRFLOW_REFERENCE_AREA
            * relative_wind_body
            * np.abs(relative_wind_body)
        )
        airflow_force = activation * (R_body_world @ airflow_force_body)
        airflow_torque = activation * (
            R_body_world
            @ np.cross(AIRFLOW_CP_OFFSET_BODY, airflow_force_body)
        )

        force_norm = np.linalg.norm(airflow_force)
        if force_norm > AIRFLOW_FORCE_LIMIT:
            airflow_force *= AIRFLOW_FORCE_LIMIT / force_norm

        torque_norm = np.linalg.norm(airflow_torque)
        if torque_norm > AIRFLOW_TORQUE_LIMIT:
            airflow_torque *= AIRFLOW_TORQUE_LIMIT / torque_norm

        self.data.xfrc_applied[self.bid_body, :3] += airflow_force
        self.data.xfrc_applied[self.bid_body, 3:] += airflow_torque

    def apply_random_disturbance(self):
        self.random_disturbance_force_mj.fill(0.0)

        if not USE_RANDOM_DISTURBANCE:
            return

        decay = math.exp(
            -1.0 / (PHYSICS_HZ * RANDOM_DISTURBANCE_TIME_CONSTANT)
        )
        self.random_disturbance_state = (
            decay * self.random_disturbance_state
            + math.sqrt(1.0 - decay * decay) * np.random.normal()
        )

        random_scale = abs(math.tanh(self.random_disturbance_state))
        disturbance_force = (
            RANDOM_DISTURBANCE_FORCE_MAX_N
            * random_scale
            * self.random_disturbance_direction_mj
        )
        self.random_disturbance_force_mj = disturbance_force
        self.data.xfrc_applied[self.bid_body, :3] += disturbance_force

    def impulse_trigger_callback(self, msg):
        if not USE_X_IMPULSE or not msg.data:
            return

        with self.lock:
            self.impulse_end_time = self.data.time + IMPULSE_DURATION_SEC
            self.get_logger().info(
                f"impulse triggered at sim t={self.data.time:.3f} s"
            )

    def apply_x_impulse(self):
        if self.impulse_end_time is None or self.data.time >= self.impulse_end_time:
            return

        impulse_force = IMPULSE_FORCE_N * self.impulse_direction_mj
        self.random_disturbance_force_mj += impulse_force
        self.data.xfrc_applied[self.bid_body, :3] += impulse_force

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

    def hexa_input_callback(self, msg):
        f = np.asarray(msg.f, dtype=float)
        alpha = np.asarray(msg.alpha, dtype=float)

        if f.shape[0] != HEXA_N_THRUST:
            self.get_logger().warn(f"f size must be {HEXA_N_THRUST}, but got {f.shape[0]}")
            return

        if alpha.shape[0] != HEXA_N_ALPHA:
            self.get_logger().warn(f"alpha size must be {HEXA_N_ALPHA}, but got {alpha.shape[0]}")
            return

        with self.lock:
            self.ctrl_recv[0:HEXA_N_ALPHA] = alpha
            self.ctrl_recv[HEXA_N_ALPHA:HEXA_N_CTRL] = np.concatenate((0.5 * f, 0.5 * f))

    def apply_control(self):
        if self.is_hexa:
            self.ctrl = self.ctrl_recv.copy()
        else:
            # BLDC thrust is applied directly.
            self.ctrl[0:HB_N_THRUST] = self.ctrl_recv[0:HB_N_THRUST]

            # Servo commands [beta1..4, alpha1..4] follow a first-order lag.
            self.ctrl[HB_N_THRUST:HB_N_CTRL] = self.hb_servo_lpf.update(
                self.ctrl_recv[HB_N_THRUST:HB_N_CTRL]
            )

        self.data.ctrl[:self.n_ctrl] = self.ctrl[:self.n_ctrl]

    def make_state_msg(self, now):
        imu_quat = self.sensing_state(self.sid_imu_quat)
        imu_gyro = add_noise(self.sensing_state(self.sid_imu_gyro), SIG_GYRO)
        imu_acc = add_noise(self.sensing_state(self.sid_imu_acc), SIG_ACC)
        pos_mj = add_noise(self.sensing_state(self.sid_body_pos), SIG_POS)
        vel_mj = add_noise(self.sensing_state(self.sid_body_linvel), SIG_VEL)

        pos = to_zdown(pos_mj)
        vel = to_zdown(vel_mj)
        disturbance_force = to_zdown(self.random_disturbance_force_mj)

        if self.is_hexa:
            beta = np.zeros(HB_N_BETA, dtype=float)
            alpha = np.zeros(HB_N_ALPHA, dtype=float)
            hexa_alpha = np.array([
                self.sensing_state(sensor_id)[0] for sensor_id in self.sid_encoder_hexa_alpha
            ], dtype=float)
            hexa_alpha = add_noise(hexa_alpha, SIG_ENCODER)
        else:
            beta = np.array([
                self.sensing_state(self.sid_encoder_beta1)[0],
                self.sensing_state(self.sid_encoder_beta2)[0],
            ], dtype=float)

            alpha = np.array([
                self.sensing_state(self.sid_encoder_alpha1)[0],
                self.sensing_state(self.sid_encoder_alpha2)[0],
                self.sensing_state(self.sid_encoder_alpha3)[0],
                self.sensing_state(self.sid_encoder_alpha4)[0],
            ], dtype=float)

            beta = add_noise(beta, SIG_ENCODER)
            alpha = add_noise(alpha, SIG_ENCODER)
            hexa_alpha = np.zeros(HEXA_N_ALPHA, dtype=float)

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
        msg.hexa_alpha = hexa_alpha.tolist()
        msg.lidar = self.lidar_ranges.tolist()

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
                    self.lidar_ranges = self.read_lidar_ranges()
                    self.apply_wall_airflow()
                    self.apply_random_disturbance()
                    self.apply_x_impulse()
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

            thrust_index = HEXA_N_ALPHA + i if self.is_hexa else i
            thrust = max(0.0, float(self.ctrl[thrust_index]))

            if thrust <= 1.0e-6:
                continue

            length = np.clip(
                THRUST_ARROW_SCALE * thrust,
                THRUST_ARROW_MIN_LEN,
                THRUST_ARROW_MAX_LEN
            )

            pos = np.array(self.data.site_xpos[site_id], dtype=np.float64)
            R = np.array(self.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
            thrust_dir = R @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
            tip = pos + length * thrust_dir

            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.eye(3, dtype=np.float64).reshape(9),
                THRUST_ARROW_RGBA
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                THRUST_ARROW_WIDTH,
                pos,
                tip
            )
            scn.ngeom += 1

    def update_lidar_rays(self, viewer):
        if not self.use_lidar or not SHOW_LIDAR_RAYS or not hasattr(viewer, "user_scn"):
            return

        scn = viewer.user_scn

        for distance, site_id in zip(self.lidar_ranges, self.lidar_site_ids):
            if scn.ngeom >= len(scn.geoms):
                return

            hit = distance >= 0.0
            length = min(float(distance), LIDAR_MAX_RANGE) if hit else LIDAR_MAX_RANGE
            pos = np.array(self.data.site_xpos[site_id], dtype=np.float64)
            R_site = np.array(
                self.data.site_xmat[site_id],
                dtype=np.float64
            ).reshape(3, 3)
            tip = pos + length * R_site[:, 2]

            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.eye(3, dtype=np.float64).reshape(9),
                LIDAR_HIT_RGBA if hit else LIDAR_NO_HIT_RGBA
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                LIDAR_RAY_WIDTH,
                pos,
                tip
            )
            scn.ngeom += 1

    def update_planning_visualization(self, viewer):
        if not self.planning_enabled or not SHOW_PLANNING_PATH or not hasattr(viewer, "user_scn"):
            return

        scn = viewer.user_scn

        # The callbacks store controller-frame positions; MuJoCo renders x-right,
        # y-left, z-up, so transform each point before drawing it.
        for waypoint in self.planning_waypoints:
            if scn.ngeom >= len(scn.geoms):
                return

            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([PLANNING_WAYPOINT_RADIUS, 0.0, 0.0], dtype=np.float64),
                to_mj(waypoint).astype(np.float64),
                np.eye(3, dtype=np.float64).reshape(9),
                PLANNING_WAYPOINT_RGBA
            )
            scn.ngeom += 1

        path_count = self.planning_path.shape[0]
        if path_count < 2 or scn.ngeom >= len(scn.geoms):
            return

        # If MuJoCo has fewer user geoms than samples, decimate while retaining
        # the first-to-last connected curve rather than truncating the route.
        available_geoms = len(scn.geoms) - scn.ngeom
        stride = max(1, int(math.ceil((path_count - 1) / available_geoms)))
        for start in range(0, path_count - 1, stride):
            end = min(start + stride, path_count - 1)
            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.eye(3, dtype=np.float64).reshape(9),
                PLANNING_PATH_RGBA
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                PLANNING_PATH_WIDTH,
                to_mj(self.planning_path[start]).astype(np.float64),
                to_mj(self.planning_path[end]).astype(np.float64)
            )
            scn.ngeom += 1

    def viewer_loop(self):
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self.set_viewer_camera(viewer)

                while viewer.is_running() and rclpy.ok() and not self.stop_event.is_set():
                    with self.lock:
                        if hasattr(viewer, "user_scn"):
                            viewer.user_scn.ngeom = 0
                        self.update_thrust_arrows(viewer)
                        self.update_lidar_rays(viewer)
                        self.update_planning_visualization(viewer)
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