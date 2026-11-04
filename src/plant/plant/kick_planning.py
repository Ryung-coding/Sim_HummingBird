#!/usr/bin/env python3
"""Incremental KICK position planner: RTAB grid, EDT, A*, IPOPT, TOPPRA."""

import heapq
import math
import time

import cyipopt
import numpy as np
import rclpy
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray

from multirotor_interfaces.msg import Cmd, MultirotorState
from plant.plant import quat_to_rotmat, to_zdown

RESOLUTION = 0.2
CLEARANCE_M = 0.43
MAX_LEG_M = 3.2
MIN_LEG_M = 0.8
MAX_SPEED_MPS = 0.55
MAX_ACCEL_MPS2 = 0.8
REPLAN_ARRIVAL_M = 0.25


def point_key(xy):
    return tuple(np.rint(np.asarray(xy) / RESOLUTION).astype(int))


def key_point(key):
    return RESOLUTION * np.asarray(key, dtype=float)


def astar(start, goal, safe):
    queue = [(0.0, start)]
    cost = {start: 0.0}
    parent = {}
    offsets = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]
    while queue and len(cost) < 12000:
        _, here = heapq.heappop(queue)
        if here == goal:
            path = [here]
            while here in parent:
                here = parent[here]
                path.append(here)
            return np.asarray([key_point(p) for p in path[::-1]])
        for dx, dy in offsets:
            nxt = (here[0] + dx, here[1] + dy)
            if nxt not in safe:
                continue
            if dx and dy and ((here[0] + dx, here[1]) not in safe or
                              (here[0], here[1] + dy) not in safe):
                continue
            candidate = cost[here] + math.hypot(dx, dy)
            if candidate < cost.get(nxt, math.inf):
                cost[nxt] = candidate
                parent[nxt] = here
                heuristic = math.hypot(nxt[0] - goal[0], nxt[1] - goal[1])
                heapq.heappush(queue, (candidate + heuristic, nxt))
    return None


def path_sample(points, spacing=0.35):
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if lengths[-1] < 1.0e-5:
        return points
    count = max(3, int(math.ceil(lengths[-1] / spacing)) + 1)
    s = np.linspace(0.0, lengths[-1], count)
    return np.column_stack([np.interp(s, lengths, points[:, dim]) for dim in range(2)])


class SpatialProblem:
    def __init__(self, initial, clearance):
        self.initial = initial
        self.clearance = clearance
        self.count = initial.shape[0]
        self.samples = np.linspace(0.0, self.count - 1.0, 5 * (self.count - 1) + 1)

    def curve(self, x):
        points = self.initial.copy()
        points[1:-1] = x.reshape(-1, 2)
        s = np.arange(self.count, dtype=float)
        return np.column_stack([PchipInterpolator(s, points[:, dim])(self.samples)
                                for dim in range(2)])

    def objective(self, x):
        points = self.curve(x)
        delta = np.diff(points, axis=0)
        bend = np.diff(delta, axis=0)
        tether = x.reshape(-1, 2) - self.initial[1:-1]
        return (0.25 * np.linalg.norm(delta, axis=1).sum() +
                8.0 * np.square(bend).sum() + 1.5 * np.square(tether).sum())

    def constraints(self, x):
        return self.clearance(self.curve(x)[1:-1]) - CLEARANCE_M

    @staticmethod
    def derivative(fun, x):
        h = 1.0e-4
        base = np.atleast_1d(fun(x))
        jac = np.empty((len(base), len(x)))
        for i in range(len(x)):
            plus = x.copy()
            minus = x.copy()
            plus[i] += h
            minus[i] -= h
            jac[:, i] = (np.atleast_1d(fun(plus)) - np.atleast_1d(fun(minus))) / (2.0 * h)
        return jac

    def gradient(self, x):
        return self.derivative(self.objective, x).ravel()

    def jacobian(self, x):
        return self.derivative(self.constraints, x).ravel()


def optimize_path(initial, clearance):
    initial = path_sample(initial, 0.45)
    if len(initial) > 11:
        initial = path_sample(initial, max(0.45, len(initial) * 0.05))
    if len(initial) <= 2:
        return initial
    problem = SpatialProblem(initial, clearance)
    x0 = initial[1:-1].ravel()
    optimizer = cyipopt.Problem(
        n=len(x0), m=len(problem.samples) - 2, problem_obj=problem,
        lb=x0 - 0.35, ub=x0 + 0.35,
        cl=np.zeros(len(problem.samples) - 2),
        cu=np.full(len(problem.samples) - 2, 1.0e19),
    )
    optimizer.add_option("hessian_approximation", "limited-memory")
    optimizer.add_option("max_iter", 80)
    optimizer.add_option("max_cpu_time", 2.0)
    optimizer.add_option("tol", 1.0e-3)
    optimizer.add_option("print_level", 0)
    solution, info = optimizer.solve(x0)
    candidate = problem.curve(solution)
    if not np.all(np.isfinite(candidate)) or np.min(clearance(candidate)) < CLEARANCE_M - 0.02:
        raise RuntimeError(f"IPOPT status={info['status']}, minimum clearance={np.min(clearance(candidate)):.2f} m")
    return candidate


class KickPlanning(Node):
    def __init__(self):
        super().__init__("kick_planning")
        default = f"{get_package_share_directory('plant')}/environment_map.yaml"
        path = self.declare_parameter("environment_file", default).value
        with open(path, encoding="utf-8") as stream:
            environment = yaml.safe_load(stream)
        self.start = np.array([environment["start"][key] for key in ("x", "y", "z")], dtype=float)
        self.goal = np.array([environment["goal"][key] for key in ("x", "y", "z")], dtype=float)
        half = np.asarray(environment["vehicle"]["half_size"], dtype=float)
        self.min_xy = np.asarray(environment["bounds"]["min"][:2], dtype=float) + half[:2]
        self.max_xy = np.asarray(environment["bounds"]["max"][:2], dtype=float) - half[:2]
        self.position = self.start.copy()
        self.hold_xy = self.start[:2].copy()
        self.R = None
        self.t = None
        self.grid = None
        self.edt_points = None
        self.edt_values = None
        self.valid = False
        self.trajectory = None
        self.traj_time = 0.0
        self.last_tick = time.monotonic()
        self.next_plan = 0.0
        self.history = []
        self.goal_reported = False
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(TransformStamped, "/slam/map_alignment", self.on_alignment, qos)
        self.create_subscription(OccupancyGrid, "/rtabmap/map", self.on_grid, qos)
        self.create_subscription(PointCloud2, "/map_edt/observed_distance", self.on_edt, qos)
        self.create_subscription(MultirotorState, "/slam_state", self.on_state, 10)
        self.create_subscription(Bool, "/slam/position_valid", self.on_valid, 10)
        self.cmd_pub = self.create_publisher(Cmd, "/cmd", 10)
        self.pose_pub = self.create_publisher(Cmd, "/planning_poses", 50)
        self.marker_pub = self.create_publisher(MarkerArray, "/kick/markers", qos)
        self.create_timer(0.02, self.on_tick)
        self.get_logger().info(f"KICK goal={self.goal.tolist()}, speed bound={MAX_SPEED_MPS:.2f} m/s")

    def on_alignment(self, msg):
        q = msg.transform.rotation
        self.R = quat_to_rotmat([q.w, q.x, q.y, q.z])
        p = msg.transform.translation
        self.t = np.array([p.x, p.y, p.z])

    def map_to_controller(self, points):
        xyz = np.atleast_2d(points)
        mapped = (self.R @ xyz.T).T + self.t
        return mapped * np.array([1.0, -1.0, -1.0])

    def controller_to_map(self, points):
        xyz = np.atleast_2d(points) * np.array([1.0, -1.0, -1.0])
        return (self.R.T @ (xyz - self.t).T).T

    def on_grid(self, msg):
        self.grid = msg

    def on_edt(self, msg):
        data = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        self.edt_points = np.asarray(data[:, :3], dtype=float)
        self.edt_values = np.asarray(data[:, 3], dtype=float)

    def on_state(self, msg):
        self.position = np.array(msg.pos, dtype=float)

    def on_valid(self, msg):
        self.valid = bool(msg.data)

    def safe_map(self):
        msg = self.grid
        data = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        rows, cols = np.where(data == 0)
        if len(rows) == 0:
            return {}
        origin = msg.info.origin.position
        q = msg.info.origin.orientation
        R = quat_to_rotmat([q.w, q.x, q.y, q.z])
        local = np.column_stack(((cols + 0.5) * msg.info.resolution,
                                 (rows + 0.5) * msg.info.resolution,
                                 np.zeros(len(rows))))
        pmap = (R @ local.T).T + np.array([origin.x, origin.y, origin.z])
        keys = {point_key(xy) for xy in self.map_to_controller(pmap)[:, :2]}
        positions = np.asarray([key_point(key) for key in keys])
        clearance = self.clearance_field(None)
        values = clearance(positions)
        safe = {key: float(value) for key, value in zip(keys, values)
                if value >= CLEARANCE_M + 0.03}
        if not safe:
            return safe
        candidates = np.asarray(list(safe))
        distance, index = cKDTree(candidates * RESOLUTION).query(self.position[:2])
        if distance <= 0.9:
            first = key_point(candidates[index])
            for alpha in np.linspace(0.0, 1.0, max(3, int(distance / 0.1) + 1)):
                key = point_key((1.0 - alpha) * self.position[:2] + alpha * first)
                value = float(clearance(key_point(key))[0])
                if value >= CLEARANCE_M - 0.01:
                    safe[key] = value
        return safe

    def free_frontiers(self):
        msg = self.grid
        data = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        free = data == 0
        unknown = data == -1
        near_unknown = np.zeros_like(free)
        near_unknown[1:] |= unknown[:-1]
        near_unknown[:-1] |= unknown[1:]
        near_unknown[:, 1:] |= unknown[:, :-1]
        near_unknown[:, :-1] |= unknown[:, 1:]
        rows, cols = np.where(free & near_unknown)
        if len(rows) == 0:
            rows, cols = np.where(free)
        if len(rows) == 0:
            return np.empty((0, 2))
        origin = msg.info.origin.position
        q = msg.info.origin.orientation
        R = quat_to_rotmat([q.w, q.x, q.y, q.z])
        local = np.column_stack(((cols + 0.5) * msg.info.resolution,
                                 (rows + 0.5) * msg.info.resolution,
                                 np.zeros(len(rows))))
        pmap = (R @ local.T).T + np.array([origin.x, origin.y, origin.z])
        return self.map_to_controller(pmap)[:, :2]

    def clearance_field(self, safe):
        points = self.map_to_controller(self.edt_points)
        altitude = np.abs(points[:, 2] - self.goal[2]) < 0.13
        observed = points[altitude, :2]
        distances = self.edt_values[altitude]
        # The vehicle occupies a known free bubble before the front camera sees the floor.
        observed = np.vstack((observed, self.position[:2]))
        distances = np.r_[distances, 0.90]
        tree = cKDTree(observed)
        neighbors = min(12, len(observed))

        def clearance(xy):
            xy = np.atleast_2d(xy)
            offsets, indices = tree.query(xy, k=neighbors)
            offsets = np.atleast_2d(offsets)
            indices = np.atleast_2d(indices)
            if offsets.shape[0] != len(xy):
                offsets = offsets.T
                indices = indices.T
            # EDT distance is 1-Lipschitz: d_i - ||p-p_i|| is a conservative bound.
            lower_bound = np.max(distances[indices] - offsets, axis=1)
            observed_bound = CLEARANCE_M + 0.30 - offsets[:, 0]
            boundary = CLEARANCE_M + np.min(np.column_stack((xy - self.min_xy,
                                                             self.max_xy - xy)), axis=1)
            return np.minimum(np.minimum(lower_bound, observed_bound), boundary)
        return clearance

    def choose_path(self, safe, clearance):
        safe = {key for key, distance in safe.items() if distance >= CLEARANCE_M - 0.01}
        if not safe:
            return None, None
        keys = np.asarray(list(safe), dtype=int)
        tree = cKDTree(keys * RESOLUTION)
        start_distance, index = tree.query(self.position[:2])
        if start_distance > 0.9:
            return None, None
        start = tuple(keys[index])
        offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        reachable = {start}
        queue = [start]
        while queue:
            here = queue.pop()
            for dx, dy in offsets:
                nxt = (here[0] + dx, here[1] + dy)
                if nxt not in safe or nxt in reachable:
                    continue
                if dx and dy and ((here[0] + dx, here[1]) not in safe or
                                  (here[0], here[1] + dy) not in safe):
                    continue
                reachable.add(nxt)
                queue.append(nxt)
        if len(reachable) < 3:
            return None, None
        frontier = self.free_frontiers()
        frontier_keys = set()
        if len(frontier):
            distances, indices = tree.query(frontier)
            frontier_keys = {tuple(keys[i]) for d, i in zip(distances, indices) if d < 0.22}
        safe_points = np.asarray([key_point(key) for key in reachable])
        old_distance = np.linalg.norm(self.goal[:2] - self.position[:2])
        final_candidate = min(reachable, key=lambda key: np.linalg.norm(key_point(key) - self.goal[:2]))
        if np.linalg.norm(key_point(final_candidate) - self.goal[:2]) > 0.25:
            final_candidate = None
        options = []
        for key in reachable:
            point = key_point(key)
            leg = np.linalg.norm(point - self.position[:2])
            progress = old_distance - np.linalg.norm(point - self.goal[:2])
            is_final = key == final_candidate
            if leg < (0.10 if is_final else MIN_LEG_M) or progress <= (0.08 if is_final else 0.45):
                continue
            ahead = safe_points[(safe_points[:, 0] >= point[0]) &
                                (safe_points[:, 0] <= point[0] + 2.0) &
                                (np.abs(safe_points[:, 1] - point[1]) <= 0.6)]
            runway = float(np.max(ahead[:, 0] - point[0])) if len(ahead) else 0.0
            score = np.linalg.norm(point - self.goal[:2]) - 0.15 * leg - 2.0 * runway
            if key in frontier_keys:
                score -= 0.25
            if is_final:
                score -= 100.0
            options.append((score, key))
        for _, distant_goal in sorted(options)[:100]:
            path = astar(start, distant_goal, safe)
            if path is None or len(path) < 3:
                continue
            lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            if lengths[-1] > MAX_LEG_M:
                end_index = max(2, int(np.searchsorted(lengths, MAX_LEG_M, side="right") - 1))
                path = path[:end_index + 1]
            is_final = distant_goal == final_candidate and lengths[-1] <= MAX_LEG_M
            if is_final:
                path[-1] = self.goal[:2]
            final_progress = old_distance - np.linalg.norm(path[-1] - self.goal[:2])
            if final_progress <= (0.08 if is_final else 0.45) or np.linalg.norm(path[-1] - self.position[:2]) < (0.10 if is_final else MIN_LEG_M):
                continue
            initial = path_sample(path, 0.45)
            trial = SpatialProblem(initial, clearance).curve(initial[1:-1].ravel())
            if np.min(clearance(trial)) >= CLEARANCE_M - 0.02:
                return path[-1].copy(), path
        return None, None

    def publish_marker(self, path):
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for index, subgoal in enumerate(self.history):
            point = self.controller_to_map(np.array([subgoal[0], subgoal[1], self.goal[2]]))[0]
            for kind in ("sphere", "text"):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = stamp
                marker.ns = f"subgoal_{kind}"
                marker.id = index
                marker.type = Marker.SPHERE if kind == "sphere" else Marker.TEXT_VIEW_FACING
                marker.action = Marker.ADD
                marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, point)
                marker.pose.orientation.w = 1.0
                marker.scale.x = marker.scale.y = marker.scale.z = 0.18 if kind == "sphere" else 0.28
                marker.color.r = 1.0
                marker.color.g = 0.65 if index + 1 == len(self.history) else 0.25
                marker.color.b = 0.05
                marker.color.a = 1.0
                if kind == "text":
                    marker.pose.position.z += 0.32
                    marker.text = f"SG {index + 1}"
                markers.markers.append(marker)
        line = Marker()
        line.header.frame_id = "map"
        line.header.stamp = stamp
        line.ns = "optimized_path"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.055
        line.color.r, line.color.g, line.color.b, line.color.a = 0.1, 1.0, 0.3, 1.0
        for xy in path:
            p = self.controller_to_map(np.array([xy[0], xy[1], self.goal[2]]))[0]
            point = Point()
            point.x, point.y, point.z = map(float, p)
            line.points.append(point)
        markers.markers.append(line)
        self.marker_pub.publish(markers)

    def make_cmd(self, xy, vel=None, acc=None):
        cmd = Cmd()
        cmd.pos_cmd = [float(xy[0]), float(xy[1]), float(self.goal[2])]
        cmd.att_cmd = [0.0, 0.0, 0.0]
        cmd.quat_cmd = [1.0, 0.0, 0.0, 0.0]
        cmd.use_quaternion = True
        cmd.vel_cmd = [float(vel[0]), float(vel[1]), 0.0] if vel is not None else [0.0] * 3
        cmd.acc_cmd = [float(acc[0]), float(acc[1]), 0.0] if acc is not None else [0.0] * 3
        return cmd

    def plan(self):
        if self.R is None or self.grid is None or self.edt_points is None:
            return
        safe = self.safe_map()
        if not safe:
            return
        clearance = self.clearance_field(safe)
        goal, path = self.choose_path(safe, clearance)
        if path is None:
            self.get_logger().warn("No reachable mapped frontier with sufficient EDT clearance", throttle_duration_sec=4.0)
            return
        self.get_logger().info(f"A* to frontier {goal}: {len(path)} cells, initial clearance={np.min(clearance(path)):.2f} m")
        try:
            optimized = optimize_path(path, clearance)
            if np.min(clearance(optimized)) < CLEARANCE_M - 0.02:
                raise RuntimeError("dense spline validation failed")
            distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(optimized, axis=0), axis=1))]
            unique = np.r_[True, np.diff(distances) > 1.0e-5]
            spline = ta.SplineInterpolator(distances[unique], optimized[unique])
            limits = [constraint.JointVelocityConstraint(np.full(2, MAX_SPEED_MPS)),
                      constraint.JointAccelerationConstraint(np.full(2, MAX_ACCEL_MPS2))]
            trajectory = algo.TOPPRA(limits, spline).compute_trajectory(0.0, 0.0)
            if trajectory is None:
                raise RuntimeError("TOPPRA found no feasible time parameterization")
            samples = np.linspace(0.0, trajectory.duration, max(20, int(trajectory.duration / 0.05)))
            if np.max(np.linalg.norm(np.asarray(trajectory(samples, 1)), axis=1)) > math.sqrt(2) * MAX_SPEED_MPS + 0.03:
                raise RuntimeError("speed bound check failed")
        except (ValueError, RuntimeError) as exc:
            self.get_logger().warn(f"KICK optimization failed: {exc}", throttle_duration_sec=3.0)
            return
        self.trajectory = trajectory
        self.traj_time = 0.0
        self.hold_xy = np.asarray(trajectory(0.0)).copy()
        self.history.append(goal.copy())
        self.publish_marker(optimized)
        for xy in optimized[::max(1, len(optimized) // 15)]:
            self.pose_pub.publish(self.make_cmd(xy))
        self.get_logger().info(f"sub-goal {len(self.history)}: x={goal[0]:.2f}, "
                               f"y={goal[1]:.2f}; A*={len(path)} points, "
                               f"IPOPT={len(optimized)} points, TOPPRA={trajectory.duration:.2f} s")

    def on_tick(self):
        now = time.monotonic()
        dt = min(now - self.last_tick, 0.1)
        self.last_tick = now
        if not self.valid:
            self.cmd_pub.publish(self.make_cmd(self.hold_xy))
            return
        if self.trajectory is None:
            self.cmd_pub.publish(self.make_cmd(self.hold_xy))
            if np.linalg.norm(self.position[:2] - self.goal[:2]) < REPLAN_ARRIVAL_M:
                if not self.goal_reported:
                    self.goal_reported = True
                    self.get_logger().info(f"KICK goal reached after {len(self.history)} sub-goals")
                return
            if now >= self.next_plan:
                self.next_plan = now + 2.0
                self.plan()
            return
        self.traj_time = min(self.traj_time + dt, self.trajectory.duration)
        xy = np.asarray(self.trajectory(self.traj_time))
        vel = np.asarray(self.trajectory(self.traj_time, 1))
        acc = np.asarray(self.trajectory(self.traj_time, 2))
        self.cmd_pub.publish(self.make_cmd(xy, vel, acc))
        self.hold_xy = xy.copy()
        if self.traj_time >= self.trajectory.duration:
            if np.linalg.norm(self.position[:2] - xy) < REPLAN_ARRIVAL_M:
                self.trajectory = None
                self.next_plan = now + 0.4
            else:
                self.cmd_pub.publish(self.make_cmd(xy))


def main():
    rclpy.init()
    node = KickPlanning()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
