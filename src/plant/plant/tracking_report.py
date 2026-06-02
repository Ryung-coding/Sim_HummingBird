#!/usr/bin/env python3
import math

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from rclpy.node import Node

from multirotor_interfaces.msg import Cmd, Input, MultirotorState


SIM_DURATION_SEC = 30.0
RAD2DEG = 180.0 / math.pi


def wrap_pi(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


class TrackingReport(Node):
    def __init__(self):
        super().__init__("tracking_report")
        self.declare_parameter("duration_sec", SIM_DURATION_SEC)

        self.duration_sec = float(self.get_parameter("duration_sec").value)

        self.latest_cmd = None
        self.latest_input = None
        self.t0 = None
        self.samples = []
        self.done = False

        self.create_subscription(Cmd, "/cmd", self.on_cmd, 10)
        self.create_subscription(Input, "/input", self.on_input, 10)
        self.create_subscription(MultirotorState, "/multirotor_state", self.on_state, 10)

        self.get_logger().info(
            f"tracking report started: collecting {self.duration_sec:.1f} s"
        )

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_cmd(self, msg):
        self.latest_cmd = (
            np.asarray(msg.pos_cmd, dtype=float),
            np.asarray(msg.att_cmd, dtype=float),
        )

    def on_input(self, msg):
        u = np.asarray(msg.u, dtype=float)
        if u.shape[0] >= 10:
            self.latest_input = u[4:10].copy()

    def on_state(self, msg):
        if self.done or self.latest_cmd is None:
            return

        now = self.now_sec()
        if self.t0 is None:
            self.t0 = now

        t = now - self.t0
        pos_cmd, att_cmd = self.latest_cmd
        pos = np.asarray(msg.pos, dtype=float)
        rpy = np.asarray(msg.rpy, dtype=float)
        servo = np.asarray([*msg.theta, *msg.phi], dtype=float)
        if self.latest_input is None:
            servo_cmd = np.full(6, np.nan, dtype=float)
        else:
            servo_cmd = self.latest_input.copy()

        pos_err = pos_cmd - pos
        rot_err = np.asarray(
            [wrap_pi(att_cmd[i] - rpy[i]) for i in range(3)],
            dtype=float,
        )

        self.samples.append(
            (t, pos_cmd.copy(), pos.copy(), pos_err, rot_err, servo_cmd, servo)
        )

        if t >= self.duration_sec:
            self.done = True
            self.write_report()
            rclpy.shutdown()

    def write_report(self):
        if not self.samples:
            self.get_logger().warn("no tracking samples collected")
            return

        t = np.asarray([s[0] for s in self.samples], dtype=float)
        pos_cmd = np.vstack([s[1] for s in self.samples])
        pos = np.vstack([s[2] for s in self.samples])
        pos_err = np.vstack([s[3] for s in self.samples])
        rot_err = np.vstack([s[4] for s in self.samples])
        servo_cmd = np.vstack([s[5] for s in self.samples])
        servo = np.vstack([s[6] for s in self.samples])

        pos_rmse = np.sqrt(np.mean(pos_err ** 2, axis=0))
        rot_rmse = np.sqrt(np.mean(rot_err ** 2, axis=0))
        pos_norm_rmse = math.sqrt(np.mean(np.sum(pos_err ** 2, axis=1)))
        rot_norm_rmse = math.sqrt(np.mean(np.sum(rot_err ** 2, axis=1)))
        servo_response = self.estimate_servo_response(t, servo_cmd, servo)

        self.plot_trajectory_3d(pos_cmd, pos)
        self.plot_errors_2d(t, pos_err, rot_err, pos_rmse, rot_rmse, pos_norm_rmse, rot_norm_rmse)
        self.plot_servo_tracking(t, servo_cmd, servo, servo_response)

        self.get_logger().info(
            "Position RMSE [m] "
            f"x={pos_rmse[0]:.4f}, y={pos_rmse[1]:.4f}, z={pos_rmse[2]:.4f}, "
            f"norm={pos_norm_rmse:.4f}"
        )
        self.get_logger().info(
            "Rotation RMSE [deg] "
            f"roll={rot_rmse[0] * RAD2DEG:.3f}, "
            f"pitch={rot_rmse[1] * RAD2DEG:.3f}, "
            f"yaw={rot_rmse[2] * RAD2DEG:.3f}, "
            f"norm={rot_norm_rmse * RAD2DEG:.3f}"
        )
        self.log_servo_response(servo_response)
        self.get_logger().info("showing tracking plots; close plot windows to finish")
        plt.show()

    def estimate_servo_response(self, t, servo_cmd, servo):
        labels = ("theta1", "theta2", "phi1", "phi2", "phi3", "phi4")
        response = []

        for i, label in enumerate(labels):
            cmd = servo_cmd[:, i]
            cur = servo[:, i]
            valid = np.isfinite(cmd) & np.isfinite(cur) & np.isfinite(t)
            if np.count_nonzero(valid) < 5:
                response.append((label, None, None, None))
                continue

            tv = t[valid]
            cmdv = cmd[valid]
            curv = cur[valid]
            err = cmdv - curv
            qdot = np.gradient(curv, tv)
            active = np.abs(err) > math.radians(0.5)

            if np.count_nonzero(active) < 5:
                response.append((label, None, None, None))
                continue

            err = err[active]
            qdot = qdot[active]
            denom = float(np.dot(err, err))
            if denom <= 1e-12:
                response.append((label, None, None, None))
                continue

            rate = float(np.dot(err, qdot) / denom)
            rmse = math.sqrt(float(np.mean((cmdv - curv) ** 2)))
            if rate <= 1e-6:
                response.append((label, None, None, rmse))
                continue

            tau = 1.0 / rate
            hz = rate / (2.0 * math.pi)
            response.append((label, tau, hz, rmse))

        return response

    def log_servo_response(self, response):
        for label, tau, hz, rmse in response:
            rmse_text = "n/a" if rmse is None else f"{rmse * RAD2DEG:.3f} deg"
            if tau is None or hz is None:
                self.get_logger().info(
                    f"{label} servo response: unavailable, rmse={rmse_text}"
                )
            else:
                self.get_logger().info(
                    f"{label} servo response: tau={tau:.3f} s, "
                    f"effective={hz:.2f} Hz, rmse={rmse_text}"
                )

    def set_equal_3d_axes(self, ax, desired, current):
        pts = np.vstack((desired, current))
        center = np.mean(pts, axis=0)
        span = np.ptp(pts, axis=0)
        radius = max(0.5 * np.max(span), 1e-3)

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    def plot_trajectory_3d(self, pos_cmd, pos):
        fig = plt.figure(figsize=(9, 7))
        ax_traj = fig.add_subplot(1, 1, 1, projection="3d")

        ax_traj.plot(
            pos_cmd[:, 0], pos_cmd[:, 1], pos_cmd[:, 2],
            label="desired",
            linewidth=2.0,
        )
        ax_traj.plot(
            pos[:, 0], pos[:, 1], pos[:, 2],
            label="current",
            linewidth=2.0,
        )
        ax_traj.scatter(
            pos_cmd[0, 0], pos_cmd[0, 1], pos_cmd[0, 2],
            marker="o",
            s=25,
            label="start",
        )
        ax_traj.set_xlabel("x [m]")
        ax_traj.set_ylabel("y [m]")
        ax_traj.set_zlabel("z [m]")
        ax_traj.set_title("3D position trajectory")
        ax_traj.legend(loc="upper right")
        self.set_equal_3d_axes(ax_traj, pos_cmd, pos)
        fig.tight_layout()

    def plot_errors_2d(self, t, pos_err, rot_err, pos_rmse, rot_rmse, pos_norm_rmse, rot_norm_rmse):
        fig, (ax_pos, ax_rot) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

        for i, label in enumerate(("x", "y", "z")):
            ax_pos.plot(t, pos_err[:, i], label=label)
        ax_pos.set_ylabel("position error [m]")
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend(loc="upper right")
        ax_pos.text(
            0.01,
            0.02,
            "RMSE [m]\n"
            f"x {pos_rmse[0]:.4f}\n"
            f"y {pos_rmse[1]:.4f}\n"
            f"z {pos_rmse[2]:.4f}\n"
            f"norm {pos_norm_rmse:.4f}",
            transform=ax_pos.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "0.8"},
        )

        rot_err_deg = rot_err * RAD2DEG
        for i, label in enumerate(("roll", "pitch", "yaw")):
            ax_rot.plot(t, rot_err_deg[:, i], label=label)
        ax_rot.set_xlabel("time [s]")
        ax_rot.set_ylabel("rotation error [deg]")
        ax_rot.grid(True, alpha=0.3)
        ax_rot.legend(loc="upper right")
        rot_rmse_deg = rot_rmse * RAD2DEG
        ax_rot.text(
            0.01,
            0.02,
            "RMSE [deg]\n"
            f"roll {rot_rmse_deg[0]:.3f}\n"
            f"pitch {rot_rmse_deg[1]:.3f}\n"
            f"yaw {rot_rmse_deg[2]:.3f}\n"
            f"norm {rot_norm_rmse * RAD2DEG:.3f}",
            transform=ax_rot.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "0.8"},
        )

        fig.suptitle(f"Trajectory tracking error ({self.duration_sec:.0f} s)")
        fig.tight_layout()

    def plot_servo_tracking(self, t, servo_cmd, servo, servo_response):
        fig, (ax_theta, ax_phi) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

        theta_labels = ("theta1", "theta2")
        phi_labels = ("phi1", "phi2", "phi3", "phi4")
        servo_cmd_deg = servo_cmd * RAD2DEG
        servo_deg = servo * RAD2DEG

        for i, label in enumerate(theta_labels):
            ax_theta.plot(t, servo_cmd_deg[:, i], "--", label=f"{label} desired")
            ax_theta.plot(t, servo_deg[:, i], label=f"{label} current")
        ax_theta.set_ylabel("theta [deg]")
        ax_theta.grid(True, alpha=0.3)
        ax_theta.legend(loc="upper right", ncol=2)

        for i, label in enumerate(phi_labels):
            j = i + 2
            ax_phi.plot(t, servo_cmd_deg[:, j], "--", label=f"{label} desired")
            ax_phi.plot(t, servo_deg[:, j], label=f"{label} current")
        ax_phi.set_xlabel("time [s]")
        ax_phi.set_ylabel("phi [deg]")
        ax_phi.grid(True, alpha=0.3)
        ax_phi.legend(loc="upper right", ncol=2)

        table_lines = []
        for label, tau, hz, _ in servo_response:
            if tau is None or hz is None:
                table_lines.append(f"{label}: n/a")
            else:
                table_lines.append(f"{label}: {hz:.2f} Hz, tau {tau:.3f}s")

        ax_phi.text(
            0.01,
            0.02,
            "\n".join(table_lines),
            transform=ax_phi.transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
        )

        fig.suptitle(f"Servo desired-current tracking response ({self.duration_sec:.0f} s)")
        fig.tight_layout()


def main():
    rclpy.init()
    node = TrackingReport()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok() and not node.done:
            node.write_report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
