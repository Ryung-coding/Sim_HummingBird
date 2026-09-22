#!/usr/bin/env python3

import os
import math
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from multirotor_interfaces.msg import MultirotorState, Cmd, Debug


# ============================================================
# EXPERIMENT SETTING
# ============================================================

RUN_IMPULSE_TEST = False
RUN_RANDOM_TEST = False
MAKE_COMPARISON_FIGURE = True

# 실제 controller의 beta_ref를 바꾸는 값이 아님.
# params.hpp의 HB_BETA_REF와 동일하게 사람이 맞춰줄 것.
BETA_REF_DEG = 0.0


# ============================================================
# TIMING
# ============================================================

HOVER_SEC = 8.0

# impulse는 plant에서 +0.1 s -> -0.1 s라고 가정
IMPULSE_POS_SEC = 0.1
IMPULSE_NEG_SEC = 0.1

IMPULSE_PRE_SEC = 1.0
IMPULSE_POST_SEC = 5.0

RANDOM_LOG_SEC = 10.0


# ============================================================
# PERFORMANCE METRIC
# ============================================================

SETTLING_BAND_M = 0.02
SETTLING_HOLD_SEC = 1.0


# ============================================================
# FIGURE
# ============================================================

FIGURE_DPI = 600
FONT_NAME = "Times New Roman"

BETA0_COLOR = "#1F4E79"
BETA20_COLOR = "#C55A11"

LINE_WIDTH = 1.7


# ============================================================
# OUTPUT
# ============================================================

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(PACKAGE_DIR, "log")


def configure_plot():
    try:
        font_manager.findfont(FONT_NAME, fallback_to_default=False)
    except ValueError:
        raise RuntimeError(
            f'"{FONT_NAME}" font is not installed. '
            f'Check with: fc-match "Times New Roman"'
        )

    plt.rcParams.update({
        "font.family": FONT_NAME,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.0,
        "lines.linewidth": LINE_WIDTH,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def variance(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.var(x))


def settling_time(time, error):
    time = np.asarray(time, dtype=float)
    error = np.asarray(error, dtype=float)

    impulse_end = IMPULSE_POS_SEC + IMPULSE_NEG_SEC
    start_index = np.searchsorted(time, impulse_end)

    for i in range(start_index, len(time)):
        if abs(error[i]) > SETTLING_BAND_M:
            continue

        end_time = time[i] + SETTLING_HOLD_SEC
        j = np.searchsorted(time, end_time)

        if j >= len(time):
            break

        if np.all(np.abs(error[i:j + 1]) <= SETTLING_BAND_M):
            return float(time[i])

    return float("nan")


def beta_tag(beta):
    if abs(beta - round(beta)) < 1.0e-9:
        return f"beta{int(round(beta))}"
    return f"beta{str(beta).replace('.', 'p')}"


def run_filename(mode, beta):
    return os.path.join(LOG_DIR, f"{mode}_{beta_tag(beta)}.npz")


def style_axis(ax):
    ax.minorticks_on()
    ax.tick_params(which="major", length=5, width=1.0)
    ax.tick_params(which="minor", length=3, width=0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


class ExperimentLogger(Node):
    def __init__(self):
        super().__init__("experiment_logger")

        if RUN_IMPULSE_TEST == RUN_RANDOM_TEST:
            raise RuntimeError("Set exactly one of RUN_IMPULSE_TEST / RUN_RANDOM_TEST to True.")

        self.mode = "impulse" if RUN_IMPULSE_TEST else "random"

        self.latest_x_cmd = None
        self.latest_fx_cmd = None
        self.latest_fx_real = None

        self.impulse_sent = False
        self.impulse_trigger_time = None

        self.saved = False

        self.time = []
        self.x = []
        self.x_cmd = []
        self.fx_cmd = []
        self.fx_real = []
        self.disturbance_fx = []

        self.create_subscription(MultirotorState, "/multirotor_state", self.state_callback, 50)
        self.create_subscription(Cmd, "/cmd", self.cmd_callback, 50)
        self.create_subscription(Debug, "/allocation_debug", self.debug_callback, 50)

        self.impulse_pub = self.create_publisher(Bool, "/test/impulse_trigger", 1)

        os.makedirs(LOG_DIR, exist_ok=True)

        self.get_logger().info("==============================================")
        self.get_logger().info(f"Experiment mode : {self.mode}")
        self.get_logger().info(f"Beta label      : {BETA_REF_DEG:.1f} deg")
        self.get_logger().info(f"Output          : {LOG_DIR}")

        if self.mode == "impulse":
            self.get_logger().info(f"Impulse trigger : sim_time >= {HOVER_SEC:.1f} s")
            self.get_logger().info(f"Logging         : {HOVER_SEC - IMPULSE_PRE_SEC:.1f} ~ {HOVER_SEC + IMPULSE_POST_SEC:.1f} s")
        else:
            self.get_logger().info(f"Logging         : {HOVER_SEC:.1f} ~ {HOVER_SEC + RANDOM_LOG_SEC:.1f} s")

        self.get_logger().info("==============================================")

    def cmd_callback(self, msg):
        self.latest_x_cmd = float(msg.pos_cmd[0])

    def debug_callback(self, msg):
        self.latest_fx_cmd = float(msg.wrench_cmd[3])
        self.latest_fx_real = float(msg.wrench_real[3])

    def state_callback(self, msg):
        if self.saved:
            return

        t = float(msg.sim_time)

        if self.mode == "impulse" and not self.impulse_sent and t >= HOVER_SEC:
            trigger = Bool()
            trigger.data = True
            self.impulse_pub.publish(trigger)

            self.impulse_sent = True
            self.impulse_trigger_time = t

            self.get_logger().info(f"Impulse triggered at sim_time = {t:.4f} s")

        if self.mode == "impulse":
            log_start = HOVER_SEC - IMPULSE_PRE_SEC
            log_end = HOVER_SEC + IMPULSE_POST_SEC
        else:
            log_start = HOVER_SEC
            log_end = HOVER_SEC + RANDOM_LOG_SEC

        if log_start <= t <= log_end:
            if self.latest_x_cmd is None or self.latest_fx_cmd is None or self.latest_fx_real is None:
                return

            self.time.append(t)
            self.x.append(float(msg.pos[0]))
            self.x_cmd.append(self.latest_x_cmd)
            self.fx_cmd.append(self.latest_fx_cmd)
            self.fx_real.append(self.latest_fx_real)
            self.disturbance_fx.append(float(msg.disturbance_force[0]))

        if t > log_end and len(self.time) > 0:
            self.save_result()

    def save_result(self):
        self.saved = True

        time = np.asarray(self.time, dtype=float)
        x = np.asarray(self.x, dtype=float)
        x_cmd = np.asarray(self.x_cmd, dtype=float)
        fx_cmd = np.asarray(self.fx_cmd, dtype=float)
        fx_real = np.asarray(self.fx_real, dtype=float)
        disturbance_fx = np.asarray(self.disturbance_fx, dtype=float)

        x_err = x_cmd - x
        fx_err = fx_cmd - fx_real

        if self.mode == "impulse":
            reference_time = self.impulse_trigger_time if self.impulse_trigger_time is not None else HOVER_SEC
        else:
            reference_time = HOVER_SEC

        time_rel = time - reference_time

        if self.mode == "impulse":
            mask = time_rel >= 0.0

            x_var = variance(x_err[mask])
            x_peak = float(np.max(np.abs(x_err[mask])))
            fx_var = variance(fx_err[mask])
            fx_peak = float(np.max(np.abs(fx_err[mask])))
            x_settling = settling_time(time_rel, x_err)
        else:
            x_var = variance(x_err)
            x_peak = float(np.max(np.abs(x_err)))
            fx_var = variance(fx_err)
            fx_peak = float(np.max(np.abs(fx_err)))
            x_settling = float("nan")

        path = run_filename(self.mode, BETA_REF_DEG)

        np.savez_compressed(
            path,
            mode=self.mode,
            beta_ref_deg=BETA_REF_DEG,
            time=time_rel,
            sim_time=time,
            x=x,
            x_cmd=x_cmd,
            x_err=x_err,
            fx_cmd=fx_cmd,
            fx_real=fx_real,
            fx_err=fx_err,
            disturbance_fx=disturbance_fx,
            x_var=x_var,
            x_peak=x_peak,
            fx_var=fx_var,
            fx_peak=fx_peak,
            x_settling=x_settling,
        )

        self.get_logger().info("----------------------------------------------")
        self.get_logger().info(f"Saved: {path}")
        self.get_logger().info(f"X variance     = {x_var:.6e} m^2")
        self.get_logger().info(f"Peak |X error| = {x_peak:.6f} m")
        self.get_logger().info(f"Fx variance    = {fx_var:.6e} N^2")
        self.get_logger().info(f"Peak |Fx err|  = {fx_peak:.6f} N")

        if self.mode == "impulse":
            if math.isnan(x_settling):
                self.get_logger().info("Settling time   = not settled")
            else:
                self.get_logger().info(f"Settling time   = {x_settling:.4f} s")

        self.get_logger().info("----------------------------------------------")
        self.get_logger().info("Experiment finished. Logger will exit.")

        rclpy.shutdown()


def load_run(mode, beta):
    path = run_filename(mode, beta)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing experiment data: {path}")

    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def plot_impulse_comparison():
    d0 = load_run("impulse", 0.0)
    d20 = load_run("impulse", 20.0)

    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.0), sharex=True)

    t0 = np.asarray(d0["time"], dtype=float)
    t20 = np.asarray(d20["time"], dtype=float)
    x0 = np.asarray(d0["x_err"], dtype=float)
    x20 = np.asarray(d20["x_err"], dtype=float)
    fx0 = np.asarray(d0["fx_err"], dtype=float)
    fx20 = np.asarray(d20["fx_err"], dtype=float)

    axes[0].plot(t0, x0, color=BETA0_COLOR, linestyle="-", label=r"$\beta_{\mathrm{ref}}=0^\circ$")
    axes[0].plot(t20, x20, color=BETA20_COLOR, linestyle="--", label=r"$\beta_{\mathrm{ref}}=20^\circ$")
    axes[1].plot(t0, fx0, color=BETA0_COLOR, linestyle="-", label=r"$\beta_{\mathrm{ref}}=0^\circ$")
    axes[1].plot(t20, fx20, color=BETA20_COLOR, linestyle="--", label=r"$\beta_{\mathrm{ref}}=20^\circ$")

    for ax in axes:
        ax.axvline(0.0, color="0.35", linewidth=0.9, linestyle=":")
        ax.axvline(IMPULSE_POS_SEC, color="0.55", linewidth=0.8, linestyle=":")
        ax.axvline(IMPULSE_POS_SEC + IMPULSE_NEG_SEC, color="0.55", linewidth=0.8, linestyle=":")
        ax.legend(loc="upper right", frameon=False)
        style_axis(ax)

    axes[0].set_ylabel(r"$e_x=x_{\mathrm{cmd}}-x$ [m]")
    axes[1].set_ylabel(r"$F_{x,\mathrm{cmd}}-F_{x,\mathrm{real}}$ [N]")
    axes[1].set_xlabel("Time after disturbance onset [s]")

    x0_eval = x0[t0 >= 0.0]
    x20_eval = x20[t20 >= 0.0]
    fx0_eval = fx0[t0 >= 0.0]
    fx20_eval = fx20[t20 >= 0.0]

    metric_x = (
        f"Variance: 0° = {variance(x0_eval):.3e} m², 20° = {variance(x20_eval):.3e} m²\n"
        f"Peak: 0° = {np.max(np.abs(x0_eval)):.4f} m, 20° = {np.max(np.abs(x20_eval)):.4f} m"
    )
    metric_fx = (
        f"Variance: 0° = {variance(fx0_eval):.3e} N², 20° = {variance(fx20_eval):.3e} N²\n"
        f"Peak: 0° = {np.max(np.abs(fx0_eval)):.3f} N, 20° = {np.max(np.abs(fx20_eval)):.3f} N"
    )

    axes[0].text(0.98, 0.04, metric_x, transform=axes[0].transAxes, ha="right", va="bottom", fontsize=8.5)
    axes[1].text(0.98, 0.04, metric_fx, transform=axes[1].transAxes, ha="right", va="bottom", fontsize=8.5)

    axes[1].set_xlim(-IMPULSE_PRE_SEC, IMPULSE_POST_SEC)

    fig.suptitle(r"Impulse Response for Different $\beta_{\mathrm{ref}}$", y=0.985)
    fig.align_ylabels(axes)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])

    output = os.path.join(LOG_DIR, "Fig01_impulse_response.png")
    fig.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {output}")


def plot_random_comparison():
    d0 = load_run("random", 0.0)
    d20 = load_run("random", 20.0)

    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.0), sharex=True)

    t0 = np.asarray(d0["time"], dtype=float)
    t20 = np.asarray(d20["time"], dtype=float)
    x0 = np.asarray(d0["x_err"], dtype=float)
    x20 = np.asarray(d20["x_err"], dtype=float)
    fx0 = np.asarray(d0["fx_err"], dtype=float)
    fx20 = np.asarray(d20["fx_err"], dtype=float)

    axes[0].plot(t0, x0, color=BETA0_COLOR, linestyle="-", label=r"$\beta_{\mathrm{ref}}=0^\circ$")
    axes[0].plot(t20, x20, color=BETA20_COLOR, linestyle="--", label=r"$\beta_{\mathrm{ref}}=20^\circ$")
    axes[1].plot(t0, fx0, color=BETA0_COLOR, linestyle="-", label=r"$\beta_{\mathrm{ref}}=0^\circ$")
    axes[1].plot(t20, fx20, color=BETA20_COLOR, linestyle="--", label=r"$\beta_{\mathrm{ref}}=20^\circ$")

    for ax in axes:
        ax.legend(loc="upper right", frameon=False)
        style_axis(ax)

    axes[0].set_ylabel(r"$e_x=x_{\mathrm{cmd}}-x$ [m]")
    axes[1].set_ylabel(r"$F_{x,\mathrm{cmd}}-F_{x,\mathrm{real}}$ [N]")
    axes[1].set_xlabel("Logging time [s]")

    metric_x = (
        f"Variance: 0° = {variance(x0):.3e} m², 20° = {variance(x20):.3e} m²\n"
        f"Peak: 0° = {np.max(np.abs(x0)):.4f} m, 20° = {np.max(np.abs(x20)):.4f} m"
    )
    metric_fx = (
        f"Variance: 0° = {variance(fx0):.3e} N², 20° = {variance(fx20):.3e} N²\n"
        f"Peak: 0° = {np.max(np.abs(fx0)):.3f} N, 20° = {np.max(np.abs(fx20)):.3f} N"
    )

    axes[0].text(0.98, 0.04, metric_x, transform=axes[0].transAxes, ha="right", va="bottom", fontsize=8.5)
    axes[1].text(0.98, 0.04, metric_fx, transform=axes[1].transAxes, ha="right", va="bottom", fontsize=8.5)

    axes[1].set_xlim(0.0, RANDOM_LOG_SEC)

    fig.suptitle(r"Random Disturbance Response for Different $\beta_{\mathrm{ref}}$", y=0.985)
    fig.align_ylabels(axes)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])

    output = os.path.join(LOG_DIR, "Fig02_random_disturbance.png")
    fig.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {output}")


def make_comparison_figures():
    os.makedirs(LOG_DIR, exist_ok=True)
    configure_plot()

    plot_impulse_comparison()
    plot_random_comparison()

    print("")
    print("Comparison figures complete.")


def main():
    if MAKE_COMPARISON_FIGURE:
        make_comparison_figures()
        return

    rclpy.init()

    node = ExperimentLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()