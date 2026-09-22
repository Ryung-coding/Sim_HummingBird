#!/usr/bin/env python3
import sys
import math
import threading
import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool
from multirotor_interfaces.msg import MultirotorState, Cmd, Input, Debug

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg


WINDOW_SEC = 10.0
MAX_SAMPLES = 5000
UPDATE_MS = 50
RAD2DEG = 180.0 / math.pi

USE_SO3_HEADING_CMD = False

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")

C_R = "#e6194b"
C_G = "#3cb44b"
C_B = "#4363d8"
C_O = "#f58231"

C4 = [C_R, C_G, C_B, C_O]


# Hummingbird rotor locations in controller body XY
ROTOR_BODY_XY = np.array([
    [ 0.175,  0.175],
    [-0.175,  0.175],
    [-0.175, -0.175],
    [ 0.175, -0.175]
], dtype=float)

# Plot horizontal axis = body y, vertical axis = body x
ROTOR_PLOT_YX = ROTOR_BODY_XY[:, [1, 0]]

# beta1 -> rotor 1,4 / beta2 -> rotor 2,3
BETA_INDEX = np.array([0, 1, 1, 0], dtype=int)

TILT_VECTOR_SCALE = 0.35
TILT_ARROW_HEAD = 0.035


class Ring:
    def __init__(self, cap, w):
        self._c = cap
        self._w = w
        self._b = np.full((cap, w), np.nan, dtype=np.float64)
        self._i = 0
        self._n = 0

    def push(self, row):
        self._b[self._i] = row
        self._i = (self._i + 1) % self._c
        if self._n < self._c:
            self._n += 1

    def get(self):
        if self._n == 0:
            return np.empty((0, self._w), dtype=np.float64)
        if self._n < self._c:
            return self._b[:self._n].copy()
        return np.concatenate([self._b[self._i:], self._b[:self._i]], axis=0)


def att_cmd_to_rpy_deg(att_cmd):
    if not USE_SO3_HEADING_CMD:
        return np.asarray(att_cmd[:3], dtype=float) * RAD2DEG

    h = np.asarray(att_cmd[:3], dtype=float)
    n = np.linalg.norm(h)
    if n < 1.0e-9:
        return np.zeros(3, dtype=float)

    h /= n
    yaw = math.atan2(h[1], h[0])
    pitch = math.atan2(-h[2], math.sqrt(h[0] * h[0] + h[1] * h[1]))
    return np.array([0.0, pitch, yaw], dtype=float) * RAD2DEG


def tilt_xy_components(alpha_deg, beta_deg):
    alpha = np.asarray(alpha_deg, dtype=float) / RAD2DEG
    beta = np.asarray(beta_deg, dtype=float)[BETA_INDEX] / RAD2DEG

    # e_i = [-sin(beta) cos(alpha), sin(alpha), -cos(beta) cos(alpha)]
    return np.column_stack((
        -np.sin(beta) * np.cos(alpha),
        np.sin(alpha)
    ))


def vector_polyline(origin, vector):
    tip = origin + TILT_VECTOR_SCALE * vector
    delta = tip - origin
    length = np.linalg.norm(delta)

    if length < 1.0e-9:
        return np.array([origin[0], tip[0]]), np.array([origin[1], tip[1]])

    unit = delta / length
    normal = np.array([-unit[1], unit[0]])

    head = min(TILT_ARROW_HEAD, 0.4 * length)
    base = tip - head * unit
    left = base + 0.55 * head * normal
    right = base - 0.55 * head * normal

    x = np.array([origin[0], tip[0], np.nan, tip[0], left[0], np.nan, tip[0], right[0]])
    y = np.array([origin[1], tip[1], np.nan, tip[1], left[1], np.nan, tip[1], right[1]])

    return x, y


def _pen(color, w=2):
    return pg.mkPen(color=color, width=w, style=QtCore.Qt.SolidLine)


def _dash_pen(color="k", w=2):
    return pg.mkPen(color=color, width=w, style=QtCore.Qt.DashLine)


def _front(curve):
    curve.setZValue(5)
    return curve


def _mkplot(glw, row, col, title, ylabel, **kwargs):
    p = glw.addPlot(row=row, col=col, title=title, **kwargs)
    p.showGrid(x=True, y=True, alpha=0.3)
    p.setLabel("left", ylabel)
    p.getAxis("left").enableAutoSIPrefix(False)
    p.getAxis("bottom").enableAutoSIPrefix(False)

    for axis in ("bottom", "left"):
        p.getAxis(axis).setPen(pg.mkPen("k"))
        p.getAxis(axis).setTextPen(pg.mkPen("k"))

    p.addLegend(offset=(-10, 5))
    return p


class VNode(Node):
    def __init__(self):
        super().__init__("multirotor_viewer")

        self.lock = threading.Lock()
        self.t0 = self.get_clock().now().nanoseconds * 1.0e-9

        # State
        self.buf_pos = Ring(MAX_SAMPLES, 4)
        self.buf_vel = Ring(MAX_SAMPLES, 4)
        self.buf_acc = Ring(MAX_SAMPLES, 4)
        self.buf_pos_cmd = Ring(MAX_SAMPLES, 4)

        self.buf_rpy = Ring(MAX_SAMPLES, 4)
        self.buf_omega = Ring(MAX_SAMPLES, 4)
        self.buf_att_cmd = Ring(MAX_SAMPLES, 4)

        # Actuator measured / command
        self.buf_alpha = Ring(MAX_SAMPLES, 5)
        self.buf_beta = Ring(MAX_SAMPLES, 3)

        self.buf_alpha_cmd = Ring(MAX_SAMPLES, 5)
        self.buf_beta_cmd = Ring(MAX_SAMPLES, 3)
        self.buf_f_cmd = Ring(MAX_SAMPLES, 5)

        # Wrench: t + cmd[6] + alloc[6] + real[6]
        self.buf_wrench = Ring(MAX_SAMPLES, 19)

        # Wdot: t + desired[6] + J*qdot[6]
        self.buf_wdot = Ring(MAX_SAMPLES, 13)

        # qdot: t + [alpha4 beta2 thrust4]
        self.buf_qdot_primary = Ring(MAX_SAMPLES, 11)
        self.buf_qdot_null = Ring(MAX_SAMPLES, 11)
        self.buf_qdot_final = Ring(MAX_SAMPLES, 11)

        # t + primary_scale + nullspace_scale
        self.buf_scale = Ring(MAX_SAMPLES, 3)

        self.create_subscription(MultirotorState, "/multirotor_state", self._cb_state, 10)
        self.create_subscription(Cmd, "/cmd", self._cb_cmd, 10)
        self.create_subscription(Input, "/input", self._cb_input, 10)
        self.create_subscription(Debug, "/allocation_debug", self._cb_debug, 10)

        self.pub_force_impulse = self.create_publisher(Bool, "/test/impulse_trigger", 1)
        self.pub_moment_impulse = self.create_publisher(Bool, "/test/moment_impulse_trigger", 1)

    def _t(self):
        return self.get_clock().now().nanoseconds * 1.0e-9 - self.t0

    def trigger_force_impulse(self):
        msg = Bool()
        msg.data = True
        self.pub_force_impulse.publish(msg)

    def trigger_moment_impulse(self):
        msg = Bool()
        msg.data = True
        self.pub_moment_impulse.publish(msg)

    def _cb_state(self, m):
        t = self._t()

        pos = np.asarray(m.pos[:3], dtype=float)
        vel = np.asarray(m.vel[:3], dtype=float)
        acc = np.asarray(m.acc[:3], dtype=float)

        rpy = np.asarray(m.rpy[:3], dtype=float) * RAD2DEG
        omega = np.asarray(m.w_rpy[:3], dtype=float) * RAD2DEG

        alpha = np.asarray(m.alpha[:4], dtype=float) * RAD2DEG
        beta = np.asarray(m.beta[:2], dtype=float) * RAD2DEG

        with self.lock:
            self.buf_pos.push([t, *pos])
            self.buf_vel.push([t, *vel])
            self.buf_acc.push([t, *acc])

            self.buf_rpy.push([t, *rpy])
            self.buf_omega.push([t, *omega])

            self.buf_alpha.push([t, *alpha])
            self.buf_beta.push([t, *beta])

    def _cb_cmd(self, m):
        t = self._t()

        pos_cmd = np.asarray(m.pos_cmd[:3], dtype=float)
        att_cmd = att_cmd_to_rpy_deg(m.att_cmd)

        with self.lock:
            self.buf_pos_cmd.push([t, *pos_cmd])
            self.buf_att_cmd.push([t, *att_cmd])

    def _cb_input(self, m):
        t = self._t()

        f = np.asarray(m.f[:4], dtype=float)
        alpha = np.asarray(m.alpha[:4], dtype=float) * RAD2DEG
        beta = np.asarray(m.beta[:2], dtype=float) * RAD2DEG

        with self.lock:
            self.buf_f_cmd.push([t, *f])
            self.buf_alpha_cmd.push([t, *alpha])
            self.buf_beta_cmd.push([t, *beta])

    def _cb_debug(self, m):
        t = self._t()

        with self.lock:
            self.buf_wrench.push([t, *m.wrench_cmd, *m.wrench_alloc, *m.wrench_real])
            self.buf_wdot.push([t, *m.w_dot_des, *m.w_dot_q])

            self.buf_qdot_primary.push([t, *m.q_dot_primary])
            self.buf_qdot_null.push([t, *m.q_dot_nullspace])
            self.buf_qdot_final.push([t, *m.q_dot_final])

            self.buf_scale.push([t, m.primary_scale, m.nullspace_scale])


class Win(QtWidgets.QMainWindow):
    def __init__(self, node):
        super().__init__()

        self.node = node
        self.paused = False

        self.cv = {}

        self.plots_state = []
        self.plots_wrench = []
        self.plots_da = []
        self.plots_act = []

        self.pause_buttons = []
        self.pause_labels = []

        self.setWindowTitle("Sim_HummingBird Viewer")
        self.resize(1850, 1200)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        self.state_glw = pg.GraphicsLayoutWidget()
        self.wrench_glw = pg.GraphicsLayoutWidget()
        self.da_glw = pg.GraphicsLayoutWidget()
        self.act_glw = pg.GraphicsLayoutWidget()

        self.tabs.addTab(self._make_tab(self.state_glw), "State")
        self.tabs.addTab(self._make_tab(self.wrench_glw), "Wrench")
        self.tabs.addTab(self._make_tab(self.da_glw), "Differential Allocation")
        self.tabs.addTab(self._make_tab(self.act_glw), "Actuator")

        self._build_state()
        self._build_wrench()
        self._build_da()
        self._build_actuator()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(UPDATE_MS)

    def _make_tab(self, glw):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        controls = QtWidgets.QHBoxLayout()

        force_button = QtWidgets.QPushButton("X Force impulse")
        moment_button = QtWidgets.QPushButton("Pitch Moment impulse")
        pause_button = QtWidgets.QPushButton("Pause")
        pause_button.setCheckable(True)

        status = QtWidgets.QLabel("LIVE")

        force_button.clicked.connect(self._trigger_force_impulse)
        moment_button.clicked.connect(self._trigger_moment_impulse)
        pause_button.clicked.connect(self._toggle_pause)

        controls.addWidget(force_button)
        controls.addWidget(moment_button)
        controls.addSpacing(25)
        controls.addWidget(pause_button)
        controls.addWidget(status)
        controls.addStretch()

        self.pause_buttons.append(pause_button)
        self.pause_labels.append(status)

        layout.addLayout(controls)
        layout.addWidget(glw, 1)

        return tab

    def _trigger_force_impulse(self):
        self.node.trigger_force_impulse()

    def _trigger_moment_impulse(self):
        self.node.trigger_moment_impulse()

    def _toggle_pause(self):
        self.paused = not self.paused

        for button in self.pause_buttons:
            button.blockSignals(True)
            button.setChecked(self.paused)
            button.setText("Resume" if self.paused else "Pause")
            button.blockSignals(False)

        for label in self.pause_labels:
            label.setText("PAUSED - drag / zoom enabled" if self.paused else "LIVE")

    # =========================================================
    # State
    # =========================================================

    def _build_state(self):
        xyz = ["x", "y", "z"]
        rpy = ["roll", "pitch", "yaw"]

        for i in range(3):
            p = _mkplot(self.state_glw, 0, i, f"Position {xyz[i]}", f"{xyz[i]} [m]")
            self.cv[f"pos{i}"] = p.plot(pen=_pen(C4[i], 3), name=xyz[i])
            self.cv[f"pos_cmd{i}"] = _front(p.plot(pen=_dash_pen("k"), name=f"{xyz[i]}_cmd"))
            self.plots_state.append(p)

        for i in range(3):
            p = _mkplot(self.state_glw, 1, i, f"Velocity / Acceleration {xyz[i]}", "[m/s, m/s²]")
            self.cv[f"vel{i}"] = p.plot(pen=_pen(C4[i], 3), name=f"v{xyz[i]}")
            self.cv[f"acc{i}"] = p.plot(pen=_pen(C_O), name=f"a{xyz[i]}")
            self.plots_state.append(p)

        for i in range(3):
            p = _mkplot(self.state_glw, 2, i, f"Attitude {rpy[i]}", "[deg, deg/s]")
            self.cv[f"rpy{i}"] = p.plot(pen=_pen(C4[i], 3), name=rpy[i])
            self.cv[f"att_cmd{i}"] = _front(p.plot(pen=_dash_pen("k"), name=f"{rpy[i]}_cmd"))
            self.cv[f"omega{i}"] = p.plot(pen=_pen(C_O), name=f"ω_{rpy[i]}")
            self.plots_state.append(p)

        for p in self.plots_state[1:]:
            p.setXLink(self.plots_state[0])

        for p in self.plots_state[:-3]:
            p.hideAxis("bottom")

        for p in self.plots_state[-3:]:
            p.setLabel("bottom", "time [s]")

    # =========================================================
    # Wrench
    # =========================================================

    def _build_wrench(self):
        force_names = ["Fx", "Fy", "Fz"]
        moment_names = ["Mx", "My", "Mz"]

        for i in range(3):
            p = _mkplot(self.wrench_glw, 0, i, force_names[i], f"{force_names[i]} [N]")
            self.cv[f"F_cmd{i}"] = p.plot(pen=_dash_pen("k"), name="cmd")
            self.cv[f"F_alloc{i}"] = p.plot(pen=_pen(C_B), name="alloc")
            self.cv[f"F_real{i}"] = p.plot(pen=_pen(C_R), name="real")
            self.plots_wrench.append(p)

        for i in range(3):
            p = _mkplot(self.wrench_glw, 1, i, moment_names[i], f"{moment_names[i]} [N·m]")
            self.cv[f"M_cmd{i}"] = p.plot(pen=_dash_pen("k"), name="cmd")
            self.cv[f"M_alloc{i}"] = p.plot(pen=_pen(C_B), name="alloc")
            self.cv[f"M_real{i}"] = p.plot(pen=_pen(C_R), name="real")
            self.plots_wrench.append(p)

        for i in range(3):
            p = _mkplot(self.wrench_glw, 2, i, f"{force_names[i]} error", f"Δ{force_names[i]} [N]")
            self.cv[f"Fe_ca{i}"] = p.plot(pen=_pen(C_B), name="cmd-alloc")
            self.cv[f"Fe_cr{i}"] = p.plot(pen=_pen(C_R), name="cmd-real")
            self.cv[f"Fe_ar{i}"] = p.plot(pen=_pen(C_G), name="alloc-real")
            self.plots_wrench.append(p)

        for i in range(3):
            p = _mkplot(self.wrench_glw, 3, i, f"{moment_names[i]} error", f"Δ{moment_names[i]} [N·m]")
            self.cv[f"Me_ca{i}"] = p.plot(pen=_pen(C_B), name="cmd-alloc")
            self.cv[f"Me_cr{i}"] = p.plot(pen=_pen(C_R), name="cmd-real")
            self.cv[f"Me_ar{i}"] = p.plot(pen=_pen(C_G), name="alloc-real")
            self.plots_wrench.append(p)

        for p in self.plots_wrench[1:]:
            p.setXLink(self.plots_wrench[0])

        for p in self.plots_wrench[:-3]:
            p.hideAxis("bottom")

        for p in self.plots_wrench[-3:]:
            p.setLabel("bottom", "time [s]")

    # =========================================================
    # Differential Allocation
    # =========================================================

    def _build_da(self):
        moment_names = ["Mx", "My", "Mz"]
        force_names = ["Fx", "Fy", "Fz"]

        # Wdot moment
        for i in range(3):
            p = _mkplot(self.da_glw, 0, i, f"{moment_names[i]} dot", f"{moment_names[i]} dot [N·m/s]")
            self.cv[f"Mdot_des{i}"] = p.plot(pen=_dash_pen("k"), name="W_dot_des")
            self.cv[f"Mdot_q{i}"] = p.plot(pen=_pen(C_R), name="J q_dot")
            self.plots_da.append(p)

        # Wdot force
        for i in range(3):
            p = _mkplot(self.da_glw, 1, i, f"{force_names[i]} dot", f"{force_names[i]} dot [N/s]")
            self.cv[f"Fdot_des{i}"] = p.plot(pen=_dash_pen("k"), name="W_dot_des")
            self.cv[f"Fdot_q{i}"] = p.plot(pen=_pen(C_R), name="J q_dot")
            self.plots_da.append(p)

        # Primary
        p = _mkplot(self.da_glw, 2, 0, "Primary alpha rate", "alpha dot [deg/s]")
        for i in range(4):
            self.cv[f"qp_a{i}"] = p.plot(pen=_pen(C4[i]), name=f"α{i+1}")
        self.plots_da.append(p)

        p = _mkplot(self.da_glw, 2, 1, "Primary beta rate", "beta dot [deg/s]")
        for i in range(2):
            self.cv[f"qp_b{i}"] = p.plot(pen=_pen(C4[i]), name=f"β{i+1}")
        self.plots_da.append(p)

        p = _mkplot(self.da_glw, 2, 2, "Primary thrust rate", "f dot [N/s]")
        for i in range(4):
            self.cv[f"qp_f{i}"] = p.plot(pen=_pen(C4[i]), name=f"f{i+1}")
        self.plots_da.append(p)

        # Nullspace
        p = _mkplot(self.da_glw, 3, 0, "Nullspace alpha rate", "alpha dot [deg/s]")
        for i in range(4):
            self.cv[f"qn_a{i}"] = p.plot(pen=_pen(C4[i]), name=f"α{i+1}")
        self.plots_da.append(p)

        p = _mkplot(self.da_glw, 3, 1, "Nullspace beta rate", "beta dot [deg/s]")
        for i in range(2):
            self.cv[f"qn_b{i}"] = p.plot(pen=_pen(C4[i]), name=f"β{i+1}")
        self.plots_da.append(p)

        p = _mkplot(self.da_glw, 3, 2, "Nullspace thrust rate", "f dot [N/s]")
        for i in range(4):
            self.cv[f"qn_f{i}"] = p.plot(pen=_pen(C4[i]), name=f"f{i+1}")
        self.plots_da.append(p)

        # Final
        p = _mkplot(self.da_glw, 4, 0, "Final alpha rate", "alpha dot [deg/s]")
        for i in range(4):
            self.cv[f"qf_a{i}"] = p.plot(pen=_pen(C4[i]), name=f"α{i+1}")
        self.plots_da.append(p)

        p = _mkplot(self.da_glw, 4, 1, "Final beta rate", "beta dot [deg/s]")
        for i in range(2):
            self.cv[f"qf_b{i}"] = p.plot(pen=_pen(C4[i]), name=f"β{i+1}")
        self.plots_da.append(p)

        p = _mkplot(self.da_glw, 4, 2, "Final thrust rate", "f dot [N/s]")
        for i in range(4):
            self.cv[f"qf_f{i}"] = p.plot(pen=_pen(C4[i]), name=f"f{i+1}")
        self.plots_da.append(p)

        # Rate scaling
        p = _mkplot(self.da_glw, 5, 0, "Rate scaling", "scale / reduction [-]", colspan=3)
        self.cv["primary_scale"] = p.plot(pen=_pen(C_R), name="primary scale")
        self.cv["null_scale"] = p.plot(pen=_pen(C_B), name="null scale")
        self.cv["primary_cut"] = p.plot(pen=_dash_pen(C_R), name="1-primary")
        self.cv["null_cut"] = p.plot(pen=_dash_pen(C_B), name="1-null")
        self.plots_da.append(p)

        for p in self.plots_da[1:]:
            p.setXLink(self.plots_da[0])

        for p in self.plots_da[:-1]:
            p.hideAxis("bottom")

        self.plots_da[-1].setLabel("bottom", "time [s]")

    # =========================================================
    # Actuator
    # =========================================================

    def _build_actuator(self):
        # Alpha measured + command
        p = _mkplot(self.act_glw, 0, 0, "Alpha measured / command", "alpha [deg]")

        for i in range(4):
            self.cv[f"act_alpha{i}"] = p.plot(pen=_pen(C4[i]), name=f"α{i+1}")
            self.cv[f"act_alpha_cmd{i}"] = _front(
                p.plot(pen=_dash_pen(C4[i]), name=f"α{i+1}_cmd")
            )

        self.plots_act.append(p)

        # Beta measured + command
        p = _mkplot(self.act_glw, 1, 0, "Beta measured / command", "beta [deg]")

        for i in range(2):
            self.cv[f"act_beta{i}"] = p.plot(pen=_pen(C4[i]), name=f"β{i+1}")
            self.cv[f"act_beta_cmd{i}"] = _front(
                p.plot(pen=_dash_pen(C4[i]), name=f"β{i+1}_cmd")
            )

        self.plots_act.append(p)

        # BLDC thrust
        p = _mkplot(self.act_glw, 2, 0, "BLDC thrust command", "f [N]")

        for i in range(4):
            self.cv[f"act_f{i}"] = p.plot(pen=_pen(C4[i]), name=f"f{i+1}")

        self.plots_act.append(p)

        # XY tilt direction
        p_xy = _mkplot(
            self.act_glw,
            0,
            1,
            "Rotor tilt direction in body XY plane",
            "body x [m]",
            rowspan=3
        )

        p_xy.setLabel("bottom", "body y [m]")
        p_xy.setAspectLocked(True)
        p_xy.setXRange(-0.58, 0.58, padding=0)
        p_xy.setYRange(-0.58, 0.58, padding=0)

        p_xy.addLine(x=0.0, pen=pg.mkPen("#b0b0b0", style=QtCore.Qt.DashLine))
        p_xy.addLine(y=0.0, pen=pg.mkPen("#b0b0b0", style=QtCore.Qt.DashLine))

        for i, (origin, color) in enumerate(zip(ROTOR_PLOT_YX, C4)):
            p_xy.plot([0.0, origin[0]], [0.0, origin[1]], pen=pg.mkPen("#888888", width=2))
            p_xy.plot([origin[0]], [origin[1]], pen=None, symbol="o",
                      symbolSize=13, symbolBrush=color, symbolPen=pg.mkPen("k"))

            rotor_label = pg.TextItem(f"R{i+1}", color=color, anchor=(0.5, 1.4))
            rotor_label.setPos(origin[0], origin[1])
            p_xy.addItem(rotor_label)

        # +y arrow
        p_xy.plot([0.0, 0.10], [0.0, 0.0], pen=pg.mkPen("#555555", width=3))
        y_label = pg.TextItem("+y", color="#333333", anchor=(0.0, 0.5))
        y_label.setPos(0.11, 0.0)
        p_xy.addItem(y_label)

        # +x arrow
        p_xy.plot([0.0, 0.0], [0.0, 0.10], pen=pg.mkPen("#555555", width=3))
        x_label = pg.TextItem("+x", color="#333333", anchor=(0.5, 1.0))
        x_label.setPos(0.0, 0.11)
        p_xy.addItem(x_label)

        self.xy_actual = []
        self.xy_cmd = []

        for i, color in enumerate(C4):
            cmd_curve = p_xy.plot(
                pen=pg.mkPen(color=color, width=2, style=QtCore.Qt.DashLine),
                name="command" if i == 0 else None,
                connect="finite"
            )

            actual_curve = p_xy.plot(
                pen=pg.mkPen(color=color, width=3),
                name="measured" if i == 0 else None,
                connect="finite"
            )

            cmd_curve.setZValue(3)
            actual_curve.setZValue(4)

            self.xy_cmd.append(cmd_curve)
            self.xy_actual.append(actual_curve)

        for p in self.plots_act[1:]:
            p.setXLink(self.plots_act[0])

        for p in self.plots_act[:-1]:
            p.hideAxis("bottom")

        self.plots_act[-1].setLabel("bottom", "time [s]")

    def _update_xy_vectors(self, alpha_data, beta_data, curves):
        if not alpha_data.shape[0] or not beta_data.shape[0]:
            return

        vectors = tilt_xy_components(alpha_data[-1, 1:5], beta_data[-1, 1:3])

        for i, curve in enumerate(curves):
            origin_yx = ROTOR_PLOT_YX[i]
            vector_yx = vectors[i, [1, 0]]

            x, y = vector_polyline(origin_yx, vector_yx)
            curve.setData(x, y, connect="finite")

    # =========================================================
    # Update
    # =========================================================

    def _update(self):
        if self.paused:
            return

        nd = self.node

        with nd.lock:
            dp = nd.buf_pos.get()
            dv = nd.buf_vel.get()
            da = nd.buf_acc.get()
            dpc = nd.buf_pos_cmd.get()

            dr = nd.buf_rpy.get()
            domega = nd.buf_omega.get()
            dac = nd.buf_att_cmd.get()

            dalpha = nd.buf_alpha.get()
            dbeta = nd.buf_beta.get()

            dalpha_c = nd.buf_alpha_cmd.get()
            dbeta_c = nd.buf_beta_cmd.get()
            df = nd.buf_f_cmd.get()

            dwr = nd.buf_wrench.get()
            dwd = nd.buf_wdot.get()

            dqp = nd.buf_qdot_primary.get()
            dqn = nd.buf_qdot_null.get()
            dqf = nd.buf_qdot_final.get()

            ds = nd.buf_scale.get()

        data_all = [
            dp, dv, da, dpc,
            dr, domega, dac,
            dalpha, dbeta, dalpha_c, dbeta_c, df,
            dwr, dwd, dqp, dqn, dqf, ds
        ]

        tn = 0.0

        for data in data_all:
            if data.shape[0]:
                tn = max(tn, data[-1, 0])

        if tn == 0.0:
            return

        tl = tn - WINDOW_SEC

        def tr(data):
            if not data.shape[0]:
                return data
            return data[data[:, 0] >= tl]

        dp = tr(dp)
        dv = tr(dv)
        da = tr(da)
        dpc = tr(dpc)

        dr = tr(dr)
        domega = tr(domega)
        dac = tr(dac)

        dalpha = tr(dalpha)
        dbeta = tr(dbeta)
        dalpha_c = tr(dalpha_c)
        dbeta_c = tr(dbeta_c)
        df = tr(df)

        dwr = tr(dwr)
        dwd = tr(dwd)

        dqp = tr(dqp)
        dqn = tr(dqn)
        dqf = tr(dqf)

        ds = tr(ds)

        cv = self.cv

        # -----------------------------------------------------
        # State
        # -----------------------------------------------------

        if dp.shape[0]:
            for i in range(3):
                cv[f"pos{i}"].setData(dp[:, 0], dp[:, 1+i])

        if dpc.shape[0]:
            for i in range(3):
                cv[f"pos_cmd{i}"].setData(dpc[:, 0], dpc[:, 1+i])

        if dv.shape[0]:
            for i in range(3):
                cv[f"vel{i}"].setData(dv[:, 0], dv[:, 1+i])

        if da.shape[0]:
            for i in range(3):
                cv[f"acc{i}"].setData(da[:, 0], da[:, 1+i])

        if dr.shape[0]:
            for i in range(3):
                cv[f"rpy{i}"].setData(dr[:, 0], dr[:, 1+i])

        if dac.shape[0]:
            for i in range(3):
                cv[f"att_cmd{i}"].setData(dac[:, 0], dac[:, 1+i])

        if domega.shape[0]:
            for i in range(3):
                cv[f"omega{i}"].setData(domega[:, 0], domega[:, 1+i])

        # -----------------------------------------------------
        # Wrench
        # order = [Mx My Mz Fx Fy Fz]
        # -----------------------------------------------------

        if dwr.shape[0]:
            t = dwr[:, 0]

            Wcmd = dwr[:, 1:7]
            Walloc = dwr[:, 7:13]
            Wreal = dwr[:, 13:19]

            for i in range(3):
                mi = i
                fi = 3 + i

                cv[f"M_cmd{i}"].setData(t, Wcmd[:, mi])
                cv[f"M_alloc{i}"].setData(t, Walloc[:, mi])
                cv[f"M_real{i}"].setData(t, Wreal[:, mi])

                cv[f"F_cmd{i}"].setData(t, Wcmd[:, fi])
                cv[f"F_alloc{i}"].setData(t, Walloc[:, fi])
                cv[f"F_real{i}"].setData(t, Wreal[:, fi])

                cv[f"Me_ca{i}"].setData(t, Wcmd[:, mi] - Walloc[:, mi])
                cv[f"Me_cr{i}"].setData(t, Wcmd[:, mi] - Wreal[:, mi])
                cv[f"Me_ar{i}"].setData(t, Walloc[:, mi] - Wreal[:, mi])

                cv[f"Fe_ca{i}"].setData(t, Wcmd[:, fi] - Walloc[:, fi])
                cv[f"Fe_cr{i}"].setData(t, Wcmd[:, fi] - Wreal[:, fi])
                cv[f"Fe_ar{i}"].setData(t, Walloc[:, fi] - Wreal[:, fi])

        # -----------------------------------------------------
        # Wdot
        # -----------------------------------------------------

        if dwd.shape[0]:
            t = dwd[:, 0]

            Wdot_des = dwd[:, 1:7]
            Wdot_q = dwd[:, 7:13]

            for i in range(3):
                cv[f"Mdot_des{i}"].setData(t, Wdot_des[:, i])
                cv[f"Mdot_q{i}"].setData(t, Wdot_q[:, i])

                cv[f"Fdot_des{i}"].setData(t, Wdot_des[:, 3+i])
                cv[f"Fdot_q{i}"].setData(t, Wdot_q[:, 3+i])

        # -----------------------------------------------------
        # qdot primary
        # -----------------------------------------------------

        if dqp.shape[0]:
            t = dqp[:, 0]

            alpha = dqp[:, 1:5] * RAD2DEG
            beta = dqp[:, 5:7] * RAD2DEG
            thrust = dqp[:, 7:11]

            for i in range(4):
                cv[f"qp_a{i}"].setData(t, alpha[:, i])
                cv[f"qp_f{i}"].setData(t, thrust[:, i])

            for i in range(2):
                cv[f"qp_b{i}"].setData(t, beta[:, i])

        # -----------------------------------------------------
        # qdot nullspace
        # -----------------------------------------------------

        if dqn.shape[0]:
            t = dqn[:, 0]

            alpha = dqn[:, 1:5] * RAD2DEG
            beta = dqn[:, 5:7] * RAD2DEG
            thrust = dqn[:, 7:11]

            for i in range(4):
                cv[f"qn_a{i}"].setData(t, alpha[:, i])
                cv[f"qn_f{i}"].setData(t, thrust[:, i])

            for i in range(2):
                cv[f"qn_b{i}"].setData(t, beta[:, i])

        # -----------------------------------------------------
        # qdot final
        # -----------------------------------------------------

        if dqf.shape[0]:
            t = dqf[:, 0]

            alpha = dqf[:, 1:5] * RAD2DEG
            beta = dqf[:, 5:7] * RAD2DEG
            thrust = dqf[:, 7:11]

            for i in range(4):
                cv[f"qf_a{i}"].setData(t, alpha[:, i])
                cv[f"qf_f{i}"].setData(t, thrust[:, i])

            for i in range(2):
                cv[f"qf_b{i}"].setData(t, beta[:, i])

        # -----------------------------------------------------
        # Scale
        # -----------------------------------------------------

        if ds.shape[0]:
            t = ds[:, 0]

            primary = ds[:, 1]
            nullspace = ds[:, 2]

            cv["primary_scale"].setData(t, primary)
            cv["null_scale"].setData(t, nullspace)

            cv["primary_cut"].setData(t, 1.0 - primary)
            cv["null_cut"].setData(t, 1.0 - nullspace)

        # -----------------------------------------------------
        # Actuator
        # -----------------------------------------------------

        if dalpha.shape[0]:
            for i in range(4):
                cv[f"act_alpha{i}"].setData(dalpha[:, 0], dalpha[:, 1+i])

        if dalpha_c.shape[0]:
            for i in range(4):
                cv[f"act_alpha_cmd{i}"].setData(dalpha_c[:, 0], dalpha_c[:, 1+i])

        if dbeta.shape[0]:
            for i in range(2):
                cv[f"act_beta{i}"].setData(dbeta[:, 0], dbeta[:, 1+i])

        if dbeta_c.shape[0]:
            for i in range(2):
                cv[f"act_beta_cmd{i}"].setData(dbeta_c[:, 0], dbeta_c[:, 1+i])

        if df.shape[0]:
            for i in range(4):
                cv[f"act_f{i}"].setData(df[:, 0], df[:, 1+i])

        self._update_xy_vectors(dalpha, dbeta, self.xy_actual)
        self._update_xy_vectors(dalpha_c, dbeta_c, self.xy_cmd)

        # -----------------------------------------------------
        # Auto-follow
        # -----------------------------------------------------

        if self.plots_state:
            self.plots_state[0].setXRange(tl, tn, padding=0)

        if self.plots_wrench:
            self.plots_wrench[0].setXRange(tl, tn, padding=0)

        if self.plots_da:
            self.plots_da[0].setXRange(tl, tn, padding=0)

        if self.plots_act:
            self.plots_act[0].setXRange(tl, tn, padding=0)


def _spin(node):
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass


def main():
    rclpy.init()

    node = VNode()
    threading.Thread(target=_spin, args=(node,), daemon=True).start()

    app = QtWidgets.QApplication(sys.argv)

    win = Win(node)
    win.show()

    code = app.exec_()

    node.destroy_node()
    rclpy.shutdown()

    sys.exit(code)


if __name__ == "__main__":
    main()