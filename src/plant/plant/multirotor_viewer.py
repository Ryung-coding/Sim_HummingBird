#!/usr/bin/env python3
import sys
import math
import threading
import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool
from multirotor_interfaces.msg import MultirotorState, Cmd, Wrench, Input, HexaInput

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

C3 = [C_R, C_G, C_B]
C4 = [C_R, C_G, C_B, C_O]
C6 = [C_R, C_G, C_B, C_O, "#911eb4", "#42d4f4"]

ROTOR_BODY_XY = np.array([
    [0.175, 0.175],
    [-0.175, 0.175],
    [-0.175, -0.175],
    [0.175, -0.175]
], dtype=float)
ROTOR_PLOT_YX = ROTOR_BODY_XY[:, [1, 0]]
BETA_INDEX = np.array([0, 1, 1, 0], dtype=int)
TILT_VECTOR_SCALE = 0.35
TILT_ARROW_HEAD = 0.035

# Same controller z-down arm order used by allocation_hexa_a6_ada().
HEXA_ARM_YAW = np.array([
    0.5 * math.pi, -0.5 * math.pi, -math.pi / 6.0,
    5.0 * math.pi / 6.0, math.pi / 6.0, -5.0 * math.pi / 6.0
], dtype=float)
HEXA_ROTOR_BODY_XY = 0.30 * np.column_stack((
    np.cos(HEXA_ARM_YAW), np.sin(HEXA_ARM_YAW)
))
HEXA_ROTOR_PLOT_YX = HEXA_ROTOR_BODY_XY[:, [1, 0]]


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


def wrap_deg(x):
    return (x + 180.0) % 360.0 - 180.0


def att_cmd_to_rpy_deg(att_cmd):
    if not USE_SO3_HEADING_CMD:
        return np.array(att_cmd[:3], dtype=float) * RAD2DEG

    h = np.array(att_cmd[:3], dtype=float)
    n = np.linalg.norm(h)
    if n < 1.0e-9:
        return np.zeros(3, dtype=float)

    h = h / n
    yaw = math.atan2(h[1], h[0])
    pitch = math.atan2(-h[2], math.sqrt(h[0] * h[0] + h[1] * h[1]))
    roll = 0.0

    return np.array([roll, pitch, yaw], dtype=float) * RAD2DEG


def tilt_xy_components(alpha_deg, beta_deg):
    alpha = np.asarray(alpha_deg, dtype=float) / RAD2DEG
    beta = np.asarray(beta_deg, dtype=float)[BETA_INDEX] / RAD2DEG

    # Same body-frame thrust direction used by allocation_a4b2().
    return np.column_stack((
        -np.sin(beta) * np.cos(alpha),
        np.sin(alpha)
    ))


def hexa_tilt_xy_components(alpha_deg):
    alpha = np.asarray(alpha_deg, dtype=float) / RAD2DEG
    tangent = np.column_stack((
        -np.sin(HEXA_ARM_YAW), np.cos(HEXA_ARM_YAW)
    ))

    # Horizontal component of e_i = tangent_i sin(alpha_i) - z cos(alpha_i).
    return np.sin(alpha)[:, None] * tangent


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


class VNode(Node):
    def __init__(self):
        super().__init__("multirotor_viewer")
        self.vehicle = self.declare_parameter("vehicle", "hummingbird").value
        if self.vehicle not in ("hummingbird", "hexa"):
            raise ValueError(f"vehicle must be hummingbird or hexa, got {self.vehicle}")
        self.is_hexa = self.vehicle == "hexa"

        self.lock = threading.Lock()
        self.t0 = self.get_clock().now().nanoseconds * 1e-9

        self.last_pos_cmd = np.zeros(3, dtype=float)
        self.last_att_cmd_deg = np.zeros(3, dtype=float)

        self.max_pos_err_abs = np.zeros(3, dtype=float)
        self.max_att_err_abs = np.zeros(3, dtype=float)
        self.max_pos_err_norm = 0.0
        self.max_att_err_norm = 0.0
        self.max_pos_err_time = 0.0
        self.max_att_err_time = 0.0

        self.buf_pos = Ring(MAX_SAMPLES, 4)
        self.buf_vel = Ring(MAX_SAMPLES, 4)
        self.buf_cmd = Ring(MAX_SAMPLES, 4)
        self.buf_pos_err = Ring(MAX_SAMPLES, 4)
        self.buf_disturbance = Ring(MAX_SAMPLES, 4)

        self.buf_rpy = Ring(MAX_SAMPLES, 4)
        self.buf_att = Ring(MAX_SAMPLES, 4)
        self.buf_att_err = Ring(MAX_SAMPLES, 4)

        self.buf_wr = Ring(MAX_SAMPLES, 7)
        self.buf_d = Ring(MAX_SAMPLES, 7)

        self.buf_beta = Ring(MAX_SAMPLES, 3)
        self.buf_alpha = Ring(MAX_SAMPLES, 5)
        self.buf_beta_cmd = Ring(MAX_SAMPLES, 3)
        self.buf_alpha_cmd = Ring(MAX_SAMPLES, 5)
        self.buf_thr = Ring(MAX_SAMPLES, 5)

        self.buf_pair_actual = Ring(MAX_SAMPLES, 5)
        self.buf_pair_cmd = Ring(MAX_SAMPLES, 5)
        self.buf_hexa_alpha = Ring(MAX_SAMPLES, 7)
        self.buf_hexa_alpha_cmd = Ring(MAX_SAMPLES, 7)
        self.buf_hexa_thr = Ring(MAX_SAMPLES, 7)

        self.create_subscription(MultirotorState, "/multirotor_state", self._cb_state, 10)
        self.create_subscription(Cmd, "/cmd", self._cb_cmd, 10)
        self.create_subscription(Wrench, "/wrench", self._cb_wrench, 10)
        self.pub_impulse_trigger = self.create_publisher(Bool, "/test/impulse_trigger", 1)
        if self.is_hexa:
            self.create_subscription(HexaInput, "/hexa_input", self._cb_hexa_input, 10)
        else:
            self.create_subscription(Input, "/input", self._cb_input, 10)

    def _t(self):
        return self.get_clock().now().nanoseconds * 1e-9 - self.t0

    def trigger_impulse(self):
        msg = Bool()
        msg.data = True
        self.pub_impulse_trigger.publish(msg)

    def _cb_state(self, m):
        t = self._t()

        pos = np.array([m.pos[0], m.pos[1], m.pos[2]], dtype=float)
        vel = np.array([m.vel[0], m.vel[1], m.vel[2]], dtype=float)
        disturbance_force = np.array([
            m.disturbance_force[0],
            m.disturbance_force[1],
            m.disturbance_force[2]
        ], dtype=float)
        rpy_deg = np.array([m.rpy[0], m.rpy[1], m.rpy[2]], dtype=float) * RAD2DEG
        if self.is_hexa:
            hexa_alpha_deg = np.array(m.hexa_alpha[:6], dtype=float) * RAD2DEG
        else:
            beta_deg = np.array([m.beta[0], m.beta[1]], dtype=float) * RAD2DEG
            alpha_deg = np.array([m.alpha[0], m.alpha[1], m.alpha[2], m.alpha[3]], dtype=float) * RAD2DEG

        pos_err = self.last_pos_cmd - pos

        att_err = np.array([
            wrap_deg(self.last_att_cmd_deg[0] - rpy_deg[0]),
            wrap_deg(self.last_att_cmd_deg[1] - rpy_deg[1]),
            wrap_deg(self.last_att_cmd_deg[2] - rpy_deg[2])
        ], dtype=float)

        pos_err_abs = np.abs(pos_err)
        att_err_abs = np.abs(att_err)
        pos_err_norm = np.linalg.norm(pos_err)
        att_err_norm = np.linalg.norm(att_err)

        with self.lock:
            self.max_pos_err_abs = np.maximum(self.max_pos_err_abs, pos_err_abs)
            self.max_att_err_abs = np.maximum(self.max_att_err_abs, att_err_abs)

            if pos_err_norm > self.max_pos_err_norm:
                self.max_pos_err_norm = pos_err_norm
                self.max_pos_err_time = t

            if att_err_norm > self.max_att_err_norm:
                self.max_att_err_norm = att_err_norm
                self.max_att_err_time = t

            self.buf_pos.push([t, pos[0], pos[1], pos[2]])
            self.buf_vel.push([t, vel[0], vel[1], vel[2]])
            self.buf_disturbance.push([t, disturbance_force[0], disturbance_force[1], disturbance_force[2]])
            self.buf_rpy.push([t, rpy_deg[0], rpy_deg[1], rpy_deg[2]])
            if self.is_hexa:
                self.buf_hexa_alpha.push([t, *hexa_alpha_deg])
            else:
                self.buf_beta.push([t, beta_deg[0], beta_deg[1]])
                self.buf_alpha.push([t, alpha_deg[0], alpha_deg[1], alpha_deg[2], alpha_deg[3]])
            self.buf_pos_err.push([t, pos_err[0], pos_err[1], pos_err[2]])
            self.buf_att_err.push([t, att_err[0], att_err[1], att_err[2]])

    def _cb_cmd(self, m):
        t = self._t()

        pos_cmd = np.array([m.pos_cmd[0], m.pos_cmd[1], m.pos_cmd[2]], dtype=float)
        att_cmd_deg = att_cmd_to_rpy_deg(m.att_cmd)

        self.last_pos_cmd = pos_cmd
        self.last_att_cmd_deg = att_cmd_deg

        with self.lock:
            self.buf_cmd.push([t, pos_cmd[0], pos_cmd[1], pos_cmd[2]])
            self.buf_att.push([t, att_cmd_deg[0], att_cmd_deg[1], att_cmd_deg[2]])

    def _cb_wrench(self, m):
        t = self._t()

        with self.lock:
            self.buf_wr.push([t, m.force[0], m.force[1], m.force[2], m.moment[0], m.moment[1], m.moment[2]])
            self.buf_d.push([
                t,
                m.d[0],
                m.d[1],
                m.d[2],
                m.d[3] * RAD2DEG,
                m.d[4] * RAD2DEG,
                m.d[5] * RAD2DEG
            ])

    def _cb_input(self, m):
        t = self._t()

        f = np.array([m.f[0], m.f[1], m.f[2], m.f[3]], dtype=float)
        beta_cmd_deg = np.array([m.beta[0], m.beta[1]], dtype=float) * RAD2DEG
        alpha_cmd_deg = np.array([m.alpha[0], m.alpha[1], m.alpha[2], m.alpha[3]], dtype=float) * RAD2DEG

        with self.lock:
            self.buf_thr.push([t, f[0], f[1], f[2], f[3]])
            self.buf_beta_cmd.push([t, beta_cmd_deg[0], beta_cmd_deg[1]])
            self.buf_alpha_cmd.push([t, alpha_cmd_deg[0], alpha_cmd_deg[1], alpha_cmd_deg[2], alpha_cmd_deg[3]])

    def _cb_hexa_input(self, m):
        t = self._t()
        f = np.array(m.f[:6], dtype=float)
        alpha_cmd_deg = np.array(m.alpha[:6], dtype=float) * RAD2DEG

        with self.lock:
            self.buf_hexa_thr.push([t, *f])
            self.buf_hexa_alpha_cmd.push([t, *alpha_cmd_deg])


def _pen(color, w=2):
    return pg.mkPen(color=color, width=w, style=QtCore.Qt.SolidLine)


def _cmd_pen(w=2):
    return pg.mkPen(color="k", width=w, style=QtCore.Qt.DashLine)


def _mkplot(glw, r, c, title, ylabel, **layout):
    p = glw.addPlot(row=r, col=c, title=title, **layout)
    p.showGrid(x=True, y=True, alpha=0.3)
    p.setLabel("left", ylabel)
    p.getAxis("left").enableAutoSIPrefix(False)
    p.getAxis("bottom").enableAutoSIPrefix(False)

    for a in ("bottom", "left"):
        p.getAxis(a).setPen(pg.mkPen("k"))
        p.getAxis(a).setTextPen(pg.mkPen("k"))

    p.addLegend(offset=(-10, 5))
    return p


def _bring_front(curve):
    curve.setZValue(5)
    return curve


class Win(QtWidgets.QMainWindow):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.is_hexa = node.is_hexa

        self.setWindowTitle(f"Sim_HummingBird Viewer ({node.vehicle})")
        self.resize(1800, 1050)

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        self._cv = {}
        self._plots_state = []
        self._plots_nullspace = []
        self._plots_disturbance = []
        self._plots_act = []

        self.state_glw = pg.GraphicsLayoutWidget()
        self.nullspace_glw = pg.GraphicsLayoutWidget()
        self.disturbance_glw = pg.GraphicsLayoutWidget()
        self.act_glw = pg.GraphicsLayoutWidget()
        self.disturbance_tab = QtWidgets.QWidget()
        disturbance_layout = QtWidgets.QVBoxLayout(self.disturbance_tab)
        disturbance_controls = QtWidgets.QHBoxLayout()
        self.impulse_button = QtWidgets.QPushButton("Trigger X impulse")
        self.impulse_button.clicked.connect(self._trigger_impulse)
        self.impulse_status = QtWidgets.QLabel("manual trigger ready")
        disturbance_controls.addWidget(self.impulse_button)
        disturbance_controls.addWidget(self.impulse_status)
        disturbance_controls.addStretch()
        self.performance_label = QtWidgets.QLabel(
            f"<b>Rolling {WINDOW_SEC:.0f} s performance</b>: waiting for state data..."
        )
        self.performance_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        disturbance_layout.addLayout(disturbance_controls)
        disturbance_layout.addWidget(self.performance_label)
        disturbance_layout.addWidget(self.disturbance_glw, 1)

        tabs.addTab(self.state_glw, "State")
        tabs.addTab(self.nullspace_glw, "Tilt Direction" if self.is_hexa else "Nullspace XY")
        tabs.addTab(self.disturbance_tab, "Disturbance / Performance")
        tabs.addTab(self.act_glw, "Actuator")

        self._build_state_tab()
        if self.is_hexa:
            self._build_hexa_nullspace_tab()
        else:
            self._build_nullspace_tab()
        self._build_disturbance_tab()
        if self.is_hexa:
            self._build_hexa_actuator_tab()
        else:
            self._build_actuator_tab()

        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._upd)
        self._timer.start(UPDATE_MS)

    def _trigger_impulse(self):
        self.node.trigger_impulse()
        self.impulse_status.setText("manual trigger published")

    def _build_state_tab(self):
        pos_lbl = ["x", "y", "z"]
        rpy_lbl = ["roll", "pitch", "yaw"]
        frc_lbl = ["Fx", "Fy", "Fz"]
        trq_lbl = ["Mx", "My", "Mz"]

        for c in range(3):
            p = _mkplot(self.state_glw, 0, c, f"{pos_lbl[c]} / {pos_lbl[c]}_cmd", f"{pos_lbl[c]} [m]")
            self._cv[f"pos{c}"] = p.plot(pen=_pen(C3[c]), name=pos_lbl[c])
            self._cv[f"cmd{c}"] = _bring_front(p.plot(pen=_cmd_pen(), name=f"{pos_lbl[c]}_cmd"))
            self._plots_state.append(p)

        for c in range(3):
            p = _mkplot(self.state_glw, 1, c, f"{pos_lbl[c]}_err = {pos_lbl[c]}_cmd - {pos_lbl[c]}", f"{pos_lbl[c]}_err [m]")
            self._cv[f"perr{c}"] = p.plot(pen=_pen(C3[c]), name=f"{pos_lbl[c]}_err")
            self._plots_state.append(p)

        for c in range(3):
            p = _mkplot(self.state_glw, 2, c, frc_lbl[c], f"{frc_lbl[c]} [N]")
            self._cv[f"F{c}"] = p.plot(pen=_pen(C3[c]), name=frc_lbl[c])
            self._plots_state.append(p)

        for c in range(3):
            p = _mkplot(self.state_glw, 3, c, f"{rpy_lbl[c]} / {rpy_lbl[c]}_cmd", f"{rpy_lbl[c]} [deg]")
            self._cv[f"rpy{c}"] = p.plot(pen=_pen(C3[c]), name=rpy_lbl[c])
            self._cv[f"acmd{c}"] = _bring_front(p.plot(pen=_cmd_pen(), name=f"{rpy_lbl[c]}_cmd"))
            self._plots_state.append(p)

        for c in range(3):
            p = _mkplot(self.state_glw, 4, c, f"{rpy_lbl[c]}_err = {rpy_lbl[c]}_cmd - {rpy_lbl[c]}", f"{rpy_lbl[c]}_err [deg]")
            self._cv[f"aerr{c}"] = p.plot(pen=_pen(C3[c]), name=f"{rpy_lbl[c]}_err")
            self._plots_state.append(p)

        for c in range(3):
            p = _mkplot(self.state_glw, 5, c, trq_lbl[c], f"{trq_lbl[c]} [N·m]")
            self._cv[f"M{c}"] = p.plot(pen=_pen(C3[c]), name=trq_lbl[c])
            self._plots_state.append(p)

        for p in self._plots_state[1:]:
            p.setXLink(self._plots_state[0])

        for p in self._plots_state:
            p.hideAxis("bottom")

        for p in self._plots_state[-3:]:
            p.showAxis("bottom")
            p.setLabel("bottom", "time [s]")

    def _build_nullspace_tab(self):
        p_xy = _mkplot(
            self.nullspace_glw,
            0,
            0,
            "Rotor tilt direction in body XY plane (x up, y right)",
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
            p_xy.plot(
                [0.0, origin[0]],
                [0.0, origin[1]],
                pen=pg.mkPen("#888888", width=2)
            )
            p_xy.plot(
                [origin[0]],
                [origin[1]],
                pen=None,
                symbol="o",
                symbolSize=13,
                symbolBrush=color,
                symbolPen=pg.mkPen("k")
            )
            rotor_label = pg.TextItem(f"R{i + 1}", color=color, anchor=(0.5, 1.4))
            rotor_label.setPos(origin[0], origin[1])
            p_xy.addItem(rotor_label)

        p_xy.plot(
            [0.0, 0.10],
            [0.0, 0.0],
            pen=pg.mkPen("#555555", width=3)
        )
        y_label = pg.TextItem("+y", color="#333333", anchor=(0.0, 0.5))
        y_label.setPos(0.11, 0.0)
        p_xy.addItem(y_label)

        p_xy.plot(
            [0.0, 0.0],
            [0.0, 0.10],
            pen=pg.mkPen("#555555", width=3)
        )
        x_label = pg.TextItem("+x", color="#333333", anchor=(0.5, 1.0))
        x_label.setPos(0.0, 0.11)
        p_xy.addItem(x_label)

        self._xy_actual = []
        self._xy_cmd = []
        for i, color in enumerate(C4):
            cmd_curve = p_xy.plot(
                pen=pg.mkPen(color=color, width=2, style=QtCore.Qt.DashLine),
                name="commanded tilt" if i == 0 else None,
                connect="finite"
            )
            actual_curve = p_xy.plot(
                pen=pg.mkPen(color=color, width=3),
                name="measured tilt" if i == 0 else None,
                connect="finite"
            )
            cmd_curve.setZValue(3)
            actual_curve.setZValue(4)
            self._xy_cmd.append(cmd_curve)
            self._xy_actual.append(actual_curve)

        p_dxy = _mkplot(
            self.nullspace_glw,
            0,
            1,
            "RMS inputs driving symmetric spreading",
            "d RMS [m]"
        )
        self._cv["null_d_x"] = p_dxy.plot(pen=_pen(C_R), name="d_x -> β spreading")
        self._cv["null_d_y"] = p_dxy.plot(pen=_pen(C_G), name="d_y -> α spreading")
        self._plots_nullspace.append(p_dxy)

        p_spread = _mkplot(
            self.nullspace_glw,
            1,
            1,
            "Symmetric tilt component",
            "spreading [deg]"
        )
        self._cv["alpha_spread"] = p_spread.plot(pen=_pen(C_G), name="Δα measured")
        self._cv["alpha_spread_cmd"] = _bring_front(
            p_spread.plot(
                pen=pg.mkPen(color=C_G, width=2, style=QtCore.Qt.DashLine),
                name="Δα commanded"
            )
        )
        self._cv["beta_spread"] = p_spread.plot(pen=_pen(C_R), name="Δβ measured")
        self._cv["beta_spread_cmd"] = _bring_front(
            p_spread.plot(
                pen=pg.mkPen(color=C_R, width=2, style=QtCore.Qt.DashLine),
                name="Δβ commanded"
            )
        )
        self._plots_nullspace.append(p_spread)

        p_spread.setXLink(p_dxy)
        p_dxy.hideAxis("bottom")
        p_spread.setLabel("bottom", "time [s]")

        self.nullspace_label = pg.LabelItem(justify="left")
        self.nullspace_label.setText(
            "<div style='font-size:13pt; color:#111;'>"
            "<b>Nullspace tilt diagnostics</b><br><br>Waiting for data..."
            "</div>"
        )
        self.nullspace_glw.addItem(self.nullspace_label, row=2, col=1)

    def _build_hexa_nullspace_tab(self):
        p_xy = _mkplot(
            self.nullspace_glw,
            0,
            0,
            "Radial tilt direction in body XY plane (x up, y right)",
            "body x [m]",
            rowspan=3
        )
        p_xy.setLabel("bottom", "body y [m]")
        p_xy.setAspectLocked(True)
        p_xy.setXRange(-0.72, 0.72, padding=0)
        p_xy.setYRange(-0.72, 0.72, padding=0)
        p_xy.addLine(x=0.0, pen=pg.mkPen("#b0b0b0", style=QtCore.Qt.DashLine))
        p_xy.addLine(y=0.0, pen=pg.mkPen("#b0b0b0", style=QtCore.Qt.DashLine))

        for i, (origin, color) in enumerate(zip(HEXA_ROTOR_PLOT_YX, C6)):
            p_xy.plot([0.0, origin[0]], [0.0, origin[1]], pen=pg.mkPen("#888888", width=2))
            p_xy.plot(
                [origin[0]],
                [origin[1]],
                pen=None,
                symbol="o",
                symbolSize=13,
                symbolBrush=color,
                symbolPen=pg.mkPen("k")
            )
            tilt_label = pg.TextItem(f"A{i}", color=color, anchor=(0.5, 1.4))
            tilt_label.setPos(origin[0], origin[1])
            p_xy.addItem(tilt_label)

        p_xy.plot([0.0, 0.10], [0.0, 0.0], pen=pg.mkPen("#555555", width=3))
        y_label = pg.TextItem("+y", color="#333333", anchor=(0.0, 0.5))
        y_label.setPos(0.11, 0.0)
        p_xy.addItem(y_label)
        p_xy.plot([0.0, 0.0], [0.0, 0.10], pen=pg.mkPen("#555555", width=3))
        x_label = pg.TextItem("+x", color="#333333", anchor=(0.5, 1.0))
        x_label.setPos(0.0, 0.11)
        p_xy.addItem(x_label)

        self._xy_actual = []
        self._xy_cmd = []
        for i, color in enumerate(C6):
            cmd_curve = p_xy.plot(
                pen=pg.mkPen(color=color, width=2, style=QtCore.Qt.DashLine),
                name="commanded tilt" if i == 0 else None,
                connect="finite"
            )
            actual_curve = p_xy.plot(
                pen=pg.mkPen(color=color, width=3),
                name="measured tilt" if i == 0 else None,
                connect="finite"
            )
            cmd_curve.setZValue(3)
            actual_curve.setZValue(4)
            self._xy_cmd.append(cmd_curve)
            self._xy_actual.append(actual_curve)

        p_dxy = _mkplot(
            self.nullspace_glw,
            0,
            1,
            "RMS inputs driving radial tilt gradient",
            "d RMS [m]"
        )
        self._cv["null_d_x"] = p_dxy.plot(pen=_pen(C_R), name="d_x")
        self._cv["null_d_y"] = p_dxy.plot(pen=_pen(C_G), name="d_y")
        self._plots_nullspace.append(p_dxy)

        p_alpha = _mkplot(
            self.nullspace_glw,
            1,
            1,
            "Radial tilt angle",
            "alpha [deg]"
        )
        for i, color in enumerate(C6):
            self._cv[f"hexa_null_alpha{i}"] = p_alpha.plot(pen=_pen(color), name=f"alpha{i}")
            self._cv[f"hexa_null_alphac{i}"] = _bring_front(
                p_alpha.plot(
                    pen=pg.mkPen(color=color, width=2, style=QtCore.Qt.DashLine),
                    name=f"alpha{i}_cmd"
                )
            )
        self._plots_nullspace.append(p_alpha)

        p_alpha.setXLink(p_dxy)
        p_dxy.hideAxis("bottom")
        p_alpha.setLabel("bottom", "time [s]")

        self.nullspace_label = pg.LabelItem(justify="left")
        self.nullspace_label.setText(
            "<div style='font-size:13pt; color:#111;'>"
            "<b>HEXA nullspace tilt diagnostics</b><br><br>Waiting for data..."
            "</div>"
        )
        self.nullspace_glw.addItem(self.nullspace_label, row=2, col=1)

    def _build_disturbance_tab(self):
        pos_lbl = ["x", "y", "z"]
        att_lbl = ["roll", "pitch", "yaw"]

        for c in range(3):
            p = _mkplot(
                self.disturbance_glw,
                0,
                c,
                f"d_{pos_lbl[c]}",
                f"d_{pos_lbl[c]} RMS [m]"
            )
            self._cv[f"dpos{c}"] = p.plot(
                pen=_pen(C3[c]),
                name=f"d_{pos_lbl[c]}"
            )
            self._plots_disturbance.append(p)

        for c in range(3):
            p = _mkplot(
                self.disturbance_glw,
                1,
                c,
                f"d_{att_lbl[c]}",
                f"d_{att_lbl[c]} RMS [deg]"
            )
            self._cv[f"datt{c}"] = p.plot(
                pen=_pen(C3[c]),
                name=f"d_{att_lbl[c]}"
            )
            self._plots_disturbance.append(p)

        p_xerr = _mkplot(self.disturbance_glw, 2, 0, "x position error", "x error [m]")
        self._cv["perf_xerr"] = p_xerr.plot(pen=_pen(C_R), name="x_cmd - x")
        self._plots_disturbance.append(p_xerr)

        p_vx = _mkplot(self.disturbance_glw, 2, 1, "x velocity", "vx [m/s]")
        self._cv["perf_vx"] = p_vx.plot(pen=_pen(C_G), name="vx")
        self._plots_disturbance.append(p_vx)

        p_fx = _mkplot(self.disturbance_glw, 2, 2, "Applied X impulse", "external Fx [N]")
        self._cv["perf_fx"] = p_fx.plot(pen=_pen(C_O), name="external Fx")
        self._plots_disturbance.append(p_fx)

        for p in self._plots_disturbance[1:]:
            p.setXLink(self._plots_disturbance[0])

        for p in self._plots_disturbance[:-3]:
            p.hideAxis("bottom")

        for p in self._plots_disturbance[-3:]:
            p.setLabel("bottom", "time [s]")

    def _build_actuator_tab(self):
        for i, cl in enumerate(C4):
            p = _mkplot(self.act_glw, 0, i, f"f{i+1}", f"f{i+1} [N]")
            self._cv[f"f_single{i}"] = p.plot(pen=_pen(cl), name=f"f{i+1}")
            self._plots_act.append(p)

        for i, cl in enumerate(C4):
            p = _mkplot(self.act_glw, 1, i, f"α{i+1} / α{i+1}_cmd", f"α{i+1} [deg]")
            self._cv[f"alpha_single{i}"] = p.plot(pen=_pen(cl), name=f"α{i+1}")
            self._cv[f"alphac_single{i}"] = _bring_front(p.plot(pen=_cmd_pen(), name=f"α{i+1}_cmd"))
            self._plots_act.append(p)

        for i, cl in enumerate(C4):
            p = _mkplot(self.act_glw, 2, i, f"β{i+1} / β{i+1}_cmd", f"β{i+1} [deg]")
            self._cv[f"beta_single{i}"] = p.plot(pen=_pen(cl), name=f"β{i+1}")
            self._cv[f"betac_single{i}"] = _bring_front(p.plot(pen=_cmd_pen(), name=f"β{i+1}_cmd"))
            self._plots_act.append(p)

        p_f_all = _mkplot(self.act_glw, 3, 0, "f1-f4", "force [N]")
        for i, cl in enumerate(C4):
            self._cv[f"f_all{i}"] = p_f_all.plot(pen=_pen(cl), name=f"f{i+1}")
        self._plots_act.append(p_f_all)

        # P2T2 pair force:
        #   f14 = average force of rotors 1 and 4
        #   f23 = average force of rotors 2 and 3
        p_f_pair = _mkplot(self.act_glw, 3, 1, "f14 / f23", "pair force [N]")
        self._cv["f_pair14"] = p_f_pair.plot(pen=_pen(C_R), name="f14 = (f1+f4)/2")
        self._cv["f_pair23"] = p_f_pair.plot(pen=_pen(C_B), name="f23 = (f2+f3)/2")
        self._plots_act.append(p_f_pair)

        # P2T2 pair theta average, including measured and commanded angles.
        p_beta_pair = _mkplot(
            self.act_glw,
            3,
            2,
            "β14_avg / β23_avg / cmd",
            "pair β avg [deg]"
        )
        self._cv["beta_pair14"] = p_beta_pair.plot(
            pen=_pen(C_R), name="β14_avg = (β1+β4)/2"
        )
        self._cv["beta_pair23"] = p_beta_pair.plot(
            pen=_pen(C_B), name="β23_avg = (β2+β3)/2"
        )
        self._cv["beta_pair14_cmd"] = _bring_front(
            p_beta_pair.plot(
                pen=pg.mkPen(color=C_R, width=2, style=QtCore.Qt.DashLine),
                name="β14_avg_cmd"
            )
        )
        self._cv["beta_pair23_cmd"] = _bring_front(
            p_beta_pair.plot(
                pen=pg.mkPen(color=C_B, width=2, style=QtCore.Qt.DashLine),
                name="β23_avg_cmd"
            )
        )
        self._plots_act.append(p_beta_pair)

        self.max_label = pg.LabelItem(justify="left")
        self.max_label.setText(
            "<div style='font-size:14pt; color:#111;'>"
            "<b>Max error since start</b><br><br>"
            "Position [m]<br>"
            "x: 0.0000&nbsp;&nbsp; y: 0.0000&nbsp;&nbsp; z: 0.0000<br>"
            "|e_pos| max: 0.0000<br><br>"
            "Attitude [deg]<br>"
            "roll: 0.000&nbsp;&nbsp; pitch: 0.000&nbsp;&nbsp; yaw: 0.000<br>"
            "|e_att| max: 0.000"
            "</div>"
        )
        self.act_glw.addItem(self.max_label, row=3, col=3)

        for p in self._plots_act[1:]:
            p.setXLink(self._plots_act[0])

        for p in self._plots_act:
            p.hideAxis("bottom")

        for p in self._plots_act[-3:]:
            p.showAxis("bottom")
            p.setLabel("bottom", "time [s]")

    def _build_hexa_actuator_tab(self):
        for i, color in enumerate(C6):
            p = _mkplot(self.act_glw, i // 3, i % 3, f"f{i} command", f"f{i} [N]")
            self._cv[f"hexa_f_single{i}"] = p.plot(pen=_pen(color), name=f"f{i}")
            self._plots_act.append(p)

        for i, color in enumerate(C6):
            p = _mkplot(
                self.act_glw,
                2 + i // 3,
                i % 3,
                f"alpha{i} / alpha{i}_cmd",
                f"alpha{i} [deg]"
            )
            self._cv[f"hexa_alpha_single{i}"] = p.plot(pen=_pen(color), name=f"alpha{i}")
            self._cv[f"hexa_alphac_single{i}"] = _bring_front(
                p.plot(pen=_cmd_pen(), name=f"alpha{i}_cmd")
            )
            self._plots_act.append(p)

        p_f_all = _mkplot(self.act_glw, 4, 0, "f0-f5 command", "pair thrust [N]", colspan=3)
        for i, color in enumerate(C6):
            self._cv[f"hexa_f_all{i}"] = p_f_all.plot(pen=_pen(color), name=f"f{i}")
        self._plots_act.append(p_f_all)

        self.max_label = pg.LabelItem(justify="left")
        self.max_label.setText(
            "<div style='font-size:14pt; color:#111;'>"
            "<b>Max error since start</b><br><br>Waiting for data..."
            "</div>"
        )
        self.act_glw.addItem(self.max_label, row=4, col=3)

        for p in self._plots_act[1:]:
            p.setXLink(self._plots_act[0])

        for p in self._plots_act:
            p.hideAxis("bottom")

        self._plots_act[-1].showAxis("bottom")
        self._plots_act[-1].setLabel("bottom", "time [s]")

    def _update_max_label(self):
        nd = self.node

        with nd.lock:
            max_pos = nd.max_pos_err_abs.copy()
            max_att = nd.max_att_err_abs.copy()
            max_pos_norm = nd.max_pos_err_norm
            max_att_norm = nd.max_att_err_norm
            max_pos_t = nd.max_pos_err_time
            max_att_t = nd.max_att_err_time

        self.max_label.setText(
            "<div style='font-size:14pt; color:#111;'>"
            "<b>Max error since start</b><br><br>"
            "<b>Position error [m]</b><br>"
            f"x_max = {max_pos[0]:.4f}<br>"
            f"y_max = {max_pos[1]:.4f}<br>"
            f"z_max = {max_pos[2]:.4f}<br>"
            f"|e_pos|_max = {max_pos_norm:.4f} @ t = {max_pos_t:.2f} s<br><br>"
            "<b>Attitude error [deg]</b><br>"
            f"roll_max = {max_att[0]:.3f}<br>"
            f"pitch_max = {max_att[1]:.3f}<br>"
            f"yaw_max = {max_att[2]:.3f}<br>"
            f"|e_att|_max = {max_att_norm:.3f} @ t = {max_att_t:.2f} s"
            "</div>"
        )

    def _update_xy_vectors(self, alpha_data, beta_data, curves):
        if not alpha_data.shape[0] or not beta_data.shape[0]:
            return

        vectors = tilt_xy_components(alpha_data[-1, 1:5], beta_data[-1, 1:3])
        for i, curve in enumerate(curves):
            origin_yx = ROTOR_PLOT_YX[i]
            vector_yx = vectors[i, [1, 0]]
            x, y = vector_polyline(origin_yx, vector_yx)
            curve.setData(x, y, connect="finite")

    def _update_hexa_xy_vectors(self, alpha_data, curves):
        if not alpha_data.shape[0]:
            return

        vectors = hexa_tilt_xy_components(alpha_data[-1, 1:7])
        for i, curve in enumerate(curves):
            origin_yx = HEXA_ROTOR_PLOT_YX[i]
            vector_yx = vectors[i, [1, 0]]
            x, y = vector_polyline(origin_yx, vector_yx)
            curve.setData(x, y, connect="finite")

    def _update_nullspace_label(self, dd, dw, dph, dth, dphc, dthc):
        if not dd.shape[0]:
            return

        d_now = dd[-1, 1:7]
        force_now = dw[-1, 1:4] if dw.shape[0] else np.full(3, np.nan)
        alpha_now = dph[-1, 1:5] if dph.shape[0] else np.full(4, np.nan)
        beta_now = dth[-1, 1:3] if dth.shape[0] else np.full(2, np.nan)
        alpha_cmd = dphc[-1, 1:5] if dphc.shape[0] else np.full(4, np.nan)
        beta_cmd = dthc[-1, 1:3] if dthc.shape[0] else np.full(2, np.nan)

        alpha_spread = 0.25 * (alpha_now[0] + alpha_now[1] - alpha_now[2] - alpha_now[3])
        beta_spread = 0.5 * (beta_now[0] - beta_now[1])
        alpha_spread_cmd = 0.25 * (alpha_cmd[0] + alpha_cmd[1] - alpha_cmd[2] - alpha_cmd[3])
        beta_spread_cmd = 0.5 * (beta_cmd[0] - beta_cmd[1])

        self.nullspace_label.setText(
            "<div style='font-size:12pt; color:#111;'>"
            "<b>Current nullspace tilt diagnostics</b><br>"
            f"d_pos RMS [m] = [{d_now[0]:.3f}, {d_now[1]:.3f}, {d_now[2]:.3f}]<br>"
            f"d_att RMS [deg] = [{d_now[3]:.2f}, {d_now[4]:.2f}, {d_now[5]:.2f}]<br>"
            f"F_body [N] = [{force_now[0]:.2f}, {force_now[1]:.2f}, {force_now[2]:.2f}]<br><br>"
            f"Δα measured / commanded = {alpha_spread:.2f} / {alpha_spread_cmd:.2f} deg<br>"
            f"Δβ measured / commanded = {beta_spread:.2f} / {beta_spread_cmd:.2f} deg<br>"
            f"α measured [deg] = [{alpha_now[0]:.1f}, {alpha_now[1]:.1f}, "
            f"{alpha_now[2]:.1f}, {alpha_now[3]:.1f}]<br>"
            f"α commanded [deg] = [{alpha_cmd[0]:.1f}, {alpha_cmd[1]:.1f}, "
            f"{alpha_cmd[2]:.1f}, {alpha_cmd[3]:.1f}]<br>"
            f"β measured / commanded [deg] = [{beta_now[0]:.1f}, {beta_now[1]:.1f}] / "
            f"[{beta_cmd[0]:.1f}, {beta_cmd[1]:.1f}]<br><br>"
            "<b>Mapping</b>: d_y -> α1,2(+), α3,4(-); d_x -> β1(+), β2(-)<br>"
            "β1: R1,R4; β2: R2,R3<br>"
            "Plot axes: x points up, y points right.<br>"
            "Arrows use e_xy = [-sin(β)cos(α), sin(α)] × 0.35 m.<br>"
            "Solid: measured, dashed: commanded"
            "</div>"
        )

    def _update_hexa_nullspace_label(self, dd, dw, dalpha, dalpha_cmd):
        if not dd.shape[0]:
            return

        d_now = dd[-1, 1:7]
        force_now = dw[-1, 1:4] if dw.shape[0] else np.full(3, np.nan)
        alpha_now = dalpha[-1, 1:7] if dalpha.shape[0] else np.full(6, np.nan)
        alpha_cmd = dalpha_cmd[-1, 1:7] if dalpha_cmd.shape[0] else np.full(6, np.nan)

        self.nullspace_label.setText(
            "<div style='font-size:12pt; color:#111;'>"
            "<b>HEXA nullspace tilt diagnostics</b><br>"
            f"d_pos RMS [m] = [{d_now[0]:.3f}, {d_now[1]:.3f}, {d_now[2]:.3f}]<br>"
            f"d_att RMS [deg] = [{d_now[3]:.2f}, {d_now[4]:.2f}, {d_now[5]:.2f}]<br>"
            f"F_body [N] = [{force_now[0]:.2f}, {force_now[1]:.2f}, {force_now[2]:.2f}]<br><br>"
            f"alpha measured [deg] = [{', '.join(f'{value:.1f}' for value in alpha_now)}]<br>"
            f"alpha commanded [deg] = [{', '.join(f'{value:.1f}' for value in alpha_cmd)}]<br><br>"
            "<b>Mapping</b>: each arrow is the horizontal thrust component "
            "sin(alpha_i) tangent_i.<br>"
            "Radial nullspace target uses d_x^2 tangent_x + d_y^2 tangent_y.<br>"
            "Solid: measured, dashed: commanded"
            "</div>"
        )

    def _update_performance_label(self, dpe, dv, dae, dext, dth, dph, df, dha, dhf):
        if not dpe.shape[0]:
            return

        def rms(x):
            return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")

        x_error = dpe[:, 1]
        z_error = dpe[:, 3]
        vx = dv[:, 1] if dv.shape[0] else np.empty(0)
        roll_error = dae[:, 1] if dae.shape[0] else np.empty(0)
        pitch_error = dae[:, 2] if dae.shape[0] else np.empty(0)
        fx = dext[:, 1] if dext.shape[0] else np.empty(0)

        if self.is_hexa:
            tilt_peak = np.max(np.abs(dha[:, 1:7])) if dha.shape[0] else float("nan")
            force_spread_rms = rms(np.ptp(dhf[:, 1:7], axis=1)) if dhf.shape[0] else float("nan")
            actuator_text = (
                f"alpha peak = {tilt_peak:.1f} deg&nbsp;&nbsp; "
                f"force spread RMS = {force_spread_rms:.2f} N"
            )
        else:
            beta_peak = np.max(np.abs(dth[:, 1:3])) if dth.shape[0] else float("nan")
            alpha_peak = np.max(np.abs(dph[:, 1:5])) if dph.shape[0] else float("nan")
            force_spread_rms = rms(np.ptp(df[:, 1:5], axis=1)) if df.shape[0] else float("nan")
            actuator_text = (
                f"beta peak = {beta_peak:.1f} deg&nbsp;&nbsp; alpha peak = {alpha_peak:.1f} deg&nbsp;&nbsp; "
                f"force spread RMS = {force_spread_rms:.2f} N"
            )

        fx_rms = rms(fx)
        fx_peak = np.max(np.abs(fx)) if fx.size else float("nan")
        self.performance_label.setText(
            f"<b>Rolling {WINDOW_SEC:.0f} s performance</b> (lower error/RMS is better)&nbsp;&nbsp; "
            f"x error RMS / peak = {rms(x_error):.4f} / {np.max(np.abs(x_error)):.4f} m&nbsp;&nbsp; "
            f"vx RMS = {rms(vx):.4f} m/s&nbsp;&nbsp; z error RMS = {rms(z_error):.4f} m<br>"
            f"roll / pitch error RMS = {rms(roll_error):.2f} / {rms(pitch_error):.2f} deg&nbsp;&nbsp; "
            f"external Fx RMS / peak = {fx_rms:.2f} / {fx_peak:.2f} N&nbsp;&nbsp; {actuator_text}"
        )

    def _upd(self):
        nd = self.node

        with nd.lock:
            dp = nd.buf_pos.get()
            dv = nd.buf_vel.get()
            dc = nd.buf_cmd.get()
            dpe = nd.buf_pos_err.get()
            dext = nd.buf_disturbance.get()
            dr = nd.buf_rpy.get()
            da = nd.buf_att.get()
            dae = nd.buf_att_err.get()
            dw = nd.buf_wr.get()
            dd = nd.buf_d.get()
            dth = nd.buf_beta.get()
            dph = nd.buf_alpha.get()
            dthc = nd.buf_beta_cmd.get()
            dphc = nd.buf_alpha_cmd.get()
            df = nd.buf_thr.get()
            dha = nd.buf_hexa_alpha.get()
            dhac = nd.buf_hexa_alpha_cmd.get()
            dhf = nd.buf_hexa_thr.get()

        tn = 0.0
        for d in (dp, dv, dc, dpe, dext, dr, da, dae, dw, dd, dth, dph, dthc, dphc, df, dha, dhac, dhf):
            if d.shape[0]:
                tn = max(tn, d[-1, 0])

        if tn == 0.0:
            return

        tl = tn - WINDOW_SEC

        def tr(a):
            return a[a[:, 0] >= tl] if a.shape[0] else a

        dp = tr(dp)
        dv = tr(dv)
        dc = tr(dc)
        dpe = tr(dpe)
        dext = tr(dext)
        dr = tr(dr)
        da = tr(da)
        dae = tr(dae)
        dw = tr(dw)
        dd = tr(dd)
        dth = tr(dth)
        dph = tr(dph)
        dthc = tr(dthc)
        dphc = tr(dphc)
        df = tr(df)
        dha = tr(dha)
        dhac = tr(dhac)
        dhf = tr(dhf)

        cv = self._cv

        if dp.shape[0]:
            for i in range(3):
                cv[f"pos{i}"].setData(dp[:, 0], dp[:, 1 + i])
        if dc.shape[0]:
            for i in range(3):
                cv[f"cmd{i}"].setData(dc[:, 0], dc[:, 1 + i])
        if dpe.shape[0]:
            for i in range(3):
                cv[f"perr{i}"].setData(dpe[:, 0], dpe[:, 1 + i])
            cv["perf_xerr"].setData(dpe[:, 0], dpe[:, 1])
        if dv.shape[0]:
            cv["perf_vx"].setData(dv[:, 0], dv[:, 1])
        if dext.shape[0]:
            cv["perf_fx"].setData(dext[:, 0], dext[:, 1])

        if dw.shape[0]:
            for i in range(3):
                cv[f"F{i}"].setData(dw[:, 0], dw[:, 1 + i])
                cv[f"M{i}"].setData(dw[:, 0], dw[:, 4 + i])

        if dd.shape[0]:
            for i in range(3):
                cv[f"dpos{i}"].setData(dd[:, 0], dd[:, 1 + i])
                cv[f"datt{i}"].setData(dd[:, 0], dd[:, 4 + i])
            cv["null_d_x"].setData(dd[:, 0], dd[:, 1])
            cv["null_d_y"].setData(dd[:, 0], dd[:, 2])

        if dr.shape[0]:
            for i in range(3):
                cv[f"rpy{i}"].setData(dr[:, 0], dr[:, 1 + i])
        if da.shape[0]:
            for i in range(3):
                cv[f"acmd{i}"].setData(da[:, 0], da[:, 1 + i])
        if dae.shape[0]:
            for i in range(3):
                cv[f"aerr{i}"].setData(dae[:, 0], dae[:, 1 + i])

        if self.is_hexa:
            if dhf.shape[0]:
                for i in range(6):
                    cv[f"hexa_f_single{i}"].setData(dhf[:, 0], dhf[:, 1 + i])
                    cv[f"hexa_f_all{i}"].setData(dhf[:, 0], dhf[:, 1 + i])

            if dha.shape[0]:
                for i in range(6):
                    cv[f"hexa_alpha_single{i}"].setData(dha[:, 0], dha[:, 1 + i])
                    cv[f"hexa_null_alpha{i}"].setData(dha[:, 0], dha[:, 1 + i])

            if dhac.shape[0]:
                for i in range(6):
                    cv[f"hexa_alphac_single{i}"].setData(dhac[:, 0], dhac[:, 1 + i])
                    cv[f"hexa_null_alphac{i}"].setData(dhac[:, 0], dhac[:, 1 + i])

            self._update_hexa_xy_vectors(dha, self._xy_actual)
            self._update_hexa_xy_vectors(dhac, self._xy_cmd)
            self._update_hexa_nullspace_label(dd, dw, dha, dhac)
        else:
            if df.shape[0]:
                for i in range(4):
                    cv[f"f_single{i}"].setData(df[:, 0], df[:, 1 + i])
                    cv[f"f_all{i}"].setData(df[:, 0], df[:, 1 + i])

                f14 = 0.5 * (df[:, 1] + df[:, 4])
                f23 = 0.5 * (df[:, 2] + df[:, 3])
                cv["f_pair14"].setData(df[:, 0], f14)
                cv["f_pair23"].setData(df[:, 0], f23)

            if dph.shape[0]:
                for i in range(4):
                    cv[f"alpha_single{i}"].setData(dph[:, 0], dph[:, 1 + i])
                alpha_spread = 0.25 * (dph[:, 1] + dph[:, 2] - dph[:, 3] - dph[:, 4])
                cv["alpha_spread"].setData(dph[:, 0], alpha_spread)

            if dphc.shape[0]:
                for i in range(4):
                    cv[f"alphac_single{i}"].setData(dphc[:, 0], dphc[:, 1 + i])
                alpha_spread_cmd = 0.25 * (dphc[:, 1] + dphc[:, 2] - dphc[:, 3] - dphc[:, 4])
                cv["alpha_spread_cmd"].setData(dphc[:, 0], alpha_spread_cmd)

            if dth.shape[0]:
                for i in range(2):
                    cv[f"beta_single{i}"].setData(dth[:, 0], dth[:, 1 + i])
                beta_spread = 0.5 * (dth[:, 1] - dth[:, 2])
                cv["beta_spread"].setData(dth[:, 0], beta_spread)

            if dthc.shape[0]:
                for i in range(2):
                    cv[f"betac_single{i}"].setData(dthc[:, 0], dthc[:, 1 + i])
                beta_spread_cmd = 0.5 * (dthc[:, 1] - dthc[:, 2])
                cv["beta_spread_cmd"].setData(dthc[:, 0], beta_spread_cmd)

            self._update_xy_vectors(dph, dth, self._xy_actual)
            self._update_xy_vectors(dphc, dthc, self._xy_cmd)
            self._update_nullspace_label(dd, dw, dph, dth, dphc, dthc)

        self._update_performance_label(dpe, dv, dae, dext, dth, dph, df, dha, dhf)
        self._update_max_label()

        if self._plots_state:
            self._plots_state[0].setXRange(tl, tn, padding=0)

        if self._plots_nullspace:
            self._plots_nullspace[0].setXRange(tl, tn, padding=0)

        if self._plots_disturbance:
            self._plots_disturbance[0].setXRange(tl, tn, padding=0)

        if self._plots_act:
            self._plots_act[0].setXRange(tl, tn, padding=0)


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
