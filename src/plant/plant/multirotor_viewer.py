#!/usr/bin/env python3
import sys
import math
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from multirotor_interfaces.msg import MultirotorState, Cmd, Wrench, Input

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


class VNode(Node):
    def __init__(self):
        super().__init__("multirotor_viewer")
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
        self.buf_cmd = Ring(MAX_SAMPLES, 4)
        self.buf_pos_err = Ring(MAX_SAMPLES, 4)

        self.buf_rpy = Ring(MAX_SAMPLES, 4)
        self.buf_att = Ring(MAX_SAMPLES, 4)
        self.buf_att_err = Ring(MAX_SAMPLES, 4)

        self.buf_wr = Ring(MAX_SAMPLES, 7)

        self.buf_theta = Ring(MAX_SAMPLES, 5)
        self.buf_phi = Ring(MAX_SAMPLES, 5)
        self.buf_theta_cmd = Ring(MAX_SAMPLES, 5)
        self.buf_phi_cmd = Ring(MAX_SAMPLES, 5)
        self.buf_thr = Ring(MAX_SAMPLES, 5)

        self.buf_pair_actual = Ring(MAX_SAMPLES, 5)
        self.buf_pair_cmd = Ring(MAX_SAMPLES, 5)

        self.create_subscription(MultirotorState, "/multirotor_state", self._cb_state, 10)
        self.create_subscription(Cmd, "/cmd", self._cb_cmd, 10)
        self.create_subscription(Wrench, "/wrench", self._cb_wrench, 10)
        self.create_subscription(Input, "/input", self._cb_input, 10)

    def _t(self):
        return self.get_clock().now().nanoseconds * 1e-9 - self.t0

    def _cb_state(self, m):
        t = self._t()

        pos = np.array([m.pos[0], m.pos[1], m.pos[2]], dtype=float)
        rpy_deg = np.array([m.rpy[0], m.rpy[1], m.rpy[2]], dtype=float) * RAD2DEG
        theta_deg = np.array([m.theta[0], m.theta[1], m.theta[2], m.theta[3]], dtype=float) * RAD2DEG
        phi_deg = np.array([m.phi[0], m.phi[1], m.phi[2], m.phi[3]], dtype=float) * RAD2DEG

        pos_err = self.last_pos_cmd - pos

        att_err = np.array([
            wrap_deg(self.last_att_cmd_deg[0] - rpy_deg[0]),
            wrap_deg(self.last_att_cmd_deg[1] - rpy_deg[1]),
            wrap_deg(self.last_att_cmd_deg[2] - rpy_deg[2])
        ], dtype=float)

        pair_actual = np.array([
            theta_deg[0] - theta_deg[3],
            theta_deg[1] - theta_deg[2],
            phi_deg[0] - phi_deg[3],
            phi_deg[1] - phi_deg[2]
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
            self.buf_rpy.push([t, rpy_deg[0], rpy_deg[1], rpy_deg[2]])
            self.buf_theta.push([t, theta_deg[0], theta_deg[1], theta_deg[2], theta_deg[3]])
            self.buf_phi.push([t, phi_deg[0], phi_deg[1], phi_deg[2], phi_deg[3]])
            self.buf_pos_err.push([t, pos_err[0], pos_err[1], pos_err[2]])
            self.buf_att_err.push([t, att_err[0], att_err[1], att_err[2]])
            self.buf_pair_actual.push([t, pair_actual[0], pair_actual[1], pair_actual[2], pair_actual[3]])

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

    def _cb_input(self, m):
        t = self._t()

        f = np.array([m.f[0], m.f[1], m.f[2], m.f[3]], dtype=float)
        theta_cmd_deg = np.array([m.theta[0], m.theta[1], m.theta[2], m.theta[3]], dtype=float) * RAD2DEG
        phi_cmd_deg = np.array([m.phi[0], m.phi[1], m.phi[2], m.phi[3]], dtype=float) * RAD2DEG

        pair_cmd = np.array([
            theta_cmd_deg[0] - theta_cmd_deg[3],
            theta_cmd_deg[1] - theta_cmd_deg[2],
            phi_cmd_deg[0] - phi_cmd_deg[3],
            phi_cmd_deg[1] - phi_cmd_deg[2]
        ], dtype=float)

        with self.lock:
            self.buf_thr.push([t, f[0], f[1], f[2], f[3]])
            self.buf_theta_cmd.push([t, theta_cmd_deg[0], theta_cmd_deg[1], theta_cmd_deg[2], theta_cmd_deg[3]])
            self.buf_phi_cmd.push([t, phi_cmd_deg[0], phi_cmd_deg[1], phi_cmd_deg[2], phi_cmd_deg[3]])
            self.buf_pair_cmd.push([t, pair_cmd[0], pair_cmd[1], pair_cmd[2], pair_cmd[3]])


def _pen(color, w=2):
    return pg.mkPen(color=color, width=w, style=QtCore.Qt.SolidLine)


def _cmd_pen(w=2):
    return pg.mkPen(color="k", width=w, style=QtCore.Qt.DashLine)


def _mkplot(glw, r, c, title, ylabel):
    p = glw.addPlot(row=r, col=c, title=title)
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

        self.setWindowTitle("Sim_HummingBird Viewer")
        self.resize(1800, 1050)

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        self._cv = {}
        self._plots_state = []
        self._plots_act = []

        self.state_glw = pg.GraphicsLayoutWidget()
        self.act_glw = pg.GraphicsLayoutWidget()

        tabs.addTab(self.state_glw, "State")
        tabs.addTab(self.act_glw, "Actuator")

        self._build_state_tab()
        self._build_actuator_tab()

        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._upd)
        self._timer.start(UPDATE_MS)

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

    def _build_actuator_tab(self):
        for i, cl in enumerate(C4):
            p = _mkplot(self.act_glw, 0, i, f"f{i+1}", f"f{i+1} [N]")
            self._cv[f"f_single{i}"] = p.plot(pen=_pen(cl), name=f"f{i+1}")
            self._plots_act.append(p)

        for i, cl in enumerate(C4):
            p = _mkplot(self.act_glw, 1, i, f"φ{i+1} / φ{i+1}_cmd", f"φ{i+1} [deg]")
            self._cv[f"phi_single{i}"] = p.plot(pen=_pen(cl), name=f"φ{i+1}")
            self._cv[f"phic_single{i}"] = _bring_front(p.plot(pen=_cmd_pen(), name=f"φ{i+1}_cmd"))
            self._plots_act.append(p)

        for i, cl in enumerate(C4):
            p = _mkplot(self.act_glw, 2, i, f"θ{i+1} / θ{i+1}_cmd", f"θ{i+1} [deg]")
            self._cv[f"tht_single{i}"] = p.plot(pen=_pen(cl), name=f"θ{i+1}")
            self._cv[f"thtc_single{i}"] = _bring_front(p.plot(pen=_cmd_pen(), name=f"θ{i+1}_cmd"))
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
        p_theta_pair = _mkplot(
            self.act_glw,
            3,
            2,
            "θ14_avg / θ23_avg / cmd",
            "pair θ avg [deg]"
        )
        self._cv["theta_pair14"] = p_theta_pair.plot(
            pen=_pen(C_R), name="θ14_avg = (θ1+θ4)/2"
        )
        self._cv["theta_pair23"] = p_theta_pair.plot(
            pen=_pen(C_B), name="θ23_avg = (θ2+θ3)/2"
        )
        self._cv["theta_pair14_cmd"] = _bring_front(
            p_theta_pair.plot(
                pen=pg.mkPen(color=C_R, width=2, style=QtCore.Qt.DashLine),
                name="θ14_avg_cmd"
            )
        )
        self._cv["theta_pair23_cmd"] = _bring_front(
            p_theta_pair.plot(
                pen=pg.mkPen(color=C_B, width=2, style=QtCore.Qt.DashLine),
                name="θ23_avg_cmd"
            )
        )
        self._plots_act.append(p_theta_pair)

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

    def _upd(self):
        nd = self.node

        with nd.lock:
            dp = nd.buf_pos.get()
            dc = nd.buf_cmd.get()
            dpe = nd.buf_pos_err.get()
            dr = nd.buf_rpy.get()
            da = nd.buf_att.get()
            dae = nd.buf_att_err.get()
            dw = nd.buf_wr.get()
            dth = nd.buf_theta.get()
            dph = nd.buf_phi.get()
            dthc = nd.buf_theta_cmd.get()
            dphc = nd.buf_phi_cmd.get()
            df = nd.buf_thr.get()

        tn = 0.0
        for d in (dp, dc, dpe, dr, da, dae, dw, dth, dph, dthc, dphc, df):
            if d.shape[0]:
                tn = max(tn, d[-1, 0])

        if tn == 0.0:
            return

        tl = tn - WINDOW_SEC

        def tr(a):
            return a[a[:, 0] >= tl] if a.shape[0] else a

        dp = tr(dp)
        dc = tr(dc)
        dpe = tr(dpe)
        dr = tr(dr)
        da = tr(da)
        dae = tr(dae)
        dw = tr(dw)
        dth = tr(dth)
        dph = tr(dph)
        dthc = tr(dthc)
        dphc = tr(dphc)
        df = tr(df)

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

        if dw.shape[0]:
            for i in range(3):
                cv[f"F{i}"].setData(dw[:, 0], dw[:, 1 + i])
                cv[f"M{i}"].setData(dw[:, 0], dw[:, 4 + i])

        if dr.shape[0]:
            for i in range(3):
                cv[f"rpy{i}"].setData(dr[:, 0], dr[:, 1 + i])
        if da.shape[0]:
            for i in range(3):
                cv[f"acmd{i}"].setData(da[:, 0], da[:, 1 + i])
        if dae.shape[0]:
            for i in range(3):
                cv[f"aerr{i}"].setData(dae[:, 0], dae[:, 1 + i])

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
                cv[f"phi_single{i}"].setData(dph[:, 0], dph[:, 1 + i])

        if dphc.shape[0]:
            for i in range(4):
                cv[f"phic_single{i}"].setData(dphc[:, 0], dphc[:, 1 + i])

        if dth.shape[0]:
            for i in range(4):
                cv[f"tht_single{i}"].setData(dth[:, 0], dth[:, 1 + i])

            theta14_avg = 0.5 * (dth[:, 1] + dth[:, 4])
            theta23_avg = 0.5 * (dth[:, 2] + dth[:, 3])
            cv["theta_pair14"].setData(dth[:, 0], theta14_avg)
            cv["theta_pair23"].setData(dth[:, 0], theta23_avg)

        if dthc.shape[0]:
            for i in range(4):
                cv[f"thtc_single{i}"].setData(dthc[:, 0], dthc[:, 1 + i])

            theta14_avg_cmd = 0.5 * (dthc[:, 1] + dthc[:, 4])
            theta23_avg_cmd = 0.5 * (dthc[:, 2] + dthc[:, 3])
            cv["theta_pair14_cmd"].setData(dthc[:, 0], theta14_avg_cmd)
            cv["theta_pair23_cmd"].setData(dthc[:, 0], theta23_avg_cmd)

        self._update_max_label()

        if self._plots_state:
            self._plots_state[0].setXRange(tl, tn, padding=0)

        if self._plots_act:
            self._plots_act[0].setXRange(tl, tn, padding=0)


def main():
    rclpy.init()
    node = VNode()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    app = QtWidgets.QApplication(sys.argv)
    win = Win(node)
    win.show()
    code = app.exec_()

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()