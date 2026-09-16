#pragma once

#include <cmath>
#include <array>

namespace params {

// System Setting -----------------------------------------------
static constexpr bool USE_SO3_HEADING_CMD = false;
static constexpr int RATE_HZ = 400;

// Task parameters-----------------------------------------------


// Model parameters -----------------------------------------------
static constexpr double mass = 3.50;
static constexpr double grav = 9.81;

static constexpr std::array<double, 3> J = {0.030, 0.030, 0.050};

static constexpr double L = 0.175;
static constexpr double zeta = 0.0500;

// position controller -----------------------------------------------
static constexpr std::array<double, 3> Kp_pos = {40.0, 40.0, 60.0};
static constexpr std::array<double, 3> Ki_pos = {0.10, 0.10, 0.10};
static constexpr std::array<double, 3> Kd_pos = {8.0, 8.0, 30.0};

static constexpr std::array<double, 3> pos_i_sat = {30.0, 30.0, 30.0};
static constexpr std::array<double, 3> force_body_sat = {50.0, 50.0, 60.0};

// attitude controller -----------------------------------------------
static constexpr std::array<double, 3> kR = {40.0, 40.0, 4.0};
static constexpr std::array<double, 3> kW = {2.0, 2.0, 1.5};
static constexpr std::array<double, 3> kI = {0.1, 0.1, 0.1};

static constexpr std::array<double, 3> att_i_sat = {1.0, 1.0, 1.0};
static constexpr std::array<double, 3> torque_sat = {3.0, 3.0, 2.0};

static constexpr double ER_NORM_MAX = 1.5;

// Disturbance observer -----------------------------------------------
static constexpr double disturbance_rms_tau = 1.0;

// Saturatation parameters -----------------------------------------------
static constexpr double f_min = 1.0e-3;
static constexpr double f_cmd_min = 1.0e-6;
static constexpr double f_cmd_max = 20.0;
static constexpr double alpha_limit_rad = M_PI / 6.0;
static constexpr double beta_limit_rad = M_PI;
static constexpr double virtual_lambda = 1.0e-4;

// Augmented Differential Allocation ----------------------------------
// (7), diag [Mx My Mz Fx Fy Fz]
static constexpr std::array<double, 6> ada_kj_diag = {60.0, 60.0, 60.0, 25.0, 25.0, 25.0};

// (5), diag [alpha1 alpha2 alpha3 alpha4 beta1 beta2 f1 f2 f3 f4]
static constexpr std::array<double, 10> ada_W_inv_diag = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3200.0, 3200.0, 3200.0, 3200.0};

// (13), order [alpha_dot1 alpha_dot2 alpha_dot3 alpha_dot4 beta_dot1 beta_dot2 f_dot1 f_dot2 f_dot3 f_dot4]
static constexpr std::array<double, 10> ada_q_dot_max = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 500.0, 500.0, 500.0, 500.0};

// Disturbance-dependent actuator weighting
static constexpr double ada_weight_rms_active = 0.0;
static constexpr double ada_weight_rms_full = 0.05;
static constexpr double ada_W_inv_f_disturbed = 10000.0;

static constexpr double check_force_tol = 1.00;
static constexpr double check_moment_tol = 2.00;
}
