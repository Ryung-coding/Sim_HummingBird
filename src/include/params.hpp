#pragma once

#include <cmath>
#include <array>

namespace params {

// System Setting -----------------------------------------------
static constexpr bool USE_SO3_HEADING_CMD = false;
static constexpr int RATE_HZ = 400;

// Task parameters-----------------------------------------------
static constexpr double HOVER_SEC = 3.0;
static constexpr double SCAN_PERIOD_SEC = 30.0;

static constexpr double APPLE_X = 1.0;
static constexpr double APPLE_Y = 0.0;
static constexpr double APPLE_Z = 3.0;
static constexpr double RADIUS = 1.5;
static constexpr double ROLL_MAX = 30.0 * M_PI / 180.0;
static constexpr double THETA_MAX = 60.0 * M_PI / 180.0;

// Model parameters -----------------------------------------------
static constexpr double mass = 5.00;
static constexpr double grav = 9.81;

static constexpr std::array<double, 3> J = {0.030, 0.030, 0.050};

static constexpr double Lx = 0.1861;
static constexpr double Ly = 0.1861;
static constexpr double d = 0.0500;
static constexpr double zeta = 0.0200;

// position controller -----------------------------------------------
static constexpr std::array<double, 3> Kp_pos = {50.0, 50.0, 50.0};
static constexpr std::array<double, 3> Ki_pos = {0.10, 0.10, 0.10};
static constexpr std::array<double, 3> Kd_pos = {30.0, 30.0, 30.0};

static constexpr std::array<double, 3> pos_i_sat = {30.0, 30.0, 30.0};
static constexpr std::array<double, 3> force_body_sat = {90.0, 60.0, 90.0};

// attitude controller -----------------------------------------------
static constexpr std::array<double, 3> kR = {40.0, 40.0, 5.0};
static constexpr std::array<double, 3> kW = {7.0, 7.0, 2.0};
static constexpr std::array<double, 3> kI = {0.1, 0.1, 0.1};

static constexpr std::array<double, 3> att_i_sat = {1.0, 1.0, 1.0};
static constexpr std::array<double, 3> torque_sat = {8.0, 8.0, 5.0};

static constexpr double ER_NORM_MAX = 1.5;

// Saturatation parameters -----------------------------------------------
static constexpr double f_min = 1.0e-3;
static constexpr double f_cmd_min = 1.0e-6;
static constexpr double f_cmd_max = 100.0;
static constexpr double theta_limit_rad = 1.57;
static constexpr double phi_limit_rad = 0.523;
static constexpr double virtual_lambda = 1.0e-4;

// allocation check tolerance ---
static constexpr double check_force_tol = 1.00;
static constexpr double check_moment_tol = 2.00;
}