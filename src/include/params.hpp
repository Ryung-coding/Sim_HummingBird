#pragma once

#include <cmath>
#include <array>

namespace params {

// System Setting -----------------------------------------------
static constexpr bool USE_SO3_HEADING_CMD = false;
static constexpr int RATE_HZ = 400;

// Position command reference, controller z-down frame -----------------
static constexpr double initial_pose_x_m =  0.0;
static constexpr double initial_pose_y_m = -1.0;
static constexpr double initial_pose_z_m = -1.0;
static constexpr double initial_pose_roll_rad = 0.0;
static constexpr double initial_pose_pitch_rad = 0.0;
static constexpr double initial_pose_yaw_rad = 0.0;

static constexpr double PITCH_MAX = 80.0 * M_PI / 180.0;

// Model parameters -----------------------------------------------
static constexpr double grav = 9.81;

static constexpr double HB_MASS = 3.50;
static constexpr std::array<double, 3> HB_J = {0.030, 0.030, 0.050};
static constexpr double HB_L = 0.175;
static constexpr double HB_ZETA = 0.0500;
static constexpr double HB_ALPHA_LIMIT_RAD = M_PI / 6.0;
static constexpr double HB_BETA_LIMIT_RAD = M_PI;

// position controller -----------------------------------------------
static constexpr std::array<double, 3> Kp_pos = {100.0, 100.0, 100.0};
static constexpr std::array<double, 3> Ki_pos = {0.50, 0.50, 0.50};
static constexpr std::array<double, 3> Kd_pos = {40.0, 40.0, 40.0};
static constexpr std::array<double, 3> pos_i_sat = {30.0, 30.0, 30.0};
static constexpr std::array<double, 3> force_body_sat = {50.0, 50.0, 60.0};

// attitude controller -----------------------------------------------
static constexpr std::array<double, 3> kR = {40.0, 40.0, 4.0};
static constexpr std::array<double, 3> kW = {2.0, 2.0, 1.5};
static constexpr std::array<double, 3> kI = {0.1, 0.1, 0.1};

static constexpr std::array<double, 3> att_i_sat = {1.0, 1.0, 1.0};
static constexpr std::array<double, 3> torque_sat = {3.0, 3.0, 2.0};

static constexpr double ER_NORM_MAX = 1.5;

// Saturatation parameters -----------------------------------------------
static constexpr double HB_F_CMD_MIN = 1.0e-6;
static constexpr double HB_F_CMD_MAX = 20.0;

static constexpr double HB_VIRTUAL_LAMBDA = 1.0e-4;


}
