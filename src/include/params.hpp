#pragma once

#include <cmath>
#include <array>
#include <Eigen/Dense>

namespace params {

// System Setting -----------------------------------------------
static constexpr bool USE_SO3_HEADING_CMD = false;
static constexpr int RATE_HZ = 400;

// Model parameters -----------------------------------------------
static constexpr double grav = 9.81;

static constexpr double HB_MASS = 3.50;
static constexpr std::array<double, 3> HB_J = {0.030, 0.030, 0.050};
static constexpr double HB_L = 0.175;
static constexpr double HB_ZETA = 0.0500;
static constexpr double HB_Q_CMD_TAU_SERVO = 0.20;
static constexpr double HB_ALPHA_LIMIT_RAD = M_PI / 6.0;
static constexpr double HB_BETA_LIMIT_RAD = M_PI;

// position controller -----------------------------------------------
static constexpr std::array<double, 3> Kp_pos = {100.0, 100.0, 100.0};
static constexpr std::array<double, 3> Ki_pos = {0.50, 0.50, 0.50};
static constexpr std::array<double, 3> Kd_pos = {40.0, 40.0, 40.0};
// static constexpr std::array<double, 3> Kp_pos = {60.0, 60.0, 60.0};
// static constexpr std::array<double, 3> Ki_pos = {0.50, 0.50, 0.10};
// static constexpr std::array<double, 3> Kd_pos = {10.0, 10.0, 30.0};

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
static constexpr double HB_F_CMD_MIN = 1.0e-6;
static constexpr double HB_F_CMD_MAX = 20.0;

static constexpr double HB_VIRTUAL_LAMBDA = 1.0e-4;

// Allocation parameters -----------------------------------------------
inline const Eigen::DiagonalMatrix<double, 6> HB_KJ = [] {Eigen::DiagonalMatrix<double, 6> K; K.diagonal() << 30.0, 30.0, 30.0, 10.0, 10.0, 10.0; return K;}();
inline const Eigen::DiagonalMatrix<double, 10> HB_W_INV = [] {Eigen::DiagonalMatrix<double, 10> W; W.diagonal() << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000.0, 1000.0, 1000.0, 1000.0; return W;}();
static constexpr std::array<double, 10> HB_QDOT_MAX = {6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 200.0, 200.0, 200.0, 200.0};
static constexpr std::array<double, 3> HB_NULL_K = {0.5, 0.5, 0.3}; // [alpha, beta, thrust]

// Planning ----------------------------------------------------------
static constexpr double PLANNING_RRT_TIME_SEC = 30.0;
static constexpr double PLANNING_RRT_RANGE = 0.35;
static constexpr double PLANNING_RRT_CLEARANCE = 0.03;
static constexpr double PLANNING_PREALIGN_FRACTION = 0.25;
static constexpr double PLANNING_PREALIGN_ANGLE_RAD = M_PI / 3.0;
static constexpr double PLANNING_RILS_MAX_RADIUS = 1.5;
static constexpr double PLANNING_RILS_MIN_RADIUS = 0.01;
static constexpr double PLANNING_SFC_MARGIN = 0.005;
static constexpr double PLANNING_AVG_SPEED = 0.35;
static constexpr double PLANNING_MIN_SEGMENT_SEC = 0.4;
static constexpr double PLANNING_MAX_SEGMENT_SEC = 30.0;
static constexpr double PLANNING_MAX_VEL = 1.0;
static constexpr double PLANNING_MAX_ACC = 1.5;
static constexpr double PLANNING_MAX_OMEGA = 1.5;
static constexpr double PLANNING_OMEGA_OPT_MARGIN_RAD_S = 0.05;
static constexpr double PLANNING_ROLL_LIMIT_RAD = 20.0 * M_PI / 180.0;
static constexpr double PLANNING_ROLL_OPT_MARGIN_RAD = 0.5 * M_PI / 180.0;
static constexpr double PLANNING_PITCH_LIMIT_RAD = 60.0 * M_PI / 180.0;
static constexpr double PLANNING_TIME_WEIGHT = 0.1;
static constexpr int PLANNING_CONSTRAINT_SAMPLES = 12;
static constexpr int PLANNING_IPOPT_MAX_ITER = 600;
static constexpr double PLANNING_IPOPT_TOLERANCE = 1.0e-4;
static constexpr double PLANNING_VALIDATION_DT = 0.01;
static constexpr double PLANNING_START_TOLERANCE = 0.12;
static constexpr double PLANNING_START_HOLD_SEC = 1.0;
static constexpr double PLANNING_VISUAL_SAMPLE_SEC = 0.75;


}

