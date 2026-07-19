#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <string>
#include <params.hpp>

namespace utils {

// struct.zip =======================================================
struct LPF {
  double y = 0.0;
  double alpha = 1.0;
  bool initialized = false;

  explicit LPF(double alpha_in = 1.0) : alpha(alpha_in) {}

  double update(double x)
  {
    if (!initialized) {
      y = x;
      initialized = true;
      return y;
    }

    y = alpha * x + (1.0 - alpha) * y;
    return y;
  }

  void reset(double x = 0.0)
  {
    y = x;
    initialized = false;
  }
};

struct TargetCMD {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
};

struct AllocationOutput {
  Eigen::Vector4d f = Eigen::Vector4d::Zero();
  Eigen::Vector4d theta = Eigen::Vector4d::Zero();
  Eigen::Vector4d phi = Eigen::Vector4d::Zero();
};

struct AllocationCheck {
  bool problem = false;
  Eigen::Vector3d moment_actual = Eigen::Vector3d::Zero();
  Eigen::Vector3d force_actual = Eigen::Vector3d::Zero();
  Eigen::Vector3d moment_error = Eigen::Vector3d::Zero();
  Eigen::Vector3d force_error = Eigen::Vector3d::Zero();
  std::string message;
};


// Math utils ========================================================
inline Eigen::Matrix3d hat(const Eigen::Vector3d& v)
{
  Eigen::Matrix3d m;
  m << 0.0, -v(2), v(1),
       v(2), 0.0, -v(0),
       -v(1), v(0), 0.0;
  return m;
}

inline Eigen::Vector3d vee(const Eigen::Matrix3d& m)
{
  Eigen::Vector3d v;
  v << m(2, 1), m(0, 2), m(1, 0);
  return v;
}

inline Eigen::Vector3d vec3(const std::array<double, 3>& a)
{
  return Eigen::Vector3d(a[0], a[1], a[2]);
}

inline Eigen::Matrix3d diag3(const std::array<double, 3>& a)
{
  Eigen::Matrix3d m;
  m << a[0], 0.0, 0.0,
       0.0, a[1], 0.0,
       0.0, 0.0, a[2];
  return m;
}

inline double meanAngle(double a, double b)
{
  return std::atan2(std::sin(a) + std::sin(b), std::cos(a) + std::cos(b));
}

inline Eigen::Vector3d clampVec3(const Eigen::Vector3d& x, const Eigen::Vector3d& lim)
{
  Eigen::Vector3d y;
  y << std::clamp(x(0), -lim(0), lim(0)),
       std::clamp(x(1), -lim(1), lim(1)),
       std::clamp(x(2), -lim(2), lim(2));
  return y;
}

inline Eigen::Matrix3d rpyToRot(const Eigen::Vector3d& rpy)
{
  const double r = rpy(0);
  const double p = rpy(1);
  const double y = rpy(2);

  const double sr = std::sin(r);
  const double cr = std::cos(r);
  const double sp = std::sin(p);
  const double cp = std::cos(p);
  const double sy = std::sin(y);
  const double cy = std::cos(y);

  Eigen::Matrix3d R;
  R << cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
       sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
           -sp,                  cp * sr,                  cp * cr;

  return R;
}

inline Eigen::Matrix3d quatToRot(const Eigen::Vector4d& q_in)
{
  Eigen::Vector4d q = q_in;
  const double n = q.norm();

  if (n < 1.0e-9) {
    return Eigen::Matrix3d::Identity();
  }

  q /= n;

  const double w = q(0);
  const double x = q(1);
  const double y = q(2);
  const double z = q(3);

  Eigen::Matrix3d R;
  R << 1.0 - 2.0 * (y * y + z * z),       2.0 * (x * y - w * z),       2.0 * (x * z + w * y),
             2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z),       2.0 * (y * z - w * x),
             2.0 * (x * z - w * y),       2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y);

  return R;
}

inline Eigen::Vector3d rotToRpy(const Eigen::Matrix3d& R)
{
  Eigen::Vector3d rpy;
  rpy << std::atan2(R(2, 1), R(2, 2)),
         std::asin(std::clamp(-R(2, 0), -1.0, 1.0)),
         std::atan2(R(1, 0), R(0, 0));
  return rpy;
}

inline Eigen::Matrix3d headingToRot(const Eigen::Vector3d& heading)
{
  Eigen::Vector3d b1 = heading;
  if (b1.norm() < 1.0e-6) b1 << 1.0, 0.0, 0.0;
  b1.normalize();

  Eigen::Vector3d b3;
  b3 << 0.0, 0.0, 1.0;

  Eigen::Vector3d b2 = b3.cross(b1);
  if (b2.norm() < 1.0e-6) b2 << 0.0, 1.0, 0.0;
  b2.normalize();

  b1 = b2.cross(b3);
  b1.normalize();

  Eigen::Matrix3d R;
  R.col(0) = b1;
  R.col(1) = b2;
  R.col(2) = b3;

  return R;
}

// Path utils =========================================================
inline TargetCMD trackApple(double t)
{
  TargetCMD cmd;

  if (t < params::HOVER_SEC) {
    const double a = t / params::HOVER_SEC;
    const double s = a * a * (3.0 - 2.0 * a);

    cmd.x = params::APPLE_X - params::RADIUS;
    cmd.y = params::APPLE_Y;
    cmd.z = params::APPLE_Z * s;
    cmd.roll = 0.0;
    cmd.pitch = 0.0;
    cmd.yaw = 0.0;

    return cmd;
  }

  const double tm = t - params::HOVER_SEC;
  const double w = 2.0 * M_PI / params::SCAN_PERIOD_SEC;
  const double yaw_phase = w * tm;

  cmd.x = params::APPLE_X - params::RADIUS * std::cos(yaw_phase);
  cmd.y = params::APPLE_Y + params::RADIUS * std::sin(yaw_phase);
  cmd.z = params::APPLE_Z + params::RADIUS * std::sin(params::THETA_MAX) * std::sin(yaw_phase);

  const double dx = params::APPLE_X - cmd.x;
  const double dy = params::APPLE_Y - cmd.y;

  cmd.roll = 0.0;
  cmd.pitch = -params::THETA_MAX * std::sin(yaw_phase);
  cmd.yaw = std::atan2(dy, dx);

  return cmd;
}

inline TargetCMD takeApple(double t)
{
  TargetCMD cmd;

  if (t < params::HOVER_SEC) {
    const double a = t / params::HOVER_SEC;
    const double s = a * a * (3.0 - 2.0 * a);

    cmd.x = params::APPLE_X - params::RADIUS;
    cmd.y = params::APPLE_Y;
    cmd.z = params::APPLE_Z * s;
    cmd.roll = 0.0;
    cmd.pitch = 0.0;
    cmd.yaw = 0.0;

    return cmd;
  }

  const double tm = t - params::HOVER_SEC;
  const double cycle = params::SCAN_PERIOD_SEC;
  const double tc = std::fmod(tm, cycle);

  const double tilt_sec = 0.25 * cycle;
  const double move_sec = 0.375 * cycle;

  const double hover_x = params::APPLE_X - params::RADIUS;
  const double hover_z = params::APPLE_Z;

  const double theta = params::THETA_MAX;

  const double tilted_x = params::APPLE_X - params::RADIUS * std::cos(theta);
  const double tilted_z = params::APPLE_Z + params::RADIUS * std::sin(theta);

  const double approach_dist = params::RADIUS * 0.5;

  const double near_x = tilted_x + approach_dist * std::cos(theta);
  const double near_z = tilted_z - approach_dist * std::sin(theta);

  cmd.y = params::APPLE_Y;
  cmd.roll = 0.0;
  cmd.yaw = 0.0;

  if (tc < tilt_sec) {
    const double a = tc / tilt_sec;
    const double s = a * a * (3.0 - 2.0 * a);
    const double th = theta * s;

    cmd.x = params::APPLE_X - params::RADIUS * std::cos(th);
    cmd.z = params::APPLE_Z + params::RADIUS * std::sin(th);
    cmd.pitch = -th;

    return cmd;
  }

  if (tc < tilt_sec + move_sec) {
    const double a = (tc - tilt_sec) / move_sec;
    const double s = a * a * (3.0 - 2.0 * a);

    cmd.x = tilted_x + (near_x - tilted_x) * s;
    cmd.z = tilted_z + (near_z - tilted_z) * s;
    cmd.pitch = -theta;

    return cmd;
  }

  {
    const double a = (tc - tilt_sec - move_sec) / move_sec;
    const double s = a * a * (3.0 - 2.0 * a);

    cmd.x = near_x + (tilted_x - near_x) * s;
    cmd.z = near_z + (tilted_z - near_z) * s;
    cmd.pitch = -theta;

    return cmd;
  }
}

inline TargetCMD positionTuningPath(double t)
{
  static constexpr double HOVER_SEC = 3.0;
  static constexpr double SEG_SEC = 1.0;
  static constexpr double XY = 0.5;
  static constexpr double Z = 3.0;

  TargetCMD cmd;

  if (t < HOVER_SEC) {
    const double a = t / HOVER_SEC;
    const double s = a * a * (3.0 - 2.0 * a);

    cmd.x = 0.0;
    cmd.y = 0.0;
    cmd.z = Z * s;
    cmd.roll = 0.0;
    cmd.pitch = 0.0;
    cmd.yaw = 0.0;

    return cmd;
  }

  const double tm = t - HOVER_SEC;
  const int phase = static_cast<int>(std::floor(tm / SEG_SEC)) % 4;

  if (phase == 0) {
    cmd.x = XY;
    cmd.y = XY;
  }
  else if (phase == 1) {
    cmd.x = -XY;
    cmd.y = XY;
  }
  else if (phase == 2) {
    cmd.x = -XY;
    cmd.y = -XY;
  }
  else {
    cmd.x = XY;
    cmd.y = -XY;
  }

  cmd.z = Z;
  cmd.roll = 0.0;
  cmd.pitch = 0.0;
  cmd.yaw = 0.0;

  return cmd;
}

inline TargetCMD attitudeTuningPath(double t)
{
  static constexpr double HOVER_SEC = 3.0;
  static constexpr double TUNE_SEC = 20.0;
  static constexpr double Z = 1.0;
  static constexpr double ROLL_AMP = 20.0 * M_PI / 180.0;
  static constexpr double PITCH_AMP = 70.0 * M_PI / 180.0;
  static constexpr double YAW_AMP = 90.0 * M_PI / 180.0;

  TargetCMD cmd;

  if (t < HOVER_SEC) {
    const double a = t / HOVER_SEC;
    const double s = a * a * (3.0 - 2.0 * a);

    cmd.x = 0.0;
    cmd.y = 0.0;
    cmd.z = Z * s;
    cmd.roll = 0.0;
    cmd.pitch = 0.0;
    cmd.yaw = 0.0;

    return cmd;
  }

  const double tm = std::fmod(t - HOVER_SEC, 3.0 * TUNE_SEC);
  const double axis_t = std::fmod(tm, TUNE_SEC);
  const double w = 2.0 * M_PI / TUNE_SEC;

  cmd.x = 0.0;
  cmd.y = 0.0;
  cmd.z = Z;
  cmd.roll = 0.0;
  cmd.pitch = 0.0;
  cmd.yaw = 0.0;

  if (tm < TUNE_SEC) {
    cmd.pitch = PITCH_AMP * std::sin(w * axis_t);
  } else if (tm < 2.0 * TUNE_SEC) {
    cmd.yaw = YAW_AMP * std::sin(w * axis_t);
  } else {
    cmd.roll = ROLL_AMP * std::sin(w * axis_t);
  }

  return cmd;
}

inline TargetCMD agilePath(double t)
{
  TargetCMD cmd;

  if (t < params::HOVER_SEC) {
    const double a = t / params::HOVER_SEC;
    const double s = a * a * (3.0 - 2.0 * a);

    cmd.x = 0.0;
    cmd.y = 0.0;
    cmd.z = params::APPLE_Z * s;
    cmd.roll = 0.0;
    cmd.pitch = 0.0;
    cmd.yaw = 0.0;

    return cmd;
  }

  const double tm = t - params::HOVER_SEC;
  const double w = 2.0 * M_PI / params::SCAN_PERIOD_SEC;
  const double p = w * tm;

  cmd.x = params::RADIUS * std::sin(p);
  cmd.y = params::RADIUS * std::sin(p) * std::cos(p);
  cmd.z = params::APPLE_Z + params::RADIUS * std::sin(params::THETA_MAX) * std::sin(p);

  cmd.roll = params::ROLL_MAX * std::sin(p);
  cmd.pitch = params::THETA_MAX * std::sin(p + 0.5 * M_PI);
  cmd.yaw = 0.5 * M_PI * std::sin(p);

  return cmd;
}

inline TargetCMD positionTrack(double t)
{
  static constexpr double HOVER_SEC = 1.0;
  static constexpr double Z = 1.0;
  static constexpr double distance_x = 1.0; // [m]
  static constexpr double distance_y = 1.0; // [m]

  static constexpr double vel_x = 1.0; // [m/s]
  static constexpr double vel_y = 1.0; // [m/s]

  const double X_SEG_SEC = distance_x / vel_x;
  const double Y_SEG_SEC = distance_y / vel_y;

  TargetCMD cmd;

  if (t < HOVER_SEC) {
    const double a = t / HOVER_SEC;
    const double s = a * a * (3.0 - 2.0 * a);

    cmd.x = 0.0;
    cmd.y = 0.0;
    cmd.z = Z * s;
    cmd.roll = 0.0;
    cmd.pitch = 60.0 * 3.141592/180.0 * s;
    cmd.yaw = 0.0;

    return cmd;
  }

  const double tm = t - HOVER_SEC;
  const double cycle = 2.0 * X_SEG_SEC + 2.0 * Y_SEG_SEC;
  const double tc = std::fmod(tm, cycle);

  cmd.x = 0.0;
  cmd.y = 0.0;
  cmd.z = Z;
  cmd.roll = 0.0;
  cmd.pitch = 60.0 * 3.141592/180.0;
  cmd.yaw = 0.0;

  if (tc < X_SEG_SEC) {
    const double a = tc / X_SEG_SEC;
    const double s = a * a * (3.0 - 2.0 * a);

    cmd.x = distance_x * s;
    cmd.y = 0.0;

    return cmd;
  }

  if (tc < X_SEG_SEC + Y_SEG_SEC) {
    const double a = (tc - X_SEG_SEC) / Y_SEG_SEC;
    const double s = a * a * (3.0 - 2.0 * a);

    cmd.x = distance_x;
    cmd.y = distance_y * s;

    return cmd;
  }

  if (tc < 2.0 * X_SEG_SEC + Y_SEG_SEC) {
    const double a = (tc - X_SEG_SEC - Y_SEG_SEC) / X_SEG_SEC;
    const double s = a * a * (3.0 - 2.0 * a);

    cmd.x = distance_x * (1.0 - s);
    cmd.y = distance_y;

    return cmd;
  }

  {
    const double a = (tc - 2.0 * X_SEG_SEC - Y_SEG_SEC) / Y_SEG_SEC;
    const double s = a * a * (3.0 - 2.0 * a);

    cmd.x = 0.0;
    cmd.y = distance_y * (1.0 - s);

    return cmd;
  }
}

inline TargetCMD steppedAttitudePath(double t)
{
  static constexpr double HOVER_SEC = 2.0;
  static constexpr double ZERO_HOLD_SEC = 2.0;
  static constexpr double RAMP_SEC = 1.0;
  static constexpr double HOLD_SEC = 3.0;
  static constexpr double Z = 1.0;

  static constexpr double SWITCH_DEG = 60.0;
  static constexpr double FIRST_STEP_DEG = 10.0;
  static constexpr double SECOND_STEP_DEG = 1.0;
  static constexpr double MAX_DEG = 90.0;
  static constexpr double DEG2RAD = M_PI / 180.0;

  static constexpr int FIRST_STAGE_COUNT =
      static_cast<int>(SWITCH_DEG / FIRST_STEP_DEG);

  TargetCMD cmd;

  cmd.x = 0.0;
  cmd.y = 0.0;
  cmd.z = Z;
  cmd.roll = 0.0;
  cmd.pitch = 0.0;
  cmd.yaw = 0.0;

  if (t < HOVER_SEC) {
    const double a = t / HOVER_SEC;
    const double s = a * a * (3.0 - 2.0 * a);

    cmd.z = Z * s;

    return cmd;
  }

  const double tm = t - HOVER_SEC;

  if (tm < ZERO_HOLD_SEC) {
    return cmd;
  }

  const double ts = tm - ZERO_HOLD_SEC;
  const double stage_sec = RAMP_SEC + HOLD_SEC;
  const int stage = static_cast<int>(std::floor(ts / stage_sec));
  const double stage_t = std::fmod(ts, stage_sec);

  double start_deg;
  double target_deg;

  if (stage < FIRST_STAGE_COUNT) {
    start_deg = stage * FIRST_STEP_DEG;
    target_deg = start_deg + FIRST_STEP_DEG;
  }
  else {
    const int second_stage = stage - FIRST_STAGE_COUNT;

    start_deg =
        SWITCH_DEG + second_stage * SECOND_STEP_DEG;

    target_deg =
        start_deg + SECOND_STEP_DEG;
  }

  start_deg = std::min(start_deg, MAX_DEG);
  target_deg = std::min(target_deg, MAX_DEG);

  const double start_angle = start_deg * DEG2RAD;
  const double target_angle = target_deg * DEG2RAD;

  if (start_deg >= MAX_DEG) {
    cmd.pitch = MAX_DEG * DEG2RAD;
    return cmd;
  }

  if (stage_t < RAMP_SEC) {
    const double a = stage_t / RAMP_SEC;
    const double s = a * a * (3.0 - 2.0 * a);

    cmd.pitch =
        start_angle + (target_angle - start_angle) * s;
  }
  else {
    cmd.pitch = target_angle;
  }

  return cmd;
}

// Control Allocation utils ===========================================
inline AllocationOutput allocation_P2T2(const Eigen::Vector3d& moment_cmd, const Eigen::Vector3d& force_cmd, const Eigen::Vector4d& theta_measured, const Eigen::Vector4d& phi_measured)
{
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Vector8d = Eigen::Matrix<double, 8, 1>;
  using Matrix68d = Eigen::Matrix<double, 6, 8>;

  const double theta_front = meanAngle(theta_measured(0), theta_measured(3));
  const double theta_back = meanAngle(theta_measured(1), theta_measured(2));

  const double phi14 = 0.5 * (phi_measured(0) + phi_measured(3));
  const double phi23 = 0.5 * (phi_measured(1) + phi_measured(2));

  const double front_ex = -std::sin(theta_front) * std::cos(phi14);
  const double front_ey = std::sin(phi14);
  const double front_ez = -std::cos(theta_front) * std::cos(phi14);

  const double back_ex = -std::sin(theta_back) * std::cos(phi23);
  const double back_ey = std::sin(phi23);
  const double back_ez = -std::cos(theta_back) * std::cos(phi23);

  const double front_x = params::Lx - params::d * std::cos(phi14) * std::sin(theta_front);
  const double front_y = params::d * std::sin(phi14);
  const double front_z = -params::d * std::cos(phi14) * std::cos(theta_front);

  const double back_x = -params::Lx - params::d * std::cos(phi23) * std::sin(theta_back);
  const double back_y = params::d * std::sin(phi23);
  const double back_z = -params::d * std::cos(phi23) * std::cos(theta_back);

  Matrix68d A;
  A.setZero();

  A(0, 1) = -front_z;
  A(0, 2) = front_y;
  A(0, 4) = -back_z;
  A(0, 5) = back_y;
  A(0, 6) = params::Ly * front_ez - params::zeta * front_ex;
  A(0, 7) = params::Ly * back_ez + params::zeta * back_ex;

  A(1, 0) = front_z;
  A(1, 2) = -front_x;
  A(1, 3) = back_z;
  A(1, 5) = -back_x;
  A(1, 6) = -params::zeta * front_ey;
  A(1, 7) = params::zeta * back_ey;

  A(2, 0) = -front_y;
  A(2, 1) = front_x;
  A(2, 3) = -back_y;
  A(2, 4) = back_x;
  A(2, 6) = -params::Ly * front_ex - params::zeta * front_ez;
  A(2, 7) = -params::Ly * back_ex + params::zeta * back_ez;

  A(3, 0) = 1.0;
  A(3, 3) = 1.0;

  A(4, 1) = 1.0;
  A(4, 4) = 1.0;

  A(5, 2) = 1.0;
  A(5, 5) = 1.0;

  Vector6d wrench;
  wrench << moment_cmd(0), moment_cmd(1), moment_cmd(2), force_cmd(0), force_cmd(1), force_cmd(2);

  const Eigen::Matrix<double, 6, 6> H = A * A.transpose() + params::virtual_lambda * params::virtual_lambda * Eigen::Matrix<double, 6, 6>::Identity();
  const Vector8d virtual_force = A.transpose() * H.ldlt().solve(wrench);

  Eigen::Vector3d front_force;
  Eigen::Vector3d back_force;

  front_force << virtual_force(0), virtual_force(1), virtual_force(2);
  back_force << virtual_force(3), virtual_force(4), virtual_force(5);

  const double f_front = std::max(front_force.norm(), params::f_min);
  const double f_back = std::max(back_force.norm(), params::f_min);

  Eigen::Vector3d front_dir = front_force / f_front;
  Eigen::Vector3d back_dir = back_force / f_back;

  if (front_force.norm() <= params::f_min) front_dir << 0.0, 0.0, -1.0;
  if (back_force.norm() <= params::f_min) back_dir << 0.0, 0.0, -1.0;

  double theta1 = std::atan2(-front_dir(0), -front_dir(2));
  double theta2 = std::atan2(-back_dir(0), -back_dir(2));
  double phi14_cmd = std::asin(std::clamp(front_dir(1), -1.0, 1.0));
  double phi23_cmd = std::asin(std::clamp(back_dir(1), -1.0, 1.0));

  theta1 = std::clamp(theta1, -params::theta_limit_rad, params::theta_limit_rad);
  theta2 = std::clamp(theta2, -params::theta_limit_rad, params::theta_limit_rad);
  phi14_cmd = std::clamp(phi14_cmd, -params::phi_limit_rad, params::phi_limit_rad);
  phi23_cmd = std::clamp(phi23_cmd, -params::phi_limit_rad, params::phi_limit_rad);

  const double df14 = std::clamp(virtual_force(6), -f_front + params::f_min, f_front - params::f_min);
  const double df23 = std::clamp(virtual_force(7), -f_back + params::f_min, f_back - params::f_min);

  AllocationOutput out;

  out.f(0) = std::clamp(0.5 * (f_front + df14), params::f_cmd_min, params::f_cmd_max);
  out.f(3) = std::clamp(0.5 * (f_front - df14), params::f_cmd_min, params::f_cmd_max);
  out.f(1) = std::clamp(0.5 * (f_back + df23), params::f_cmd_min, params::f_cmd_max);
  out.f(2) = std::clamp(0.5 * (f_back - df23), params::f_cmd_min, params::f_cmd_max);

  out.theta(0) = theta1;
  out.theta(1) = theta2;
  out.theta(2) = theta2;
  out.theta(3) = theta1;

  out.phi(0) = phi14_cmd;
  out.phi(1) = phi23_cmd;
  out.phi(2) = phi23_cmd;
  out.phi(3) = phi14_cmd;

  return out;
}


inline AllocationOutput allocation_P2T2_renewal(const Eigen::Vector3d& moment_cmd, const Eigen::Vector3d& force_cmd, const Eigen::Vector4d& theta_measured, const Eigen::Vector4d& phi_measured)
{
  (void)theta_measured;
  (void)phi_measured;

  const double force_cmd_norm = force_cmd.norm();
  const double force_norm = std::max(force_cmd_norm, params::f_min);

  Eigen::Vector3d common_dir;
  if (force_cmd_norm > params::f_min) {
    common_dir = force_cmd / force_cmd_norm;
  }
  else {
    common_dir << 0.0, 0.0, -1.0;
  }

  double theta_common = std::atan2(-common_dir(0), -common_dir(2));
  double phi_common = std::asin(std::clamp(common_dir(1), -1.0, 1.0));

  theta_common = std::clamp(theta_common, -params::theta_limit_rad, params::theta_limit_rad);
  phi_common = std::clamp(phi_common, -params::phi_limit_rad, params::phi_limit_rad);

  Eigen::Vector3d e_cmd;
  e_cmd << -std::sin(theta_common) * std::cos(phi_common),
            std::sin(phi_common),
           -std::cos(theta_common) * std::cos(phi_common);

  constexpr std::array<double, 4> x_signs = {1.0, -1.0, -1.0, 1.0};
  constexpr std::array<double, 4> y_signs = {1.0, 1.0, -1.0, -1.0};
  constexpr std::array<double, 4> spins = {1.0, -1.0, 1.0, -1.0};

  Eigen::Matrix4d B;
  B.setZero();

  for (int i = 0; i < 4; ++i) {
    Eigen::Vector3d r_i;
    r_i << x_signs[i] * params::Lx, y_signs[i] * params::Ly, 0.0;

    const Eigen::Vector3d moment_per_newton =
      r_i.cross(e_cmd) - spins[i] * params::zeta * e_cmd;

    B(0, i) = moment_per_newton(0);
    B(1, i) = moment_per_newton(1);
    B(2, i) = moment_per_newton(2);
    B(3, i) = 1.0;
  }

  Eigen::Vector4d target;
  target << moment_cmd(0), moment_cmd(1), moment_cmd(2), force_norm;

  const Eigen::Matrix4d H =
    B * B.transpose() +
    params::virtual_lambda * params::virtual_lambda * Eigen::Matrix4d::Identity();

  const Eigen::Vector4d thrust_raw = B.transpose() * H.ldlt().solve(target);

  AllocationOutput out;
  for (int i = 0; i < 4; ++i) {
    out.f(i) = std::clamp(thrust_raw(i), params::f_cmd_min, params::f_cmd_max);
  }

  out.theta.setConstant(theta_common);
  out.phi.setConstant(phi_common);

  return out;
}

inline AllocationOutput allocation_P4T4(const Eigen::Vector3d& moment_cmd, const Eigen::Vector3d& force_cmd, const Eigen::Vector4d& theta_measured, const Eigen::Vector4d& phi_measured)
{
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Vector12d = Eigen::Matrix<double, 12, 1>;
  using Matrix612d = Eigen::Matrix<double, 6, 12>;

  Matrix612d A;
  A.setZero();

  for (int i = 0; i < 4; ++i) {
    const double x_sign = (i == 0 || i == 3) ? 1.0 : -1.0;
    const double y_sign = (i == 0 || i == 1) ? 1.0 : -1.0;
    const double spin = (i % 2 == 0) ? 1.0 : -1.0;

    const double theta = theta_measured(i);
    const double phi = phi_measured(i);

    Eigen::Vector3d r;
    r << x_sign * params::Lx - params::d * std::cos(phi) * std::sin(theta), y_sign * params::Ly + params::d * std::sin(phi), -params::d * std::cos(phi) * std::cos(theta);

    Eigen::Matrix3d B;
    B << 0.0, -r(2), r(1),
         r(2), 0.0, -r(0),
         -r(1), r(0), 0.0;

    B -= spin * params::zeta * Eigen::Matrix3d::Identity();

    A.block<3, 3>(0, 3 * i) = B;
    A.block<3, 3>(3, 3 * i) = Eigen::Matrix3d::Identity();
  }

  Vector6d wrench;
  wrench << moment_cmd(0), moment_cmd(1), moment_cmd(2), force_cmd(0), force_cmd(1), force_cmd(2);

  const Eigen::Matrix<double, 6, 6> H = A * A.transpose() + params::virtual_lambda * params::virtual_lambda * Eigen::Matrix<double, 6, 6>::Identity();
  const Vector12d virtual_force = A.transpose() * H.ldlt().solve(wrench);

  AllocationOutput out;

  for (int i = 0; i < 4; ++i) {
    Eigen::Vector3d force_i;
    force_i << virtual_force(3 * i + 0), virtual_force(3 * i + 1), virtual_force(3 * i + 2);

    const double f_raw = force_i.norm();
    const double f = std::max(f_raw, params::f_min);

    Eigen::Vector3d dir = force_i / f;
    if (f_raw <= params::f_min) dir << 0.0, 0.0, -1.0;

    double theta = std::atan2(-dir(0), -dir(2));
    double phi = std::asin(std::clamp(dir(1), -1.0, 1.0));

    theta = std::clamp(theta, -params::theta_limit_rad, params::theta_limit_rad);
    phi = std::clamp(phi, -params::phi_limit_rad, params::phi_limit_rad);

    out.f(i) = std::clamp(f, params::f_cmd_min, params::f_cmd_max);
    out.theta(i) = theta;
    out.phi(i) = phi;
  }


  return out;
}

inline AllocationOutput allocation_P2T2_ADA(const Eigen::Vector3d& moment_cmd, const Eigen::Vector3d& force_cmd, const Eigen::Vector3d& att_cmd, const Eigen::Vector4d& theta_measured, const Eigen::Vector4d& phi_measured)
{
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Vector8d = Eigen::Matrix<double, 8, 1>;
  using Matrix68d = Eigen::Matrix<double, 6, 8>;
  using Matrix86d = Eigen::Matrix<double, 8, 6>;
  using Matrix88d = Eigen::Matrix<double, 8, 8>;
  using Matrix66d = Eigen::Matrix<double, 6, 6>;

  // Current system has theta/phi feedback, but no thrust feedback.
  // Therefore f_meas ~= f_cmd_prev.
  static bool initialized = false;
  static AllocationOutput prev_cmd;

  // P2T2 measured pair angles
  // phi_f=phi1=phi4, phi_b=phi2=phi3, theta_f=theta1=theta4, theta_b=theta2=theta3
  const double theta_f_meas = meanAngle(theta_measured(0), theta_measured(3));
  const double theta_b_meas = meanAngle(theta_measured(1), theta_measured(2));
  const double phi_f_meas = 0.5 * (phi_measured(0) + phi_measured(3));
  const double phi_b_meas = 0.5 * (phi_measured(1) + phi_measured(2));

  if (!initialized) {
    const double f_init = std::clamp(0.25 * force_cmd.norm(), params::f_cmd_min, params::f_cmd_max);

    prev_cmd.f << f_init, f_init, f_init, f_init;
    prev_cmd.theta << theta_f_meas, theta_b_meas, theta_b_meas, theta_f_meas;
    prev_cmd.phi << phi_f_meas, phi_b_meas, phi_b_meas, phi_f_meas;

    initialized = true;
  }

  // q_meas = [f1 f2 f3 f4 phi_f phi_b theta_f theta_b]^T
  Vector8d q_meas;
  q_meas << prev_cmd.f(0), prev_cmd.f(1), prev_cmd.f(2), prev_cmd.f(3),
            phi_f_meas, phi_b_meas, theta_f_meas, theta_b_meas;

  const double f1 = q_meas(0);
  const double f2 = q_meas(1);
  const double f3 = q_meas(2);
  const double f4 = q_meas(3);
  const double phi_f = q_meas(4);
  const double phi_b = q_meas(5);
  const double theta_f = q_meas(6);
  const double theta_b = q_meas(7);

  // e(theta,phi)=[-sin(theta)cos(phi), sin(phi), -cos(theta)cos(phi)]^T
  const double efx = -std::sin(theta_f) * std::cos(phi_f);
  const double efy =  std::sin(phi_f);
  const double efz = -std::cos(theta_f) * std::cos(phi_f);

  const double ebx = -std::sin(theta_b) * std::cos(phi_b);
  const double eby =  std::sin(phi_b);
  const double ebz = -std::cos(theta_b) * std::cos(phi_b);

  // A(q) is frozen at current q_meas.
  const double x_f = params::Lx - params::d * std::cos(phi_f) * std::sin(theta_f);
  const double y_f = params::d * std::sin(phi_f);
  const double z_f = -params::d * std::cos(phi_f) * std::cos(theta_f);
  const double x_b = -params::Lx - params::d * std::cos(phi_b) * std::sin(theta_b);
  const double y_b = params::d * std::sin(phi_b);
  const double z_b = -params::d * std::cos(phi_b) * std::cos(theta_b);

  // w=A(q)u, w=[Mx My Mz Fx Fy Fz]^T, u=[Ffx Ffy Ffz Fbx Fby Fbz df14 df23]^T
  Matrix68d A;
  A.setZero();

  A(0, 1) = -z_f;
  A(0, 2) = y_f;
  A(0, 4) = -z_b;
  A(0, 5) = y_b;
  A(0, 6) = params::Ly * efz - params::zeta * efx;
  A(0, 7) = params::Ly * ebz + params::zeta * ebx;

  A(1, 0) = z_f;
  A(1, 2) = -x_f;
  A(1, 3) = z_b;
  A(1, 5) = -x_b;
  A(1, 6) = -params::zeta * efy;
  A(1, 7) = params::zeta * eby;

  A(2, 0) = -y_f;
  A(2, 1) = x_f;
  A(2, 3) = -y_b;
  A(2, 4) = x_b;
  A(2, 6) = -params::Ly * efx - params::zeta * efz;
  A(2, 7) = -params::Ly * ebx + params::zeta * ebz;

  A(3, 0) = 1.0;
  A(3, 3) = 1.0;

  A(4, 1) = 1.0;
  A(4, 4) = 1.0;

  A(5, 2) = 1.0;
  A(5, 5) = 1.0;

  // u(q)=[(f1+f4)efx (f1+f4)efy (f1+f4)efz (f2+f3)ebx (f2+f3)eby (f2+f3)ebz f1-f4 f2-f3]^T
  const double s_f = f1 + f4;
  const double s_b = f2 + f3;

  Vector8d u_meas;
  u_meas << s_f * efx, s_f * efy, s_f * efz, s_b * ebx, s_b * eby, s_b * ebz, f1 - f4, f2 - f3;

  // w_now = A(q_meas)u(q_meas)
  const Vector6d w_now = A * u_meas;

  // w_d = [moment_cmd force_cmd]^T
  Vector6d w_d;
  w_d << moment_cmd(0), moment_cmd(1), moment_cmd(2), force_cmd(0), force_cmd(1), force_cmd(2);

  // ADA: w_dot_des = k_j(w_d - w_now)
  const Vector6d w_dot_des = params::ada_k_j * (w_d - w_now);

  // D(q)=du/dq, row:u=[Ffx Ffy Ffz Fbx Fby Fbz df14 df23], col:q=[f1 f2 f3 f4 phi_f phi_b theta_f theta_b]
  Matrix88d D;
  D.setZero();

  D(0, 0) = efx;
  D(0, 3) = efx;
  D(0, 4) = s_f * std::sin(theta_f) * std::sin(phi_f);
  D(0, 6) = -s_f * std::cos(theta_f) * std::cos(phi_f);

  D(1, 0) = efy;
  D(1, 3) = efy;
  D(1, 4) = s_f * std::cos(phi_f);

  D(2, 0) = efz;
  D(2, 3) = efz;
  D(2, 4) = s_f * std::cos(theta_f) * std::sin(phi_f);
  D(2, 6) = s_f * std::sin(theta_f) * std::cos(phi_f);

  D(3, 1) = ebx;
  D(3, 2) = ebx;
  D(3, 5) = s_b * std::sin(theta_b) * std::sin(phi_b);
  D(3, 7) = -s_b * std::cos(theta_b) * std::cos(phi_b);

  D(4, 1) = eby;
  D(4, 2) = eby;
  D(4, 5) = s_b * std::cos(phi_b);

  D(5, 1) = ebz;
  D(5, 2) = ebz;
  D(5, 5) = s_b * std::cos(theta_b) * std::sin(phi_b);
  D(5, 7) = s_b * std::sin(theta_b) * std::cos(phi_b);

  D(6, 0) = 1.0;
  D(6, 3) = -1.0;

  D(7, 1) = 1.0;
  D(7, 2) = -1.0;

  // J(q)=A(q)D(q), w_dot=J(q)q_dot
  const Matrix68d J = A * D;

  // J_dagger = W^{-1}J^T(JW^{-1}J^T + lambda_J^2 I_6)^{-1}
  Vector8d q_dot_max;
  q_dot_max << params::ada_f_dot_max, params::ada_f_dot_max, params::ada_f_dot_max, params::ada_f_dot_max,
               params::ada_phi_dot_max, params::ada_phi_dot_max, params::ada_theta_dot_max, params::ada_theta_dot_max;

  Matrix88d W_inv;
  W_inv.setZero();

  for (int i = 0; i < 8; ++i) {
    W_inv(i, i) = q_dot_max(i) * q_dot_max(i);
  }

  const Matrix66d H = J * W_inv * J.transpose() + params::ada_lambda_j * params::ada_lambda_j * Matrix66d::Identity();
  const Matrix66d H_inv = H.ldlt().solve(Matrix66d::Identity());
  const Matrix86d J_dagger = W_inv * J.transpose() * H_inv;

  // q_dot_star = secondary objective, projected by N_J=(I_8-J_dagger J)
  Vector8d q_dot_star;
  q_dot_star.setZero();

  const double ds = s_f - s_b;

  q_dot_star(0) += -params::ada_k_star_f * ds;
  q_dot_star(1) +=  params::ada_k_star_f * ds;
  q_dot_star(2) +=  params::ada_k_star_f * ds;
  q_dot_star(3) += -params::ada_k_star_f * ds;

  const double phi_nav = std::clamp(att_cmd(0), -params::phi_limit_rad, params::phi_limit_rad);
  const double theta_nav = std::clamp(att_cmd(1), -params::theta_limit_rad, params::theta_limit_rad);

  q_dot_star(4) += -params::ada_k_star_phi * (phi_f - phi_nav);
  q_dot_star(5) += -params::ada_k_star_phi * (phi_b - phi_nav);
  q_dot_star(6) += -params::ada_k_star_theta * (theta_f - theta_nav);
  q_dot_star(7) += -params::ada_k_star_theta * (theta_b - theta_nav);

  const double dtheta_fb = theta_f - theta_b;

  q_dot_star(6) += -params::ada_k_star_theta_diff * dtheta_fb;
  q_dot_star(7) +=  params::ada_k_star_theta_diff * dtheta_fb;

  // q_dot_ADA = J_dagger w_dot_des + (I_8 - J_dagger J) q_dot_star
  const Matrix88d N_J = Matrix88d::Identity() - J_dagger * J;
  const Vector8d q_dot_ADA = J_dagger * w_dot_des + N_J * q_dot_star;

  // First-order inverse: q_dot=K_act(q_cmd-q_meas), K_act=diag(1/tau_i), q_cmd=q_meas+K_act^{-1}q_dot_ADA
  Vector8d tau_act;
  tau_act << params::ada_tau_act_f, params::ada_tau_act_f, params::ada_tau_act_f, params::ada_tau_act_f,
             params::ada_tau_act_phi, params::ada_tau_act_phi, params::ada_tau_act_theta, params::ada_tau_act_theta;

  Matrix88d K_act_inv;
  K_act_inv.setZero();

  for (int i = 0; i < 8; ++i) {
    K_act_inv(i, i) = std::max(tau_act(i), 1.0e-6);
  }

  const Vector8d q_cmd = q_meas + K_act_inv * q_dot_ADA;

  // q_cmd projection to physical actuator bounds, not NDA saturation
  const double f1_cmd = std::clamp(q_cmd(0), params::f_cmd_min, params::f_cmd_max);
  const double f2_cmd = std::clamp(q_cmd(1), params::f_cmd_min, params::f_cmd_max);
  const double f3_cmd = std::clamp(q_cmd(2), params::f_cmd_min, params::f_cmd_max);
  const double f4_cmd = std::clamp(q_cmd(3), params::f_cmd_min, params::f_cmd_max);

  const double phi_f_cmd = std::clamp(q_cmd(4), -params::phi_limit_rad, params::phi_limit_rad);
  const double phi_b_cmd = std::clamp(q_cmd(5), -params::phi_limit_rad, params::phi_limit_rad);
  const double theta_f_cmd = std::clamp(q_cmd(6), -params::theta_limit_rad, params::theta_limit_rad);
  const double theta_b_cmd = std::clamp(q_cmd(7), -params::theta_limit_rad, params::theta_limit_rad);

  // q_cmd=[f1 f2 f3 f4 phi_f phi_b theta_f theta_b]^T -> plant input f=[f1 f2 f3 f4]^T, theta=[theta_f theta_b theta_b theta_f]^T, phi=[phi_f phi_b phi_b phi_f]^T
  AllocationOutput out;

  out.f << f1_cmd, f2_cmd, f3_cmd, f4_cmd;
  out.theta << theta_f_cmd, theta_b_cmd, theta_b_cmd, theta_f_cmd;
  out.phi << phi_f_cmd, phi_b_cmd, phi_b_cmd, phi_f_cmd;

  prev_cmd = out;

  return out;
}


inline AllocationOutput allocation_P2T2_ADA_renewal(const Eigen::Vector3d& moment_cmd, const Eigen::Vector3d& force_cmd, const Eigen::Vector3d& att_cmd, const Eigen::Vector4d& theta_measured, const Eigen::Vector4d& phi_measured)
{
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix66d = Eigen::Matrix<double, 6, 6>;

  static bool initialized = false;
  static AllocationOutput prev_cmd;

  const double theta_front_meas = meanAngle(theta_measured(0), theta_measured(3));
  const double theta_back_meas = meanAngle(theta_measured(1), theta_measured(2));
  const double theta_common_meas = meanAngle(theta_front_meas, theta_back_meas);
  const double phi_common_meas = 0.25 * phi_measured.sum();

  if (!initialized) {
    const double f_init = std::clamp(0.25 * force_cmd.norm(), params::f_cmd_min, params::f_cmd_max);

    prev_cmd.f << f_init, f_init, f_init, f_init;
    prev_cmd.theta.setConstant(theta_common_meas);
    prev_cmd.phi.setConstant(phi_common_meas);

    initialized = true;
  }

  const double f1 = prev_cmd.f(0);
  const double f2 = prev_cmd.f(1);
  const double f3 = prev_cmd.f(2);
  const double f4 = prev_cmd.f(3);
  const double phi = std::clamp(phi_common_meas, -params::phi_limit_rad, params::phi_limit_rad);
  const double theta = std::clamp(theta_common_meas, -params::theta_limit_rad, params::theta_limit_rad);

  const double s_total = f1 + f2 + f3 + f4;

  Eigen::Vector3d e;
  e << -std::sin(theta) * std::cos(phi),
        std::sin(phi),
       -std::cos(theta) * std::cos(phi);

  Eigen::Vector3d de_dphi;
  de_dphi << std::sin(theta) * std::sin(phi),
             std::cos(phi),
             std::cos(theta) * std::sin(phi);

  Eigen::Vector3d de_dtheta;
  de_dtheta << -std::cos(theta) * std::cos(phi),
                0.0,
                std::sin(theta) * std::cos(phi);

  constexpr std::array<double, 4> x_signs = {1.0, -1.0, -1.0, 1.0};
  constexpr std::array<double, 4> y_signs = {1.0, 1.0, -1.0, -1.0};
  constexpr std::array<double, 4> spins = {1.0, -1.0, 1.0, -1.0};

  const Eigen::Vector4d f_vec(f1, f2, f3, f4);

  Vector6d w_now;
  w_now.setZero();

  Matrix66d J;
  J.setZero();

  Eigen::Vector3d dm_dphi_sum = Eigen::Vector3d::Zero();
  Eigen::Vector3d dm_dtheta_sum = Eigen::Vector3d::Zero();

  for (int i = 0; i < 4; ++i) {
    Eigen::Vector3d r_i;
    r_i << x_signs[i] * params::Lx, y_signs[i] * params::Ly, 0.0;

    const Eigen::Vector3d moment_per_newton = r_i.cross(e) - spins[i] * params::zeta * e;
    const Eigen::Vector3d dm_dphi_per_newton = r_i.cross(de_dphi) - spins[i] * params::zeta * de_dphi;
    const Eigen::Vector3d dm_dtheta_per_newton = r_i.cross(de_dtheta) - spins[i] * params::zeta * de_dtheta;

    w_now.segment<3>(0) += f_vec(i) * moment_per_newton;
    w_now.segment<3>(3) += f_vec(i) * e;

    J.block<3, 1>(0, i) = moment_per_newton;
    J.block<3, 1>(3, i) = e;

    dm_dphi_sum += f_vec(i) * dm_dphi_per_newton;
    dm_dtheta_sum += f_vec(i) * dm_dtheta_per_newton;
  }

  J.block<3, 1>(0, 4) = dm_dphi_sum;
  J.block<3, 1>(3, 4) = s_total * de_dphi;
  J.block<3, 1>(0, 5) = dm_dtheta_sum;
  J.block<3, 1>(3, 5) = s_total * de_dtheta;

  Vector6d w_d;
  w_d << moment_cmd(0), moment_cmd(1), moment_cmd(2), force_cmd(0), force_cmd(1), force_cmd(2);

  const Vector6d w_dot_des = params::ada_k_j * (w_d - w_now);

  Vector6d q_dot_max;
  q_dot_max << params::ada_f_dot_max,
               params::ada_f_dot_max,
               params::ada_f_dot_max,
               params::ada_f_dot_max,
               params::ada_phi_dot_max,
               params::ada_theta_dot_max;

  Matrix66d W_inv;
  W_inv.setZero();

  for (int i = 0; i < 6; ++i) {
    W_inv(i, i) = q_dot_max(i) * q_dot_max(i);
  }

  const Matrix66d H = J * W_inv * J.transpose() + params::ada_lambda_j * params::ada_lambda_j * Matrix66d::Identity();
  const Matrix66d H_inv = H.ldlt().solve(Matrix66d::Identity());
  const Matrix66d J_dagger = W_inv * J.transpose() * H_inv;

  Vector6d q_dot_star;
  q_dot_star.setZero();

  const double f_mean = 0.25 * s_total;
  q_dot_star(0) += -params::ada_k_star_f * (f1 - f_mean);
  q_dot_star(1) += -params::ada_k_star_f * (f2 - f_mean);
  q_dot_star(2) += -params::ada_k_star_f * (f3 - f_mean);
  q_dot_star(3) += -params::ada_k_star_f * (f4 - f_mean);

  const double phi_nav = std::clamp(att_cmd(0), -params::phi_limit_rad, params::phi_limit_rad);
  const double theta_nav = std::clamp(att_cmd(1), -params::theta_limit_rad, params::theta_limit_rad);

  q_dot_star(4) += -params::ada_k_star_phi * (phi - phi_nav);
  q_dot_star(5) += -params::ada_k_star_theta * (theta - theta_nav);

  const Matrix66d N_J = Matrix66d::Identity() - J_dagger * J;
  const Vector6d q_dot_ADA = J_dagger * w_dot_des + N_J * q_dot_star;

  Vector6d q_meas;
  q_meas << f1, f2, f3, f4, phi, theta;

  Vector6d tau_act;
  tau_act << params::ada_tau_act_f,
             params::ada_tau_act_f,
             params::ada_tau_act_f,
             params::ada_tau_act_f,
             params::ada_tau_act_phi,
             params::ada_tau_act_theta;

  Matrix66d K_act_inv;
  K_act_inv.setZero();

  for (int i = 0; i < 6; ++i) {
    K_act_inv(i, i) = std::max(tau_act(i), 1.0e-6);
  }

  const Vector6d q_cmd = q_meas + K_act_inv * q_dot_ADA;

  AllocationOutput out;

  out.f(0) = std::clamp(q_cmd(0), params::f_cmd_min, params::f_cmd_max);
  out.f(1) = std::clamp(q_cmd(1), params::f_cmd_min, params::f_cmd_max);
  out.f(2) = std::clamp(q_cmd(2), params::f_cmd_min, params::f_cmd_max);
  out.f(3) = std::clamp(q_cmd(3), params::f_cmd_min, params::f_cmd_max);

  const double phi_cmd = std::clamp(q_cmd(4), -params::phi_limit_rad, params::phi_limit_rad);
  const double theta_cmd = std::clamp(q_cmd(5), -params::theta_limit_rad, params::theta_limit_rad);

  out.theta.setConstant(theta_cmd);
  out.phi.setConstant(phi_cmd);

  prev_cmd = out;

  return out;
}

inline AllocationCheck checkAllocation(const AllocationOutput& alloc, const Eigen::Vector3d& moment_cmd, const Eigen::Vector3d& force_cmd)
{
  AllocationCheck result;

  for (int i = 0; i < 4; ++i) {
    const double x_sign = (i == 0 || i == 3) ? 1.0 : -1.0;
    const double y_sign = (i == 0 || i == 1) ? 1.0 : -1.0;
    const double spin = (i % 2 == 0) ? 1.0 : -1.0;

    const double theta = alloc.theta(i);
    const double phi = alloc.phi(i);

    Eigen::Vector3d e_i;
    e_i << -std::sin(theta) * std::cos(phi), std::sin(phi), -std::cos(theta) * std::cos(phi);

    Eigen::Vector3d r_i;
    r_i << x_sign * params::Lx - params::d * std::cos(phi) * std::sin(theta), y_sign * params::Ly + params::d * std::sin(phi), -params::d * std::cos(phi) * std::cos(theta);

    const Eigen::Vector3d force_i = alloc.f(i) * e_i;
    const Eigen::Vector3d moment_i = r_i.cross(force_i) - spin * params::zeta * alloc.f(i) * e_i;

    result.force_actual += force_i;
    result.moment_actual += moment_i;
  }

  result.moment_error = moment_cmd - result.moment_actual;
  result.force_error = force_cmd - result.force_actual;

  const double err_m_norm = result.moment_error.norm();
  const double err_f_norm = result.force_error.norm();

  result.problem = (!result.moment_actual.allFinite()) || (!result.force_actual.allFinite()) || (err_f_norm > params::check_force_tol) || (err_m_norm > params::check_moment_tol);

  if (result.problem) {
    std::ostringstream ss;
    ss << "\033[31m"
       << "\n[CA CHECK]"
       << "\n  error norm"
       << "\n    err_m_norm = " << err_m_norm << " Nm"
       << "\n    err_f_norm = " << err_f_norm << " N"
       << "\n"
       << "\n  moment [Nm]"
       << "\n    cmd    = " << moment_cmd.transpose()
       << "\n    actual = " << result.moment_actual.transpose()
       << "\n    error  = " << result.moment_error.transpose()
       << "\n"
       << "\n  force [N]"
       << "\n    cmd    = " << force_cmd.transpose()
       << "\n    actual = " << result.force_actual.transpose()
       << "\n    error  = " << result.force_error.transpose()
       << "\n"
       << "\n  actuator"
       << "\n    f     = " << alloc.f.transpose()
       << "\n    theta = " << alloc.theta.transpose()
       << "\n    phi   = " << alloc.phi.transpose()
       << "\033[0m";

    result.message = ss.str();
  }

  return result;
}

}
