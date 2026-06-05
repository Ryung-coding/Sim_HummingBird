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
  static constexpr double PITCH_SEC = 10.0;
  static constexpr double YAW_SEC = 30.0;
  static constexpr double Z = 1.0;
  static constexpr double PITCH_AMP = 70.0 * M_PI / 180.0;
  static constexpr double YAW_AMP = 179.0 * M_PI / 180.0;

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
  const double cycle = PITCH_SEC + YAW_SEC;
  const double tc = std::fmod(tm, cycle);

  cmd.x = 0.0;
  cmd.y = 0.0;
  cmd.z = Z;
  cmd.roll = 0.0;
  cmd.pitch = 0.0;
  cmd.yaw = 0.0;

  if (tc < PITCH_SEC) {
    const double w = 2.0 * M_PI / PITCH_SEC;
    cmd.pitch = PITCH_AMP * std::sin(w * tc);
  }
  else {
    const double ty = tc - PITCH_SEC;
    const double w = 2.0 * M_PI / YAW_SEC;
    cmd.yaw = YAW_AMP * std::sin(w * ty);
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

// Control Allocation utils ===========================================
inline AllocationOutput allocation_P2T2(const Eigen::Vector3d& moment_cmd, const Eigen::Vector3d& force_cmd, const Eigen::Vector4d& theta_measured, const Eigen::Vector4d& phi_measured)
{
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Vector8d = Eigen::Matrix<double, 8, 1>;
  using Matrix68d = Eigen::Matrix<double, 6, 8>;

  const double theta_front = 0.5 * (theta_measured(0) + theta_measured(3));
  const double theta_back = 0.5 * (theta_measured(1) + theta_measured(2));

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

  const double back_x = -params::Lx - params::d * std::cos(phi23) * std::sin(theta_back);
  const double back_y = params::d * std::sin(phi23);

  Matrix68d A;
  A.setZero();

  A(0, 2) = front_y;
  A(0, 5) = back_y;
  A(0, 6) = params::Ly * front_ez + params::zeta * front_ex;
  A(0, 7) = params::Ly * back_ez - params::zeta * back_ex;

  A(1, 2) = -front_x;
  A(1, 5) = -back_x;
  A(1, 6) = params::zeta * front_ey;
  A(1, 7) = -params::zeta * back_ey;

  A(2, 0) = -front_y;
  A(2, 1) = front_x;
  A(2, 3) = -back_y;
  A(2, 4) = back_x;
  A(2, 6) = -params::Ly * front_ex + params::zeta * front_ez;
  A(2, 7) = -params::Ly * back_ex - params::zeta * back_ez;

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
    r << x_sign * params::Lx - params::d * std::cos(phi) * std::sin(theta), y_sign * params::Ly + params::d * std::sin(phi), 0.0;

    Eigen::Matrix3d B;
    B << 0.0, -r(2), r(1),
         r(2), 0.0, -r(0),
         -r(1), r(0), 0.0;

    B += spin * params::zeta * Eigen::Matrix3d::Identity();

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
    r_i << x_sign * params::Lx - params::d * std::cos(phi) * std::sin(theta), y_sign * params::Ly + params::d * std::sin(phi), 0.0;

    const Eigen::Vector3d force_i = alloc.f(i) * e_i;
    const Eigen::Vector3d moment_i = r_i.cross(force_i) + spin * params::zeta * alloc.f(i) * e_i;

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