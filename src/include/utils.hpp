#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <params.hpp>

namespace utils {

// struct.zip =======================================================
struct TargetCMD {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  Eigen::Vector3d vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();
  Eigen::Vector4d quat{1.0, 0.0, 0.0, 0.0};
  Eigen::Vector3d omega = Eigen::Vector3d::Zero();
  Eigen::Vector3d omega_dot = Eigen::Vector3d::Zero();
  bool use_quaternion = false;
};

struct AllocationOutput {

  Eigen::Vector4d f = Eigen::Vector4d::Zero();
  Eigen::Vector2d beta = Eigen::Vector2d::Zero();
  Eigen::Vector4d alpha = Eigen::Vector4d::Zero();

  Eigen::Matrix<double, 6, 1> wrench_cmd = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> wrench_alloc = Eigen::Matrix<double, 6, 1>::Zero();
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

inline Eigen::Matrix<double, 6, 1> evaluateA4B2Wrench(const Eigen::Vector4d& f, const Eigen::Vector2d& beta, const Eigen::Vector4d& alpha)
{
  Eigen::Matrix<double, 6, 1> W = Eigen::Matrix<double, 6, 1>::Zero();

  static constexpr std::array<int, 4> beta_index = {0, 1, 1, 0};
  static constexpr std::array<double, 4> x_sign = {1.0, -1.0, -1.0, 1.0};
  static constexpr std::array<double, 4> y_sign = {1.0, 1.0, -1.0, -1.0};
  static constexpr std::array<double, 4> reaction_sign = {1.0, -1.0, 1.0, -1.0};

  for (int i = 0; i < 4; ++i)
  {
    const double a = alpha(i);
    const double b = beta(beta_index[i]);

    const Eigen::Vector3d r_i(x_sign[i] * params::HB_L, y_sign[i] * params::HB_L, 0.0);
    const Eigen::Vector3d e_i(-std::sin(b) * std::cos(a), std::sin(a), -std::cos(b) * std::cos(a));

    const Eigen::Vector3d force_i = f(i) * e_i;
    const Eigen::Vector3d moment_i = r_i.cross(force_i) + reaction_sign[i] * params::HB_ZETA * force_i;

    W.head<3>() += moment_i;
    W.tail<3>() += force_i;
  }

  return W;
}

// Path utils =========================================================
inline TargetCMD posPath(double t)
{
  static constexpr bool TWO_POINT = false;

  static constexpr double SEG_SEC = 2.0;
  static constexpr double HOLD_SEC = 3.0;
  static constexpr double XY = 1.0;

  TargetCMD cmd;

  const double CYCLE_SEC = SEG_SEC + HOLD_SEC;
  const int phase = static_cast<int>(std::floor(t / CYCLE_SEC)) % (TWO_POINT ? 2 : 4);
  const double local_t = std::fmod(t, CYCLE_SEC);

  const std::array<Eigen::Vector2d, 4> corners =
  {
    Eigen::Vector2d{XY, 0.0},
    Eigen::Vector2d{XY, XY},
    Eigen::Vector2d{0.0, XY},
    Eigen::Vector2d{0.0, 0.0}
  };

  Eigen::Vector2d p0;
  Eigen::Vector2d p1;

  if (TWO_POINT)
  {
    p0 = (phase == 0) ? Eigen::Vector2d{0.0, 0.0} : Eigen::Vector2d{XY, 0.0};
    p1 = (phase == 0) ? Eigen::Vector2d{XY, 0.0} : Eigen::Vector2d{0.0, 0.0};
  }
  else
  {
    p0 = (phase == 0) ? Eigen::Vector2d{0.0, 0.0} : corners[phase - 1];
    p1 = corners[phase];
  }

  double s = 1.0;

  if (local_t < SEG_SEC)
  {
    const double a = local_t / SEG_SEC;
    s = a * a * (3.0 - 2.0 * a);
  }

  cmd.x = p0(0) + (p1(0) - p0(0)) * s;
  cmd.y = p0(1) + (p1(1) - p0(1)) * s;
  cmd.z = 0.0;
  cmd.roll = 0.0;
  cmd.pitch = 0.0;
  cmd.yaw = 0.0;

  return cmd;
}

inline TargetCMD attPath(double t)
{
  static constexpr double TUNE_SEC = 40.0;

  TargetCMD cmd;

  const double tm = std::fmod(t, TUNE_SEC);
  const double axis_t = tm;
  const double w = 2.0 * M_PI / TUNE_SEC;

  cmd.x = 0.0;
  cmd.y = 0.0;
  cmd.z = 0.0;
  cmd.roll = 0.0;
  cmd.pitch = params::PITCH_MAX * std::sin(w * axis_t);
  cmd.yaw = 0.0;

  return cmd;
}

inline TargetCMD stepAttPath(double t)
{
  static constexpr double ZERO_HOLD_SEC = 2.0;
  static constexpr double RAMP_SEC = 3.0;
  static constexpr double HOLD_SEC = 5.0;

  static constexpr double SWITCH_DEG = 60.0;
  static constexpr double FIRST_STEP_DEG = 5.0;
  static constexpr double SECOND_STEP_DEG = 1.0;
  static constexpr double MAX_DEG = 60.0;
  static constexpr double DEG2RAD = M_PI / 180.0;

  static constexpr int FIRST_STAGE_COUNT = static_cast<int>(SWITCH_DEG / FIRST_STEP_DEG);

  TargetCMD cmd;

  cmd.x = 0.0;
  cmd.y = 0.0;
  cmd.z = 0.0;
  cmd.roll = 0.0;
  cmd.pitch = 0.0;
  cmd.yaw = 0.0;

  const double stage_sec = RAMP_SEC + HOLD_SEC;
  const int UP_STAGE_COUNT = FIRST_STAGE_COUNT + static_cast<int>((MAX_DEG - SWITCH_DEG) / SECOND_STEP_DEG);
  const double cycle_sec = ZERO_HOLD_SEC + 2.0 * UP_STAGE_COUNT * stage_sec + ZERO_HOLD_SEC;
  const double cycle_t = std::fmod(t, cycle_sec);

  if (cycle_t < ZERO_HOLD_SEC) return cmd;

  const double ts = cycle_t - ZERO_HOLD_SEC;
  if (ts >= 2.0 * UP_STAGE_COUNT * stage_sec) return cmd;

  const int raw_stage = static_cast<int>(std::floor(ts / stage_sec));
  const double stage_t = std::fmod(ts, stage_sec);
  const bool down = raw_stage >= UP_STAGE_COUNT;
  const int stage = down ? 2 * UP_STAGE_COUNT - raw_stage - 1 : raw_stage;

  double start_deg;
  double target_deg;

  if (stage < FIRST_STAGE_COUNT)
  {
    start_deg = stage * FIRST_STEP_DEG;
    target_deg = start_deg + FIRST_STEP_DEG;
  }
  else
  {
    const int second_stage = stage - FIRST_STAGE_COUNT;

    start_deg = SWITCH_DEG + second_stage * SECOND_STEP_DEG;
    target_deg = start_deg + SECOND_STEP_DEG;
  }

  start_deg = std::min(start_deg, MAX_DEG);
  target_deg = std::min(target_deg, MAX_DEG);

  if (down) std::swap(start_deg, target_deg);

  const double start_angle = start_deg * DEG2RAD;
  const double target_angle = target_deg * DEG2RAD;

  if (stage_t < RAMP_SEC)
  {
    const double a = stage_t / RAMP_SEC;
    const double s = a * a * (3.0 - 2.0 * a);

    cmd.pitch = start_angle + (target_angle - start_angle) * s;
  }
  else cmd.pitch = target_angle;

  return cmd;
}

inline TargetCMD circle_demo(double t)
{
  static constexpr double HOVER_SEC = 3.0;
  static constexpr double PERIOD = 30.0;
  static constexpr double RADIUS = 0.5;
  static constexpr double PITCH_AMP = 0.0 * M_PI / 180.0;

  TargetCMD cmd;

  if (t < HOVER_SEC) {
    cmd.x = 0.0;
    cmd.y = 0.0;
    cmd.z = 0.0;
    cmd.roll = 0.0;
    cmd.pitch = 0.0;
    cmd.yaw = 0.0;

    return cmd;
  }

  const double tm = t - HOVER_SEC;
  const double w = 2.0 * M_PI / PERIOD;

  cmd.x = RADIUS * (std::cos(w * tm) - 1.0);
  cmd.y = RADIUS * std::sin(w * tm);
  cmd.z = 0.0;

  cmd.roll = 0.0;
  cmd.pitch = PITCH_AMP * std::sin(w * tm);
  cmd.yaw = 0.0;

  return cmd;
}

inline TargetCMD demo_straight(double t)
{
  static constexpr double MOVE_SEC = 1.0;
  static constexpr double HOLD_SEC = 5.0;
  static constexpr double DIST = 1.0;
  static constexpr double PERIOD = 2.0 * (MOVE_SEC + HOLD_SEC);

  TargetCMD cmd;

  const double tm = std::fmod(std::max(0.0, t), PERIOD);

  cmd.x = 0.0;
  cmd.y = 0.0;
  cmd.z = 0.0;
  cmd.roll = 0.0;
  cmd.pitch = 0.0;
  cmd.yaw = 0.0;

  if (tm < HOLD_SEC) {
    cmd.y = 0.0;
  }
  else if (tm < HOLD_SEC + MOVE_SEC) {
    const double a = (tm - HOLD_SEC) / MOVE_SEC;
    const double s = a * a * (3.0 - 2.0 * a);
    cmd.y = DIST * s;
  }
  else if (tm < 2.0 * HOLD_SEC + MOVE_SEC) {
    cmd.y = DIST;
  }
  else {
    const double a = (tm - 2.0 * HOLD_SEC - MOVE_SEC) / MOVE_SEC;
    const double s = a * a * (3.0 - 2.0 * a);
    cmd.y = DIST * (1.0 - s);
  }

  return cmd;
}

// Control Allocation utils ===========================================
inline AllocationOutput allocation_a1b1(const Eigen::Vector3d& moment_cmd, const Eigen::Vector3d& force_cmd)
{
  const double force_norm = std::max(force_cmd.norm(), 1.0e-3);
  Eigen::Vector3d force_dir;
  if (force_norm > 1.0e-3) force_dir = force_cmd / force_norm;
  else force_dir << 0.0, 0.0, -1.0;

  double beta_des = std::atan2(-force_dir(0), -force_dir(2));
  double alpha_des = std::asin(std::clamp(force_dir(1), -1.0, 1.0));

  beta_des = std::clamp(beta_des, -params::HB_BETA_LIMIT_RAD, params::HB_BETA_LIMIT_RAD);
  alpha_des = std::clamp(alpha_des, -params::HB_ALPHA_LIMIT_RAD, params::HB_ALPHA_LIMIT_RAD);

  Eigen::Vector3d e_cmd;
  e_cmd << -std::sin(beta_des) * std::cos(alpha_des),
            std::sin(alpha_des),
           -std::cos(beta_des) * std::cos(alpha_des);

  const double L = params::HB_L;
  const double zeta  = params::HB_ZETA;

  Eigen::Matrix4d B;
  Eigen::Vector4d Wrench;

  Wrench << moment_cmd(0), moment_cmd(1), moment_cmd(2), force_norm;

  B <<
                    L * e_cmd(2) + zeta * e_cmd(0),                     L * e_cmd(2) - zeta * e_cmd(0),                     -L * e_cmd(2) + zeta * e_cmd(0),                    -L * e_cmd(2) - zeta * e_cmd(0),

                   -L * e_cmd(2) + zeta * e_cmd(1),                     L * e_cmd(2) - zeta * e_cmd(1),                      L * e_cmd(2) + zeta * e_cmd(1),                    -L * e_cmd(2) - zeta * e_cmd(1),

    L * e_cmd(1) - L * e_cmd(0) + zeta * e_cmd(2),    -L * e_cmd(1) - L * e_cmd(0) - zeta * e_cmd(2),     -L * e_cmd(1) + L * e_cmd(0) + zeta * e_cmd(2),     L * e_cmd(1) + L * e_cmd(0) - zeta * e_cmd(2),

                                                1.0,                                                 1.0,                                                   1.0,                                                1.0;


  const Eigen::Matrix4d H = B * B.transpose() + params::HB_VIRTUAL_LAMBDA * params::HB_VIRTUAL_LAMBDA * Eigen::Matrix4d::Identity();
  const Eigen::Vector4d thrust_raw = B.transpose() * H.ldlt().solve(Wrench);

  AllocationOutput out;

  for (int i = 0; i < 4; ++i) out.f(i) = std::clamp(thrust_raw(i), params::HB_F_CMD_MIN, params::HB_F_CMD_MAX);

  out.beta << beta_des, beta_des;
  out.alpha.setConstant(alpha_des);

  out.wrench_cmd << moment_cmd, force_cmd;
  out.wrench_alloc = evaluateA4B2Wrench(out.f, out.beta, out.alpha);

  return out;
}


}
