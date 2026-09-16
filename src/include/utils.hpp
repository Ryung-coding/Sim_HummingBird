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
  Eigen::Vector2d beta = Eigen::Vector2d::Zero();
  Eigen::Vector4d alpha = Eigen::Vector4d::Zero();
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
inline TargetCMD posPath(double t)
{
  static constexpr double HOVER_SEC = 0.5;
  static constexpr double SEG_SEC = 1.0;
  static constexpr double XY = 0.5;
  static constexpr double Z = 0.5;

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

  const auto smooth = [](double a) {
    a = std::clamp(a, 0.0, 1.0);
    return a * a * a * (a * (a * 6.0 - 15.0) + 10.0);
  };

  const double corners[4][2] = {
    { XY,  XY},
    {-XY,  XY},
    {-XY, -XY},
    { XY, -XY},
  };

  const double tm = t - HOVER_SEC;
  double x0 = 0.0;
  double y0 = 0.0;
  double x1 = corners[0][0];
  double y1 = corners[0][1];
  double seg_t = tm;

  if (tm >= SEG_SEC) {
    const double loop_t = tm - SEG_SEC;
    const int phase = static_cast<int>(std::floor(loop_t / SEG_SEC)) % 4;
    const int next_phase = (phase + 1) % 4;

    x0 = corners[phase][0];
    y0 = corners[phase][1];
    x1 = corners[next_phase][0];
    y1 = corners[next_phase][1];
    seg_t = std::fmod(loop_t, SEG_SEC);
  }

  const double s = smooth(seg_t / SEG_SEC);

  cmd.x = x0 + (x1 - x0) * s;
  cmd.y = y0 + (y1 - y0) * s;
  cmd.z = Z;
  cmd.roll = 0.0;
  cmd.pitch = 0.0;
  cmd.yaw = 0.0;

  return cmd;
}

inline TargetCMD attPath(double t)
{
  static constexpr double HOVER_SEC = 1.0;
  static constexpr double TUNE_SEC = 60.0;
  static constexpr double Z = 1.0;
  static constexpr double ROLL_AMP = 20.0 * M_PI / 180.0;
  static constexpr double PITCH_AMP = 60.0 * M_PI / 180.0;

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
    cmd.roll = ROLL_AMP * std::sin(w * axis_t);
  }

  return cmd;
}

inline TargetCMD stepAttPath(double t)
{
  static constexpr double HOVER_SEC = 2.0;
  static constexpr double ZERO_HOLD_SEC = 2.0;
  static constexpr double RAMP_SEC = 1.0;
  static constexpr double HOLD_SEC = 3.0;
  static constexpr double Z = 1.0;

  static constexpr double SWITCH_DEG = 80.0;
  static constexpr double FIRST_STEP_DEG = 40.0;
  static constexpr double SECOND_STEP_DEG = 10.0;
  static constexpr double MAX_DEG = 89.0;
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

inline TargetCMD throughWallPath(double t)
{
  static constexpr double TAKEOFF_SEC = 1.0;
  static constexpr double STRAIGHT_SEC = 3.0;
  static constexpr double TILT_SEC = 3.0;
  static constexpr double DIAG_SEC = 3.0;
  static constexpr double EXIT_SEC = 2.0;

  static constexpr double Z0 = 1.0;
  static constexpr double X_TURN = 1.5;
  static constexpr double X_DIAG_END = 3.0;
  static constexpr double Z_DIAG_END = 2.5;
  static constexpr double X_END = 4.0;

  static constexpr double PITCH_45 = M_PI / 4.0;

  static constexpr double ONE_WAY_SEC =
      STRAIGHT_SEC + TILT_SEC + DIAG_SEC + TILT_SEC + EXIT_SEC;

  static constexpr double CYCLE_SEC = 2.0 * ONE_WAY_SEC;

  const auto smooth = [](double a) {
    a = std::clamp(a, 0.0, 1.0);
    return a * a * a * (a * (a * 6.0 - 15.0) + 10.0);
  };

  TargetCMD cmd;
  cmd.x = 0.0;
  cmd.y = 0.0;
  cmd.z = 0.0;
  cmd.roll = 0.0;
  cmd.pitch = 0.0;
  cmd.yaw = 0.0;

  // Initial takeoff only once
  if (t < TAKEOFF_SEC) {
    const double s = smooth(t / TAKEOFF_SEC);
    cmd.z = Z0 * s;
    return cmd;
  }

  // Repeat 0 -> 4 -> 0
  double tm = std::fmod(t - TAKEOFF_SEC, CYCLE_SEC);

  // =========================================================
  // Forward : x = 0 -> 4
  // =========================================================
  if (tm < ONE_WAY_SEC) {

    if (tm < STRAIGHT_SEC) {
      const double s = smooth(tm / STRAIGHT_SEC);
      cmd.x = X_TURN * s;
      cmd.z = Z0;
      return cmd;
    }

    tm -= STRAIGHT_SEC;

    if (tm < TILT_SEC) {
      const double s = smooth(tm / TILT_SEC);
      cmd.x = X_TURN;
      cmd.z = Z0;
      cmd.pitch = PITCH_45 * s;
      return cmd;
    }

    tm -= TILT_SEC;

    if (tm < DIAG_SEC) {
      const double s = smooth(tm / DIAG_SEC);
      cmd.x = X_TURN + (X_DIAG_END - X_TURN) * s;
      cmd.z = Z0 + (Z_DIAG_END - Z0) * s;
      cmd.pitch = PITCH_45;
      return cmd;
    }

    tm -= DIAG_SEC;

    if (tm < TILT_SEC) {
      const double s = smooth(tm / TILT_SEC);
      cmd.x = X_DIAG_END;
      cmd.z = Z_DIAG_END;
      cmd.pitch = PITCH_45 * (1.0 - s);
      return cmd;
    }

    tm -= TILT_SEC;

    const double s = smooth(tm / EXIT_SEC);
    cmd.x = X_DIAG_END + (X_END - X_DIAG_END) * s;
    cmd.z = Z_DIAG_END;
    return cmd;
  }

  // =========================================================
  // Return : x = 4 -> 0
  // =========================================================
  tm -= ONE_WAY_SEC;

  // 4.0 -> 3.0
  if (tm < EXIT_SEC) {
    const double s = smooth(tm / EXIT_SEC);
    cmd.x = X_END + (X_DIAG_END - X_END) * s;
    cmd.z = Z_DIAG_END;
    return cmd;
  }

  tm -= EXIT_SEC;

  // At x = 3.0, tilt 0 -> 45 deg
  if (tm < TILT_SEC) {
    const double s = smooth(tm / TILT_SEC);
    cmd.x = X_DIAG_END;
    cmd.z = Z_DIAG_END;
    cmd.pitch = PITCH_45 * s;
    return cmd;
  }

  tm -= TILT_SEC;

  // Diagonal backward
  if (tm < DIAG_SEC) {
    const double s = smooth(tm / DIAG_SEC);
    cmd.x = X_DIAG_END + (X_TURN - X_DIAG_END) * s;
    cmd.z = Z_DIAG_END + (Z0 - Z_DIAG_END) * s;
    cmd.pitch = PITCH_45;
    return cmd;
  }

  tm -= DIAG_SEC;

  // At x = 1.5, tilt 45 -> 0 deg
  if (tm < TILT_SEC) {
    const double s = smooth(tm / TILT_SEC);
    cmd.x = X_TURN;
    cmd.z = Z0;
    cmd.pitch = PITCH_45 * (1.0 - s);
    return cmd;
  }

  tm -= TILT_SEC;

  // 1.5 -> 0.0
  const double s = smooth(tm / STRAIGHT_SEC);
  cmd.x = X_TURN * (1.0 - s);
  cmd.z = Z0;

  return cmd;
}

inline TargetCMD circularWallPath(double t)
{
  static constexpr double R = 2.5;
  static constexpr double Z = 1.2;

  static constexpr double ORIENT_SEC = 2.0;
  static constexpr double RAMP_SEC = 3.0;
  static constexpr double LAP_SEC = 14.0;

  static constexpr double BANK_RAD = 35.0 * M_PI / 180.0;
  static constexpr double OMEGA = 2.0 * M_PI / LAP_SEC;

  const auto smooth = [](double a) {
    a = std::clamp(a, 0.0, 1.0);
    return a * a * a * (a * (a * 6.0 - 15.0) + 10.0);
  };

  TargetCMD cmd;

  cmd.x = 0.0;
  cmd.y = 0.0;
  cmd.z = Z;

  cmd.roll = 0.0;
  cmd.pitch = 0.0;
  cmd.yaw = 0.0;

  if (t < ORIENT_SEC) {
    const double s = smooth(t / ORIENT_SEC);

    cmd.x = 0.0;
    cmd.y = 0.0;
    cmd.z = Z;

    cmd.roll = 0.0;
    cmd.pitch = 0.0;
    cmd.yaw = 0.5 * M_PI * s;

    return cmd;
  }

  t -= ORIENT_SEC;

  double phase = 0.0;
  double bank = BANK_RAD;

  if (t < RAMP_SEC) {
    const double u = std::clamp(t / RAMP_SEC, 0.0, 1.0);
    const double s = smooth(u);

    const double phase_integral =
        2.5 * std::pow(u, 4)
      - 3.0 * std::pow(u, 5)
      +       std::pow(u, 6);

    phase = OMEGA * RAMP_SEC * phase_integral;
    bank = BANK_RAD * s;
  }
  else {
    phase = OMEGA * (t - 0.5 * RAMP_SEC);
    bank = BANK_RAD;
  }

  cmd.x = R * (1.0 - std::cos(phase));
  cmd.y = R * std::sin(phase);
  cmd.z = Z;

  // Tangential heading
  const double yaw = 0.5 * M_PI - phase;

  cmd.roll = 0.0;
  cmd.pitch = 0.0;
  cmd.yaw = std::atan2(std::sin(yaw), std::cos(yaw));

  return cmd;
}

// Control Allocation utils ===========================================
inline AllocationOutput allocation_a1b1(const Eigen::Vector3d& moment_cmd, const Eigen::Vector3d& force_cmd)
{
  const double force_norm = std::max(force_cmd.norm(), params::f_min);

  Eigen::Vector3d force_dir;
  if (force_norm > params::f_min) force_dir = force_cmd / force_norm;
  else force_dir << 0.0, 0.0, -1.0;

  double beta_des = std::atan2(-force_dir(0), -force_dir(2));
  double alpha_des = std::asin(std::clamp(force_dir(1), -1.0, 1.0));

  beta_des = std::clamp(beta_des, -params::beta_limit_rad, params::beta_limit_rad);
  alpha_des = std::clamp(alpha_des, -params::alpha_limit_rad, params::alpha_limit_rad);

  Eigen::Vector3d e_cmd;
  e_cmd << -std::sin(beta_des) * std::cos(alpha_des),
            std::sin(alpha_des),
           -std::cos(beta_des) * std::cos(alpha_des);

  const double L = params::L;
  const double zeta  = params::zeta;

  Eigen::Matrix4d B;
  Eigen::Vector4d Wrench;

  Wrench << moment_cmd(0), moment_cmd(1), moment_cmd(2), force_norm;

  B <<
                    L * e_cmd(2) + zeta * e_cmd(0),                     L * e_cmd(2) - zeta * e_cmd(0),                     -L * e_cmd(2) + zeta * e_cmd(0),                    -L * e_cmd(2) - zeta * e_cmd(0),

                   -L * e_cmd(2) + zeta * e_cmd(1),                     L * e_cmd(2) - zeta * e_cmd(1),                      L * e_cmd(2) + zeta * e_cmd(1),                    -L * e_cmd(2) - zeta * e_cmd(1),

    L * e_cmd(1) - L * e_cmd(0) + zeta * e_cmd(2),    -L * e_cmd(1) - L * e_cmd(0) - zeta * e_cmd(2),     -L * e_cmd(1) + L * e_cmd(0) + zeta * e_cmd(2),     L * e_cmd(1) + L * e_cmd(0) - zeta * e_cmd(2),

                                                1.0,                                                 1.0,                                                   1.0,                                                1.0;


  const Eigen::Matrix4d H = B * B.transpose() + params::virtual_lambda * params::virtual_lambda * Eigen::Matrix4d::Identity();
  const Eigen::Vector4d thrust_raw = B.transpose() * H.ldlt().solve(Wrench);

  AllocationOutput out;

  for (int i = 0; i < 4; ++i) out.f(i) = std::clamp(thrust_raw(i), params::f_cmd_min, params::f_cmd_max);

  out.beta << beta_des, beta_des;
  out.alpha.setConstant(alpha_des);

  return out;
}

inline AllocationOutput allocation_a4b2(const Eigen::Vector3d& moment_cmd, const Eigen::Vector3d& force_cmd, const Eigen::Matrix<double, 6, 1>& d, const Eigen::Vector3d& att_cmd, const Eigen::Vector4d& alpha_measured, const Eigen::Vector2d& beta_measured, bool servo_read, double dt)
{
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Vector10d = Eigen::Matrix<double, 10, 1>;
  using Matrix66d = Eigen::Matrix<double, 6, 6>;
  using Matrix610d = Eigen::Matrix<double, 6, 10>;
  using Matrix106d = Eigen::Matrix<double, 10, 6>;
  using Matrix1010d = Eigen::Matrix<double, 10, 10>;

  static bool initialized = false;
  static AllocationOutput previous_cmd;

  if (!initialized) 
  {
    previous_cmd.alpha = alpha_measured;
    previous_cmd.beta = beta_measured;
    previous_cmd.f.setConstant(std::clamp(0.25 * force_cmd.norm(), params::f_cmd_min, params::f_cmd_max));
    initialized = true;
  }

  // q = [alpha1 alpha2 alpha3 alpha4 beta1 beta2 f1 f2 f3 f4]^T
  Vector10d q;
  q.segment<4>(0) = servo_read ? alpha_measured : previous_cmd.alpha;
  q.segment<2>(4) = servo_read ? beta_measured : previous_cmd.beta;
  q.segment<4>(6) = previous_cmd.f;

  constexpr std::array<int, 4> beta_index = {0, 1, 1, 0};
  constexpr std::array<double, 4> x_sign = {1.0, -1.0, -1.0, 1.0};
  constexpr std::array<double, 4> y_sign = {1.0, 1.0, -1.0, -1.0};
  constexpr std::array<double, 4> reaction_sign = {1.0, -1.0, 1.0, -1.0};

  // W(q) = sum_i f_i [r_i x e_i + sigma_i zeta e_i; e_i]
  Vector6d W_now = Vector6d::Zero();
  Matrix610d J = Matrix610d::Zero();

  for (int i = 0; i < 4; ++i) 
  {
    const double alpha = q(i);
    const double beta = q(4 + beta_index[i]);
    const double f = q(6 + i);

    const double sa = std::sin(alpha);
    const double ca = std::cos(alpha);
    const double sb = std::sin(beta);
    const double cb = std::cos(beta);

    const Eigen::Vector3d r_i(x_sign[i] * params::L, y_sign[i] * params::L, 0.0);
    const Eigen::Vector3d e_i(-sb * ca, sa, -cb * ca);
    const Eigen::Vector3d de_dalpha(sb * sa, ca, cb * sa);
    const Eigen::Vector3d de_dbeta(-cb * ca, 0.0, sb * ca);

    // D_i(v) = [r_i x v + sigma_i zeta v; v]
    const auto wrenchDirection = [&](const Eigen::Vector3d& v) 
    {
      Vector6d D;
      D.segment<3>(0) = r_i.cross(v) + reaction_sign[i] * params::zeta * v;
      D.segment<3>(3) = v;
      return D;
    };

    const Vector6d D_e = wrenchDirection(e_i);
    const Vector6d D_alpha = wrenchDirection(de_dalpha);
    const Vector6d D_beta = wrenchDirection(de_dbeta);

    W_now += f * D_e;

    // J = dW/dq = [dW/dalpha(4) dW/dbeta(2) dW/df(4)]
    J.col(i) = f * D_alpha;
    J.col(4 + beta_index[i]) += f * D_beta;
    J.col(6 + i) = D_e;
  }

  Vector6d W_des;
  W_des << moment_cmd, force_cmd;

  // (7) W_dot_des = Kj (W_des - W(q))
  Matrix66d Kj = Matrix66d::Zero();
  for (int i = 0; i < 6; ++i) Kj(i, i) = params::ada_kj_diag[i];
  const Vector6d W_dot_des = Kj * (W_des - W_now);

  // (5) J_dagger = W_q^{-1} J^T (J W_q^{-1} J^T)^{-1}
  const double disturbance_rms = d.maxCoeff();
  const double weight_ratio = std::clamp(
      (disturbance_rms - params::ada_weight_rms_active)
      / (params::ada_weight_rms_full - params::ada_weight_rms_active),
      0.0,
      1.0);
  const double weight_blend = weight_ratio * weight_ratio * (3.0 - 2.0 * weight_ratio);
  const double f_inverse_weight = params::ada_W_inv_diag[6] + weight_blend * (params::ada_W_inv_f_disturbed - params::ada_W_inv_diag[6]);

  Matrix1010d W_q_inv = Matrix1010d::Zero();
  for (int i = 0; i < 6; ++i) W_q_inv(i, i) = params::ada_W_inv_diag[i];
  for (int i = 6; i < 10; ++i) W_q_inv(i, i) = f_inverse_weight;

  const Matrix66d JWJ = J * W_q_inv * J.transpose();
  const Matrix66d JWJ_inv = JWJ.completeOrthogonalDecomposition().solve(Matrix66d::Identity());
  const Matrix106d J_dagger = W_q_inv * J.transpose() * JWJ_inv;

  Vector10d q_dot = J_dagger * W_dot_des;

  // (13) sat(q_dot) = k_s q_dot
  double k_s = 1.0;
  for (int i = 0; i < 10; ++i) if (std::abs(q_dot(i)) > params::ada_q_dot_max[i]) k_s = std::min(k_s, params::ada_q_dot_max[i] / std::abs(q_dot(i)));
  q_dot *= k_s;

  // W(q), J(q) use the selected servo state; q_dot integration keeps the previous command state.
  Vector10d q_cmd_prev = q;
  q_cmd_prev.segment<4>(0) = previous_cmd.alpha;
  q_cmd_prev.segment<2>(4) = previous_cmd.beta;
  const Vector10d q_cmd = q_cmd_prev + std::max(dt, 0.0) * q_dot;


  AllocationOutput out;
  for (int i = 0; i < 4; ++i) 
  {
    out.alpha(i) = std::clamp(q_cmd(i), -params::alpha_limit_rad, params::alpha_limit_rad);
    out.f(i) = std::clamp(q_cmd(6 + i), params::f_cmd_min, params::f_cmd_max);
  }
  for (int i = 0; i < 2; ++i) out.beta(i) = std::clamp(q_cmd(4 + i), -params::beta_limit_rad, params::beta_limit_rad);

  previous_cmd = out;
  return out;
}

inline AllocationCheck checkAllocation(const AllocationOutput& alloc, const Eigen::Vector3d& moment_cmd, const Eigen::Vector3d& force_cmd)
{
  AllocationCheck result;

  for (int i = 0; i < 4; ++i) {
    const double x_sign = (i == 0 || i == 3) ? 1.0 : -1.0;
    const double y_sign = (i == 0 || i == 1) ? 1.0 : -1.0;
    const double spin = (i % 2 == 0) ? 1.0 : -1.0;

    const double beta = (i == 0 || i == 3) ? alloc.beta(0) : alloc.beta(1);
    const double alpha = alloc.alpha(i);

    Eigen::Vector3d e_i;
    e_i << -std::sin(beta) * std::cos(alpha), std::sin(alpha), -std::cos(beta) * std::cos(alpha);

    Eigen::Vector3d r_i;
    r_i << x_sign * params::L, y_sign * params::L, 0.0;

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
       << "\n    beta = " << alloc.beta.transpose()
       << "\n    alpha   = " << alloc.alpha.transpose()
       << "\033[0m";

    result.message = ss.str();
  }

  return result;
}

}
