#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
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
  double primary_scale = 1.0;
  double nullspace_scale = 1.0;
};

struct HexaAllocationOutput {
  Eigen::Matrix<double, 6, 1> f = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> alpha = Eigen::Matrix<double, 6, 1>::Zero();
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
  static constexpr double HOVER_SEC = 3.0;
  static constexpr double SEG_SEC = 5.0;
  static constexpr double XY = 0.0;
  static constexpr double Z = 1.0;

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
  if (t < RAMP_SEC) {
    const double u = std::clamp(t / RAMP_SEC, 0.0, 1.0);
    const double phase_integral =
        2.5 * std::pow(u, 4)
      - 3.0 * std::pow(u, 5)
      +       std::pow(u, 6);

    phase = OMEGA * RAMP_SEC * phase_integral;
  }
  else {
    phase = OMEGA * (t - 0.5 * RAMP_SEC);
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

inline TargetCMD stepPath(double t)
{
  static constexpr double HOVER_SEC = 3.0;
  static constexpr double HOLD_SEC  = 3.0;

  static constexpr double X_delta = 0.5;
  static constexpr double Z = 1.0;

  TargetCMD cmd;

  if (t < HOVER_SEC) 
  {
    const double a = t / HOVER_SEC;
    const double s = a * a * (3.0 - 2.0 * a);

    cmd.x = 0.0;
    cmd.y = 0.0;
    cmd.z = Z * s;

    cmd.roll  = 0.0;
    cmd.pitch = 0.0;
    cmd.yaw   = 0.0;

    return cmd;
  }

  cmd.x = t - HOVER_SEC < HOLD_SEC ? X_delta : 0.0;
  cmd.y = 0.0;
  cmd.z = Z;

  cmd.roll  = 0.0;
  cmd.pitch = 0.0;
  cmd.yaw   = 0.0;

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

  return out;
}

inline AllocationOutput allocation_a4b2(const Eigen::Vector3d& moment_cmd, const Eigen::Vector3d& force_cmd, const Eigen::Vector2d& beta_ref, const Eigen::Vector4d& alpha_measured, const Eigen::Vector2d& beta_measured, bool servo_read, double dt)
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
    previous_cmd.beta = beta_ref;
    previous_cmd.f.setConstant(std::clamp(0.25 * force_cmd.norm(), params::HB_F_CMD_MIN, params::HB_F_CMD_MAX));
    initialized = true;
  }

  // q = [alpha1 alpha2 alpha3 alpha4 beta1 beta2 f1 f2 f3 f4]^T
  Vector10d q;
  q.segment<4>(0) = servo_read ? alpha_measured : previous_cmd.alpha;
  q.segment<2>(4) = servo_read ? beta_measured : previous_cmd.beta;
  q.segment<4>(6) = previous_cmd.f;

  const double alpha1 = q(0);
  const double alpha2 = q(1);
  const double alpha3 = q(2);
  const double alpha4 = q(3);
  const double beta1 = q(4);
  const double beta2 = q(5);
  const double f1 = q(6);
  const double f2 = q(7);
  const double f3 = q(8);
  const double f4 = q(9);

  const Eigen::Vector3d r1(params::HB_L, params::HB_L, 0.0);
  const Eigen::Vector3d r2(-params::HB_L, params::HB_L, 0.0);
  const Eigen::Vector3d r3(-params::HB_L, -params::HB_L, 0.0);
  const Eigen::Vector3d r4(params::HB_L, -params::HB_L, 0.0);

  const Eigen::Vector3d e1(-std::sin(beta1) * std::cos(alpha1), std::sin(alpha1), -std::cos(beta1) * std::cos(alpha1));
  const Eigen::Vector3d e2(-std::sin(beta2) * std::cos(alpha2), std::sin(alpha2), -std::cos(beta2) * std::cos(alpha2));
  const Eigen::Vector3d e3(-std::sin(beta2) * std::cos(alpha3), std::sin(alpha3), -std::cos(beta2) * std::cos(alpha3));
  const Eigen::Vector3d e4(-std::sin(beta1) * std::cos(alpha4), std::sin(alpha4), -std::cos(beta1) * std::cos(alpha4));

  const Eigen::Vector3d de_dalpha1(std::sin(beta1) * std::sin(alpha1), std::cos(alpha1), std::cos(beta1) * std::sin(alpha1));
  const Eigen::Vector3d de_dalpha2(std::sin(beta2) * std::sin(alpha2), std::cos(alpha2), std::cos(beta2) * std::sin(alpha2));
  const Eigen::Vector3d de_dalpha3(std::sin(beta2) * std::sin(alpha3), std::cos(alpha3), std::cos(beta2) * std::sin(alpha3));
  const Eigen::Vector3d de_dalpha4(std::sin(beta1) * std::sin(alpha4), std::cos(alpha4), std::cos(beta1) * std::sin(alpha4));

  const Eigen::Vector3d de_dbeta1(-std::cos(beta1) * std::cos(alpha1), 0.0, std::sin(beta1) * std::cos(alpha1));
  const Eigen::Vector3d de_dbeta2(-std::cos(beta2) * std::cos(alpha2), 0.0, std::sin(beta2) * std::cos(alpha2));
  const Eigen::Vector3d de_dbeta3(-std::cos(beta2) * std::cos(alpha3), 0.0, std::sin(beta2) * std::cos(alpha3));
  const Eigen::Vector3d de_dbeta4(-std::cos(beta1) * std::cos(alpha4), 0.0, std::sin(beta1) * std::cos(alpha4));

  const Eigen::Vector3d m_e1 = r1.cross(e1) + params::HB_ZETA * e1;
  const Eigen::Vector3d m_e2 = r2.cross(e2) - params::HB_ZETA * e2;
  const Eigen::Vector3d m_e3 = r3.cross(e3) + params::HB_ZETA * e3;
  const Eigen::Vector3d m_e4 = r4.cross(e4) - params::HB_ZETA * e4;

  const Eigen::Vector3d m_alpha1 = r1.cross(de_dalpha1) + params::HB_ZETA * de_dalpha1;
  const Eigen::Vector3d m_alpha2 = r2.cross(de_dalpha2) - params::HB_ZETA * de_dalpha2;
  const Eigen::Vector3d m_alpha3 = r3.cross(de_dalpha3) + params::HB_ZETA * de_dalpha3;
  const Eigen::Vector3d m_alpha4 = r4.cross(de_dalpha4) - params::HB_ZETA * de_dalpha4;

  const Eigen::Vector3d m_beta1 = r1.cross(de_dbeta1) + params::HB_ZETA * de_dbeta1;
  const Eigen::Vector3d m_beta2 = r2.cross(de_dbeta2) - params::HB_ZETA * de_dbeta2;
  const Eigen::Vector3d m_beta3 = r3.cross(de_dbeta3) + params::HB_ZETA * de_dbeta3;
  const Eigen::Vector3d m_beta4 = r4.cross(de_dbeta4) - params::HB_ZETA * de_dbeta4;

  Vector6d W_now;
  W_now.head<3>() = f1 * m_e1 + f2 * m_e2 + f3 * m_e3 + f4 * m_e4;
  W_now.tail<3>() = f1 * e1 + f2 * e2 + f3 * e3 + f4 * e4;

  Matrix610d J = Matrix610d::Zero();
  J.col(0) << f1 * m_alpha1, f1 * de_dalpha1;
  J.col(1) << f2 * m_alpha2, f2 * de_dalpha2;
  J.col(2) << f3 * m_alpha3, f3 * de_dalpha3;
  J.col(3) << f4 * m_alpha4, f4 * de_dalpha4;
  J.col(4) << f1 * m_beta1 + f4 * m_beta4, f1 * de_dbeta1 + f4 * de_dbeta4;
  J.col(5) << f2 * m_beta2 + f3 * m_beta3, f2 * de_dbeta2 + f3 * de_dbeta3;
  J.col(6) << m_e1, e1;
  J.col(7) << m_e2, e2;
  J.col(8) << m_e3, e3;
  J.col(9) << m_e4, e4;

  Vector6d W_des;
  W_des << moment_cmd, force_cmd;

  // (7) W_dot_des = Kj (W_des - W(q))
  const Vector6d W_dot_des = params::HB_KJ * (W_des - W_now);

  const Matrix66d JWJ = J * params::HB_W_INV * J.transpose();
  const Matrix66d JWJ_inv = JWJ.completeOrthogonalDecomposition().solve(Matrix66d::Identity());
  const Matrix106d J_pesudo = params::HB_W_INV * J.transpose() * JWJ_inv;

  Vector10d q_dot_star = Vector10d::Zero();
  for (int i = 0; i < 4; ++i) q_dot_star(i)  = params::HB_NULL_K[0] * (0.0                    - q(i));
  for (int i = 4; i < 6; ++i) q_dot_star(i)  = params::HB_NULL_K[1] * (beta_ref(i-4)          - q(i));
  for (int i = 6; i < 10; ++i) q_dot_star(i) = params::HB_NULL_K[2] * (force_cmd.norm() / 4.0 - q(i));

  // q_dot = J^# W_dot_des + (I - J^# J) q_dot_star
  const Vector10d q_dot_primary = J_pesudo * W_dot_des;

  const Vector10d q_dot_nullspace = (Matrix1010d::Identity() - J_pesudo * J) * q_dot_star;

  Vector10d q_dot_max;
  for (int i = 0; i < 4; ++i) q_dot_max(i) = params::HB_QDOT_MAX[i];
  for (int i = 4; i < 6; ++i) q_dot_max(i) = params::HB_QDOT_MAX[i];
  for (int i = 6; i < 10; ++i) q_dot_max(i) = params::HB_QDOT_MAX[i];

  const auto nullspace_interval = [&](double primary_scale, double& lower, double& upper)
  {
    lower = 0.0;
    upper = 1.0;

    for (int i = 0; i < 10; ++i)
    {
      const double primary_rate = primary_scale * q_dot_primary(i);
      if (std::abs(q_dot_nullspace(i)) < 1.0e-12)
      {
        if (std::abs(primary_rate) > q_dot_max(i) + 1.0e-12) return false;
        continue;
      }

      double bound_1 = (-q_dot_max(i) - primary_rate) / q_dot_nullspace(i);
      double bound_2 = ( q_dot_max(i) - primary_rate) / q_dot_nullspace(i);
      if (bound_1 > bound_2) std::swap(bound_1, bound_2);

      lower = std::max(lower, bound_1);
      upper = std::min(upper, bound_2);
      if (lower > upper + 1.0e-12) return false;
    }

    return true;
  };

  double primary_scale = 1.0;
  double nullspace_lower = 0.0;
  double nullspace_upper = 1.0;
  if (!nullspace_interval(primary_scale, nullspace_lower, nullspace_upper))
  {
    double feasible_scale = 0.0;
    double infeasible_scale = 1.0;
    for (int iteration = 0; iteration < 40; ++iteration)
    {
      const double candidate = 0.5 * (feasible_scale + infeasible_scale);
      double candidate_lower = 0.0;
      double candidate_upper = 1.0;
      if (nullspace_interval(candidate, candidate_lower, candidate_upper))
      {
        feasible_scale = candidate;
      }
      else
      {
        infeasible_scale = candidate;
      }
    }

    primary_scale = feasible_scale;
    nullspace_interval(primary_scale, nullspace_lower, nullspace_upper);
  }

  const double nullspace_scale = std::clamp(nullspace_upper, 0.0, 1.0);
  const Vector10d q_dot = primary_scale * q_dot_primary + nullspace_scale * q_dot_nullspace;

  Vector10d q_cmd;
  for (int i = 0; i < 6; ++i) q_cmd(i) = q(i) + params::HB_Q_CMD_TAU_SERVO * q_dot(i);
  for (int i = 6; i < 10; ++i) q_cmd(i) = q(i) + std::max(dt, 1.0e-6) * q_dot(i);

  AllocationOutput out;
  for (int i = 0; i < 4; ++i) out.alpha(i) = std::clamp(q_cmd(i), -params::HB_ALPHA_LIMIT_RAD, params::HB_ALPHA_LIMIT_RAD);
  for (int i = 0; i < 2; ++i) out.beta(i) = std::clamp(q_cmd(4 + i), -params::HB_BETA_LIMIT_RAD, params::HB_BETA_LIMIT_RAD);
  for (int i = 0; i < 4; ++i) out.f(i) = std::clamp(q_cmd(6 + i), params::HB_F_CMD_MIN, params::HB_F_CMD_MAX);
  out.primary_scale = primary_scale;
  out.nullspace_scale = nullspace_scale;

  previous_cmd = out;
  return out;
}

inline HexaAllocationOutput allocation_hexa_a6_ada(const Eigen::Vector3d& moment_cmd, const Eigen::Vector3d& force_cmd, const Eigen::Matrix<double, 6, 1>& d, const Eigen::Vector3d& att_cmd, const Eigen::Matrix<double, 6, 1>& alpha_measured, bool servo_read, double dt)
{
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Vector12d = Eigen::Matrix<double, 12, 1>;
  using Matrix66d = Eigen::Matrix<double, 6, 6>;
  using Matrix612d = Eigen::Matrix<double, 6, 12>;
  using Matrix126d = Eigen::Matrix<double, 12, 6>;
  using Matrix1212d = Eigen::Matrix<double, 12, 12>;

  static bool initialized = false;
  static HexaAllocationOutput previous_cmd;
  static Vector6d rms_gradient = Vector6d::Zero();

  if (!initialized)
  {
    previous_cmd.alpha = alpha_measured;
    previous_cmd.f.setConstant(force_cmd.norm() / 6.0);
    initialized = true;
  }

  // q = [alpha_0 ... alpha_5 f_0 ... f_5]^T, where f_i is one coaxial arm pair's total thrust.
  Vector12d q;
  q.segment<6>(0) = servo_read ? alpha_measured : previous_cmd.alpha;
  q.segment<6>(6) = previous_cmd.f;

  // XML arms transformed from MuJoCo z-up to the controller's z-down body frame.
  static const std::array<double, 6> arm_yaw = {
    0.5 * M_PI, -0.5 * M_PI, -M_PI / 6.0,
    5.0 * M_PI / 6.0, M_PI / 6.0, -5.0 * M_PI / 6.0
  };
  static const std::array<double, 6> reaction_sign = {1.0, -1.0, 1.0, -1.0, -1.0, 1.0};

  Vector6d W_now = Vector6d::Zero();
  Matrix612d J = Matrix612d::Zero();

  for (int i = 0; i < 6; ++i) {
    const double alpha = q(i);
    const double f = q(6 + i);
    const double yaw = arm_yaw[i];
    const double sa = std::sin(alpha);
    const double ca = std::cos(alpha);

    const Eigen::Vector3d r_i(
      params::HEXA_L * std::cos(yaw),
      params::HEXA_L * std::sin(yaw),
      0.0);
    const Eigen::Vector3d e_i(-std::sin(yaw) * sa, std::cos(yaw) * sa, -ca);
    const Eigen::Vector3d de_dalpha(-std::sin(yaw) * ca, std::cos(yaw) * ca, sa);

    const auto wrench_direction = [&](const Eigen::Vector3d& v) {
      Vector6d D;
      D.segment<3>(0) = r_i.cross(v) + reaction_sign[i] * params::HEXA_ZETA * v;
      D.segment<3>(3) = v;
      return D;
    };

    const Vector6d D_e = wrench_direction(e_i);
    W_now += f * D_e;
    J.col(i) = f * wrench_direction(de_dalpha);
    J.col(6 + i) = D_e;
  }

  Vector6d W_des;
  W_des << moment_cmd, force_cmd;

  // Paper (7): augment wrench tracking into a differential wrench command.
  const Vector6d W_dot_des = params::HEXA_KJ * (W_des - W_now);

  // Paper (5): fixed weighted pseudoinverse. The nullspace term below selects posture.
  const Matrix66d JWJ_inv = (J * params::HEXA_W_INV * J.transpose()).completeOrthogonalDecomposition().solve(Matrix66d::Identity());
  const Matrix126d J_dagger = params::HEXA_W_INV * J.transpose() * JWJ_inv;

  // Transplanted from the previous a4b2 nullspace-gradient implementation.
  // q_dot_star is integrated only through N(q) = I - J_dagger J, preserving the primary wrench.
  Vector12d q_dot_star = Vector12d::Zero();
  const Eigen::Vector3d thrust_dir_ref = rpyToRot(att_cmd).transpose() * Eigen::Vector3d(0.0, 0.0, -1.0);
  const double f_ref = force_cmd.norm() / 6.0;
  const double rms = d.head<2>().norm();

  if (rms > params::HEXA_RMS_ACTIVE) {
    for (int i = 0; i < 6; ++i) {
      const double yaw = arm_yaw[i];
      const Eigen::Vector3d tangent(-std::sin(yaw), std::cos(yaw), 0.0);
      rms_gradient(i) += 10.0 * dt * (d(0) * d(0) * tangent(0) + d(1) * d(1) * tangent(1));
    }
  }
  else {
    rms_gradient *= std::exp(-dt / params::HEXA_RMS_DECAY_TAU);
  }

  for (int i = 0; i < 6; ++i) {
    const double yaw = arm_yaw[i];
    const Eigen::Vector3d tangent(-std::sin(yaw), std::cos(yaw), 0.0);
    const double alpha_ref = std::atan2(tangent.dot(thrust_dir_ref), -thrust_dir_ref(2));
    const double alpha_target = alpha_ref + rms_gradient(i);

    q_dot_star(i) = params::HEXA_NULL_K[0] * (alpha_target - q(i));
    q_dot_star(6 + i) = params::HEXA_NULL_K[1] * (f_ref - q(6 + i));
  }

  const Matrix1212d nullspace = Matrix1212d::Identity() - J_dagger * J;
  Vector12d q_dot = J_dagger * W_dot_des + nullspace * q_dot_star;

  Vector12d q_cmd_prev = q;
  q_cmd_prev.segment<6>(0) = previous_cmd.alpha;
  const Vector12d q_cmd = q_cmd_prev + dt * q_dot;

  HexaAllocationOutput out;
  for (int i = 0; i < 6; ++i) {
    out.alpha(i) = q_cmd(i);
    out.f(i) = q_cmd(6 + i);
  }

  previous_cmd = out;
  return out;
}

}
