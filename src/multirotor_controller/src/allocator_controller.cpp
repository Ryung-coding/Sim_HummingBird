#include <rclcpp/rclcpp.hpp>
#include <multirotor_interfaces/msg/wrench.hpp>
#include <multirotor_interfaces/msg/input.hpp>
#include <multirotor_interfaces/msg/multirotor_state.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>

class AllocatorController : public rclcpp::Node
{
public:
  AllocatorController() : rclcpp::Node("allocator_controller")
  {
    allocation_mode_ = this->declare_parameter<std::string>("allocation_mode", "A");
    for (auto & c : allocation_mode_) {
      c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }

    if (allocation_mode_ != "A" && allocation_mode_ != "B") {
      RCLCPP_WARN(
        this->get_logger(),
        "Unknown allocation_mode='%s'. Use allocation_mode='A' instead.",
        allocation_mode_.c_str()
      );
      allocation_mode_ = "A";
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Allocator mode = %s (%s)",
      allocation_mode_.c_str(),
      allocation_mode_ == "A" ? "original git two-stage allocation" : "new 6x12 force-component allocation"
    );

    wrench_subscription_ = this->create_subscription<multirotor_interfaces::msg::Wrench>(
      "/wrench",
      10,
      std::bind(&AllocatorController::onWrench, this, std::placeholders::_1)
    );

    state_subscription_ = this->create_subscription<multirotor_interfaces::msg::MultirotorState>(
      "/multirotor_state",
      10,
      std::bind(&AllocatorController::onState, this, std::placeholders::_1)
    );

    input_publisher_ = this->create_publisher<multirotor_interfaces::msg::Input>(
      "/input",
      10
    );

    thrust_.setConstant(10.0);
    theta_measured_.setZero();
    phi_measured_.setZero();
    theta_desired_.setZero();
    phi_desired_.setZero();
  }

private:
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Vector8d = Eigen::Matrix<double, 8, 1>;
  using Vector12d = Eigen::Matrix<double, 12, 1>;
  using Matrix68d = Eigen::Matrix<double, 6, 8>;
  using Matrix612d = Eigen::Matrix<double, 6, 12>;

  static constexpr double Lx = 0.1861;
  static constexpr double Ly = 0.1861;
  static constexpr double d = 0.0500;
  static constexpr double zeta = 0.0200;

  static constexpr double f_min = 1.0e-3;
  static constexpr double f_cmd_min = 1.0e-6;
  static constexpr double f_cmd_max = 50.0;

  static constexpr double angle_limit_rad = 1.57;  // < 90 deg
  static constexpr double virtual_lambda = 1.0e-4;
  static constexpr double new_lambda = 1.0e-8;

  static constexpr double check_force_tol = 1.00;
  static constexpr double check_moment_tol = 2.00;

  struct PlantCheckResult
  {
    bool problem = false;
    double tau_z_thrust = 0.0;
    double tau_z_reaction = 0.0;
    Eigen::Vector3d moment_actual = Eigen::Vector3d::Zero();
    Eigen::Vector3d force_actual = Eigen::Vector3d::Zero();
    Eigen::Vector3d moment_error = Eigen::Vector3d::Zero();
    Eigen::Vector3d force_error = Eigen::Vector3d::Zero();
    std::string message;
  };

  void onState(const multirotor_interfaces::msg::MultirotorState::SharedPtr msg)
  {
    theta_measured_(0) = static_cast<double>(msg->theta[0]);
    theta_measured_(1) = static_cast<double>(msg->theta[1]);

    phi_measured_(0) = static_cast<double>(msg->phi[0]);
    phi_measured_(1) = static_cast<double>(msg->phi[1]);
    phi_measured_(2) = static_cast<double>(msg->phi[2]);
    phi_measured_(3) = static_cast<double>(msg->phi[3]);
  }

  void onWrench(const multirotor_interfaces::msg::Wrench::SharedPtr msg)
  {
    Eigen::Vector3d moment_cmd;
    Eigen::Vector3d force_cmd;

    moment_cmd << static_cast<double>(msg->moment[0]),
                  static_cast<double>(msg->moment[1]),
                  static_cast<double>(msg->moment[2]);

    force_cmd << static_cast<double>(msg->force[0]),
                 static_cast<double>(msg->force[1]),
                 static_cast<double>(msg->force[2]);

    if (allocation_mode_ == "A") {
      // Original git method:
      //   W -> virtual front/back force + df
      //   virtual force -> thrust, theta, phi
      Vector8d virtual_force = calcA1(moment_cmd, force_cmd, theta_measured_, phi_measured_);
      calcA2(virtual_force, thrust_, theta_desired_, phi_desired_);
    }
    else {
      // New method:
      //   W = A_new * T
      //   T = [F1x F1y F1z ... F4x F4y F4z]
      //   T -> pair direction -> thrust, theta, phi
      calcNewAllocation(moment_cmd, force_cmd, thrust_, theta_desired_, phi_desired_);
    }

    PlantCheckResult check = checkPlantWrench(
      thrust_,
      theta_desired_,
      phi_desired_,
      moment_cmd,
      force_cmd
    );

    if (check.problem) {
      RCLCPP_ERROR_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        500,
        "%s",
        check.message.c_str()
      );
    }

    multirotor_interfaces::msg::Input out;

    out.u[0] = static_cast<float>(thrust_(0));
    out.u[1] = static_cast<float>(thrust_(1));
    out.u[2] = static_cast<float>(thrust_(2));
    out.u[3] = static_cast<float>(thrust_(3));

    out.u[4] = static_cast<float>(theta_desired_(0));
    out.u[5] = static_cast<float>(theta_desired_(1));

    out.u[6] = static_cast<float>(phi_desired_(0));
    out.u[7] = static_cast<float>(phi_desired_(1));
    out.u[8] = static_cast<float>(phi_desired_(2));
    out.u[9] = static_cast<float>(phi_desired_(3));

    input_publisher_->publish(out);
  }

  // ============================================================
  // Common geometry / force convention
  // ============================================================

  Eigen::Matrix3d skew(const Eigen::Vector3d& r) const
  {
    Eigen::Matrix3d S;
    S << 0.0, -r.z(), r.y(),
         r.z(), 0.0, -r.x(),
        -r.y(), r.x(), 0.0;
    return S;
  }

  Eigen::Vector3d forceDirectionFromThetaPhi(double theta, double phi) const
  {
    Eigen::Vector3d e;
    e << -std::sin(theta) * std::cos(phi),
          std::sin(phi),
         -std::cos(theta) * std::cos(phi);
    return e;
  }

  void thetaPhiFromDirection(
    const Eigen::Vector3d& direction_raw,
    double& theta_out,
    double& phi_out,
    Eigen::Vector3d& direction_clamped_out
  ) const
  {
    Eigen::Vector3d e = direction_raw;
    const double n = e.norm();

    if (!(n > f_min) || !e.allFinite()) {
      e << 0.0, 0.0, -1.0;
    }
    else {
      e /= n;
    }

    // Inverse of:
    //   e = [-sin(theta)cos(phi), sin(phi), -cos(theta)cos(phi)]
    phi_out = std::asin(std::clamp(e.y(), -1.0, 1.0));
    theta_out = std::atan2(-e.x(), -e.z());

    theta_out = std::clamp(theta_out, -angle_limit_rad, angle_limit_rad);
    phi_out = std::clamp(phi_out, -angle_limit_rad, angle_limit_rad);

    direction_clamped_out = forceDirectionFromThetaPhi(theta_out, phi_out);
  }

  // ============================================================
  // Mode A: original git method
  // ============================================================

  Vector8d calcA1(
    const Eigen::Vector3d& moment_cmd,
    const Eigen::Vector3d& force_cmd,
    const Eigen::Vector2d& theta,
    const Eigen::Vector4d& phi
  )
  {
    const double theta1 = theta(0);
    const double theta2 = theta(1);

    const double phi14 = 0.5 * (phi(0) + phi(3));
    const double phi23 = 0.5 * (phi(1) + phi(2));

    const double Fx14 = -std::sin(theta1) * std::cos(phi14);
    const double Fy14 = +std::sin(phi14);
    const double Fz14 = -std::cos(theta1) * std::cos(phi14);

    const double Fx23 = -std::sin(theta2) * std::cos(phi23);
    const double Fy23 = +std::sin(phi23);
    const double Fz23 = -std::cos(theta2) * std::cos(phi23);

    Matrix68d A1;
    A1.setZero();

    // C = [front_Fx; front_Fy; front_Fz; back_Fx; back_Fy; back_Fz; df14; df23]
    // W = [Mx; My; Mz; Fx; Fy; Fz]

    // Mx
    A1(0, 6) = (Ly - d * std::sin(theta1)) * Fz14 + zeta * Fx14;
    A1(0, 7) = (Ly - d * std::sin(theta2)) * Fz23 - zeta * Fx23;

    // My
    A1(1, 2) = -Lx;
    A1(1, 5) = +Lx;
    A1(1, 6) = zeta * Fy14;
    A1(1, 7) = -zeta * Fy23;

    // Mz
    A1(2, 1) = +Lx;
    A1(2, 4) = -Lx;
    A1(2, 6) = -(Ly - d * std::sin(theta1)) * Fx14 + zeta * Fz14;
    A1(2, 7) = -(Ly - d * std::sin(theta2)) * Fx23 - zeta * Fz23;

    // Fx = front_Fx + back_Fx
    A1(3, 0) = 1.0;
    A1(3, 3) = 1.0;

    // Fy = front_Fy + back_Fy
    A1(4, 1) = 1.0;
    A1(4, 4) = 1.0;

    // Fz = front_Fz + back_Fz
    A1(5, 2) = 1.0;
    A1(5, 5) = 1.0;

    Vector6d B1;
    B1 << moment_cmd(0), moment_cmd(1), moment_cmd(2),
          force_cmd(0), force_cmd(1), force_cmd(2);

    Eigen::Matrix<double, 6, 6> solve_matrix =
      A1 * A1.transpose()
      + virtual_lambda * virtual_lambda * Eigen::Matrix<double, 6, 6>::Identity();

    Vector8d virtual_force;
    virtual_force.setZero();

    Eigen::FullPivLU<Eigen::Matrix<double, 6, 6>> lu_1(solve_matrix);

    if (lu_1.isInvertible()) {
      virtual_force = A1.transpose() * lu_1.solve(B1);
    }
    else {
      std::cout << "[WARN] A1 virtual singular (rank=" << lu_1.rank()
                << ")\nA1:\n" << A1
                << "\nB1: " << B1.transpose() << std::endl;

      virtual_force = A1.transpose() * solve_matrix.ldlt().solve(B1);
    }

    return virtual_force;
  }

  void calcA2(
    const Vector8d& virtual_force,
    Eigen::Vector4d& thrust_out,
    Eigen::Vector2d& theta_out,
    Eigen::Vector4d& phi_out
  )
  {
    Eigen::Vector3d front_force;
    Eigen::Vector3d back_force;

    front_force << virtual_force(0), virtual_force(1), virtual_force(2);
    back_force << virtual_force(3), virtual_force(4), virtual_force(5);

    double front_force_norm = front_force.norm();
    double back_force_norm = back_force.norm();

    double safe_front_force_norm = std::max(front_force_norm, f_min);
    double safe_back_force_norm = std::max(back_force_norm, f_min);

    Eigen::Vector3d front_direction = front_force / safe_front_force_norm;
    Eigen::Vector3d back_direction = back_force / safe_back_force_norm;

    if (front_force_norm <= f_min) {
      front_direction << 0.0, 0.0, -1.0;
    }

    if (back_force_norm <= f_min) {
      back_direction << 0.0, 0.0, -1.0;
    }

    double theta1 = 0.0;
    double theta2 = 0.0;
    double phi14 = 0.0;
    double phi23 = 0.0;
    Eigen::Vector3d front_direction_clamped;
    Eigen::Vector3d back_direction_clamped;

    thetaPhiFromDirection(front_direction, theta1, phi14, front_direction_clamped);
    thetaPhiFromDirection(back_direction, theta2, phi23, back_direction_clamped);

    double df14 = std::clamp(
      virtual_force(6),
      -safe_front_force_norm + f_min,
      safe_front_force_norm - f_min
    );

    double df23 = std::clamp(
      virtual_force(7),
      -safe_back_force_norm + f_min,
      safe_back_force_norm - f_min
    );

    const double f1 = 0.5 * (safe_front_force_norm + df14);
    const double f4 = 0.5 * (safe_front_force_norm - df14);
    const double f2 = 0.5 * (safe_back_force_norm + df23);
    const double f3 = 0.5 * (safe_back_force_norm - df23);

    thrust_out(0) = std::clamp(f1, f_cmd_min, f_cmd_max);
    thrust_out(1) = std::clamp(f2, f_cmd_min, f_cmd_max);
    thrust_out(2) = std::clamp(f3, f_cmd_min, f_cmd_max);
    thrust_out(3) = std::clamp(f4, f_cmd_min, f_cmd_max);

    theta_out(0) = theta1;
    theta_out(1) = theta2;

    phi_out(0) = phi14;
    phi_out(1) = phi23;
    phi_out(2) = phi23;
    phi_out(3) = phi14;
  }

  // ============================================================
  // Mode B: new 6x12 force-component allocation
  // ============================================================

  Matrix612d buildNewAllocationMatrix() const
  {
    Matrix612d A;
    A.setZero();

    // Constant geometry for the new allocation.
    // This keeps W = A*T constant.
    //
    // T = [F1x F1y F1z F2x F2y F2z F3x F3y F3z F4x F4y F4z]^T
    //
    // W = [Mx My Mz Fx Fy Fz]^T
    //
    // M_i = r_i x F_i + spin_i*zeta*F_i
    //
    // Note:
    // The original plant check uses y = Ly - d*sin(theta).
    // The new allocation intentionally ignores this angle-dependent d term
    // so that A remains constant.

    const std::array<Eigen::Vector3d, 4> r = {
      Eigen::Vector3d(+Lx, +Ly, 0.0),
      Eigen::Vector3d(-Lx, +Ly, 0.0),
      Eigen::Vector3d(-Lx, -Ly, 0.0),
      Eigen::Vector3d(+Lx, -Ly, 0.0)
    };

    const std::array<double, 4> spin = {
      +1.0, -1.0, +1.0, -1.0
    };

    for (int i = 0; i < 4; ++i) {
      const int c = 3 * i;

      Eigen::Matrix3d moment_block = skew(r[i]) + spin[i] * zeta * Eigen::Matrix3d::Identity();

      A.block<3, 3>(0, c) = moment_block;
      A.block<3, 3>(3, c) = Eigen::Matrix3d::Identity();
    }

    return A;
  }

  void calcNewAllocation(
    const Eigen::Vector3d& moment_cmd,
    const Eigen::Vector3d& force_cmd,
    Eigen::Vector4d& thrust_out,
    Eigen::Vector2d& theta_out,
    Eigen::Vector4d& phi_out
  )
  {
    Matrix612d A = buildNewAllocationMatrix();

    Vector6d W;
    W << moment_cmd(0), moment_cmd(1), moment_cmd(2),
         force_cmd(0), force_cmd(1), force_cmd(2);

    Eigen::Matrix<double, 6, 6> solve_matrix =
      A * A.transpose()
      + new_lambda * new_lambda * Eigen::Matrix<double, 6, 6>::Identity();

    Vector12d T;
    T.setZero();

    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(solve_matrix);
    if (ldlt.info() == Eigen::Success) {
      T = A.transpose() * ldlt.solve(W);
    }
    else {
      Eigen::FullPivLU<Eigen::Matrix<double, 6, 6>> lu(solve_matrix);
      T = A.transpose() * lu.solve(W);
    }

    const Eigen::Vector3d lambda1 = T.segment<3>(0);
    const Eigen::Vector3d lambda2 = T.segment<3>(3);
    const Eigen::Vector3d lambda3 = T.segment<3>(6);
    const Eigen::Vector3d lambda4 = T.segment<3>(9);

    // Shared direction constraint:
    // rotor 1 and 4 share theta1, phi14
    // rotor 2 and 3 share theta2, phi23
    const Eigen::Vector3d lambda_front = lambda1 + lambda4;
    const Eigen::Vector3d lambda_back = lambda2 + lambda3;

    double theta1 = 0.0;
    double theta2 = 0.0;
    double phi14 = 0.0;
    double phi23 = 0.0;

    Eigen::Vector3d front_direction;
    Eigen::Vector3d back_direction;

    thetaPhiFromDirection(lambda_front, theta1, phi14, front_direction);
    thetaPhiFromDirection(lambda_back, theta2, phi23, back_direction);

    // Project each rotor force component onto the implementable pair direction.
    double f1 = lambda1.dot(front_direction);
    double f4 = lambda4.dot(front_direction);
    double f2 = lambda2.dot(back_direction);
    double f3 = lambda3.dot(back_direction);

    // Positive thrust only.
    f1 = std::clamp(f1, f_cmd_min, f_cmd_max);
    f2 = std::clamp(f2, f_cmd_min, f_cmd_max);
    f3 = std::clamp(f3, f_cmd_min, f_cmd_max);
    f4 = std::clamp(f4, f_cmd_min, f_cmd_max);

    thrust_out(0) = f1;
    thrust_out(1) = f2;
    thrust_out(2) = f3;
    thrust_out(3) = f4;

    theta_out(0) = theta1;
    theta_out(1) = theta2;

    phi_out(0) = phi14;
    phi_out(1) = phi23;
    phi_out(2) = phi23;
    phi_out(3) = phi14;
  }

  // ============================================================
  // Plant check
  // ============================================================

  PlantCheckResult checkPlantWrench(
    const Eigen::Vector4d& thrust,
    const Eigen::Vector2d& theta,
    const Eigen::Vector4d& phi,
    const Eigen::Vector3d& moment_cmd,
    const Eigen::Vector3d& force_cmd
  )
  {
    PlantCheckResult result;

    Eigen::Vector3d r1(+Lx, +(Ly - d * std::sin(theta(0))), 0.0);
    Eigen::Vector3d r2(-Lx, +(Ly - d * std::sin(theta(1))), 0.0);
    Eigen::Vector3d r3(-Lx, -(Ly - d * std::sin(theta(1))), 0.0);
    Eigen::Vector3d r4(+Lx, -(Ly - d * std::sin(theta(0))), 0.0);

    for (int i = 0; i < 4; ++i) {
      Eigen::Vector3d r_i;
      double theta_i = 0.0;
      double phi_i = phi(i);
      double spin_i = 1.0;

      if (i == 0) {
        r_i = r1;
        theta_i = theta(0);
        spin_i = +1.0;
      }

      if (i == 1) {
        r_i = r2;
        theta_i = theta(1);
        spin_i = -1.0;
      }

      if (i == 2) {
        r_i = r3;
        theta_i = theta(1);
        spin_i = +1.0;
      }

      if (i == 3) {
        r_i = r4;
        theta_i = theta(0);
        spin_i = -1.0;
      }

      Eigen::Vector3d force_i;
      force_i << thrust(i) * (-std::sin(theta_i) * std::cos(phi_i)),
                 thrust(i) * std::sin(phi_i),
                 thrust(i) * (-std::cos(theta_i) * std::cos(phi_i));

      Eigen::Vector3d moment_thrust_i = r_i.cross(force_i);

      Eigen::Vector3d moment_reaction_i = Eigen::Vector3d::Zero();
      const double force_norm = force_i.norm();
      if (force_norm > 1.0e-12) {
        moment_reaction_i = spin_i * zeta * thrust(i) * force_i.normalized();
      }

      result.force_actual += force_i;
      result.moment_actual += moment_thrust_i + moment_reaction_i;
      result.tau_z_thrust += moment_thrust_i(2);
      result.tau_z_reaction += moment_reaction_i(2);
    }

    result.moment_error = moment_cmd - result.moment_actual;
    result.force_error = force_cmd - result.force_actual;

    const double err_m_norm = result.moment_error.norm();
    const double err_f_norm = result.force_error.norm();

    result.problem =
      (!result.moment_actual.allFinite())
      || (!result.force_actual.allFinite())
      || (err_f_norm > check_force_tol)
      || (err_m_norm > check_moment_tol);

    if (result.problem) {
      std::ostringstream ss;
      ss << "\n[CA CHECK]"
         << "\n  mode = " << allocation_mode_
         << "\n"
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
         << "\n    thrust = " << thrust.transpose()
         << "\n    theta  = " << theta.transpose()
         << "\n    phi    = " << phi.transpose()
         << "\n"
         << "\n  yaw split"
         << "\n    tau_z_thrust   = " << result.tau_z_thrust << " Nm"
         << "\n    tau_z_reaction = " << result.tau_z_reaction << " Nm";

      result.message = ss.str();
    }

    return result;
  }

  rclcpp::Subscription<multirotor_interfaces::msg::Wrench>::SharedPtr wrench_subscription_;
  rclcpp::Subscription<multirotor_interfaces::msg::MultirotorState>::SharedPtr state_subscription_;
  rclcpp::Publisher<multirotor_interfaces::msg::Input>::SharedPtr input_publisher_;

  Eigen::Vector4d thrust_;
  Eigen::Vector2d theta_measured_;
  Eigen::Vector4d phi_measured_;
  Eigen::Vector2d theta_desired_;
  Eigen::Vector4d phi_desired_;

  std::string allocation_mode_{"A"};
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AllocatorController>());
  rclcpp::shutdown();
  return 0;
}