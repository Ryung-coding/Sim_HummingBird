#include <rclcpp/rclcpp.hpp>
#include <multirotor_interfaces/msg/wrench.hpp>
#include <multirotor_interfaces/msg/input.hpp>
#include <multirotor_interfaces/msg/multirotor_state.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

class AllocatorController : public rclcpp::Node
{
public:
  AllocatorController() : rclcpp::Node("allocator_controller")
  {
    wrench_subscription_ = this->create_subscription<multirotor_interfaces::msg::Wrench>("/wrench", 10, std::bind(&AllocatorController::onWrench, this, std::placeholders::_1));
    state_subscription_ = this->create_subscription<multirotor_interfaces::msg::MultirotorState>("/multirotor_state", 10, std::bind(&AllocatorController::onState, this, std::placeholders::_1));
    input_publisher_ = this->create_publisher<multirotor_interfaces::msg::Input>("/input", 10);

    thrust_.setConstant(10.0);
    theta_measured_.setZero();
    phi_measured_.setZero();
    theta_desired_.setZero();
    phi_desired_.setZero();
  }

private:
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Vector8d = Eigen::Matrix<double, 8, 1>;
  using Matrix68d = Eigen::Matrix<double, 6, 8>;

  static constexpr double Lx = 0.1861;
  static constexpr double Ly = 0.1861;
  static constexpr double d = 0.0500;
  static constexpr double zeta = 0.0200;
  static constexpr double f_min = 1.0e-3;
  static constexpr double f_cmd_min = 1.0e-6;
  static constexpr double f_cmd_max = 100.0;
  static constexpr double angle_limit_rad = 1.57;
  static constexpr double virtual_lambda = 1.0e-4;
  static constexpr double check_tau_z_thrust_tol = 0.50;
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

    moment_cmd << static_cast<double>(msg->moment[0]), static_cast<double>(msg->moment[1]), static_cast<double>(msg->moment[2]);
    force_cmd << static_cast<double>(msg->force[0]), static_cast<double>(msg->force[1]), static_cast<double>(msg->force[2]);

    Vector8d virtual_force = calcA1(moment_cmd, force_cmd, theta_measured_, phi_measured_);

    calcA2(virtual_force, thrust_, theta_desired_, phi_desired_);

    PlantCheckResult check = checkPlantWrench(thrust_, theta_desired_, phi_desired_, moment_cmd, force_cmd);
    if (check.problem) {RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 500, "%s", check.message.c_str());}

    multirotor_interfaces::msg::Input out;

    out.u[0] = thrust_(0);
    out.u[1] = thrust_(1);
    out.u[2] = thrust_(2);
    out.u[3] = thrust_(3);
    out.u[4] = theta_desired_(0);
    out.u[5] = theta_desired_(1);
    out.u[6] = phi_desired_(0);
    out.u[7] = phi_desired_(1);
    out.u[8] = phi_desired_(2);
    out.u[9] = phi_desired_(3);

    input_publisher_->publish(out);
  }

  Vector8d calcA1(const Eigen::Vector3d& moment_cmd, const Eigen::Vector3d& force_cmd, const Eigen::Vector2d& theta, const Eigen::Vector4d& phi)
  {
    const double theta1 = theta(0);
    const double theta2 = theta(1);

    const double phi14 = 0.5 * (phi(0) + phi(3));
    const double phi23 = 0.5 * (phi(1) + phi(2));

    const double front_ex = -std::sin(theta1) * std::cos(phi14);
    const double front_ey = +std::sin(phi14);
    const double front_ez = -std::cos(theta1) * std::cos(phi14);

    const double back_ex = -std::sin(theta2) * std::cos(phi23);
    const double back_ey = +std::sin(phi23);
    const double back_ez = -std::cos(theta2) * std::cos(phi23);

    const double front_x = +Lx - d * std::cos(phi14) * std::sin(theta1);
    const double front_y = +d * std::sin(phi14);

    const double back_x = -Lx - d * std::cos(phi23) * std::sin(theta2);
    const double back_y = +d * std::sin(phi23);

    Matrix68d A1;
    A1.setZero();

    // C = [front_Fx; front_Fy; front_Fz; back_Fx; back_Fy; back_Fz; df14; df23]
    // W = [Mx; My; Mz; Fx; Fy; Fz]
    //
    // front force acts at:
    // r_front_avg = [front_x, front_y, 0]
    //
    // back force acts at:
    // r_back_avg = [back_x, back_y, 0]
    //
    // M = r x F
    // Mx = y Fz
    // My = -x Fz
    // Mz = x Fy - y Fx
    //
    // df14 = f1 - f4
    // df23 = f2 - f3
    //
    // df thrust moment:
    // front: [Ly * front_ez, 0, -Ly * front_ex]
    // back : [Ly * back_ez,  0, -Ly * back_ex]
    //
    // df reaction moment:
    // front: +zeta * [front_ex, front_ey, front_ez]
    // back : -zeta * [back_ex,  back_ey,  back_ez]

    // Mx
    A1(0, 0) = 0.0;
    A1(0, 1) = 0.0;
    A1(0, 2) = front_y;
    A1(0, 3) = 0.0;
    A1(0, 4) = 0.0;
    A1(0, 5) = back_y;
    A1(0, 6) = Ly * front_ez + zeta * front_ex;
    A1(0, 7) = Ly * back_ez - zeta * back_ex;

    // My
    A1(1, 0) = 0.0;
    A1(1, 1) = 0.0;
    A1(1, 2) = -front_x;
    A1(1, 3) = 0.0;
    A1(1, 4) = 0.0;
    A1(1, 5) = -back_x;
    A1(1, 6) = zeta * front_ey;
    A1(1, 7) = -zeta * back_ey;

    // Mz
    A1(2, 0) = -front_y;
    A1(2, 1) = +front_x;
    A1(2, 2) = 0.0;
    A1(2, 3) = -back_y;
    A1(2, 4) = +back_x;
    A1(2, 5) = 0.0;
    A1(2, 6) = -Ly * front_ex + zeta * front_ez;
    A1(2, 7) = -Ly * back_ex - zeta * back_ez;

    // Fx = front_Fx + back_Fx
    A1(3, 0) = 1.0;
    A1(3, 1) = 0.0;
    A1(3, 2) = 0.0;
    A1(3, 3) = 1.0;
    A1(3, 4) = 0.0;
    A1(3, 5) = 0.0;
    A1(3, 6) = 0.0;
    A1(3, 7) = 0.0;

    // Fy = front_Fy + back_Fy
    A1(4, 0) = 0.0;
    A1(4, 1) = 1.0;
    A1(4, 2) = 0.0;
    A1(4, 3) = 0.0;
    A1(4, 4) = 1.0;
    A1(4, 5) = 0.0;
    A1(4, 6) = 0.0;
    A1(4, 7) = 0.0;

    // Fz = front_Fz + back_Fz
    A1(5, 0) = 0.0;
    A1(5, 1) = 0.0;
    A1(5, 2) = 1.0;
    A1(5, 3) = 0.0;
    A1(5, 4) = 0.0;
    A1(5, 5) = 1.0;
    A1(5, 6) = 0.0;
    A1(5, 7) = 0.0;

    Vector6d B1;
    B1 << moment_cmd(0), moment_cmd(1), moment_cmd(2), force_cmd(0), force_cmd(1), force_cmd(2);

    Eigen::Matrix<double, 6, 6> solve_matrix = A1 * A1.transpose() + virtual_lambda * virtual_lambda * Eigen::Matrix<double, 6, 6>::Identity();

    Vector8d virtual_force;
    virtual_force.setZero();

    Eigen::FullPivLU<Eigen::Matrix<double, 6, 6>> lu_1(solve_matrix);

    if (lu_1.isInvertible()) {
      virtual_force = A1.transpose() * lu_1.solve(B1);
    }
    else {
      std::cout << "[WARN] A1 virtual singular (rank=" << lu_1.rank() << ")\nA1:\n" << A1 << "\nB1: " << B1.transpose() << std::endl;
      virtual_force = A1.transpose() * solve_matrix.ldlt().solve(B1);
    }

    return virtual_force;
  }

  void calcA2(const Vector8d& virtual_force, Eigen::Vector4d& thrust_out, Eigen::Vector2d& theta_out, Eigen::Vector4d& phi_out)
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

    if (front_force_norm <= f_min) {front_direction << 0.0, 0.0, -1.0;}
    if (back_force_norm <= f_min) {back_direction << 0.0, 0.0, -1.0;}

    double phi14 = std::asin(std::clamp(front_direction(1), -1.0, 1.0));
    double phi23 = std::asin(std::clamp(back_direction(1), -1.0, 1.0));

    double theta1 = std::atan2(-front_direction(0), -front_direction(2));
    double theta2 = std::atan2(-back_direction(0), -back_direction(2));

    theta1 = std::clamp(theta1, -angle_limit_rad, angle_limit_rad);
    theta2 = std::clamp(theta2, -angle_limit_rad, angle_limit_rad);
    phi14 = std::clamp(phi14, -angle_limit_rad, angle_limit_rad);
    phi23 = std::clamp(phi23, -angle_limit_rad, angle_limit_rad);

    double df14 = std::clamp(virtual_force(6), -safe_front_force_norm + f_min, safe_front_force_norm - f_min);
    double df23 = std::clamp(virtual_force(7), -safe_back_force_norm + f_min, safe_back_force_norm - f_min);

    const double f1 = 0.5 * (safe_front_force_norm + df14);
    const double f4 = 0.5 * (safe_front_force_norm - df14);
    const double f2 = 0.5 * (safe_back_force_norm + df23);
    const double f3 = 0.5 * (safe_back_force_norm - df23);

    thrust_out(0) = std::clamp(f1, f_cmd_min, f_cmd_max);
    thrust_out(1) = std::clamp(f2, f_cmd_min, f_cmd_max);
    thrust_out(2) = std::clamp(f3, f_cmd_min, f_cmd_max);
    thrust_out(3) = std::clamp(f4, f_cmd_min, f_cmd_max);

    const double f1_raw = 0.5 * (safe_front_force_norm + df14);
    const double f4_raw = 0.5 * (safe_front_force_norm - df14);
    const double f2_raw = 0.5 * (safe_back_force_norm + df23);
    const double f3_raw = 0.5 * (safe_back_force_norm - df23);

    thrust_out(0) = std::clamp(f1_raw, f_cmd_min, f_cmd_max);
    thrust_out(1) = std::clamp(f2_raw, f_cmd_min, f_cmd_max);
    thrust_out(2) = std::clamp(f3_raw, f_cmd_min, f_cmd_max);
    thrust_out(3) = std::clamp(f4_raw, f_cmd_min, f_cmd_max);

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
        "\n[THRUST]"
        "\n  raw     = %.4f %.4f %.4f %.4f"
        "\n  clamped = %.4f %.4f %.4f %.4f"
        "\n  front_norm = %.4f, back_norm = %.4f"
        "\n  df14 = %.4f, df23 = %.4f"
        "\n  theta = %.4f %.4f"
        "\n  phi   = %.4f %.4f",
        f1_raw, f2_raw, f3_raw, f4_raw, thrust_out(0), thrust_out(1), thrust_out(2), thrust_out(3), safe_front_force_norm, safe_back_force_norm, df14, df23, theta1, theta2, phi14, phi23 );

    theta_out(0) = theta1;
    theta_out(1) = theta2;

    phi_out(0) = phi14;
    phi_out(1) = phi23;
    phi_out(2) = phi23;
    phi_out(3) = phi14;
  }

PlantCheckResult checkPlantWrench(const Eigen::Vector4d& thrust, const Eigen::Vector2d& theta, const Eigen::Vector4d& phi, const Eigen::Vector3d& moment_cmd, const Eigen::Vector3d& force_cmd)
{
  PlantCheckResult result;

  Eigen::Vector3d r1(+Lx - d * std::cos(phi(0)) * std::sin(theta(0)), +Ly + d * std::sin(phi(0)), 0.0);
  Eigen::Vector3d r2(-Lx - d * std::cos(phi(1)) * std::sin(theta(1)), +Ly + d * std::sin(phi(1)), 0.0);
  Eigen::Vector3d r3(-Lx - d * std::cos(phi(2)) * std::sin(theta(1)), -Ly + d * std::sin(phi(2)), 0.0);
  Eigen::Vector3d r4(+Lx - d * std::cos(phi(3)) * std::sin(theta(0)), -Ly + d * std::sin(phi(3)), 0.0);

  for (int i = 0; i < 4; ++i) {
    Eigen::Vector3d r_i;
    double theta_i = 0.0;
    double phi_i = phi(i);
    double spin_i = 1.0;

    if (i == 0) {r_i = r1; theta_i = theta(0); spin_i = +1.0;}
    if (i == 1) {r_i = r2; theta_i = theta(1); spin_i = -1.0;}
    if (i == 2) {r_i = r3; theta_i = theta(1); spin_i = +1.0;}
    if (i == 3) {r_i = r4; theta_i = theta(0); spin_i = -1.0;}

    Eigen::Vector3d direction_i;
    direction_i << -std::sin(theta_i) * std::cos(phi_i), std::sin(phi_i), -std::cos(theta_i) * std::cos(phi_i);

    Eigen::Vector3d force_i = thrust(i) * direction_i;
    Eigen::Vector3d moment_thrust_i = r_i.cross(force_i);
    Eigen::Vector3d moment_reaction_i = spin_i * zeta * thrust(i) * direction_i;

    result.force_actual += force_i;
    result.moment_actual += moment_thrust_i + moment_reaction_i;
  }

  result.moment_error = moment_cmd - result.moment_actual;
  result.force_error = force_cmd - result.force_actual;

  const double err_m_norm = result.moment_error.norm();
  const double err_f_norm = result.force_error.norm();

  result.problem = (!result.moment_actual.allFinite()) || (!result.force_actual.allFinite()) || (err_f_norm > check_force_tol) || (err_m_norm > check_moment_tol);

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
       << "\n    thrust = " << thrust.transpose()
       << "\n    theta  = " << theta.transpose()
       << "\n    phi    = " << phi.transpose()
       << "\033[0m";

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
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AllocatorController>());
  rclcpp::shutdown();
  return 0;
}