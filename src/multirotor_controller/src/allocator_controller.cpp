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
  static constexpr double Lx = 0.1861;
  static constexpr double Ly = 0.1861;
  static constexpr double d = 0.0500;
  static constexpr double zeta = 0.0200;
  static constexpr double f_min = 1.0e-3;
  static constexpr double f_cmd_min = 1.0e-6;
  static constexpr double f_cmd_max = 50.0;
  static constexpr double angle_limit_rad = 1.5707963267948966;
  static constexpr double eps_cos = 1.0e-6;
  static constexpr double check_tau_z_thrust_tol = 0.10;
  static constexpr double check_force_tol = 5.00;
  static constexpr double check_moment_tol = 0.50;

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

    Eigen::Vector4d B1;
    B1 << moment_cmd(0), moment_cmd(1), moment_cmd(2), force_cmd(2);

    Eigen::Matrix4d A1 = calcA1(theta_measured_, phi_measured_);
    Eigen::FullPivLU<Eigen::Matrix4d> lu_1(A1);

    if (lu_1.isInvertible()) {
      thrust_ = lu_1.solve(B1);
    }
    else {
      std::cout << "[WARN] A1 singular (rank=" << lu_1.rank() << ")\nA1:\n" << A1 << "\nB1: " << B1.transpose() << std::endl;
      thrust_ = (A1.transpose() * A1 + 1.0e-8 * Eigen::Matrix4d::Identity()).ldlt().solve(A1.transpose() * B1);
    }

    for (int i = 0; i < 4; ++i) {
      thrust_(i) = std::clamp(thrust_(i), f_cmd_min, f_cmd_max);
    }

    //----------------------

    calcA2(force_cmd(0), force_cmd(1), thrust_, theta_measured_, theta_desired_, phi_desired_);

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

    // const double f_total = std::max(0.0, -force_cmd(2));
    // const double f_each = std::clamp(f_total / 4.0, 0.0, 30.0);

    // out.u[0] = f_each;
    // out.u[1] = f_each;
    // out.u[2] = f_each;
    // out.u[3] = f_each;

    // out.u[4] = 0.0;
    // out.u[5] = 0.0;

    // out.u[6] = 0.0;
    // out.u[7] = 0.0;
    // out.u[8] = 0.0;
    // out.u[9] = 0.0;

    input_publisher_->publish(out);
  }

  Eigen::Matrix4d calcA1(const Eigen::Vector2d& theta, const Eigen::Vector4d& phi)
  {
    Eigen::Matrix4d A1;
    A1.setZero();

    const double theta1 = theta(0);
    const double theta2 = theta(1);

    const double arm_y_14 = Ly - d * std::sin(theta1);
    const double arm_y_23 = Ly - d * std::sin(theta2);

    Eigen::Vector3d r1(+Lx, +arm_y_14, 0.0);
    Eigen::Vector3d r2(-Lx, +arm_y_23, 0.0);
    Eigen::Vector3d r3(-Lx, -arm_y_23, 0.0);
    Eigen::Vector3d r4(+Lx, -arm_y_14, 0.0);

    for (int i = 0; i < 4; ++i) {
      Eigen::Vector3d r_i;
      double theta_i = 0.0;
      double phi_i = phi(i);
      double spin_i = 1.0;

      if (i == 0) {r_i = r1; theta_i = theta1; spin_i = +1.0;}
      if (i == 1) {r_i = r2; theta_i = theta2; spin_i = -1.0;}
      if (i == 2) {r_i = r3; theta_i = theta2; spin_i = +1.0;}
      if (i == 3) {r_i = r4; theta_i = theta1; spin_i = -1.0;}

      const double Fx_i = -std::sin(theta_i) * std::cos(phi_i);
      const double Fy_i = +std::sin(phi_i);
      const double Fz_i = -std::cos(theta_i) * std::cos(phi_i);

      A1(0, i) = r_i(1) * Fz_i + spin_i * zeta * Fx_i;
      A1(1, i) = -r_i(0) * Fz_i + spin_i * zeta * Fy_i;
      A1(2, i) = spin_i * zeta * Fz_i;
      A1(3, i) = Fz_i;
    }

    return A1;
  }

  void calcA2(double Fx_cmd, double Fy_cmd, const Eigen::Vector4d& thrust, const Eigen::Vector2d& theta_measured, Eigen::Vector2d& theta_out, Eigen::Vector4d& phi_out)
  {
    const double theta1_measured = theta_measured(0);
    const double theta2_measured = theta_measured(1);

    const double arm_y_14 = Ly - d * std::sin(theta1_measured);
    const double arm_y_23 = Ly - d * std::sin(theta2_measured);

    Eigen::Vector3d r1(+Lx, +arm_y_14, 0.0);
    Eigen::Vector3d r2(-Lx, +arm_y_23, 0.0);
    Eigen::Vector3d r3(-Lx, -arm_y_23, 0.0);
    Eigen::Vector3d r4(+Lx, -arm_y_14, 0.0);

    const double f1 = std::max(thrust(0), f_min);
    const double f2 = std::max(thrust(1), f_min);
    const double f3 = std::max(thrust(2), f_min);
    const double f4 = std::max(thrust(3), f_min);

    const double f14 = std::max(f1 + f4, f_min);
    const double f23 = std::max(f2 + f3, f_min);

    Eigen::Matrix4d A2;
    A2.setZero();

    A2(0, 0) = 0.0;
    A2(0, 1) = 0.0;
    A2(0, 2) = 1.0;
    A2(0, 3) = 1.0;

    A2(1, 0) = 1.0;
    A2(1, 1) = 1.0;
    A2(1, 2) = 0.0;
    A2(1, 3) = 0.0;

    A2(2, 0) = (r1(0) * f1 + r4(0) * f4) / f14;
    A2(2, 1) = (r2(0) * f2 + r3(0) * f3) / f23;
    A2(2, 2) = -(r1(1) * f1 + r4(1) * f4) / f14;
    A2(2, 3) = -(r2(1) * f2 + r3(1) * f3) / f23;

    A2(3, 0) = 1.0e-4;
    A2(3, 1) = -1.0e-4;
    A2(3, 2) = 1.0e-4;
    A2(3, 3) = -1.0e-4;

    Eigen::Vector4d B2;
    B2 << Fx_cmd, Fy_cmd, 0.0, 0.0;

    Eigen::Vector4d C2_des;
    C2_des.setZero();

    Eigen::FullPivLU<Eigen::Matrix4d> lu_2(A2);
    if (lu_2.isInvertible()) {C2_des = lu_2.solve(B2);}
    else {
      std::cout << "[WARN] A2 singular (rank=" << lu_2.rank() << ")\nA2:\n" << A2 << "\nB2: " << B2.transpose() << "\nthrust: " << thrust.transpose() << std::endl;
      C2_des = (A2.transpose() * A2 + 1.0e-8 * Eigen::Matrix4d::Identity()).ldlt().solve(A2.transpose() * B2);
    }

    convertVirtualForceToAngle(C2_des, thrust, theta_out, phi_out);
  }

  void convertVirtualForceToAngle(const Eigen::Vector4d& C2_des, const Eigen::Vector4d& thrust, Eigen::Vector2d& theta_out, Eigen::Vector4d& phi_out)
  {
    const double f1 = std::max(thrust(0), f_min);
    const double f2 = std::max(thrust(1), f_min);
    const double f3 = std::max(thrust(2), f_min);
    const double f4 = std::max(thrust(3), f_min);

    const double f14 = std::max(f1 + f4, f_min);
    const double f23 = std::max(f2 + f3, f_min);

    double Fy_14 = C2_des(0);
    double Fy_23 = C2_des(1);
    double Fx_14 = C2_des(2);
    double Fx_23 = C2_des(3);

    double Fx_direction_14 = Fx_14 / f14;
    double Fy_direction_14 = Fy_14 / f14;
    double Fx_direction_23 = Fx_23 / f23;
    double Fy_direction_23 = Fy_23 / f23;

    Fx_direction_14 = std::clamp(Fx_direction_14, -1.0, 1.0);
    Fy_direction_14 = std::clamp(Fy_direction_14, -1.0, 1.0);
    Fx_direction_23 = std::clamp(Fx_direction_23, -1.0, 1.0);
    Fy_direction_23 = std::clamp(Fy_direction_23, -1.0, 1.0);

    double normalized_horizontal_14 = std::sqrt(Fx_direction_14 * Fx_direction_14 + Fy_direction_14 * Fy_direction_14);
    double normalized_horizontal_23 = std::sqrt(Fx_direction_23 * Fx_direction_23 + Fy_direction_23 * Fy_direction_23);

    if (normalized_horizontal_14 > 1.0) {
      Fx_direction_14 = Fx_direction_14 / normalized_horizontal_14;
      Fy_direction_14 = Fy_direction_14 / normalized_horizontal_14;
    }

    if (normalized_horizontal_23 > 1.0) {
      Fx_direction_23 = Fx_direction_23 / normalized_horizontal_23;
      Fy_direction_23 = Fy_direction_23 / normalized_horizontal_23;
    }

    double phi14 = std::asin(Fy_direction_14);
    double phi23 = std::asin(Fy_direction_23);

    double cos_phi14 = std::cos(phi14);
    double cos_phi23 = std::cos(phi23);

    if (std::abs(cos_phi14) < eps_cos) {cos_phi14 = (cos_phi14 >= 0.0) ? eps_cos : -eps_cos;}
    if (std::abs(cos_phi23) < eps_cos) {cos_phi23 = (cos_phi23 >= 0.0) ? eps_cos : -eps_cos;}

    double theta1 = std::asin(std::clamp(-Fx_direction_14 / cos_phi14, -1.0, 1.0));
    double theta2 = std::asin(std::clamp(-Fx_direction_23 / cos_phi23, -1.0, 1.0));

    theta1 = std::clamp(theta1, -angle_limit_rad, angle_limit_rad);
    theta2 = std::clamp(theta2, -angle_limit_rad, angle_limit_rad);
    phi14 = std::clamp(phi14, -angle_limit_rad, angle_limit_rad);
    phi23 = std::clamp(phi23, -angle_limit_rad, angle_limit_rad);

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

    const double arm_y_14 = Ly - d * std::sin(theta(0));
    const double arm_y_23 = Ly - d * std::sin(theta(1));

    Eigen::Vector3d r1(+Lx, +arm_y_14, 0.0);
    Eigen::Vector3d r2(-Lx, +arm_y_23, 0.0);
    Eigen::Vector3d r3(-Lx, -arm_y_23, 0.0);
    Eigen::Vector3d r4(+Lx, -arm_y_14, 0.0);

    for (int i = 0; i < 4; ++i) {
      Eigen::Vector3d r_i;
      double theta_i = 0.0;
      double phi_i = phi(i);
      double spin_i = 1.0;

      if (i == 0) {r_i = r1; theta_i = theta(0); spin_i = +1.0;}
      if (i == 1) {r_i = r2; theta_i = theta(1); spin_i = -1.0;}
      if (i == 2) {r_i = r3; theta_i = theta(1); spin_i = +1.0;}
      if (i == 3) {r_i = r4; theta_i = theta(0); spin_i = -1.0;}

      Eigen::Vector3d force_i;
      force_i << thrust(i) * (-std::sin(theta_i) * std::cos(phi_i)), thrust(i) * std::sin(phi_i), thrust(i) * (-std::cos(theta_i) * std::cos(phi_i));

      Eigen::Vector3d moment_thrust_i = r_i.cross(force_i);
      Eigen::Vector3d moment_reaction_i = spin_i * zeta * thrust(i) * force_i.normalized();

      result.force_actual += force_i;
      result.moment_actual += moment_thrust_i + moment_reaction_i;
      result.tau_z_thrust += moment_thrust_i(2);
      result.tau_z_reaction += moment_reaction_i(2);
    }

    result.moment_error = moment_cmd - result.moment_actual;
    result.force_error = force_cmd - result.force_actual;

    result.problem = (!result.moment_actual.allFinite()) || (!result.force_actual.allFinite()) || (std::abs(result.tau_z_thrust) > check_tau_z_thrust_tol) || (result.force_error.norm() > check_force_tol) || (result.moment_error.norm() > check_moment_tol);

    if (result.problem) {
      std::ostringstream ss;
      ss << "[CA CHECK]"
        << "\n  tau_z_thrust  = " << result.tau_z_thrust
        << "\n  tau_z_reaction= " << result.tau_z_reaction
        << "\n  moment_cmd    = " << moment_cmd.transpose()
        << "\n  moment_actual = " << result.moment_actual.transpose()
        << "\n  moment_error  = " << result.moment_error.transpose()
        << "\n  force_cmd     = " << force_cmd.transpose()
        << "\n  force_actual  = " << result.force_actual.transpose()
        << "\n  force_error   = " << result.force_error.transpose()
        << "\n  thrust        = " << thrust.transpose()
        << "\n  theta         = " << theta.transpose()
        << "\n  phi           = " << phi.transpose();

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