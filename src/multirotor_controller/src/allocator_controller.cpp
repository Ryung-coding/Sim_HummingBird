#include <rclcpp/rclcpp.hpp>
#include <multirotor_interfaces/msg/wrench.hpp>
#include <multirotor_interfaces/msg/input.hpp>
#include <multirotor_interfaces/msg/multirotor_state.hpp>

#include <Eigen/Dense>
#include <functional>
#include <params.hpp>
#include <utils.hpp>

class AllocatorController : public rclcpp::Node
{
public:
  AllocatorController() : rclcpp::Node("allocator_controller")
  {
    wrench_subscription_ = this->create_subscription<multirotor_interfaces::msg::Wrench>("/wrench", 10, std::bind(&AllocatorController::onWrench, this, std::placeholders::_1));
    state_subscription_ = this->create_subscription<multirotor_interfaces::msg::MultirotorState>("/multirotor_state", 10, std::bind(&AllocatorController::onState, this, std::placeholders::_1));
    input_publisher_ = this->create_publisher<multirotor_interfaces::msg::Input>("/input", 10);

    theta_measured_.setZero();
    phi_measured_.setZero();
  }

private:
  void onState(const multirotor_interfaces::msg::MultirotorState::SharedPtr msg)
  {
    theta_measured_(0) = static_cast<double>(msg->theta[0]);
    theta_measured_(1) = static_cast<double>(msg->theta[1]);
    theta_measured_(2) = static_cast<double>(msg->theta[2]);
    theta_measured_(3) = static_cast<double>(msg->theta[3]);

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

    const auto alloc = utils::allocation_P2T2(moment_cmd, force_cmd, theta_measured_, phi_measured_);
    // const auto alloc = utils::allocation_P4T4(moment_cmd, force_cmd, theta_measured_, phi_measured_);
    const auto check = utils::checkAllocation(alloc, moment_cmd, force_cmd);

    if (check.problem) RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 500, "%s", check.message.c_str());

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
      "\n[ALLOCATION]"
      "\n  f     = %.4f %.4f %.4f %.4f"
      "\n  theta = %.4f %.4f %.4f %.4f"
      "\n  phi   = %.4f %.4f %.4f %.4f",
      alloc.f(0), alloc.f(1), alloc.f(2), alloc.f(3),
      alloc.theta(0), alloc.theta(1), alloc.theta(2), alloc.theta(3),
      alloc.phi(0), alloc.phi(1), alloc.phi(2), alloc.phi(3));

    multirotor_interfaces::msg::Input out;

    out.f[0] = alloc.f(0);
    out.f[1] = alloc.f(1);
    out.f[2] = alloc.f(2);
    out.f[3] = alloc.f(3);

    out.theta[0] = alloc.theta(0);
    out.theta[1] = alloc.theta(1);
    out.theta[2] = alloc.theta(2);
    out.theta[3] = alloc.theta(3);

    out.phi[0] = alloc.phi(0);
    out.phi[1] = alloc.phi(1);
    out.phi[2] = alloc.phi(2);
    out.phi[3] = alloc.phi(3);

    input_publisher_->publish(out);
  }

  rclcpp::Subscription<multirotor_interfaces::msg::Wrench>::SharedPtr wrench_subscription_;
  rclcpp::Subscription<multirotor_interfaces::msg::MultirotorState>::SharedPtr state_subscription_;
  rclcpp::Publisher<multirotor_interfaces::msg::Input>::SharedPtr input_publisher_;

  Eigen::Vector4d theta_measured_;
  Eigen::Vector4d phi_measured_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AllocatorController>());
  rclcpp::shutdown();
  return 0;
}