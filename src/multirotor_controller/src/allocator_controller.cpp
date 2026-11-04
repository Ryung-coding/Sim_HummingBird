#include <rclcpp/rclcpp.hpp>
#include <multirotor_interfaces/msg/input.hpp>
#include <multirotor_interfaces/msg/multirotor_state.hpp>
#include <multirotor_interfaces/msg/wrench.hpp>
#include <multirotor_interfaces/msg/debug.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <functional>

#include <params.hpp>
#include <utils.hpp>

class AllocatorController : public rclcpp::Node
{
public:
  AllocatorController() : rclcpp::Node("allocator_controller")
  {
    wrench_subscription_ = create_subscription<multirotor_interfaces::msg::Wrench>("/wrench", 10, std::bind(&AllocatorController::onWrench, this, std::placeholders::_1));
    state_subscription_ = create_subscription<multirotor_interfaces::msg::MultirotorState>("/multirotor_state", 10, std::bind(&AllocatorController::onState, this, std::placeholders::_1));
    input_publisher_ = create_publisher<multirotor_interfaces::msg::Input>("/input", 10);
    debug_publisher_ = create_publisher<multirotor_interfaces::msg::Debug>("/allocation_debug", 10);

    alpha_measured_.setZero();
    beta_measured_.setZero();
  }

private:
  void onState(const multirotor_interfaces::msg::MultirotorState::SharedPtr msg)
  {
    for (int i = 0; i < 4; ++i) alpha_measured_(i) = msg->alpha[i];
    for (int i = 0; i < 2; ++i) beta_measured_(i) = msg->beta[i];
    have_state_ = true;
  }

  void onWrench(const multirotor_interfaces::msg::Wrench::SharedPtr msg)
  {
    if (!have_state_) return;

    Eigen::Vector3d moment_cmd;
    Eigen::Vector3d force_cmd;

    moment_cmd << msg->moment[0], msg->moment[1], msg->moment[2];
    force_cmd << msg->force[0], msg->force[1], msg->force[2];

    const auto alloc = utils::allocation_a1b1(moment_cmd, force_cmd);
    // No thrust sensor: estimate applied wrench using the previous thrust command and measured servo angles.
    const auto wrench_real = utils::evaluateA4B2Wrench(last_f_cmd_, beta_measured_, alpha_measured_);
    last_f_cmd_ = alloc.f;
    multirotor_interfaces::msg::Input out;

    out.f[0] = alloc.f(0);
    out.f[1] = alloc.f(1);
    out.f[2] = alloc.f(2);
    out.f[3] = alloc.f(3);

    out.alpha[0] = alloc.alpha(0);
    out.alpha[1] = alloc.alpha(1);
    out.alpha[2] = alloc.alpha(2);
    out.alpha[3] = alloc.alpha(3);

    out.beta[0] = alloc.beta(0);
    out.beta[1] = alloc.beta(1);

    input_publisher_->publish(out);

    multirotor_interfaces::msg::Debug debug;

    for (int i = 0; i < 6; ++i)
    {
      debug.wrench_cmd[i] = alloc.wrench_cmd(i);
      debug.wrench_alloc[i] = alloc.wrench_alloc(i);
      debug.wrench_real[i] = wrench_real(i);
    }

    debug_publisher_->publish(debug);

  }

  rclcpp::Subscription<multirotor_interfaces::msg::Wrench>::SharedPtr wrench_subscription_;
  rclcpp::Subscription<multirotor_interfaces::msg::MultirotorState>::SharedPtr state_subscription_;
  rclcpp::Publisher<multirotor_interfaces::msg::Input>::SharedPtr input_publisher_;
  rclcpp::Publisher<multirotor_interfaces::msg::Debug>::SharedPtr debug_publisher_;

  Eigen::Vector4d last_f_cmd_ = Eigen::Vector4d::Zero();
  Eigen::Vector4d alpha_measured_;
  Eigen::Vector2d beta_measured_;

  bool have_state_{false};

};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AllocatorController>());
  rclcpp::shutdown();
  return 0;
}
