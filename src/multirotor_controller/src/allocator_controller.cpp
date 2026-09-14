#include <rclcpp/rclcpp.hpp>
#include <multirotor_interfaces/msg/cmd.hpp>
#include <multirotor_interfaces/msg/input.hpp>
#include <multirotor_interfaces/msg/multirotor_state.hpp>
#include <multirotor_interfaces/msg/wrench.hpp>
#include <Eigen/Dense>
#include <functional>
#include <params.hpp>
#include <utils.hpp>

class AllocatorController : public rclcpp::Node
{
public:
  AllocatorController() : rclcpp::Node("allocator_controller")
  {
    cmd_subscription_ = create_subscription<multirotor_interfaces::msg::Cmd>("/cmd", 10, std::bind(&AllocatorController::onCmd, this, std::placeholders::_1));
    wrench_subscription_ = create_subscription<multirotor_interfaces::msg::Wrench>("/wrench", 10, std::bind(&AllocatorController::onWrench, this, std::placeholders::_1));
    state_subscription_ = create_subscription<multirotor_interfaces::msg::MultirotorState>("/multirotor_state", 10, std::bind(&AllocatorController::onState, this, std::placeholders::_1));
    input_publisher_ = create_publisher<multirotor_interfaces::msg::Input>("/input", 10);

    att_cmd_.setZero();
    alpha_measured_.setZero();
    beta_measured_.setZero();
    last_time_ = now();
  }

private:
  void onCmd(const multirotor_interfaces::msg::Cmd::SharedPtr msg)
  {
    att_cmd_ << msg->att_cmd[0], msg->att_cmd[1], msg->att_cmd[2];
    have_cmd_ = true;
  }

  void onState(const multirotor_interfaces::msg::MultirotorState::SharedPtr msg)
  {
    for (int i = 0; i < 4; ++i) alpha_measured_(i) = msg->alpha[i];
    for (int i = 0; i < 2; ++i) beta_measured_(i) = msg->beta[i];
    have_state_ = true;
  }

  void onWrench(const multirotor_interfaces::msg::Wrench::SharedPtr msg)
  {
    if (!have_cmd_ || !have_state_) return;

    Eigen::Vector3d moment_cmd;
    Eigen::Vector3d force_cmd;
    Eigen::Matrix<double, 6, 1> d;

    moment_cmd << msg->moment[0], msg->moment[1], msg->moment[2];
    force_cmd << msg->force[0], msg->force[1], msg->force[2];
    for (int i = 0; i < 6; ++i) d(i) = msg->d[i];

    const rclcpp::Time current_time = now();
    double dt = (current_time - last_time_).seconds();
    last_time_ = current_time;

    if (!(dt > 0.0) || dt > 0.2) dt = 1.0 / static_cast<double>(params::RATE_HZ);

    // const auto alloc = utils::allocation_a1b1(moment_cmd, force_cmd);
    const auto alloc = utils::allocation_a4b2(moment_cmd, force_cmd, d, att_cmd_, alpha_measured_, beta_measured_, servo_read, dt);

    const auto check = utils::checkAllocation(alloc, moment_cmd, force_cmd);
    if (check.problem) RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 500, "%s", check.message.c_str());

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
  }

  rclcpp::Subscription<multirotor_interfaces::msg::Cmd>::SharedPtr cmd_subscription_;
  rclcpp::Subscription<multirotor_interfaces::msg::Wrench>::SharedPtr wrench_subscription_;
  rclcpp::Subscription<multirotor_interfaces::msg::MultirotorState>::SharedPtr state_subscription_;
  rclcpp::Publisher<multirotor_interfaces::msg::Input>::SharedPtr input_publisher_;

  Eigen::Vector3d att_cmd_;
  Eigen::Vector4d alpha_measured_;
  Eigen::Vector2d beta_measured_;
  rclcpp::Time last_time_;

  bool have_cmd_{false};
  bool servo_read{true};
  bool have_state_{false};
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AllocatorController>());
  rclcpp::shutdown();
  return 0;
}
