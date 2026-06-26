#include <chrono>
#include <cmath>
#include <functional>
#include <rclcpp/rclcpp.hpp>
#include <multirotor_interfaces/msg/cmd.hpp>
#include <params.hpp>
#include <utils.hpp>

class PositionCmd : public rclcpp::Node {
public:
  PositionCmd() : rclcpp::Node("position_cmd")
  {
    using multirotor_interfaces::msg::Cmd;

    pub_cmd_ = this->create_publisher<Cmd>("/cmd", 10);

    auto period_ns = std::chrono::nanoseconds(static_cast<int64_t>(1.0e9 / static_cast<double>(params::RATE_HZ)));
    t0_ = std::chrono::steady_clock::now();
    timer_ = this->create_wall_timer(period_ns, std::bind(&PositionCmd::onTick, this));
  }

private:
  void onTick()
  {
    using multirotor_interfaces::msg::Cmd;
    Cmd msg;

    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    // const auto cmd = utils::trackApple(t);
    // const auto cmd = utils::takeApple(t);
    // const auto cmd = utils::positionTuningPath(t);
    const auto cmd = utils::attitudeTuningPath(t);
    // const auto cmd = utils::agilePath(t);
    // const auto cmd = utils::positionTrack(t);

    msg.pos_cmd[0] = cmd.x;
    msg.pos_cmd[1] = cmd.y;
    msg.pos_cmd[2] = -cmd.z;

    if constexpr (params::USE_SO3_HEADING_CMD) {
      msg.att_cmd[0] = std::cos(cmd.yaw);
      msg.att_cmd[1] = std::sin(cmd.yaw);
      msg.att_cmd[2] = 0.0;
    }
    else {
      msg.att_cmd[0] = cmd.roll;
      msg.att_cmd[1] = cmd.pitch;
      msg.att_cmd[2] = cmd.yaw;
    }

    pub_cmd_->publish(msg);
  }

  rclcpp::Publisher<multirotor_interfaces::msg::Cmd>::SharedPtr pub_cmd_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::chrono::steady_clock::time_point t0_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PositionCmd>());
  rclcpp::shutdown();
  return 0;
}