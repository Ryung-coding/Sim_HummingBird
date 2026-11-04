#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <rclcpp/rclcpp.hpp>
#include <multirotor_interfaces/msg/cmd.hpp>
#include <std_msgs/msg/bool.hpp>
#include <params.hpp>
#include <utils.hpp>

class PositionCmd : public rclcpp::Node {
public:
  PositionCmd() : rclcpp::Node("position_cmd")
  {
    using multirotor_interfaces::msg::Cmd;

    pub_cmd_ = this->create_publisher<Cmd>("/cmd", 10);
    trajectory_ = this->declare_parameter<std::string>("trajectory", "attPath");
    wait_for_slam_ = this->declare_parameter<bool>("wait_for_slam", false);
    if (wait_for_slam_) {
      sub_slam_valid_ = this->create_subscription<std_msgs::msg::Bool>(
        "/slam/position_valid", 10, [this](const std_msgs::msg::Bool::SharedPtr msg) {
          slam_valid_ = msg->data;
          ever_slam_valid_ |= msg->data;
        });
    }

    auto period_ns = std::chrono::nanoseconds(static_cast<int64_t>(1.0e9 / static_cast<double>(params::RATE_HZ)));
    t0_ = std::chrono::steady_clock::now();
    last_tick_ = t0_;
    timer_ = this->create_wall_timer(period_ns, std::bind(&PositionCmd::onTick, this));
  }

private:
  void onTick()
  {
    using multirotor_interfaces::msg::Cmd;
    Cmd msg;

    const auto now = std::chrono::steady_clock::now();
    if (wait_for_slam_ && (slam_valid_ || (!ever_slam_valid_ && trajectory_time_ < 8.5))) {
      trajectory_time_ += std::chrono::duration<double>(now - last_tick_).count();
      if (!ever_slam_valid_) trajectory_time_ = std::min(trajectory_time_, 8.5);
    }
    const double t = wait_for_slam_ ? trajectory_time_ :
      std::chrono::duration<double>(now - t0_).count();
    last_tick_ = now;

    const auto cmd = trajectory_ == "mapCHK" ? utils::mapCHK(t) : utils::attPath(t);
    // const auto cmd = utils::posPath(t);
    // const auto cmd = utils::stepAttPath(t);
    // const auto cmd = utils::throughWallPath(t);
    // const auto cmd = utils::circularWallPath(t);
    // const auto cmd = utils::stepPath(t);

    msg.pos_cmd[0] = cmd.x;
    msg.pos_cmd[1] = cmd.y;
    msg.pos_cmd[2] = -cmd.z;

    if constexpr (params::USE_SO3_HEADING_CMD) 
    {
      msg.att_cmd[0] = std::cos(cmd.yaw);
      msg.att_cmd[1] = std::sin(cmd.yaw);
      msg.att_cmd[2] = 0.0;
    }
    else 
    {
      msg.att_cmd[0] = cmd.roll;
      msg.att_cmd[1] = cmd.pitch;
      msg.att_cmd[2] = cmd.yaw;
    }

    pub_cmd_->publish(msg);
  }

  rclcpp::Publisher<multirotor_interfaces::msg::Cmd>::SharedPtr pub_cmd_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_slam_valid_;
  std::chrono::steady_clock::time_point t0_;
  std::chrono::steady_clock::time_point last_tick_;
  double trajectory_time_{0.0};
  bool wait_for_slam_{false};
  bool slam_valid_{false};
  bool ever_slam_valid_{false};
  std::string trajectory_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PositionCmd>());
  rclcpp::shutdown();
  return 0;
}
