#include <chrono>
#include <functional>
#include <stdexcept>
#include <string>
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
    path_ = this->declare_parameter<std::string>("path", "att");
    if (path_ != "pos" && path_ != "att" && path_ != "step_att" &&
        path_ != "circle_demo" && path_ != "demo_straight") {
      throw std::invalid_argument("unknown path: " + path_);
    }
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
    utils::TargetCMD cmd;
    if (path_ == "pos") cmd = utils::posPath(t);
    else if (path_ == "att") cmd = utils::attPath(t);
    else if (path_ == "step_att") cmd = utils::stepAttPath(t);
    else if (path_ == "circle_demo") cmd = utils::circle_demo(t);
    else cmd = utils::demo_straight(t);

    msg.pos_cmd[0] = params::initial_pose_x_m + cmd.x;
    msg.pos_cmd[1] = params::initial_pose_y_m + cmd.y;
    msg.pos_cmd[2] = params::initial_pose_z_m + cmd.z;
    msg.att_cmd[0] = params::initial_pose_roll_rad + cmd.roll;
    msg.att_cmd[1] = params::initial_pose_pitch_rad + cmd.pitch;
    msg.att_cmd[2] = params::initial_pose_yaw_rad + cmd.yaw;

    pub_cmd_->publish(msg);
  }

  rclcpp::Publisher<multirotor_interfaces::msg::Cmd>::SharedPtr pub_cmd_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::chrono::steady_clock::time_point t0_;
  std::string path_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PositionCmd>());
  rclcpp::shutdown();
  return 0;
}
