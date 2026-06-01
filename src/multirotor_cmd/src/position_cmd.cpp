#include <chrono>
#include <cmath>
#include <algorithm>

#include <rclcpp/rclcpp.hpp>
#include <multirotor_interfaces/msg/cmd.hpp>

static constexpr int RATE_HZ = 400;
static constexpr double HOVER_SEC = 5.0;
static constexpr double HOVER_ALT = -2.0;

static constexpr double PITCH_MAX = 70.0 * M_PI / 180.0;
static constexpr double ROLL_MAX = 20.0 * M_PI / 180.0;
static constexpr double YAW_MAX = 10.0 * M_PI / 180.0;

static double clampValue(double x, double lo, double hi)
{
  return std::max(lo, std::min(hi, x));
}

class PositionCmd : public rclcpp::Node {
public:
  PositionCmd() : rclcpp::Node("position_cmd")
  {
    using multirotor_interfaces::msg::Cmd;

    pub_cmd_ = this->create_publisher<Cmd>("/cmd", 10);

    auto period_ms = std::chrono::milliseconds(1000 / (RATE_HZ > 0 ? RATE_HZ : 1));
    t0_ = std::chrono::steady_clock::now();
    timer_ = this->create_wall_timer(period_ms, std::bind(&PositionCmd::onTick, this));
  }

private:
  void onTick()
  {
    using multirotor_interfaces::msg::Cmd;
    Cmd msg;

    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();

    if (t < HOVER_SEC) {
      msg.pos_cmd[0] = 0.0;
      msg.pos_cmd[1] = 0.0;
      msg.pos_cmd[2] = HOVER_ALT;

      msg.att_cmd[0] = 0.0;
      msg.att_cmd[1] = 0.0;
      msg.att_cmd[2] = 0.0;

      pub_cmd_->publish(msg);
      return;
    }

    double tm = t - HOVER_SEC;
    double phase_t = std::fmod(tm, 58.0);

    double x_cmd = 0.0;
    double y_cmd = 0.0;
    double z_cmd = HOVER_ALT;

    double roll_cmd = 0.0;
    double pitch_cmd = 0.0;
    double yaw_cmd = 0.0;

    if (phase_t < 12.0) {
      double tau = phase_t;
      double w = 2.0 * M_PI / 12.0;

      x_cmd = 1.5 * std::cos(w * tau);
      y_cmd = 1.5 * std::sin(w * tau);
      z_cmd = HOVER_ALT;

      roll_cmd = 0.25 * ROLL_MAX * std::sin(w * tau);
      pitch_cmd = 0.35 * PITCH_MAX * std::cos(w * tau);
      yaw_cmd = 0.20 * YAW_MAX * std::sin(w * tau);
    }
    else if (phase_t < 24.0) {
      double tau = phase_t - 12.0;
      double w = 2.0 * M_PI / 12.0;

      x_cmd = 1.8 * std::sin(w * tau);
      y_cmd = 1.0 * std::sin(2.0 * w * tau);
      z_cmd = HOVER_ALT + 0.4 * std::sin(w * tau);

      roll_cmd = 0.50 * ROLL_MAX * std::sin(2.0 * w * tau);
      pitch_cmd = 0.45 * PITCH_MAX * std::sin(w * tau);
      yaw_cmd = 0.30 * YAW_MAX * std::cos(w * tau);
    }
    else if (phase_t < 34.0) {
      double tau = phase_t - 24.0;
      double w = 2.0 * M_PI / 5.0;

      x_cmd = 1.2 * std::sin(w * tau);
      y_cmd = 0.8 * std::sin(0.5 * w * tau);
      z_cmd = HOVER_ALT + 0.25 * std::sin(1.5 * w * tau);

      roll_cmd = 0.70 * ROLL_MAX * std::sin(w * tau);
      pitch_cmd = 0.55 * PITCH_MAX * std::sin(0.8 * w * tau);
      yaw_cmd = 0.40 * YAW_MAX * std::sin(0.5 * w * tau);
    }
    else if (phase_t < 46.0) {
      double tau = phase_t - 34.0;
      double w = 2.0 * M_PI / 12.0;

      x_cmd = 2.0 * std::sin(0.5 * w * tau);
      y_cmd = 0.5 * std::sin(w * tau);
      z_cmd = HOVER_ALT + 0.2 * std::sin(w * tau);

      roll_cmd = 0.30 * ROLL_MAX * std::sin(w * tau);
      pitch_cmd = PITCH_MAX * std::sin(w * tau);
      yaw_cmd = 0.20 * YAW_MAX * std::sin(w * tau);
    }
    else {
      double tau = phase_t - 46.0;
      double w = 2.0 * M_PI / 12.0;

      x_cmd = 0.8 * std::sin(w * tau);
      y_cmd = 1.8 * std::sin(0.5 * w * tau);
      z_cmd = HOVER_ALT + 0.3 * std::cos(w * tau);

      roll_cmd = ROLL_MAX * std::sin(w * tau);
      pitch_cmd = 0.40 * PITCH_MAX * std::sin(0.5 * w * tau);
      yaw_cmd = 0.50 * YAW_MAX * std::sin(w * tau);
    }

    msg.pos_cmd[0] = x_cmd;
    msg.pos_cmd[1] = y_cmd;
    msg.pos_cmd[2] = z_cmd;

    msg.att_cmd[0] = clampValue(roll_cmd, -ROLL_MAX, ROLL_MAX);
    msg.att_cmd[1] = clampValue(pitch_cmd, -PITCH_MAX, PITCH_MAX);
    msg.att_cmd[2] = clampValue(yaw_cmd, -YAW_MAX, YAW_MAX);

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