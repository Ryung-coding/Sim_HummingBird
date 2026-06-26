#include <rclcpp/rclcpp.hpp>
#include <multirotor_interfaces/msg/cmd.hpp>
#include <multirotor_interfaces/msg/multirotor_state.hpp>
#include <multirotor_interfaces/msg/wrench.hpp>

#include <Eigen/Dense>
#include <functional>
#include <params.hpp>
#include <utils.hpp>

class WrenchController : public rclcpp::Node
{
public:
  WrenchController() : rclcpp::Node("wrench_controller")
  {
    sub_cmd_ = this->create_subscription<multirotor_interfaces::msg::Cmd>("/cmd", 10, std::bind(&WrenchController::onCmd, this, std::placeholders::_1));
    sub_state_ = this->create_subscription<multirotor_interfaces::msg::MultirotorState>("/multirotor_state", 10, std::bind(&WrenchController::onState, this, std::placeholders::_1));
    pub_wrench_ = this->create_publisher<multirotor_interfaces::msg::Wrench>("/wrench", 10);

    last_time_ = this->now();

    pos_cmd_.setZero();
    att_cmd_.setZero();
    pos_.setZero();
    vel_.setZero();
    rpy_.setZero();
    W_.setZero();
    pos_i_.setZero();
    att_i_.setZero();
  }

private:
  void onCmd(const multirotor_interfaces::msg::Cmd::SharedPtr msg)
  {
    pos_cmd_ << static_cast<double>(msg->pos_cmd[0]), static_cast<double>(msg->pos_cmd[1]), static_cast<double>(msg->pos_cmd[2]);
    att_cmd_ << static_cast<double>(msg->att_cmd[0]), static_cast<double>(msg->att_cmd[1]), static_cast<double>(msg->att_cmd[2]);

    have_cmd_ = true;
    tryPublish();
  }

  void onState(const multirotor_interfaces::msg::MultirotorState::SharedPtr msg)
  {
    pos_ << static_cast<double>(msg->pos[0]), static_cast<double>(msg->pos[1]), static_cast<double>(msg->pos[2]);
    vel_ << static_cast<double>(msg->vel[0]), static_cast<double>(msg->vel[1]), static_cast<double>(msg->vel[2]);
    rpy_ << static_cast<double>(msg->rpy[0]), static_cast<double>(msg->rpy[1]), static_cast<double>(msg->rpy[2]);
    W_ << static_cast<double>(msg->w_rpy[0]), static_cast<double>(msg->w_rpy[1]), static_cast<double>(msg->w_rpy[2]);

    have_state_ = true;
    tryPublish();
  }

  Eigen::Vector3d positionController(double dt)
  {
    const Eigen::Vector3d Kp = utils::vec3(params::Kp_pos);
    const Eigen::Vector3d Ki = utils::vec3(params::Ki_pos);
    const Eigen::Vector3d Kd = utils::vec3(params::Kd_pos);
    const Eigen::Vector3d i_sat = utils::vec3(params::pos_i_sat);

    const Eigen::Vector3d e = pos_cmd_ - pos_;

    pos_i_ += e * dt;
    pos_i_ = utils::clampVec3(pos_i_, i_sat);

    Eigen::Vector3d F_world = Kp.cwiseProduct(e) + Ki.cwiseProduct(pos_i_) - Kd.cwiseProduct(vel_);
    F_world(2) -= params::mass * params::grav;

    return F_world;
  }

  Eigen::Vector3d attitudeController(const Eigen::Matrix3d& R, const Eigen::Matrix3d& Rd, double dt)
  {
    const Eigen::Vector3d Wd = Eigen::Vector3d::Zero();
    const Eigen::Vector3d Wd_dot = Eigen::Vector3d::Zero();

    const Eigen::Matrix3d RtRd = R.transpose() * Rd;

    Eigen::Vector3d eR = 0.5 * utils::vee(RtRd.transpose() - RtRd);

    const double eR_norm = eR.norm();
    if (eR_norm > params::ER_NORM_MAX) eR = eR * (params::ER_NORM_MAX / eR_norm);

    const Eigen::Vector3d eW = W_ - R.transpose() * Rd * Wd;

    att_i_ += (eW + eR) * dt;
    att_i_ = utils::clampVec3(att_i_, utils::vec3(params::att_i_sat));

    const Eigen::Matrix3d J = utils::diag3(params::J);
    const Eigen::Matrix3d kR = utils::diag3(params::kR);
    const Eigen::Matrix3d kW = utils::diag3(params::kW);
    const Eigen::Matrix3d kI = utils::diag3(params::kI);

    Eigen::Vector3d M = -kR * eR - kW * eW - kI * att_i_ - J * utils::hat(W_) * RtRd * Wd + J * RtRd * Wd_dot;
    M = utils::clampVec3(M, utils::vec3(params::torque_sat));

    return M;
  }

  void tryPublish()
  {
    if (!have_state_ || !have_cmd_) return;

    const rclcpp::Time now = this->now();
    double dt = (now - last_time_).seconds();
    last_time_ = now;

    if (!(dt > 0.0) || dt > 0.2) dt = 1.0 / static_cast<double>(params::RATE_HZ);

    const Eigen::Matrix3d R = utils::rpyToRot(rpy_);
    const Eigen::Matrix3d Rd = params::USE_SO3_HEADING_CMD ? utils::headingToRot(att_cmd_) : utils::rpyToRot(att_cmd_);

    const Eigen::Vector3d F_world = positionController(dt);
    Eigen::Vector3d F_body = R.transpose() * F_world;
    F_body = utils::clampVec3(F_body, utils::vec3(params::force_body_sat));

    const Eigen::Vector3d M_body = attitudeController(R, Rd, dt);

    multirotor_interfaces::msg::Wrench msg;

    msg.force[0] = static_cast<float>(F_body(0));
    msg.force[1] = static_cast<float>(F_body(1));
    msg.force[2] = static_cast<float>(F_body(2));

    msg.moment[0] = static_cast<float>(M_body(0));
    msg.moment[1] = static_cast<float>(M_body(1));
    msg.moment[2] = static_cast<float>(M_body(2));

    pub_wrench_->publish(msg);
  }

  rclcpp::Subscription<multirotor_interfaces::msg::Cmd>::SharedPtr sub_cmd_;
  rclcpp::Subscription<multirotor_interfaces::msg::MultirotorState>::SharedPtr sub_state_;
  rclcpp::Publisher<multirotor_interfaces::msg::Wrench>::SharedPtr pub_wrench_;

  rclcpp::Time last_time_;

  Eigen::Vector3d pos_cmd_;
  Eigen::Vector3d att_cmd_;
  Eigen::Vector3d pos_;
  Eigen::Vector3d vel_;
  Eigen::Vector3d rpy_;
  Eigen::Vector3d W_;

  Eigen::Vector3d pos_i_;
  Eigen::Vector3d att_i_;

  bool have_cmd_{false};
  bool have_state_{false};
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WrenchController>());
  rclcpp::shutdown();
  return 0;
}
