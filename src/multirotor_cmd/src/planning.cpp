#include <chrono>
#include <functional>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>

#include <multirotor_interfaces/msg/cmd.hpp>
#include <multirotor_interfaces/msg/multirotor_state.hpp>

#include <params.hpp>
#include <planning.hpp>

class Planning : public rclcpp::Node
{
public:
  Planning() : rclcpp::Node("planning")
  {
    const std::string default_file = ament_index_cpp::get_package_share_directory("plant") + "/environment.yaml";
    const std::string file = declare_parameter<std::string>("environment_file", default_file);

    env_ = planning::loadEnvironment(file);
    pub_cmd_ = create_publisher<multirotor_interfaces::msg::Cmd>("/cmd", 10);
    pub_pose_ = create_publisher<multirotor_interfaces::msg::Cmd>("/planning_poses", 50);
    
    sub_state_ = create_subscription<multirotor_interfaces::msg::MultirotorState>("/multirotor_state", 10, std::bind(&Planning::onState, this, std::placeholders::_1));
    
    const auto period = std::chrono::nanoseconds(static_cast<int64_t>(1.0e9 / params::RATE_HZ));
    timer_ = create_wall_timer(period, std::bind(&Planning::onTick, this));
    RCLCPP_INFO(get_logger(), "planning environment: %s, boxes=%zu", file.c_str(), env_.obstacles.size());
  }

private:
  void publishCmd(const utils::TargetCMD& target,const rclcpp::Publisher<multirotor_interfaces::msg::Cmd>::SharedPtr& pub)
  {
    multirotor_interfaces::msg::Cmd msg;
    msg.pos_cmd[0] = target.x; msg.pos_cmd[1] = target.y; msg.pos_cmd[2] = target.z;

    for (int i = 0; i < 3; ++i) 
    {
      msg.vel_cmd[i] = target.vel(i);
      msg.acc_cmd[i] = target.acc(i);
      msg.angular_velocity_cmd[i] = target.omega(i);
      msg.angular_acceleration_cmd[i] = target.omega_dot(i);
    }

    msg.att_cmd[0] = target.roll; msg.att_cmd[1] = target.pitch; msg.att_cmd[2] = target.yaw;
    for (int i = 0; i < 4; ++i) msg.quat_cmd[i] = target.quat(i);
    msg.use_quaternion = target.use_quaternion;
    pub->publish(msg);
  }

  void onState(const multirotor_interfaces::msg::MultirotorState::SharedPtr msg)
  {
    position_ << msg->pos[0], msg->pos[1], msg->pos[2];

    have_state_ = true;
    
    if (have_plan_) 
    {
      const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time_).count();
      const auto target = trajectory_.sample(elapsed);

      const Eigen::Matrix3d R = utils::quatToRot(Eigen::Vector4d(msg->quat[0], msg->quat[1], msg->quat[2], msg->quat[3]));
      const Eigen::Matrix3d Rd = utils::quatToRot(target.quat);
      const Eigen::Vector3d rpy = utils::rotToRpy(R);
      
      max_error_ = std::max(max_error_, (position_ - Eigen::Vector3d(target.x, target.y, target.z)).norm());
      max_attitude_error_ = std::max(max_attitude_error_, Eigen::AngleAxisd(Rd.transpose() * R).angle());
      max_attitude_change_ = std::max(max_attitude_change_, Eigen::AngleAxisd(utils::rpyToRot(env_.start_rpy).transpose() * R).angle());
      max_roll_ = std::max(max_roll_, std::abs(rpy(0)));
      max_pitch_ = std::max(max_pitch_, std::abs(rpy(1)));

      if (!collision_ && !env_.collisionFree(position_, R)) 
      {
        collision_ = true;
        RCLCPP_WARN(get_logger(), "collision envelope at t=%.3f p=(%.3f %.3f %.3f) rpy=(%.3f %.3f %.3f) target=(%.3f %.3f %.3f)",
          elapsed, position_(0), position_(1), position_(2), rpy(0), rpy(1), rpy(2), target.x, target.y, target.z);
      }

      if (!reported_ && elapsed > trajectory_.duration() + 1.0) 
      {
        reported_ = true;
        const double goal_attitude_error = Eigen::AngleAxisd(utils::rpyToRot(env_.goal_rpy).transpose() * R).angle();

        RCLCPP_INFO(get_logger(),
          "tracking result: goal_error=%.3f m, goal_attitude_error=%.2f deg, max_error=%.3f m, max_attitude_error=%.2f deg, attitude_change=%.2f deg, max_roll=%.2f deg, max_pitch=%.2f deg, collision_flag=%s",
          (position_ - env_.goal).norm(), goal_attitude_error * 180.0 / M_PI, max_error_,
          max_attitude_error_ * 180.0 / M_PI, max_attitude_change_ * 180.0 / M_PI,
          max_roll_ * 180.0 / M_PI, max_pitch_ * 180.0 / M_PI, collision_ ? "true" : "false");
      }
    }
  }

  void onTick()
  {
    if (!have_state_) return;

    if (!have_plan_) 
    {
      utils::TargetCMD hold;
      hold.x = env_.start(0); hold.y = env_.start(1); hold.z = env_.start(2);
      hold.roll = env_.start_rpy(0); hold.pitch = env_.start_rpy(1); hold.yaw = env_.start_rpy(2);
      publishCmd(hold, pub_cmd_);

      if ((position_ - env_.start).norm() > params::PLANNING_START_TOLERANCE) 
      {
        settled_ = false;
        return;
      }

      const auto now = std::chrono::steady_clock::now();
      if (!settled_) { settled_ = true; settled_time_ = now; return; }
      if (std::chrono::duration<double>(now - settled_time_).count() < params::PLANNING_START_HOLD_SEC) return;
      if (now < next_attempt_) return;
      
      next_attempt_ = now + std::chrono::seconds(3);
      try 
      {
        // Liu et al. (2024), Sec. III: RRT -> SFC -> MINCO/IPOPT -> validation -> tracking.
        std::vector<Eigen::Quaterniond> path_attitude;
        const auto path = planning::RRT(env_, path_attitude);
        if (path.empty()) throw std::runtime_error("OMPL SE(3) RRT found no path");
        
        const auto corridor = planning::SFC(env_, path);
        planning::Trajectory candidate;
        if (!planning::optimize(env_, path, path_attitude, corridor, candidate)) throw std::runtime_error("IPOPT failed");
        if (!planning::validate(env_, candidate, corridor)) throw std::runtime_error("trajectory validation failed");

        trajectory_ = std::move(candidate);
        have_plan_ = true;
        
        start_time_ = std::chrono::steady_clock::now();
        
        for (double t = params::PLANNING_VISUAL_SAMPLE_SEC; t < trajectory_.duration(); t += params::PLANNING_VISUAL_SAMPLE_SEC) 
        {
          publishCmd(trajectory_.sample(t), pub_pose_);
        }

        RCLCPP_INFO(get_logger(), "planning success: RRT points=%zu, SFC=%zu, duration=%.2f s", path.size(), corridor.size(), trajectory_.duration());
      } catch (const std::exception& e) { RCLCPP_ERROR(get_logger(), "planning failed: %s", e.what()); }
      
      return;
    }
    
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time_).count();
    const auto cmd = trajectory_.sample(t);
    publishCmd(cmd, pub_cmd_);
  }

  rclcpp::Publisher<multirotor_interfaces::msg::Cmd>::SharedPtr pub_cmd_;
  rclcpp::Publisher<multirotor_interfaces::msg::Cmd>::SharedPtr pub_pose_;
  rclcpp::Subscription<multirotor_interfaces::msg::MultirotorState>::SharedPtr sub_state_;
  rclcpp::TimerBase::SharedPtr timer_;
  planning::Environment env_;
  planning::Trajectory trajectory_;
  Eigen::Vector3d position_ = Eigen::Vector3d::Zero();
  std::chrono::steady_clock::time_point settled_time_{};
  std::chrono::steady_clock::time_point start_time_{};
  std::chrono::steady_clock::time_point next_attempt_{};
  double max_error_{0.0};
  double max_attitude_error_{0.0};
  double max_attitude_change_{0.0};
  double max_roll_{0.0};
  double max_pitch_{0.0};
  bool have_state_{false};
  bool settled_{false};
  bool have_plan_{false};
  bool reported_{false};
  bool collision_{false};
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Planning>());
  rclcpp::shutdown();
  return 0;
}
