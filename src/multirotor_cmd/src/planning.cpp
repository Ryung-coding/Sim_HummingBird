#include <chrono>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>

#include <multirotor_interfaces/msg/cmd.hpp>
#include <multirotor_interfaces/msg/multirotor_state.hpp>

#include <params.hpp>
#include <planning.hpp>

class Planning : public rclcpp::Node
{
public:
  Planning() : rclcpp::Node("planning")
  {
    using std::placeholders::_1;

    pub_cmd_ = create_publisher<multirotor_interfaces::msg::Cmd>("/cmd", 10);
    const auto planning_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    pub_path_ = create_publisher<nav_msgs::msg::Path>(
      "/planning/path", planning_qos);
    pub_waypoints_ = create_publisher<nav_msgs::msg::Path>(
      "/planning/waypoints", planning_qos);

    sub_ogm_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/planning/ogm",
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local(),
      std::bind(&Planning::onOgm, this, _1));
    sub_state_ = create_subscription<multirotor_interfaces::msg::MultirotorState>(
      "/multirotor_state", 10, std::bind(&Planning::onState, this, _1));

    const auto period = std::chrono::nanoseconds(
      static_cast<int64_t>(1.0e9 / static_cast<double>(params::RATE_HZ)));
    timer_ = create_wall_timer(period, std::bind(&Planning::onTick, this));
  }

private:
  Eigen::Vector3d plannedStart() const
  {
    return utils::vec3(params::PLANNING_START);
  }

  void updateStartReady()
  {
    const double error = (position_ - plannedStart()).norm();
    const auto now = std::chrono::steady_clock::now();

    if (error > params::PLANNING_START_TOLERANCE) {
      start_ready_ = false;
      have_start_settle_time_ = false;
      return;
    }

    if (!have_start_settle_time_) {
      start_settle_time_ = now;
      have_start_settle_time_ = true;
      return;
    }

    start_ready_ = std::chrono::duration<double>(now - start_settle_time_).count()
      >= params::PLANNING_START_HOLD_SEC;
  }

  void onOgm(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {
    ogm_.resolution = msg->info.resolution;
    ogm_.origin << msg->info.origin.position.x, msg->info.origin.position.y;
    ogm_.width = static_cast<int>(msg->info.width);
    ogm_.height = static_cast<int>(msg->info.height);
    ogm_.occupied.assign(msg->data.size(), false);

    for (std::size_t i = 0; i < msg->data.size(); ++i) {
      // The plant publishes a complete global map. Unknown cells stay unsafe.
      ogm_.occupied[i] = msg->data[i] != 0;
    }

    have_ogm_ = ogm_.valid();
    have_trajectory_ = false;
    tryPlan();
  }

  void onState(const multirotor_interfaces::msg::MultirotorState::SharedPtr msg)
  {
    position_ << msg->pos[0], msg->pos[1], msg->pos[2];
    have_state_ = true;
    updateStartReady();
    tryPlan();
  }

  void tryPlan()
  {
    if (!have_ogm_ || !have_state_ || !start_ready_ || have_trajectory_) {
      return;
    }

    const Eigen::Vector3d start = plannedStart();
    const Eigen::Vector3d goal = utils::vec3(params::PLANNING_GOAL);
    const Eigen::Vector2d start_xz(start(0), start(2));
    const Eigen::Vector2d goal_xz(goal(0), goal(2));
    Eigen::Vector2i start_ij;
    Eigen::Vector2i goal_ij;

    if (!ogm_.worldToGrid(start_xz, start_ij) || !ogm_.worldToGrid(goal_xz, goal_ij)) {
      RCLCPP_ERROR(get_logger(), "planning start or goal is outside /planning/ogm");
      return;
    }

    const auto a_star_path = planning::aStar(ogm_, start_ij, goal_ij);
    if (a_star_path.empty()) {
      RCLCPP_ERROR(get_logger(), "A* failed: start or goal is occupied, or no collision-free path exists");
      return;
    }

    const auto essential_ij = planning::selectEssentialWaypoints(ogm_, a_star_path);
    std::vector<Eigen::Vector3d> waypoints;
    waypoints.reserve(essential_ij.size());
    for (const auto& ij : essential_ij) {
      const Eigen::Vector2d xz = ogm_.gridToWorld(ij);
      waypoints.emplace_back(xz(0), 0.0, xz(1));
    }

    // Start sigma = [x, y, z, psi]^T at the settled hover point (paper Eq. (3));
    // all later sigma positions lie on the A* route.
    waypoints.front() = start;
    waypoints.back() = goal;
    trajectory_ = planning::makeMinimumSnapTrajectory(waypoints);
    have_trajectory_ = trajectory_.valid();
    trajectory_start_ = std::chrono::steady_clock::now();

    if (have_trajectory_) {
      publishPlanVisualization(waypoints);
      RCLCPP_INFO(
        get_logger(),
        "OGM %dx%d: A* cells=%zu, essential waypoints=%zu, duration=%.2f s",
        ogm_.width, ogm_.height, a_star_path.size(), essential_ij.size(),
        trajectory_.cumulative_time.back());
    }
  }

  void publishPlanVisualization(const std::vector<Eigen::Vector3d>& waypoints)
  {
    nav_msgs::msg::Path waypoint_msg;
    const auto now_ns = get_clock()->now().nanoseconds();
    waypoint_msg.header.stamp.sec = static_cast<int32_t>(now_ns / 1000000000LL);
    waypoint_msg.header.stamp.nanosec = static_cast<uint32_t>(now_ns % 1000000000LL);
    waypoint_msg.header.frame_id = "world_zdown";
    waypoint_msg.poses.reserve(waypoints.size());
    for (const auto& waypoint : waypoints) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = waypoint_msg.header;
      pose.pose.position.x = waypoint(0);
      pose.pose.position.y = waypoint(1);
      pose.pose.position.z = waypoint(2);
      pose.pose.orientation.w = 1.0;
      waypoint_msg.poses.push_back(pose);
    }
    pub_waypoints_->publish(waypoint_msg);

    // This samples the actual seventh-order trajectory, rather than drawing
    // only A* chords, so the viewer shows the command sent to the controller.
    constexpr double visualization_dt = 0.05;
    const double total_time = trajectory_.cumulative_time.back();
    const int sample_count = std::max(
      1, static_cast<int>(std::ceil(total_time / visualization_dt)));

    nav_msgs::msg::Path path_msg;
    path_msg.header = waypoint_msg.header;
    path_msg.poses.reserve(static_cast<std::size_t>(sample_count + 1));
    for (int sample = 0; sample <= sample_count; ++sample) {
      const auto target = trajectory_.sample(total_time * sample / sample_count);
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path_msg.header;
      pose.pose.position.x = target.x;
      pose.pose.position.y = target.y;
      pose.pose.position.z = target.z;
      pose.pose.orientation.w = 1.0;
      path_msg.poses.push_back(pose);
    }
    pub_path_->publish(path_msg);
  }

  void publishTarget(const utils::TargetCMD& target)
  {
    multirotor_interfaces::msg::Cmd msg;
    msg.pos_cmd[0] = static_cast<float>(target.x);
    msg.pos_cmd[1] = static_cast<float>(target.y);
    msg.pos_cmd[2] = static_cast<float>(target.z);

    if constexpr (params::USE_SO3_HEADING_CMD) {
      msg.att_cmd[0] = static_cast<float>(std::cos(target.yaw));
      msg.att_cmd[1] = static_cast<float>(std::sin(target.yaw));
      msg.att_cmd[2] = 0.0F;
    }
    else {
      msg.att_cmd[0] = static_cast<float>(target.roll);
      msg.att_cmd[1] = static_cast<float>(target.pitch);
      msg.att_cmd[2] = static_cast<float>(target.yaw);
    }

    pub_cmd_->publish(msg);
  }

  void onTick()
  {
    if (!have_state_) {
      return;
    }

    if (!have_trajectory_) {
      utils::TargetCMD start_target;
      const Eigen::Vector3d start = plannedStart();
      start_target.x = start(0);
      start_target.y = start(1);
      start_target.z = start(2);
      publishTarget(start_target);
      return;
    }

    const double t = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - trajectory_start_).count();
    publishTarget(trajectory_.sample(t));
  }

  rclcpp::Publisher<multirotor_interfaces::msg::Cmd>::SharedPtr pub_cmd_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_waypoints_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_ogm_;
  rclcpp::Subscription<multirotor_interfaces::msg::MultirotorState>::SharedPtr sub_state_;
  rclcpp::TimerBase::SharedPtr timer_;

  planning::OccupancyGrid ogm_;
  planning::MinimumSnapTrajectory trajectory_;
  Eigen::Vector3d position_{Eigen::Vector3d::Zero()};
  std::chrono::steady_clock::time_point trajectory_start_{};
  std::chrono::steady_clock::time_point start_settle_time_{};
  bool have_start_settle_time_{false};
  bool start_ready_{false};
  bool have_ogm_{false};
  bool have_state_{false};
  bool have_trajectory_{false};
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Planning>());
  rclcpp::shutdown();
  return 0;
}