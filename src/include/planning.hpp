#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include <params.hpp>
#include <utils.hpp>

namespace planning {

// A 2-D slice of the paper's binary OGM [i, j, k]^T. Obstacles in this
// simulator span the commanded flight altitude, so k is fixed for this task.
struct OccupancyGrid
{
  double resolution{0.0};
  Eigen::Vector2d origin{Eigen::Vector2d::Zero()};
  int width{0};
  int height{0};
  std::vector<bool> occupied;

  bool valid() const
  {
    return resolution > 0.0 && width > 0 && height > 0
      && occupied.size() == static_cast<std::size_t>(width * height);
  }

  bool inBounds(int i, int j) const
  {
    return i >= 0 && i < width && j >= 0 && j < height;
  }

  int flatIndex(int i, int j) const
  {
    return j * width + i;
  }

  bool isOccupied(int i, int j) const
  {
    return !inBounds(i, j) || occupied[flatIndex(i, j)];
  }

  bool worldToGrid(const Eigen::Vector2d& p, Eigen::Vector2i& ij) const
  {
    ij(0) = static_cast<int>(std::floor((p(0) - origin(0)) / resolution));
    ij(1) = static_cast<int>(std::floor((p(1) - origin(1)) / resolution));
    return inBounds(ij(0), ij(1));
  }

  Eigen::Vector2d gridToWorld(const Eigen::Vector2i& ij) const
  {
    return origin + resolution * (ij.cast<double>() + Eigen::Vector2d::Constant(0.5));
  }
};

// Paper Sec. 3.2 uses non-diagonal adjacent cells. This is its fixed-altitude
// 2-D specialization: front, back, left and right, each with unit cost.
inline std::vector<Eigen::Vector2i> aStar(
  const OccupancyGrid& ogm,
  const Eigen::Vector2i& start,
  const Eigen::Vector2i& goal)
{
  if (!ogm.valid() || ogm.isOccupied(start(0), start(1)) || ogm.isOccupied(goal(0), goal(1))) {
    return {};
  }

  const int node_count = ogm.width * ogm.height;
  const int start_id = ogm.flatIndex(start(0), start(1));
  const int goal_id = ogm.flatIndex(goal(0), goal(1));
  const double inf = std::numeric_limits<double>::infinity();

  std::vector<double> g_cost(node_count, inf);
  std::vector<int> parent(node_count, -1);
  std::vector<bool> closed(node_count, false);
  using QueueEntry = std::pair<double, int>;
  std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> open;

  const auto heuristic = [&ogm, &goal](int id) {
    const int i = id % ogm.width;
    const int j = id / ogm.width;
    return std::hypot(static_cast<double>(goal(0) - i), static_cast<double>(goal(1) - j));
  };

  g_cost[start_id] = 0.0;
  open.emplace(heuristic(start_id), start_id);

  static constexpr int d_i[4] = {1, -1, 0, 0};
  static constexpr int d_j[4] = {0, 0, 1, -1};

  while (!open.empty()) {
    const int current_id = open.top().second;
    open.pop();

    if (closed[current_id]) {
      continue;
    }
    closed[current_id] = true;

    if (current_id == goal_id) {
      break;
    }

    const int current_i = current_id % ogm.width;
    const int current_j = current_id / ogm.width;
    for (int n = 0; n < 4; ++n) {
      const int next_i = current_i + d_i[n];
      const int next_j = current_j + d_j[n];
      if (ogm.isOccupied(next_i, next_j)) {
        continue;
      }

      const int next_id = ogm.flatIndex(next_i, next_j);
      const double next_g = g_cost[current_id] + 1.0;
      if (next_g >= g_cost[next_id]) {
        continue;
      }

      g_cost[next_id] = next_g;
      parent[next_id] = current_id;
      open.emplace(next_g + heuristic(next_id), next_id);
    }
  }

  if (!closed[goal_id]) {
    return {};
  }

  std::vector<Eigen::Vector2i> path;
  for (int id = goal_id; id >= 0; id = parent[id]) {
    path.emplace_back(id % ogm.width, id / ogm.width);
    if (id == start_id) {
      break;
    }
  }
  std::reverse(path.begin(), path.end());
  return path;
}

inline bool lineIsFree(
  const OccupancyGrid& ogm,
  const Eigen::Vector2i& from,
  const Eigen::Vector2i& to)
{
  const int coarse_steps = std::max(std::abs(to(0) - from(0)), std::abs(to(1) - from(1)));
  const int step_count = std::max(1, 4 * coarse_steps);
  for (int step = 0; step <= step_count; ++step) {
    const double ratio = static_cast<double>(step) / step_count;
    const double i = from(0) + ratio * (to(0) - from(0));
    const double j = from(1) + ratio * (to(1) - from(1));

    // Check every OGM cell touched by the shortcut, not only a rounded cell.
    // This prevents a diagonal minimum-snap segment from cutting an inflated obstacle.
    for (int ii = static_cast<int>(std::floor(i)); ii <= static_cast<int>(std::ceil(i)); ++ii) {
      for (int jj = static_cast<int>(std::floor(j)); jj <= static_cast<int>(std::ceil(j)); ++jj) {
        if (ogm.isOccupied(ii, jj)) {
          return false;
        }
      }
    }
  }
  return true;
}

// Paper Sec. 3.2 / Fig. 1C-D: retain a waypoint only when skipping it would
// make its predecessor-successor segment intersect an occupied OGM cell.
inline std::vector<Eigen::Vector2i> selectEssentialWaypoints(
  const OccupancyGrid& ogm,
  const std::vector<Eigen::Vector2i>& a_star_path)
{
  if (a_star_path.size() <= 2) {
    return a_star_path;
  }

  std::vector<Eigen::Vector2i> essential;
  essential.push_back(a_star_path.front());
  std::size_t anchor = 0;

  while (anchor + 1 < a_star_path.size()) {
    std::size_t farthest = anchor + 1;
    for (std::size_t candidate = anchor + 2; candidate < a_star_path.size(); ++candidate) {
      if (!lineIsFree(ogm, a_star_path[anchor], a_star_path[candidate])) {
        break;
      }
      farthest = candidate;
    }
    essential.push_back(a_star_path[farthest]);
    anchor = farthest;
  }

  return essential;
}

// Paper Eq. (5), sigma_i(t) = sum_n p_{n,i} t^n. With zero endpoint
// velocity, acceleration and jerk, this seventh-order polynomial is the
// closed-form minimum-snap segment used here instead of solving a QP online.
struct MinimumSnapSegment
{
  Eigen::Vector3d p0{Eigen::Vector3d::Zero()};
  Eigen::Vector3d p1{Eigen::Vector3d::Zero()};
  double duration{0.0};

  utils::TargetCMD sample(double t) const
  {
    const double s = std::clamp(t / duration, 0.0, 1.0);
    const double blend = 35.0 * std::pow(s, 4) - 84.0 * std::pow(s, 5)
      + 70.0 * std::pow(s, 6) - 20.0 * std::pow(s, 7);
    const double blend_dot = (140.0 * std::pow(s, 3) - 420.0 * std::pow(s, 4)
      + 420.0 * std::pow(s, 5) - 140.0 * std::pow(s, 6)) / duration;

    const Eigen::Vector3d position = p0 + blend * (p1 - p0);
    Eigen::Vector2d tangent = (blend_dot * (p1 - p0)).head<2>();
    if (tangent.norm() < 1.0e-6) {
      tangent = (p1 - p0).head<2>();
    }

    utils::TargetCMD cmd;
    cmd.x = position(0);
    cmd.y = position(1);
    cmd.z = position(2);
    cmd.yaw = tangent.norm() > 1.0e-6 ? std::atan2(tangent(1), tangent(0)) : 0.0;
    return cmd;
  }
};

struct MinimumSnapTrajectory
{
  std::vector<MinimumSnapSegment> segments;
  std::vector<double> cumulative_time;

  bool valid() const
  {
    return !segments.empty() && segments.size() == cumulative_time.size();
  }

  utils::TargetCMD sample(double t) const
  {
    if (t <= 0.0) {
      return segments.front().sample(0.0);
    }

    for (std::size_t i = 0; i < segments.size(); ++i) {
      if (t <= cumulative_time[i]) {
        const double segment_start = i == 0 ? 0.0 : cumulative_time[i - 1];
        return segments[i].sample(t - segment_start);
      }
    }

    return segments.back().sample(segments.back().duration);
  }
};

// Paper Sec. 3.3: T_total = ||p_goal - p_start|| / v, then each segment gets
// a share proportional to sqrt(||p_{m+1} - p_m||).
inline MinimumSnapTrajectory makeMinimumSnapTrajectory(const std::vector<Eigen::Vector3d>& waypoints)
{
  MinimumSnapTrajectory trajectory;
  if (waypoints.size() < 2) {
    return trajectory;
  }

  std::vector<double> segment_length;
  double sqrt_length_sum = 0.0;
  for (std::size_t i = 1; i < waypoints.size(); ++i) {
    const double length = (waypoints[i] - waypoints[i - 1]).norm();
    segment_length.push_back(length);
    sqrt_length_sum += std::sqrt(length);
  }

  const double direct_distance = (waypoints.back() - waypoints.front()).norm();
  const double total_time = direct_distance / params::PLANNING_AVG_SPEED;
  double elapsed = 0.0;

  for (std::size_t i = 0; i < segment_length.size(); ++i) {
    const double duration = std::max(
      0.25,
      total_time * std::sqrt(segment_length[i]) / std::max(sqrt_length_sum, 1.0e-6));
    trajectory.segments.push_back({waypoints[i], waypoints[i + 1], duration});
    elapsed += duration;
    trajectory.cumulative_time.push_back(elapsed);
  }

  return trajectory;
}

}  // namespace planning
