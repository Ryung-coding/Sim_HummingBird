#pragma once

#include <Eigen/Dense>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/SE3StateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <yaml-cpp/yaml.h>
#include <IpIpoptApplication.hpp>
#include <IpSolveStatistics.hpp>
#include <IpTNLP.hpp>

#include <algorithm>
#include <utility>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <params.hpp>
#include <utils.hpp>

namespace planning {

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix86d = Eigen::Matrix<double, 8, 6>;

// Environment ======================================================
struct Box
{
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d half_size = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
};

struct Environment
{
  Eigen::Vector3d lower = Eigen::Vector3d::Zero();
  Eigen::Vector3d upper = Eigen::Vector3d::Zero();
  // HB_model.xml rotor centers and propeller span are enclosed by this cuboid.
  Eigen::Vector3d vehicle_half_size = Eigen::Vector3d::Zero();
  Eigen::Vector3d start = Eigen::Vector3d::Zero();
  Eigen::Vector3d goal = Eigen::Vector3d::Zero();
  Eigen::Vector3d start_rpy = Eigen::Vector3d::Zero();
  Eigen::Vector3d goal_rpy = Eigen::Vector3d::Zero();
  std::vector<Box> obstacles;

  bool collisionFree(const Eigen::Vector3d& p, const Eigen::Matrix3d& R) const
  {
    // Liu et al. (2024), Eq. (18): full vehicle cuboid inside map bounds and outside obstacle boxes.
    for (const auto& v : utils::boxVertices(vehicle_half_size))
    {
      const Eigen::Vector3d x = p + R * v;
      if ((x.array() < lower.array()).any() || (x.array() > upper.array()).any()) return false;
    }
    for (const auto& box : obstacles)
    {
      if (utils::orientedBoxesOverlap(p, R, vehicle_half_size, box.center, box.R, box.half_size)) return false;
    }
    return true;
  }
};

inline Environment loadEnvironment(const std::string& file)
{
  const YAML::Node yaml = YAML::LoadFile(file);
  if (yaml["frame"].as<std::string>() != "controller_z_down") throw std::runtime_error("environment frame must be controller_z_down");
  const auto vec = [](const YAML::Node& n) {
    if (!n || n.size() != 3) throw std::runtime_error("environment vector must have 3 elements");
    return Eigen::Vector3d(n[0].as<double>(), n[1].as<double>(), n[2].as<double>());
  };
  Environment env;
  env.lower = vec(yaml["bounds"]["min"]);
  env.upper = vec(yaml["bounds"]["max"]);
  env.start << yaml["start"]["x"].as<double>(), yaml["start"]["y"].as<double>(), yaml["start"]["z"].as<double>();
  env.goal << yaml["goal"]["x"].as<double>(), yaml["goal"]["y"].as<double>(), yaml["goal"]["z"].as<double>();
  env.start_rpy << yaml["start"]["roll"].as<double>(), yaml["start"]["pitch"].as<double>(), yaml["start"]["yaw"].as<double>();
  env.goal_rpy << yaml["goal"]["roll"].as<double>(), yaml["goal"]["pitch"].as<double>(), yaml["goal"]["yaw"].as<double>();
  env.vehicle_half_size = vec(yaml["vehicle"]["half_size"]);
  for (const auto& item : yaml["obstacles"]) {
    const Eigen::Vector3d size = vec(item["size"]);
    if ((size.array() <= 0.0).any()) throw std::runtime_error("obstacle size must be positive");
    // The cylinder's enclosing box keeps RRT, RILS and validation conservative.
    if (item["type"] && item["type"].as<std::string>() != "cylinder") throw std::runtime_error("unsupported obstacle type");
    if (item["type"] && std::abs(size(0) - size(1)) > 1.0e-9) throw std::runtime_error("cylinder x/y diameter must match");
    Box box{vec(item["center"]), 0.5 * size};
    if (item["rpy"]) box.R = utils::rpyToRot(vec(item["rpy"]));
    env.obstacles.push_back(box);
  }
  if ((env.upper.array() <= env.lower.array()).any() || (env.vehicle_half_size.array() <= 0.0).any()) {
    throw std::runtime_error("invalid planning bounds or vehicle size");
  }
  return env;
}

// RRT ===============================================================
inline std::vector<Eigen::Vector3d> RRT(const Environment& env, std::vector<Eigen::Quaterniond>& path_attitude)
{
  namespace ob = ompl::base;
  namespace og = ompl::geometric;
  Environment search_env = env;
  search_env.vehicle_half_size.array() += params::PLANNING_RRT_CLEARANCE;
  const auto segmentFree = [&](const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                               const Eigen::Quaterniond& qa, const Eigen::Quaterniond& qb)
  {
    const int steps = std::max(2, static_cast<int>(std::ceil((b - a).norm() / 0.03)));
    for (int i = 0; i <= steps; ++i)
    {
      const double u = static_cast<double>(i) / steps;
      if (!search_env.collisionFree(a + (b - a) * u, qa.slerp(u, qb).toRotationMatrix())) return false;
    }
    return true;
  };
  auto space = std::make_shared<ob::SE3StateSpace>();
  ob::RealVectorBounds bounds(3);
  for (int i = 0; i < 3; ++i) {
    bounds.setLow(i, env.lower(i) + search_env.vehicle_half_size(i));
    bounds.setHigh(i, env.upper(i) - search_env.vehicle_half_size(i));
  }
  space->setBounds(bounds);
  og::SimpleSetup setup(space);
  setup.setStateValidityChecker([&](const ob::State* state) {
    const auto* s = state->as<ob::SE3StateSpace::StateType>();
    const Eigen::Vector3d p(s->getX(), s->getY(), s->getZ());
    const auto& r = s->rotation();
    Eigen::Quaterniond q(r.w, r.x, r.y, r.z);
    const Eigen::Matrix3d R = q.normalized().toRotationMatrix();
    const Eigen::Vector3d rpy = utils::rotToRpy(R);
    return std::abs(rpy(0)) <= params::PLANNING_ROLL_LIMIT_RAD
      && std::abs(rpy(1)) <= params::PLANNING_PITCH_LIMIT_RAD
      && search_env.collisionFree(p, R);
  });
  auto planner = std::make_shared<og::RRT>(setup.getSpaceInformation());
  planner->setRange(params::PLANNING_RRT_RANGE);
  setup.setPlanner(planner);
  ob::ScopedState<ob::SE3StateSpace> start(space), goal(space);
  start->setXYZ(env.start(0), env.start(1), env.start(2));
  goal->setXYZ(env.goal(0), env.goal(1), env.goal(2));
  Eigen::Quaterniond qs(utils::rpyToRot(env.start_rpy)), qg(utils::rpyToRot(env.goal_rpy));
  start->rotation().x = qs.x(); start->rotation().y = qs.y();
  start->rotation().z = qs.z(); start->rotation().w = qs.w();
  goal->rotation().x = qg.x(); goal->rotation().y = qg.y();
  goal->rotation().z = qg.z(); goal->rotation().w = qg.w();
  setup.setStartAndGoalStates(start, goal);
  if (setup.solve(params::PLANNING_RRT_TIME_SEC) != ob::PlannerStatus::EXACT_SOLUTION) return {};
  setup.getSolutionPath().interpolate();
  std::vector<Eigen::Vector3d> raw;
  std::vector<Eigen::Quaterniond> raw_q;
  for (const auto* state : setup.getSolutionPath().getStates()) {
    const auto* s = state->as<ob::SE3StateSpace::StateType>();
    raw.emplace_back(s->getX(), s->getY(), s->getZ());
    const auto& r = s->rotation();
    raw_q.emplace_back(r.w, r.x, r.y, r.z);
  }
  std::vector<Eigen::Vector3d> path{env.start};
  path_attitude = {qs};
  const Eigen::Vector3d turn = env.start + params::PLANNING_PREALIGN_FRACTION * (env.goal - env.start);
  if (segmentFree(env.start, turn, qs, qg)
      && segmentFree(turn, env.goal, qg, qg)) {
    if (qs.angularDistance(qg) > params::PLANNING_PREALIGN_ANGLE_RAD)
    {
      path.push_back(0.5 * (env.start + turn));
      path_attitude.push_back(qs.slerp(0.5, qg));
    }
    path.push_back(turn);
    path_attitude.push_back(qg);
    path.push_back(env.goal);
    path_attitude.push_back(qg);
  } else {
    std::size_t anchor = 0;
    while (anchor + 1 < raw.size()) {
      std::size_t next = raw.size() - 1;
      while (next > anchor + 1 && !segmentFree(raw[anchor], raw[next], raw_q[anchor], raw_q[next])) --next;
      path.push_back(raw[next]);
      path_attitude.push_back(raw_q[next]);
      anchor = next;
    }
    path.back() = env.goal;
    path_attitude.back() = qg;
  }
  return path;
}

// SFC ===============================================================
struct Plane
{
  Eigen::Vector3d n = Eigen::Vector3d::Zero();
  double d = 0.0; // n.dot(x) <= d
};

using Corridor = std::vector<Plane>;

inline std::vector<Corridor> SFC(const Environment& env, const std::vector<Eigen::Vector3d>& path)
{
  // Liu et al. (2017), Sec. III-B: RILS fitted ellipsoid and dilated tangent planes.
  // For a Box, its closest point in the ellipsoid metric replaces occupied voxels.
  std::vector<Corridor> result;
  for (std::size_t i = 1; i < path.size(); ++i) {
    const Eigen::Vector3d p0 = path[i - 1], p1 = path[i];
    const double length = (p1 - p0).norm();
    if (length < 1.0e-6) throw std::runtime_error("zero-length RRT segment");
    const Eigen::Vector3d center = 0.5 * (p0 + p1);
    const Eigen::Vector3d e1 = (p1 - p0) / length;
    Eigen::Vector3d nearest_direction = e1.unitOrthogonal();
    double nearest_distance = std::numeric_limits<double>::infinity();
    for (const auto& box : env.obstacles) {
      const auto nearest = [&](double u) {
        const Eigen::Vector3d p = p0 + u * (p1 - p0);
        const Eigen::Vector3d local = box.R.transpose() * (p - box.center);
        const Eigen::Vector3d x = box.center + box.R * local.cwiseMax(-box.half_size).cwiseMin(box.half_size);
        return std::pair<double, Eigen::Vector3d>((p - x).squaredNorm(), p - x);
      };
      double lo = 0.0, hi = 1.0;
      for (int k = 0; k < 40; ++k) {
        const double u = lo + (hi - lo) / 3.0, v = hi - (hi - lo) / 3.0;
        if (nearest(u).first < nearest(v).first) hi = v; else lo = u;
      }
      const auto point = nearest(0.5 * (lo + hi));
      if (point.first < nearest_distance) {
        nearest_distance = point.first;
        nearest_direction = point.second;
      }
    }
    Eigen::Vector3d e2 = nearest_direction - e1 * e1.dot(nearest_direction);
    if (e2.norm() < 1.0e-8) e2 = e1.unitOrthogonal();
    e2.normalize();
    const Eigen::Vector3d e3 = e1.cross(e2).normalized();
    Eigen::Matrix3d basis;
    basis.col(0) = e1; basis.col(1) = e2; basis.col(2) = e3;
    const double a = 0.5 * length + 0.01;
    const auto ellipsoid = [&](double b, double c) {
      Eigen::Vector3d inverse_square(1.0 / (a * a), 1.0 / (b * b), 1.0 / (c * c));
      return Eigen::Matrix3d(basis * inverse_square.asDiagonal() * basis.transpose());
    };
    const auto clear = [&](double b, double c) {
      const Eigen::Matrix3d Q = ellipsoid(b, c);
      for (const auto& box : env.obstacles) {
        if (utils::nearestBoxPointInEllipsoid(box.center, box.half_size, box.R, center, Q).first <= 1.0) return false;
      }
      return true;
    };
    if (!clear(params::PLANNING_RILS_MIN_RADIUS, params::PLANNING_RILS_MIN_RADIUS)) {
      throw std::runtime_error("RILS seed ellipsoid intersects obstacle");
    }
    double lo = params::PLANNING_RILS_MIN_RADIUS, hi = params::PLANNING_RILS_MAX_RADIUS;
    for (int k = 0; k < 32; ++k) {
      const double b = 0.5 * (lo + hi);
      if (clear(b, b)) lo = b; else hi = b;
    }
    const double b = 0.98 * lo;
    lo = b; hi = params::PLANNING_RILS_MAX_RADIUS;
    for (int k = 0; k < 32; ++k) {
      const double c = 0.5 * (lo + hi);
      if (clear(b, c)) lo = c; else hi = c;
    }
    const Eigen::Matrix3d Q = ellipsoid(b, std::max(b, 0.98 * lo));

    Corridor region;
    for (int axis = 0; axis < 3; ++axis) {
      Eigen::Vector3d n = Eigen::Vector3d::Zero();
      n(axis) = 1.0;
      region.push_back({n, env.upper(axis)});
      region.push_back({-n, -env.lower(axis)});
    }
    for (const auto& box : env.obstacles) {
      const auto point = utils::nearestBoxPointInEllipsoid(box.center, box.half_size, box.R, center, Q);
      Eigen::Vector3d n = Q * (point.second - center);
      if (point.first <= 1.0 || n.norm() < 1.0e-9) {
        throw std::runtime_error("RILS ellipsoid intersects obstacle");
      }
      n.normalize();
      region.push_back({n, n.dot(point.second) - params::PLANNING_SFC_MARGIN});
    }
    result.push_back(region);
  }
  return result;
}

// MINCO =============================================================
struct Segment
{
  Matrix86d c = Matrix86d::Zero();
  double T = 1.0;

  Vector6d derivative(double t, int order) const
  {
    const double u = std::clamp(t / T, 0.0, 1.0);
    Vector6d z = Vector6d::Zero();
    for (int k = order; k < 8; ++k) {
      double factor = 1.0;
      for (int m = 0; m < order; ++m) factor *= k - m;
      z += c.row(k).transpose() * factor * std::pow(u, k - order) / std::pow(T, order);
    }
    return z;
  }
};

struct Trajectory
{
  std::vector<Segment> segments;

  double duration() const
  {
    double t = 0.0;
    for (const auto& s : segments) t += s.T;
    return t;
  }

  utils::TargetCMD sample(double t) const
  {
    if (segments.empty()) throw std::runtime_error("empty trajectory");
    t = std::clamp(t, 0.0, duration());
    std::size_t i = 0;
    while (i + 1 < segments.size() && t > segments[i].T) t -= segments[i++].T;
    const Vector6d z = segments[i].derivative(t, 0);
    const Vector6d dz = segments[i].derivative(t, 1);
    const Vector6d ddz = segments[i].derivative(t, 2);
    const Eigen::Vector3d sigma = z.tail<3>();
    const Eigen::Vector4d q = utils::sigmaToQuat(sigma);
    utils::TargetCMD cmd;
    cmd.x = z(0); cmd.y = z(1); cmd.z = z(2);
    cmd.vel = dz.head<3>(); cmd.acc = ddz.head<3>();
    cmd.quat = q;
    cmd.omega = utils::sigmaToBodyOmega(sigma, dz.tail<3>());
    cmd.omega_dot = utils::sigmaToBodyOmegaDot(sigma, dz.tail<3>(), ddz.tail<3>());
    cmd.use_quaternion = true;
    const Eigen::Vector3d rpy = utils::rotToRpy(utils::quatToRot(q));
    cmd.roll = rpy(0); cmd.pitch = rpy(1); cmd.yaw = rpy(2);
    return cmd;
  }
};

inline Trajectory MINCO(const std::vector<Vector6d>& q, const std::vector<double>& T)
{
  // Liu et al. (2024), Eqs. (19)-(23c); MINCO coefficients are uniquely
  // determined by q, T, endpoint derivatives and C6 knot continuity.
  const int M = static_cast<int>(T.size());
  if (static_cast<int>(q.size()) != M + 1 || M < 1) throw std::runtime_error("invalid MINCO dimensions");
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(8 * M, 8 * M);
  Eigen::MatrixXd b = Eigen::MatrixXd::Zero(8 * M, 6);
  const auto row = [&](int r, int piece, int order, double u, double sign) {
    for (int k = order; k < 8; ++k) {
      double factor = sign / std::pow(T[piece], order);
      for (int m = 0; m < order; ++m) factor *= k - m;
      A(r, 8 * piece + k) = factor * std::pow(u, k - order);
    }
  };
  int r = 0;
  for (int d = 0; d < 4; ++d, ++r) { row(r, 0, d, 0.0, 1.0); if (d == 0) b.row(r) = q.front().transpose(); }
  for (int i = 0; i < M - 1; ++i) {
    row(r, i, 0, 1.0, 1.0); b.row(r++) = q[i + 1].transpose();
    row(r, i + 1, 0, 0.0, 1.0); b.row(r++) = q[i + 1].transpose();
    for (int d = 1; d <= 6; ++d, ++r) {
      row(r, i, d, 1.0, 1.0);
      row(r, i + 1, d, 0.0, -1.0);
    }
  }
  for (int d = 0; d < 4; ++d, ++r) { row(r, M - 1, d, 1.0, 1.0); if (d == 0) b.row(r) = q.back().transpose(); }
  const Eigen::MatrixXd c = A.colPivHouseholderQr().solve(b);
  if (!c.allFinite() || (A * c - b).norm() > 1.0e-4) throw std::runtime_error("MINCO solve failed");
  Trajectory trajectory;
  for (int i = 0; i < M; ++i) {
    Segment s;
    s.c = c.block(8 * i, 0, 8, 6);
    s.T = T[i];
    trajectory.segments.push_back(s);
  }
  return trajectory;
}

// Constrained trajectory optimization ===============================
class Optimization : public Ipopt::TNLP
{
public:
  Optimization(const Environment& env, std::vector<Vector6d> q0,
               std::vector<double> T0, std::vector<Corridor> corridor)
  : env_(env), q0_(std::move(q0)), T0_(std::move(T0)), corridor_(std::move(corridor))
  {
    M_ = static_cast<int>(T0_.size());
    n_ = 6 * (M_ - 1) + M_;
    for (int i = 1; i < M_; ++i) g_count_ += static_cast<int>(corridor_[i - 1].size() + corridor_[i].size());
    for (const auto& poly : corridor_) g_count_ += (params::PLANNING_CONSTRAINT_SAMPLES + 1) * (7 + 8 * static_cast<int>(poly.size()));
  }

  bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g,
                    Ipopt::Index& nnz_h_lag, IndexStyleEnum& index_style) override
  {
    n = n_; m = g_count_; nnz_jac_g = n_ * g_count_; nnz_h_lag = 0; index_style = C_STYLE;
    return true;
  }

  bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
                       Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u) override
  {
    for (int i = 0; i < M_ - 1; ++i) {
      for (int d = 0; d < 3; ++d) {
        x_l[6 * i + d] = env_.lower(d);
        x_u[6 * i + d] = env_.upper(d);
      }
      for (int d = 3; d < 6; ++d) {
        x_l[6 * i + d] = -3.0;
        x_u[6 * i + d] = 3.0;
      }
    }
    for (int i = 0; i < M_; ++i) {
      x_l[6 * (M_ - 1) + i] = params::PLANNING_MIN_SEGMENT_SEC;
      x_u[6 * (M_ - 1) + i] = params::PLANNING_MAX_SEGMENT_SEC;
    }
    for (Ipopt::Index j = 0; j < m; ++j) { g_l[j] = -1.0e19; g_u[j] = 0.0; }
    return n == n_ && m == g_count_;
  }

  bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x,
                          bool init_z, Ipopt::Number*, Ipopt::Number*,
                          Ipopt::Index, bool init_lambda, Ipopt::Number*) override
  {
    if (!init_x || init_z || init_lambda || n != n_) return false;
    for (int i = 1; i < M_; ++i) for (int d = 0; d < 6; ++d) x[6 * (i - 1) + d] = q0_[i](d);
    for (int i = 0; i < M_; ++i) x[6 * (M_ - 1) + i] = T0_[i];
    return true;
  }

  Trajectory trajectory(const Ipopt::Number* x) const
  {
    std::vector<Vector6d> q = q0_;
    std::vector<double> T = T0_;
    for (int i = 1; i < M_; ++i) for (int d = 0; d < 6; ++d) q[i](d) = x[6 * (i - 1) + d];
    for (int i = 0; i < M_; ++i) T[i] = x[6 * (M_ - 1) + i];
    return MINCO(q, T);
  }

  double cost(const Ipopt::Number* x) const
  {
    // Liu et al. (2024), Eq. (23a): snap integral and segment-time regularizer.
    const Trajectory tr = trajectory(x);
    double objective = params::PLANNING_TIME_WEIGHT * tr.duration();
    for (const auto& seg : tr.segments) {
      const int K = params::PLANNING_CONSTRAINT_SAMPLES;
      for (int j = 0; j <= K; ++j) {
        const double weight = (j == 0 || j == K) ? 0.5 : 1.0;
        objective += weight * seg.derivative(seg.T * j / K, 4).squaredNorm() * seg.T / K;
      }
    }
    return objective;
  }

  void constraints(const Ipopt::Number* x, std::vector<double>& g) const
  {
    // Liu et al. (2024), Eqs. (23d)-(23k): direct sampled inequalities.
    const Trajectory tr = trajectory(x);
    g.clear();
    g.reserve(g_count_);
    for (int i = 1; i < M_; ++i) {
      const Eigen::Vector3d p = tr.segments[i - 1].derivative(tr.segments[i - 1].T, 0).head<3>();
      for (const auto& plane : corridor_[i - 1]) g.push_back(plane.n.dot(p) - plane.d);
      for (const auto& plane : corridor_[i]) g.push_back(plane.n.dot(p) - plane.d);
    }
    const auto body_vertices = utils::boxVertices((env_.vehicle_half_size.array() + params::PLANNING_RRT_CLEARANCE).matrix());
    for (int i = 0; i < M_; ++i) {
      const auto& seg = tr.segments[i];
      for (int j = 0; j <= params::PLANNING_CONSTRAINT_SAMPLES; ++j) {
        const double t = seg.T * j / params::PLANNING_CONSTRAINT_SAMPLES;
        const Vector6d z = seg.derivative(t, 0);
        const Vector6d dz = seg.derivative(t, 1);
        const Vector6d ddz = seg.derivative(t, 2);
        const Eigen::Matrix3d R = utils::quatToRot(utils::sigmaToQuat(z.tail<3>()));
        const Eigen::Vector3d rpy = utils::rotToRpy(R);
        const Eigen::Vector3d omega = utils::sigmaToBodyOmega(z.tail<3>(), dz.tail<3>());
        g.push_back(dz.head<3>().squaredNorm() - std::pow(params::PLANNING_MAX_VEL, 2)); // Eq. (23h)
        g.push_back(ddz.head<3>().squaredNorm() - std::pow(params::PLANNING_MAX_ACC, 2)); // Eq. (23i)
        g.push_back(omega.squaredNorm() - std::pow(params::PLANNING_MAX_OMEGA - params::PLANNING_OMEGA_OPT_MARGIN_RAD_S, 2)); // Eq. (23j)
        g.push_back(rpy(0) - params::PLANNING_ROLL_LIMIT_RAD + params::PLANNING_ROLL_OPT_MARGIN_RAD);
        g.push_back(-rpy(0) - params::PLANNING_ROLL_LIMIT_RAD + params::PLANNING_ROLL_OPT_MARGIN_RAD);
        g.push_back(rpy(1) - params::PLANNING_PITCH_LIMIT_RAD);
        g.push_back(-rpy(1) - params::PLANNING_PITCH_LIMIT_RAD);
        for (const auto& v : body_vertices) {
          const Eigen::Vector3d vertex = z.head<3>() + R * v;
          for (const auto& plane : corridor_[i]) g.push_back(plane.n.dot(vertex) - plane.d); // Eq. (23k)
        }
      }
    }
  }

  bool eval_f(Ipopt::Index, const Ipopt::Number* x, bool, Ipopt::Number& value) override
  {
    value = cost(x);
    return std::isfinite(value);
  }

  bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool, Ipopt::Number* grad) override
  {
    std::vector<double> probe(x, x + n);
    for (int k = 0; k < n; ++k) {
      const double h = 1.0e-5 * std::max(1.0, std::abs(x[k]));
      probe[k] = x[k] + h; const double up = cost(probe.data());
      probe[k] = x[k] - h; const double down = cost(probe.data());
      probe[k] = x[k];
      grad[k] = (up - down) / (2.0 * h);
    }
    return true;
  }

  bool eval_g(Ipopt::Index, const Ipopt::Number* x, bool, Ipopt::Index m, Ipopt::Number* g) override
  {
    std::vector<double> values;
    constraints(x, values);
    if (static_cast<Ipopt::Index>(values.size()) != m) return false;
    std::copy(values.begin(), values.end(), g);
    return true;
  }

  bool eval_jac_g(Ipopt::Index n, const Ipopt::Number* x, bool, Ipopt::Index m,
                  Ipopt::Index, Ipopt::Index* i_row, Ipopt::Index* j_col, Ipopt::Number* values) override
  {
    if (!values) {
      for (int row = 0; row < m; ++row) for (int col = 0; col < n; ++col) {
        i_row[row * n + col] = row; j_col[row * n + col] = col;
      }
      return true;
    }
    std::vector<double> probe(x, x + n), up, down;
    for (int col = 0; col < n; ++col) {
      const double h = 1.0e-5 * std::max(1.0, std::abs(x[col]));
      probe[col] = x[col] + h; constraints(probe.data(), up);
      probe[col] = x[col] - h; constraints(probe.data(), down);
      probe[col] = x[col];
      if (static_cast<int>(up.size()) != m || static_cast<int>(down.size()) != m) return false;
      for (int row = 0; row < m; ++row) values[row * n + col] = (up[row] - down[row]) / (2.0 * h);
    }
    return true;
  }

  void finalize_solution(Ipopt::SolverReturn status, Ipopt::Index n, const Ipopt::Number* x,
                         const Ipopt::Number*, const Ipopt::Number*, Ipopt::Index,
                         const Ipopt::Number*, const Ipopt::Number*, Ipopt::Number,
                         const Ipopt::IpoptData*, Ipopt::IpoptCalculatedQuantities*) override
  {
    status_ = status;
    solution_.assign(x, x + n);
  }

  Ipopt::SolverReturn status() const { return status_; }
  const std::vector<double>& solution() const { return solution_; }

private:
  const Environment& env_;
  std::vector<Vector6d> q0_;
  std::vector<double> T0_;
  std::vector<Corridor> corridor_;
  int M_{0}, n_{0}, g_count_{0};
  Ipopt::SolverReturn status_{Ipopt::UNASSIGNED};
  std::vector<double> solution_;
};

inline bool optimize(const Environment& env, const std::vector<Eigen::Vector3d>& path,
                     const std::vector<Eigen::Quaterniond>& path_attitude, const std::vector<Corridor>& corridor, Trajectory& result)
{
  if (path.size() < 2 || path_attitude.size() != path.size() || corridor.size() + 1 != path.size()) return false;
  std::vector<Vector6d> q(path.size(), Vector6d::Zero());
  std::vector<double> T(corridor.size());
  for (std::size_t i = 0; i < path.size(); ++i) q[i].head<3>() = path[i];
  q.front().tail<3>() = utils::rpyToSigma(env.start_rpy);
  q.back().tail<3>() = utils::rpyToSigma(env.goal_rpy);
  for (std::size_t i = 1; i + 1 < path.size(); ++i)
    q[i].tail<3>() = utils::rpyToSigma(utils::rotToRpy(path_attitude[i].toRotationMatrix()));
  for (std::size_t i = 0; i < T.size(); ++i) {
    T[i] = std::max(params::PLANNING_MIN_SEGMENT_SEC, (path[i + 1] - path[i]).norm() / params::PLANNING_AVG_SPEED);
  }
  Ipopt::SmartPtr<Optimization> nlp = new Optimization(env, q, T, corridor);
  Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();
  app->Options()->SetStringValue("hessian_approximation", "limited-memory");
  app->Options()->SetStringValue("mu_strategy", "adaptive");
  app->Options()->SetIntegerValue("max_iter", params::PLANNING_IPOPT_MAX_ITER);
  app->Options()->SetNumericValue("tol", params::PLANNING_IPOPT_TOLERANCE);
  app->Options()->SetIntegerValue("print_level", 0);
  if (app->Initialize() != Ipopt::Solve_Succeeded) return false;
  const auto status = app->OptimizeTNLP(nlp);
  if (status != Ipopt::Solve_Succeeded && status != Ipopt::Solved_To_Acceptable_Level) {
    if (status == Ipopt::Maximum_Iterations_Exceeded && !nlp->solution().empty()) {
      const auto statistics = app->Statistics();
      if (Ipopt::IsValid(statistics)) {
        Ipopt::Number dual, constraint, complementarity, kkt;
        statistics->ScaledInfeasibilities(dual, constraint, complementarity, kkt);
        std::fprintf(stderr, "IPOPT iterations=%d scaled dual=%.6g constraint=%.6g complementarity=%.6g KKT=%.6g\n",
          static_cast<int>(statistics->IterationCount()), dual, constraint, complementarity, kkt);
      }
      std::vector<double> g;
      nlp->constraints(nlp->solution().data(), g);
      double violation = 0.0;
      for (const double value : g) {
        if (!std::isfinite(value)) return false;
        violation = std::max(violation, value);
      }
      if (!g.empty() && violation <= 1.0e-4) {
        result = nlp->trajectory(nlp->solution().data());
        std::fprintf(stderr, "IPOPT iteration limit; feasible candidate (sampled violation=%.6g) sent to dense validation\n", violation);
        return true;
      }
      std::fprintf(stderr, "IPOPT sampled constraint violation=%.6g\n", violation);
    }
    std::fprintf(stderr, "IPOPT status=%d, segments=%zu\n", static_cast<int>(status), corridor.size());
    return false;
  }
  result = nlp->trajectory(nlp->solution().data());
  return true;
}

// Trajectory validation =============================================
inline bool validate(const Environment& env, const Trajectory& tr, const std::vector<Corridor>& corridor)
{
  if (tr.segments.empty() || tr.segments.size() != corridor.size()) return false;
  Environment clearance_env = env;
  clearance_env.vehicle_half_size.array() += params::PLANNING_RRT_CLEARANCE;
  for (std::size_t i = 0; i < tr.segments.size(); ++i) {
    const auto& seg = tr.segments[i];
    const int count = std::max(1, static_cast<int>(std::ceil(seg.T / params::PLANNING_VALIDATION_DT)));
    for (int j = 0; j <= count; ++j) {
      const double t = seg.T * j / count;
      const Vector6d z = seg.derivative(t, 0), dz = seg.derivative(t, 1), ddz = seg.derivative(t, 2);
      const Eigen::Matrix3d R = utils::quatToRot(utils::sigmaToQuat(z.tail<3>()));
      const Eigen::Vector3d rpy = utils::rotToRpy(R);
      if (dz.head<3>().norm() > params::PLANNING_MAX_VEL + 0.01
          || ddz.head<3>().norm() > params::PLANNING_MAX_ACC + 0.01
          || utils::sigmaToBodyOmega(z.tail<3>(), dz.tail<3>()).norm() > params::PLANNING_MAX_OMEGA + 0.01
          || std::abs(rpy(0)) > params::PLANNING_ROLL_LIMIT_RAD + 0.005
          || std::abs(rpy(1)) > params::PLANNING_PITCH_LIMIT_RAD + 0.005) { std::fprintf(stderr, "validation dynamics seg=%zu t=%.3f v=%.3f a=%.3f w=%.3f roll=%.3f pitch=%.3f\n", i, t, dz.head<3>().norm(), ddz.head<3>().norm(), utils::sigmaToBodyOmega(z.tail<3>(), dz.tail<3>()).norm(), rpy(0), rpy(1)); return false; }
      for (const auto& v : utils::boxVertices(clearance_env.vehicle_half_size)) {
        const Eigen::Vector3d vertex = z.head<3>() + R * v;
        for (const auto& plane : corridor[i]) if (plane.n.dot(vertex) > plane.d + 0.005) { std::fprintf(stderr, "validation SFC seg=%zu t=%.3f violation=%.4f\n", i, t, plane.n.dot(vertex) - plane.d); return false; }
      }
      if (!clearance_env.collisionFree(z.head<3>(), R)) { std::fprintf(stderr, "validation collision seg=%zu t=%.3f p=(%.3f %.3f %.3f)\n", i, t, z(0), z(1), z(2)); return false; }
    }
  }
  return true;
}

} // namespace planning
