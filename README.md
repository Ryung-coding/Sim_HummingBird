# Sim_HummingBird README (current working version)

This repository is a ROS 2 + MuJoCo simulation workspace for a tilting multirotor / HummingBird-style UAV.

The current direction is to make the project simpler and more flight-oriented. ROS 2 is kept minimal because heavy ROS tools such as rqt can disturb plant timing/synchronization. The long-term direction is to keep the wrench-level controller on-board and later connect downstream PX4/Pixhawk control through a separate DDS-bridge branch.

This note is based on the current working structure and the latest controller/allocation/viewer changes.

---

## 1. Package Structure

```text
src
├── include
│   ├── params.hpp
│   └── utils.hpp
├── multirotor_cmd
│   ├── launch/run.py
│   ├── package.xml
│   ├── CMakeLists.txt
│   └── src/position_cmd.cpp
├── multirotor_controller
│   ├── package.xml
│   ├── CMakeLists.txt
│   └── src
│       ├── wrench_controller.cpp
│       └── allocator_controller.cpp
├── multirotor_interfaces
│   ├── msg
│   │   ├── Cmd.msg
│   │   ├── Input.msg
│   │   ├── MultirotorState.msg
│   │   └── Wrench.msg
│   └── package.xml
└── plant
    ├── plant
    │   ├── plant.py
    │   └── multirotor_viewer.py
    ├── setup.py
    └── xml
        ├── HummingBird.xml
        ├── scene.xml
        ├── BODY.stl
        ├── ARM.stl
        └── PROP.stl
```

### Main packages

- `multirotor_cmd`: publishes command trajectories.
- `multirotor_controller`: contains wrench control and control allocation.
- `multirotor_interfaces`: custom ROS 2 message definitions.
- `plant`: MuJoCo plant simulation and PyQtGraph viewer.
- `src/include`: shared C++ headers used by command/controller packages.

---

## 2. Build

From the workspace root:

```bash
cd ~/Desktop/Sim_HummingBird
colcon build --symlink-install
source install/setup.bash
```

User alias example:

```bash
alias cm='cd ~/Desktop/Sim_HummingBird && colcon build --symlink-install && source install/setup.bash'
```

If message generation or symlink install gets corrupted after changing `.msg` files:

```bash
cd ~/Desktop/Sim_HummingBird
rm -rf build install log
colcon build --symlink-install
source install/setup.bash
```

If the terminal still prints old underlay warnings after deleting `install`, open a new terminal or reset the ROS environment:

```bash
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
source /opt/ros/jazzy/setup.bash
```

---

## 3. Run

Typical launch:

```bash
ros2 launch multirotor_cmd run.py
```

Run the viewer separately:

```bash
ros2 run plant multirotor_viewer
```

Direct plant run:

```bash
ros2 run plant plant
```

Recommended workflow for debugging is to run only the required nodes. Avoid rqt/rqt_plot if the plant becomes unstable due to synchronization or GUI load.

---

## 4. Coordinate Convention

MuJoCo world is z-up, but the controller/state convention is z-down.

```text
body x : front
body y : right
body z : down
```

The plant publishes state in the controller convention:

```text
pos = [x_mj, -y_mj, -z_mj]
vel = [vx_mj, -vy_mj, -vz_mj]
```

Therefore, hover/upward force appears as negative body-z force in the wrench convention.

---

## 5. Message Definitions

### Cmd.msg

```text
float32[3] pos_cmd
float32[3] att_cmd
```

`att_cmd` has two modes:

```text
USE_SO3_HEADING_CMD = false:
  att_cmd = [roll, pitch, yaw] [rad]

USE_SO3_HEADING_CMD = true:
  att_cmd = heading direction vector [hx, hy, hz]
```

For attitude tuning using roll/pitch/yaw directly, set:

```cpp
static constexpr bool USE_SO3_HEADING_CMD = false;
```

For heading-vector based SO(3) command, set:

```cpp
static constexpr bool USE_SO3_HEADING_CMD = true;
```

If SO3 heading mode is true, pitch will not appear unless the heading vector itself contains the intended 3D direction and `headingToRot()` preserves that direction.

### Input.msg

Current input is separated into thrust, arm angle, and BLDC tilt angle:

```text
float64[4] f      # f1 f2 f3 f4 [N]
float64[4] theta  # theta1 theta2 theta3 theta4 [rad]
float64[4] phi    # phi1 phi2 phi3 phi4 [rad]
```

Input order in the plant:

```text
ctrl[0:4]   = f1 f2 f3 f4
ctrl[4:8]   = theta1 theta2 theta3 theta4
ctrl[8:12]  = phi1 phi2 phi3 phi4
```

### MultirotorState.msg

Current state includes:

```text
float64[3] pos
float64[3] vel
float64[3] acc

float64[3] rpy
float64[3] w_rpy
float64[3] a_rpy

float64[3] imu_acc

float64[4] theta
float64[4] phi
```

### Wrench.msg

Wrench is:

```text
float64[3] force
float64[3] moment
```

Order:

```text
force  = [Fx, Fy, Fz]
moment = [Mx, My, Mz]
```

---

## 6. Shared Headers

### `src/include/params.hpp`

Stores shared constants:

- trajectory/path parameters
- model parameters
- position controller gains
- attitude controller gains
- saturation values
- allocation parameters
- allocation check tolerances

Current model parameters are set around a 5 kg class model:

```cpp
static constexpr double mass = 4.62;
static constexpr double grav = 9.81;
static constexpr std::array<double, 3> J = {0.030, 0.030, 0.050};
```

If matching the current MuJoCo XML directly, check the body inertial values in:

```text
src/plant/xml/HummingBird.xml
```

### `src/include/utils.hpp`

Contains shared utilities:

- math utilities:
  - `hat()`
  - `vee()`
  - `vec3()`
  - `diag3()`
  - `clampVec3()`
  - `rpyToRot()`
  - `headingToRot()`
- trajectory helpers:
  - `trackApple(t)`
  - `takeApple(t)`
  - `positionTuningPath(t)`
  - `attitudeTuningPath(t)`
  - `agilePath(t)`
- allocation helpers:
  - `allocation_P2T2(...)`
  - `allocation_P4T4(...)`
  - `checkAllocation(...)`
- filter:
  - `LPF`

---

## 7. Command Path Selection

Command generation is in:

```text
src/multirotor_cmd/src/position_cmd.cpp
```

The path is selected by changing one line:

```cpp
const auto cmd = utils::trackApple(t);
```

Available examples:

```cpp
const auto cmd = utils::trackApple(t);
const auto cmd = utils::takeApple(t);
const auto cmd = utils::positionTuningPath(t);
const auto cmd = utils::attitudeTuningPath(t);
const auto cmd = utils::agilePath(t);
```

### `trackApple(t)`

Circular apple-tracking path.

- Hover first.
- Move around the apple.
- Position follows a circular/yaw scan path.
- Pitch changes with the scan.
- Useful for camera-like apple inspection.

### `takeApple(t)`

Apple approach path.

- Hover phase is same as `trackApple`.
- Tilt smoothly to `THETA_MAX`.
- Hold yaw fixed.
- Move in the tilted heading direction toward the apple.
- Move backward to the original hover/tilted position.
- Repeat with `SCAN_PERIOD_SEC`.

### `positionTuningPath(t)`

Position controller tuning path.

- Hover to target altitude.
- Move in a square path on the horizontal plane.
- Uses `±1 m` style step movements.
- Attitude command is zero.

Use this to tune `Kp_pos`, `Ki_pos`, `Kd_pos`, `force_body_sat`.

### `attitudeTuningPath(t)`

Attitude controller tuning path.

- Hover at 1 m.
- Alternate between sinusoidal pitch and sinusoidal yaw.
- Useful for testing `kR`, `kW`, `kI`, and `torque_sat`.

For this path, use:

```cpp
static constexpr bool USE_SO3_HEADING_CMD = false;
```

### `agilePath(t)`

Aggressive tuning path.

- Figure-eight position motion in x-y.
- z also moves.
- roll uses `±ROLL_MAX`.
- pitch uses `±THETA_MAX`.
- yaw uses `±90 deg`.

For this path, RPY mode is recommended:

```cpp
static constexpr bool USE_SO3_HEADING_CMD = false;
```

---

## 8. Wrench Controller

File:

```text
src/multirotor_controller/src/wrench_controller.cpp
```

The controller has two main parts:

```cpp
positionController(dt)
attitudeController(R, Rd, dt)
```

### Position controller

The position controller computes world-frame force:

```text
F_world = Kp_pos * pos_err + Ki_pos * integral(pos_err) - Kd_pos * vel
F_world.z -= mass * grav
```

Then it converts to body frame:

```cpp
F_body = R.transpose() * F_world;
```

Saturation is applied in body frame:

```cpp
F_body = utils::clampVec3(F_body, utils::vec3(params::force_body_sat));
```

This means x/y/z force limits are applied to body-frame `Fx, Fy, Fz`.

### Attitude controller

The attitude controller is an FDCL-style SO(3) feedback controller directly implemented in `wrench_controller.cpp`.

Current first implementation uses:

```cpp
Wd = Eigen::Vector3d::Zero();
Wd_dot = Eigen::Vector3d::Zero();
```

Attitude error:

```cpp
RtRd = R.transpose() * Rd;
eR = 0.5 * vee(RtRd.transpose() - RtRd);
```

Angular-rate error:

```cpp
eW = W - R.transpose() * Rd * Wd;
```

Moment command:

```cpp
M = -kR * eR - kW * eW - kI * att_i
    - J * hat(W) * RtRd * Wd
    + J * RtRd * Wd_dot;
```

Moment saturation is applied like force saturation:

```cpp
M = utils::clampVec3(M, utils::vec3(params::torque_sat));
```

### SO3 true/false mode

Desired rotation is selected as:

```cpp
const Eigen::Matrix3d Rd =
    params::USE_SO3_HEADING_CMD ? utils::headingToRot(att_cmd_) : utils::rpyToRot(att_cmd_);
```

Use RPY mode when the command path directly sets roll/pitch/yaw.

Use SO3 heading mode when `att_cmd` is a heading vector.

---

## 9. Control Allocation

File:

```text
src/include/utils.hpp
```

Allocator is called in:

```text
src/multirotor_controller/src/allocator_controller.cpp
```

Change allocator mode by changing one line:

```cpp
const auto alloc = utils::allocation_P2T2(moment_cmd, force_cmd, theta_measured_, phi_measured_);
```

or:

```cpp
const auto alloc = utils::allocation_P4T4(moment_cmd, force_cmd, theta_measured_, phi_measured_);
```

### P2T2 allocation

P2T2 groups front/back pairs:

```text
theta1 = theta4
theta2 = theta3

phi1 = phi4
phi2 = phi3
```

Virtual variables:

```text
C = [front_Fx, front_Fy, front_Fz,
     back_Fx,  back_Fy,  back_Fz,
     df14, df23]
```

Wrench order:

```text
W = [Mx, My, Mz, Fx, Fy, Fz]
```

This is useful for pair-constrained allocation and comparison with the previous two-arm style.

### P4T4 allocation

P4T4 uses each rotor independently.

Virtual variables:

```text
C = [Fx1, Fy1, Fz1,
     Fx2, Fy2, Fz2,
     Fx3, Fy3, Fz3,
     Fx4, Fy4, Fz4]
```

Then each rotor force vector is converted to:

```text
f_i
theta_i
phi_i
```

This is useful to compare the benefit of fully independent `theta1~theta4` and `phi1~phi4`.

### Rotor reaction torque sign

The allocator and MuJoCo XML must use the same rotor-spin convention.

If allocator uses:

```cpp
const double spin = (i % 2 == 0) ? 1.0 : -1.0;
```

then the MuJoCo `gear` torque sign should match:

```text
BLDC1: +
BLDC2: -
BLDC3: +
BLDC4: -
```

Example:

```xml
<general name="BLDC1_thrust" site="prop1_site" gear="0 0 -1 0 0 +0.02" .../>
<general name="BLDC2_thrust" site="prop2_site" gear="0 0 -1 0 0 -0.02" .../>
<general name="BLDC3_thrust" site="prop3_site" gear="0 0 -1 0 0 +0.02" .../>
<general name="BLDC4_thrust" site="prop4_site" gear="0 0 -1 0 0 -0.02" .../>
```

If yaw becomes unstable, check this sign first.

---

## 10. Plant

File:

```text
src/plant/plant/plant.py
```

Current direction:

- Delay is removed/disabled.
- Noise can be kept for sensor-like testing.
- Input order is `f[4], theta[4], phi[4]`.
- State publishes `theta[4]` and `phi[4]`.

Typical plant log messages should reflect:

```text
input order: f[4], theta[4], phi[4] -> ctrl[0:4], ctrl[4:8], ctrl[8:12]
```

---

## 11. Viewer

File:

```text
src/plant/plant/multirotor_viewer.py
```

Run:

```bash
ros2 run plant multirotor_viewer
```

The viewer is a PyQtGraph-based replacement for rqt plots.

It subscribes to:

```text
/multirotor_state
/cmd
/wrench
/input
```

### Tabs

The viewer uses two tabs:

```text
State
Actuator
```

### State tab

Plots:

```text
x / x_cmd [m]
y / y_cmd [m]
z / z_cmd [m]

x_err [m]
y_err [m]
z_err [m]

Fx [N]
Fy [N]
Fz [N]

roll / roll_cmd [deg]
pitch / pitch_cmd [deg]
yaw / yaw_cmd [deg]

roll_err [deg]
pitch_err [deg]
yaw_err [deg]

Mx [N·m]
My [N·m]
Mz [N·m]
```

### Actuator tab

The actuator tab is arranged as 4 x 4:

```text
row 0: f1, f2, f3, f4
row 1: phi1 / phi1_cmd, phi2 / phi2_cmd, phi3 / phi3_cmd, phi4 / phi4_cmd
row 2: theta1 / theta1_cmd, theta2 / theta2_cmd, theta3 / theta3_cmd, theta4 / theta4_cmd
row 3: f1-f4, phi1-phi4 / cmd, theta1-theta4 / cmd, max error text
```

The bottom-right panel shows maximum error since start:

```text
Position error [m]
x_max
y_max
z_max
|e_pos|_max

Attitude error [deg]
roll_max
pitch_max
yaw_max
|e_att|_max
```

Command curves are black dashed lines and drawn above actual curves.

---

## 12. Common Checks

### Pitch does not tilt

Check `USE_SO3_HEADING_CMD`.

If it is true, `att_cmd` is interpreted as a heading vector, not `[roll, pitch, yaw]`.

For roll/pitch/yaw tests:

```cpp
static constexpr bool USE_SO3_HEADING_CMD = false;
```

If using SO3 mode, make sure the heading vector contains the intended 3D direction and that `headingToRot()` preserves its z component.

### Yaw is unstable

Check rotor reaction torque signs.

MuJoCo XML `gear` torque sign and allocator `spin` sign must match.

### P2T2 but theta1 and theta4 look different

First check `/input` directly:

```bash
ros2 topic echo /input
```

For P2T2, command should satisfy:

```text
theta1_cmd = theta4_cmd
theta2_cmd = theta3_cmd
phi1_cmd = phi4_cmd
phi2_cmd = phi3_cmd
```

Actual measured angles can differ because joints are physically separated and driven by independent position actuators. Command equality and actual equality are not the same.

### rqt breaks plant timing

Use `multirotor_viewer.py` instead of rqt/rqt_plot for live debugging.

---

## 13. Development Notes

- `multirotor_interfaces` should remain stable unless the input/state structure changes again.
- `plant.py` should stay simple and avoid delay unless explicitly needed.
- `scene.xml` may change for apple/tree/camera testing.
- Viewer/report tools may later be integrated.
- Overall package structure should remain unchanged for now.
- Future PX4/Pixhawk integration should be done in a separate branch with DDS bridge support.
