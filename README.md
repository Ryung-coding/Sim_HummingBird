# Sim_HummingBird

This repository is a ROS 2 + MuJoCo simulation workspace for a tilting multirotor / HummingBird-style UAV. The project is still under development, so some parameters and allocation logic are experimental.

> Note: the current MuJoCo model mass is set around **10 kg** to avoid the controller repeatedly driving thrust to zero during simulation tests.

---

## 1. Package Structure

```text
src
├── multirotor_cmd
│   ├── launch/run.py
│   └── src/position_cmd.cpp
├── multirotor_controller
│   └── src
│       ├── allocator_controller.cpp
│       └── wrench_controller.cpp
├── multirotor_interfaces
│   └── msg
│       ├── Cmd.msg
│       ├── Input.msg
│       ├── MultirotorState.msg
│       └── Wrench.msg
└── plant
    ├── plant
    │   ├── multirotor_viewer.py
    │   └── plant.py
    └── xml
        ├── HummingBird.xml
        ├── scene.xml
        ├── BODY.stl
        ├── ARM.stl
        └── PROP.stl
```

### Main packages

- `multirotor_cmd`: publishes reference commands.
- `multirotor_controller`: contains wrench control and control allocation.
- `multirotor_interfaces`: custom ROS 2 message definitions.
- `plant`: MuJoCo-based plant simulation and viewer.

---

## 2. Requirements

This workspace assumes:

- Ubuntu 22.04
- ROS 2 Humble
- MuJoCo Python package
- Eigen / standard ROS 2 build tools

Basic Python dependencies depend on your environment, but MuJoCo must be installed for the `plant` package.

---

## 3. Build

From the workspace root:

```bash
cd ~/Desktop/Sim_HummingBird
colcon build --symlink-install
source install/setup.bash
```

If custom messages are not found, rebuild from a clean state:

```bash
rm -rf build install log
colcon build --symlink-install
source install/setup.bash
```

---

## 4. Run

Launch the simulation stack:

```bash
ros2 launch multirotor_cmd run.py
```

The launch file starts the command publisher, controller nodes, and the MuJoCo plant/viewer depending on the current launch configuration.

---

## 5. Coordinate Convention

The body frame uses a **z-down** convention:

```text
x : front
y : right
z : down
```

Therefore, upward hover force appears as negative body-z force in the wrench convention.

---

## 6. Command Parameters

The command node currently uses:

```cpp
static constexpr int RATE_HZ = 400;
static constexpr double HOVER_SEC = 3.0;
static constexpr double HOVER_ALT = -1.0;
static constexpr double TILT_ANGLE = 70.0 * M_PI / 180.0;
```

The maximum commanded tilt angle is currently set to **70 deg**.

---

## 7. Plant / Simulation Parameters

The MuJoCo plant is currently configured around:

```python
PHYSICS_HZ = 400.0
PUB_HZ = 400.0

USE_DELAY = False
USE_NOISE = True

DELAY_TIME = 0.01

SIG_POS = 0.005       # 5 mm
SIG_VEL = 0.03        # 3 cm/s
SIG_GYRO = 0.01       # 0.57 deg/s
SIG_ACC = 0.10        # 0.1 m/s^2
SIG_ENCODER = 0.002   # 0.11 deg

N_CTRL = 10

USE_FIXED_CAMERA = False
VIEW_CAMERA_NAME = "front_camera"
```

### Notes

- `USE_NOISE = True` adds sensor-like noise to the simulated state feedback.
- `USE_DELAY` is currently disabled. The delay function is experimental and should not be used for normal testing.
- Camera view can be changed using `USE_FIXED_CAMERA` and `VIEW_CAMERA_NAME`.

---

## 8. Allocator Parameters

The current allocator uses the following parameters:

```cpp
static constexpr double Lx = 0.1861;
static constexpr double Ly = 0.1861;
static constexpr double d = 0.0500;
static constexpr double zeta = 0.0200;
static constexpr double f_min = 1.0e-3;
static constexpr double f_cmd_min = 1.0e-6;
static constexpr double f_cmd_max = 50.0;
static constexpr double angle_limit_rad = 1.57;  // < 90 deg
static constexpr double virtual_lambda = 1.0e-4;
static constexpr double check_tau_z_thrust_tol = 0.50;
static constexpr double check_force_tol = 1.00;
static constexpr double check_moment_tol = 2.00;
```

Meaning:

- `Lx`, `Ly`: nominal motor arm distances in x/y direction.
- `d`: offset between the arm rotation axis and roll-tilt position.
- `zeta`: rotor reaction torque coefficient.
- `f_cmd_min`, `f_cmd_max`: thrust command limits.
- `virtual_lambda`: damping factor for virtual allocation.
- `check_force_tol`, `check_moment_tol`: plant-check warning thresholds.

---

## 9. Control Allocation Summary

The current allocator uses a virtual-force allocation form. The virtual variable is:

```text
C = [front_Fx; front_Fy; front_Fz; back_Fx; back_Fy; back_Fz; df14; df23]
```

The wrench order is:

```text
W = [Mx; My; Mz; Fx; Fy; Fz]
```

The allocation is written as:

```text
W = A1 * C
```

with:

```text
Mx = (y14*Fz14 + zeta*Fx14)*df14 + (y23*Fz23 - zeta*Fx23)*df23
My = -Lx*front_Fz + Lx*back_Fz + zeta*Fy14*df14 - zeta*Fy23*df23
Mz = Lx*front_Fy - Lx*back_Fy + (-y14*Fx14 + zeta*Fz14)*df14 + (-y23*Fx23 - zeta*Fz23)*df23
Fx = front_Fx + back_Fx
Fy = front_Fy + back_Fy
Fz = front_Fz + back_Fz
```

where:

```text
y14 = Ly - d*sin(theta1)
y23 = Ly - d*sin(theta2)
```

After solving the virtual force, the allocator converts `front_force` and `back_force` into tilt angles and thrust commands:

```text
front_force -> theta1, phi14, f1, f4
back_force  -> theta2, phi23, f2, f3
```

Final input order:

```text
u = [f1; f2; f3; f4; theta1; theta2; phi1; phi2; phi3; phi4]
```

with:

```text
phi1 = phi4 = phi14
phi2 = phi3 = phi23
```

---

## 10. Development Notes

- This project is experimental and intended for control-allocation testing.
- The allocation logic is still being tuned.
- Current focus is stable simulation behavior rather than final hardware-ready allocation.
- If the plant-check warning is too frequent, first check force/moment error norms before changing the allocation logic.