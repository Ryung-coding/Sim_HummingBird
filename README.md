# Sim_HummingBird

ROS 2 + MuJoCo HummingBird simulation. The live viewer is kept, while CSV logging and log plotting are removed.

## Build

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Run

```bash
ros2 launch multirotor_cmd run.launch.py
```

The launch starts the MuJoCo plant, wrench controller, a1b1 allocator, and command publisher.

## Actuator convention

- `f[0:4]`: rotor thrust in N.
- `beta[0:2]`: `beta_1`, `beta_2`; the plant expands them as `[beta_1, beta_2, beta_2, beta_1]`.
- `alpha[0:4]`: `alpha_1` through `alpha_4` in rad.
- Plant control order is `f[4]`, expanded beta `ctrl[4:8]`, and alpha `ctrl[8:12]`.
- MuJoCo uses z-up; ROS state/control uses z-down.

## Allocation

- `allocation_a1b1()` is the only active allocator and is based on the renewal formulation.
- `allocation_a2b4()` is an empty placeholder for future work.
- Arm geometry is unified as `L = 0.175 m`; `Lx` and `Ly` are removed.
- Model body mass and controller mass are `3.5 kg`.

## Paths

Only the three path helpers matching the Desktop `px4_hummingbird` workspace remain:

- `posPath(t)`
- `attPath(t)`
- `stepAttPath(t)`

The current command publisher selects `attPath(t)`. Apple/agile path commands and path parameters are removed.

## Viewer

```bash
ros2 run plant multirotor_viewer
```

The viewer displays position, attitude, wrench, thrust, beta, and alpha in real time without creating log files.
