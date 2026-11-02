# MuJoCo world: z-up
# IMU: z-down, x-axis same as body x-axis

float64 sim_time  # MuJoCo simulation time [s]

float64[3] pos       # [m]
float64[3] vel       # [m/s]
float64[3] acc       # [m/s^2]

float64[3] rpy       # [roll, pitch, yaw] [rad]
float64[4] quat      # z-down body attitude quaternion [w, x, y, z]
float64[3] w_rpy     # [w_roll, w_pitch, w_yaw] [rad/s]
float64[3] a_rpy     # [a_roll, a_pitch, a_yaw] [rad/s^2]

float64[3] imu_acc   # z-down IMU acceleration [m/s^2]
float64[3] disturbance_force   # total applied disturbance in controller world frame [N]
float64[3] disturbance_moment  # total applied disturbance in controller world frame [N*m]
float64[4] rotor_wall_distance    # center-to-wall distance x [m], inf when no wall
float64[4] rotor_ground_distance  # center-to-surface distance z [m]
float64[4] rotor_ground_gain      # K_G [-]
float64[4] rotor_wall_gain        # K_W [-]
float64[4] rotor_thrust_delta     # deterministic delta T [N]
float64[4] rotor_thrust_noise     # axial noise force [N]

float64[2] beta     # beta_1, beta_2 [rad]
float64[4] alpha   # alpha_1 ... alpha_4 [rad]
