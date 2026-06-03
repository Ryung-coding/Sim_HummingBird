# MuJoCo world: z-up
# IMU: z-down, x-axis same as body x-axis

float64[3] pos       # [m]
float64[3] vel       # [m/s]
float64[3] acc       # [m/s^2]

float64[3] rpy       # [roll, pitch, yaw] [rad]
float64[3] w_rpy     # [w_roll, w_pitch, w_yaw] [rad/s]
float64[3] a_rpy     # [a_roll, a_pitch, a_yaw] [rad/s^2]

float64[3] imu_acc   # z-down IMU acceleration [m/s^2]

float64[4] theta     # [theta1, theta2, theta3, theta4] [rad]
float64[4] phi       # [phi1, phi2, phi3, phi4] [rad]