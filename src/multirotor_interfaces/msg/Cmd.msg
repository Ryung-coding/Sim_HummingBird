# position command [x, y, z] [m], z-down convention
float32[3] pos_cmd
float32[3] vel_cmd
float32[3] acc_cmd

# att_cmd when USE_SO3_HEADING_CMD = false: => [roll, pitch, yaw] [rad]
# att_cmd when USE_SO3_HEADING_CMD = true: => heading direction unit vector [hx, hy, hz]  
float32[3] att_cmd

# Hamilton quaternion [w, x, y, z] and body angular feedforward
float32[4] quat_cmd
float32[3] angular_velocity_cmd
float32[3] angular_acceleration_cmd
bool use_quaternion
