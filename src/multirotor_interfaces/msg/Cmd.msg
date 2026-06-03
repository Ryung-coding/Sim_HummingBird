# position command [x, y, z] [m], z-down convention
float32[3] pos_cmd   

# att_cmd when USE_SO3_HEADING_CMD = false: => [roll, pitch, yaw] [rad]
# att_cmd when USE_SO3_HEADING_CMD = true: => heading direction unit vector [hx, hy, hz]  
float32[3] att_cmd