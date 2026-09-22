# Wrench order:
# [Mx, My, Mz, Fx, Fy, Fz]

float64[6] wrench_cmd
float64[6] wrench_alloc
float64[6] wrench_real

float64[6] w_dot_des
float64[6] w_dot_q

# q order:
# [alpha1, alpha2, alpha3, alpha4, beta1, beta2, f1, f2, f3, f4]

float64[10] q_dot_primary
float64[10] q_dot_nullspace
float64[10] q_dot_final

float64 primary_scale
float64 nullspace_scale