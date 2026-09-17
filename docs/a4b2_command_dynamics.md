# A4B2 command dynamics

The differential allocator computes the actuator velocity that tracks the
wrench:

    q_dot_DA = J_dagger W_dot_des + (I - J_dagger J) q_dot_ns

The nullspace posture velocity is secondary, so its projected component does
not change the primary wrench at the current configuration:

    q_dot_ns = K (q_ref - q)

`q_dot_DA` is rate-limited once by `HB_QDOT_MAX`. The actuator command then follows Eq. (14) of the paper:

    q_dot = K (q_cmd - q)

Solving the actuator model for the command gives:

    q_cmd = q + K^-1 q_dot

The implementation uses `HB_Q_CMD_TAU_SERVO` and `HB_Q_CMD_TAU_THRUST` as the diagonal entries of `K^-1` for the servo and thrust coordinates, respectively. The former simple Euler integration is retained only as a commented experimental alternative in `utils.hpp`.

The final output clamp is physical: alpha is limited to +/-30 deg, beta to
+/-180 deg, and thrust to [0, 20] N, matching HB_model.xml. There is no
reference-centred beta clamp; `beta_ref` affects beta only through the
nullspace term.
