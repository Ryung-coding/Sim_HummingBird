#!/usr/bin/env python3
import numpy as np


# ============================================================
# Convention
# ============================================================
# W = [Mx, My, Mz, Fx, Fy, Fz]^T
#
# Body frame:
#   x: forward
#   y: left
#   z: up
#
# Rotor order:
#   F1: 11시 = front-left
#   F2: 7시  = rear-left
#   F3: 5시  = rear-right
#   F4: 1시  = front-right
#
# Force component:
#
#   lambda_i = [Tix, Tiy, Tiz]^T
#
# Tilt convention:
#
#   alpha: rotation about x-axis
#          positive alpha tilts thrust toward +y
#
#   beta : rotation about y-axis
#          positive beta tilts thrust toward +x
#
# Direction model:
#
#   lambda_i = Fi [
#       sin(beta_i) cos(alpha_i),
#       sin(alpha_i),
#       cos(beta_i) cos(alpha_i)
#   ]^T
#
# Therefore:
#   alpha = 0, beta = 0  ->  +z direction
#
# Front pair:
#   rotor 1 and 4 share alpha1, beta1
#
# Rear pair:
#   rotor 2 and 3 share alpha2, beta2
# ============================================================


# ============================================================
# Settings
# ============================================================

Lx = 0.1861
Ly = 0.1861
zeta = 0.0200

USE_REACTION_TORQUE = True

DAMPING_NEW = 1.0e-8
DAMPING_GIT = 1.0e-4

F_MIN = 1.0e-9
F_CMD_MIN = 1.0e-6
F_CMD_MAX = 50.0

CLAMP_THRUST = False


# ============================================================
# Utility
# ============================================================

def skew(r):
    x, y, z = r
    return np.array([
        [0.0, -z,  y],
        [z,  0.0, -x],
        [-y, x,  0.0],
    ], dtype=float)


def damped_pinv_right(A, damping):
    """
    A# = A^T (A A^T + damping^2 I)^(-1)
    """
    m = A.shape[0]
    return A.T @ np.linalg.inv(A @ A.T + damping**2 * np.eye(m))


def deg(x):
    return np.rad2deg(x)


def fmt_arr(x, width=8, prec=3):
    return "[" + " ".join(f"{v:{width}.{prec}f}" for v in x) + "]"


# ============================================================
# Correct alpha-beta direction
# ============================================================

def direction_from_alpha_beta(alpha, beta):
    """
    Direction vector:

      e = [
          sin(beta) cos(alpha),
          sin(alpha),
          cos(beta) cos(alpha)
      ]

    alpha = 0, beta = 0 -> e = [0, 0, 1]
    """
    return np.array([
        np.sin(beta) * np.cos(alpha),
        np.sin(alpha),
        np.cos(beta) * np.cos(alpha),
    ], dtype=float)


def alpha_beta_from_direction(v, eps=1.0e-12):
    """
    Given v = [Tx, Ty, Tz], find alpha, beta such that

      v / ||v|| = [
          sin(beta) cos(alpha),
          sin(alpha),
          cos(beta) cos(alpha)
      ]

    Therefore:

      alpha = asin(Ty / ||T||)
      beta  = atan2(Tx, Tz)
    """
    n = np.linalg.norm(v)

    if n < eps:
        alpha = 0.0
        beta = 0.0
        e = np.array([0.0, 0.0, 1.0])
        return alpha, beta, e

    e = v / n
    ex, ey, ez = e

    alpha = np.arcsin(np.clip(ey, -1.0, 1.0))
    beta = np.arctan2(ex, ez)

    e_rec = direction_from_alpha_beta(alpha, beta)
    return alpha, beta, e_rec


# ============================================================
# Geometry
# ============================================================

def rotor_positions():
    """
    Rotor order:
      1: front-left
      2: rear-left
      3: rear-right
      4: front-right
    """
    r1 = np.array([+Lx, +Ly, 0.0])
    r2 = np.array([-Lx, +Ly, 0.0])
    r3 = np.array([-Lx, -Ly, 0.0])
    r4 = np.array([+Lx, -Ly, 0.0])
    return [r1, r2, r3, r4]


def rotor_spins():
    """
    Same spin convention used before:
      rotor 1: +
      rotor 2: -
      rotor 3: +
      rotor 4: -
    """
    return [+1.0, -1.0, +1.0, -1.0]


def wrench_from_lambdas(lambdas, use_reaction_torque=USE_REACTION_TORQUE):
    positions = rotor_positions()
    spins = rotor_spins()

    M = np.zeros(3)
    F = np.zeros(3)

    for i in range(4):
        lam = lambdas[i]

        M += np.cross(positions[i], lam)

        if use_reaction_torque:
            M += spins[i] * zeta * lam

        F += lam

    return np.array([M[0], M[1], M[2], F[0], F[1], F[2]], dtype=float)


# ============================================================
# Method 1: New 6x12 constant allocation
# ============================================================

def build_A_new_6x12(use_reaction_torque=USE_REACTION_TORQUE):
    """
    T = [
      T1x,T1y,T1z,
      T2x,T2y,T2z,
      T3x,T3y,T3z,
      T4x,T4y,T4z
    ]^T

    W = A T
    """
    positions = rotor_positions()
    spins = rotor_spins()

    A = np.zeros((6, 12), dtype=float)

    for i, r in enumerate(positions):
        c = 3 * i

        M_block = skew(r)

        if use_reaction_torque:
            M_block = M_block + spins[i] * zeta * np.eye(3)

        A[0:3, c:c+3] = M_block
        A[3:6, c:c+3] = np.eye(3)

    return A


def allocate_new(W_cmd):
    """
    New method:
      1. Solve T_raw = A_new# W_cmd.
      2. Front pair direction = lambda1 + lambda4.
      3. Rear pair direction  = lambda2 + lambda3.
      4. Individual thrusts are projections onto pair directions.
    """
    W_cmd = np.asarray(W_cmd, dtype=float).reshape(6)

    A_new = build_A_new_6x12()
    T_raw = damped_pinv_right(A_new, DAMPING_NEW) @ W_cmd

    lam1 = T_raw[0:3]
    lam2 = T_raw[3:6]
    lam3 = T_raw[6:9]
    lam4 = T_raw[9:12]

    lam_front = lam1 + lam4
    lam_rear = lam2 + lam3

    alpha1, beta1, e_front = alpha_beta_from_direction(lam_front)
    alpha2, beta2, e_rear = alpha_beta_from_direction(lam_rear)

    F1 = float(lam1 @ e_front)
    F4 = float(lam4 @ e_front)
    F2 = float(lam2 @ e_rear)
    F3 = float(lam3 @ e_rear)

    if CLAMP_THRUST:
        F1 = float(np.clip(F1, F_CMD_MIN, F_CMD_MAX))
        F2 = float(np.clip(F2, F_CMD_MIN, F_CMD_MAX))
        F3 = float(np.clip(F3, F_CMD_MIN, F_CMD_MAX))
        F4 = float(np.clip(F4, F_CMD_MIN, F_CMD_MAX))

    lambdas_impl = [
        F1 * e_front,
        F2 * e_rear,
        F3 * e_rear,
        F4 * e_front,
    ]

    W_impl = wrench_from_lambdas(lambdas_impl)
    err = np.linalg.norm(W_cmd - W_impl)

    return {
        "F": np.array([F1, F2, F3, F4]),
        "alpha_deg": np.array([deg(alpha1), deg(alpha2)]),
        "beta_deg": np.array([deg(beta1), deg(beta2)]),
        "W_impl": W_impl,
        "err": err,
    }


# ============================================================
# Method 2: Git-style two-stage allocation, corrected alpha-beta axes
# ============================================================

def build_A_git_style(alpha_state, beta_state):
    """
    Git-style virtual allocation.

    C = [
      front_Fx, front_Fy, front_Fz,
      rear_Fx,  rear_Fy,  rear_Fz,
      df14,
      df23
    ]^T

    W = A_git C

    Current angle state is used only to build the df columns.
    Here:
      alpha_state = [0, 0]
      beta_state  = [0, 0]
    """

    alpha_front = alpha_state[0]
    alpha_rear = alpha_state[1]
    beta_front = beta_state[0]
    beta_rear = beta_state[1]

    e_f = direction_from_alpha_beta(alpha_front, beta_front)
    e_r = direction_from_alpha_beta(alpha_rear, beta_rear)

    ex_f, ey_f, ez_f = e_f
    ex_r, ey_r, ez_r = e_r

    A_git = np.zeros((6, 8), dtype=float)

    # ------------------------------------------------------------
    # Front pair: rotor 1 and 4
    #
    #   lambda1 = f1 e_f
    #   lambda4 = f4 e_f
    #   front_force = lambda1 + lambda4
    #   df14 = f1 - f4
    #
    # Rear pair: rotor 2 and 3
    #
    #   lambda2 = f2 e_r
    #   lambda3 = f3 e_r
    #   rear_force = lambda2 + lambda3
    #   df23 = f2 - f3
    # ------------------------------------------------------------

    # Mx
    # front: Ly * ez_f * df14
    # rear : Ly * ez_r * df23
    A_git[0, 6] = Ly * ez_f
    A_git[0, 7] = Ly * ez_r

    # My
    # front: -Lx * front_Fz
    # rear : +Lx * rear_Fz
    A_git[1, 2] = -Lx
    A_git[1, 5] = +Lx

    # Mz
    # front: +Lx * front_Fy - Ly * ex_f * df14
    # rear : -Lx * rear_Fy  - Ly * ex_r * df23
    A_git[2, 1] = +Lx
    A_git[2, 4] = -Lx
    A_git[2, 6] = -Ly * ex_f
    A_git[2, 7] = -Ly * ex_r

    # Reaction torque:
    #
    # front: spin1=+, spin4=- -> +zeta * df14 * e_f
    # rear : spin2=-, spin3=+ -> -zeta * df23 * e_r
    if USE_REACTION_TORQUE:
        A_git[0, 6] += zeta * ex_f
        A_git[1, 6] += zeta * ey_f
        A_git[2, 6] += zeta * ez_f

        A_git[0, 7] += -zeta * ex_r
        A_git[1, 7] += -zeta * ey_r
        A_git[2, 7] += -zeta * ez_r

    # Fx = front_Fx + rear_Fx
    A_git[3, 0] = 1.0
    A_git[3, 3] = 1.0

    # Fy = front_Fy + rear_Fy
    A_git[4, 1] = 1.0
    A_git[4, 4] = 1.0

    # Fz = front_Fz + rear_Fz
    A_git[5, 2] = 1.0
    A_git[5, 5] = 1.0

    return A_git


def allocate_git_style(W_cmd):
    """
    Existing git-style logic:
      1. Current tilt state is fixed to zero.
      2. Build A_git from current angle state.
      3. Solve C = A_git# W.
      4. Convert front/rear force vector to alpha1,beta1,alpha2,beta2.
      5. Split pair thrust by df14, df23.
    """
    W_cmd = np.asarray(W_cmd, dtype=float).reshape(6)

    # current tilt state = zero
    alpha_state = np.array([0.0, 0.0])
    beta_state = np.array([0.0, 0.0])

    A_git = build_A_git_style(alpha_state, beta_state)
    C = damped_pinv_right(A_git, DAMPING_GIT) @ W_cmd

    front_force = C[0:3]
    rear_force = C[3:6]
    df14 = C[6]
    df23 = C[7]

    front_norm = np.linalg.norm(front_force)
    rear_norm = np.linalg.norm(rear_force)

    safe_front_norm = max(front_norm, F_MIN)
    safe_rear_norm = max(rear_norm, F_MIN)

    alpha1, beta1, e_front = alpha_beta_from_direction(front_force)
    alpha2, beta2, e_rear = alpha_beta_from_direction(rear_force)

    # Keep pair thrusts positive
    df14 = float(np.clip(df14, -safe_front_norm + F_CMD_MIN, safe_front_norm - F_CMD_MIN))
    df23 = float(np.clip(df23, -safe_rear_norm + F_CMD_MIN, safe_rear_norm - F_CMD_MIN))

    F1 = 0.5 * (safe_front_norm + df14)
    F4 = 0.5 * (safe_front_norm - df14)

    F2 = 0.5 * (safe_rear_norm + df23)
    F3 = 0.5 * (safe_rear_norm - df23)

    if CLAMP_THRUST:
        F1 = float(np.clip(F1, F_CMD_MIN, F_CMD_MAX))
        F2 = float(np.clip(F2, F_CMD_MIN, F_CMD_MAX))
        F3 = float(np.clip(F3, F_CMD_MIN, F_CMD_MAX))
        F4 = float(np.clip(F4, F_CMD_MIN, F_CMD_MAX))

    lambdas_impl = [
        F1 * e_front,
        F2 * e_rear,
        F3 * e_rear,
        F4 * e_front,
    ]

    W_impl = wrench_from_lambdas(lambdas_impl)
    err = np.linalg.norm(W_cmd - W_impl)

    return {
        "F": np.array([F1, F2, F3, F4]),
        "alpha_deg": np.array([deg(alpha1), deg(alpha2)]),
        "beta_deg": np.array([deg(beta1), deg(beta2)]),
        "W_impl": W_impl,
        "err": err,
    }


# ============================================================
# Print
# ============================================================

def print_compare(W):
    W = np.asarray(W, dtype=float).reshape(6)

    new = allocate_new(W)
    git = allocate_git_style(W)

    print(
        f"W={fmt_arr(W)} | "
        f"new: F={fmt_arr(new['F'])}, "
        f"alpha={fmt_arr(new['alpha_deg'])}deg, "
        f"beta={fmt_arr(new['beta_deg'])}deg, "
        f"err={new['err']:.3e} | "
        f"git: F={fmt_arr(git['F'])}, "
        f"alpha={fmt_arr(git['alpha_deg'])}deg, "
        f"beta={fmt_arr(git['beta_deg'])}deg, "
        f"err={git['err']:.3e}"
    )


if __name__ == "__main__":
    # W = [Mx, My, Mz, Fx, Fy, Fz]
    wrench_tests = [
        # mostly +z force with small +y component
        [0.0, 0.0, 0.0, 0.0, 4.0, 40.0],
        [0.0, 0.0, 0.0, 5.0, 4.0, 40.0],
        [0.0, 0.0, 0.0, 0.0, 4.0, 40.0],
        [0.0, 0.0, 0.0, 5.0, 4.0, 40.0],

        # roll
        [0.3, 0.0, 0.0, 0.0, 4.0, 40.0],
        [-0.3, 0.0, 0.0, 0.0, 0.0, 40.0],

        # pitch
        [0.0, 0.3, 0.0, 0.0, 4.0, 40.0],
        [0.0, -0.3, 0.0, 0.0, 4.0, 40.0],

        # yaw
        [0.0, 0.0, 0.3, 0.0, 4.0, 40.0],
        [0.0, 0.0, -0.3, 0.0, 4.0, 40.0],

        # combined
        [0.3, 0.5, 0.2, 3.0, 4.0, 40.0],
        [-0.3, 0.5, -0.2, -3.0, 4.0, 40.0],
        [0.5, -0.3, 0.5, 5.0, 3.0, 40.0],
        [0.3, 0.3, 0.3, 10.0, 5.0, 40.0],
    ]

    for W in wrench_tests:
        print_compare(W)