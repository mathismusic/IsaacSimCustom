"""Test time-reversal physics in the Newton shuffleboard environment.

1. Reset the environment with puck placed far from robot
2. Let puck settle on table
3. Give the puck an initial velocity along X (via joint_qd)
4. Let the puck slide freely for N steps (forward)
5. Reverse physics for N steps (negative dt + negated kd)
6. Check that the puck returns close to its pre-kick position

Uses the kd-negation trick: negating both dt and kd keeps the implicit
system matrix positive-definite (M + (-dt)*(-kd) = M + dt*kd) while
the anti-damping adds back dissipated energy → exact time reversal.

Usage:
    conda run -n env_isaacsim python test_negative_dt.py
"""

import sys
import numpy as np
import warp as wp

sys.path.insert(0, "/home/krishna/Documents/sim2real/IsaacSimCustom")

import newton
from newton_shuffleboard.env import NewtonShuffleboardEnv, PUCK_Z


def main():
    # Fine substeps to resolve elastic contact dynamics (zero damping).
    # dt=1/1200 per substep, 20 substeps/frame → control rate 1/60 Hz.
    # With contact_stiffness=1e4, dt_stable = pi*sqrt(m/k) = 0.01s >> 8.3e-4.
    print("Creating NewtonShuffleboardEnv with fine substeps...")
    env = NewtonShuffleboardEnv(
        dt=1.0 / 1200.0,
        decimation=20,
        episode_length=0,
    )

    obs = env.reset(randomize_puck=False)

    # Move puck far from robot to avoid finger contact during test.
    puck_jq_start = env._find_puck_joint_q_start()
    joint_q = env.state_0.joint_q.numpy()
    joint_q[puck_jq_start + 0] = 1.5
    joint_q[puck_jq_start + 1] = 0.0
    joint_q[puck_jq_start + 2] = PUCK_Z  # At EEF height (gravity-compensated)
    joint_q[puck_jq_start + 3:puck_jq_start + 7] = [0.0, 0.0, 0.0, 1.0]
    wp.copy(env.state_0.joint_q, wp.array(joint_q, dtype=wp.float32))
    newton.eval_fk(env.model, env.state_0.joint_q, env.state_0.joint_qd, env.state_0)

    # Record the initial state (no settling — elastic contacts don't dissipate,
    # so settling would never converge). The puck is placed at rest on the table.
    body_q = env.state_0.body_q.numpy()
    puck_start = body_q[env._puck_body_idx, :3].copy()
    puck_vel_start = env.state_0.body_qd.numpy()[env._puck_body_idx, 3:6].copy()
    print(f"Initial puck pos: {puck_start}")
    print(f"Initial puck vel: {puck_vel_start} (speed: {np.linalg.norm(puck_vel_start[:2]):.6f})")

    # Set puck velocity via joint_qd (MuJoCo reads joint-space velocities)
    # MuJoCo free joint DOF layout: [vx, vy, vz, wx, wy, wz] starting at DOF 9
    joint_qd = env.state_0.joint_qd.numpy()
    joint_qd[9 + 0] = 5.0  # vx = 5.0 m/s
    wp.copy(env.state_0.joint_qd, wp.array(joint_qd, dtype=wp.float32))
    newton.eval_fk(env.model, env.state_0.joint_q, env.state_0.joint_qd, env.state_0)

    body_qd_check = env.state_0.body_qd.numpy()[env._puck_body_idx]
    print(f"\nPuck body_qd after kick: {body_qd_check}")

    # ---- Phase 1: Let puck slide freely ----
    N_SLIDE = 30  # 30 control steps at 60Hz = 0.5s
    print(f"\nPhase 1: Free sliding ({N_SLIDE} control steps = {N_SLIDE/60:.2f}s)...")

    for i in range(N_SLIDE):
        env._step_physics()
        if i % 10 == 0:
            bq = env.state_0.body_q.numpy()
            bqd = env.state_0.body_qd.numpy()
            p = bq[env._puck_body_idx, :3]
            v = bqd[env._puck_body_idx, 3:6]
            print(f"  Step {i}: pos={p}, vel={v[:2]}")

    body_q = env.state_0.body_q.numpy()
    puck_after_slide = body_q[env._puck_body_idx, :3].copy()
    body_qd = env.state_0.body_qd.numpy()
    puck_vel = body_qd[env._puck_body_idx, 3:6].copy()

    print(f"Puck pos after slide: {puck_after_slide}")
    print(f"Puck velocity:        {puck_vel[:2]} (speed: {np.linalg.norm(puck_vel[:2]):.4f} m/s)")

    distance_slid = np.linalg.norm(puck_after_slide[:2] - puck_start[:2])
    print(f"Distance puck slid:   {distance_slid:.4f}m")

    # Also record robot state for comparison
    robot_q_after_slide = env.state_0.joint_q.numpy()[:9].copy()
    robot_qd_after_slide = env.state_0.joint_qd.numpy()[:9].copy()
    print(f"Robot joint_q after slide:  {robot_q_after_slide}")
    print(f"Robot joint_qd after slide: {robot_qd_after_slide}")

    # ---- Phase 2: Reverse physics (kd-negation) ----
    print(f"\nPhase 2: Reversing physics ({N_SLIDE} steps, -dt and -kd)...")
    reversed_obs = env.step_reverse(n_steps=N_SLIDE)

    body_q = env.state_0.body_q.numpy()
    puck_after_reverse = body_q[env._puck_body_idx, :3].copy()
    body_qd = env.state_0.body_qd.numpy()
    puck_vel_after = body_qd[env._puck_body_idx, 3:6].copy()

    print(f"Puck pos after reversal:  {puck_after_reverse}")
    print(f"Puck velocity after:      {puck_vel_after[:2]}")

    # Robot state after reversal
    robot_q_after_rev = env.state_0.joint_q.numpy()[:9].copy()
    robot_qd_after_rev = env.state_0.joint_qd.numpy()[:9].copy()
    print(f"Robot joint_q after rev:  {robot_q_after_rev}")
    print(f"Robot joint_qd after rev: {robot_qd_after_rev}")

    reversal_error = np.linalg.norm(puck_after_reverse[:2] - puck_start[:2])
    print(f"\nReversal error (XY distance from pre-kick): {reversal_error:.6f}m")

    if distance_slid > 0.05:
        if reversal_error < 0.01:
            print("PASS: Puck returned close to pre-kick position!")
        elif reversal_error < 0.05:
            print(f"MARGINAL: Reversal error {reversal_error:.4f}m (< 0.05m but > 0.01m)")
        else:
            print(f"FAIL: Reversal error too large ({reversal_error:.4f}m > 0.05m)")
    else:
        print("INCONCLUSIVE: Puck didn't slide far enough for a meaningful test.")

    # ---- Summary ----
    print("\n=== Summary ===")
    print(f"  Puck pre-kick:      {puck_start}")
    print(f"  Puck after slide:   {puck_after_slide}")
    print(f"  Puck after reverse: {puck_after_reverse}")
    print(f"  Slide distance:     {distance_slid:.4f}m")
    print(f"  Reversal error:     {reversal_error:.6f}m")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
