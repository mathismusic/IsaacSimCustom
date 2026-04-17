"""Test time-reversal with raw torques only (no PD controller).

Apply sinusoidal torques for 30 frames, then reverse with the same
torques in reverse order and negative dt. Env uses RK4 integrator.

Usage:
    conda run -n env_isaacsim python test_torque_reverse.py
"""

import sys
import numpy as np
import warp as wp

sys.path.insert(0, "/home/krishna/Documents/sim2real/IsaacSimCustom")

import newton
from newton_shuffleboard.env import NewtonShuffleboardEnv


def disable_pd_controller(env):
    """Zero out gainprm and biasprm so the PD actuator produces zero force."""
    mjw = env._mjw_model
    mj = env._mj_model
    if mjw is not None:
        gp = mjw.actuator_gainprm.numpy()
        bp = mjw.actuator_biasprm.numpy()
        gp[:] = 0.0
        bp[:] = 0.0
        wp.copy(mjw.actuator_gainprm, wp.array(gp, dtype=wp.float32))
        wp.copy(mjw.actuator_biasprm, wp.array(bp, dtype=wp.float32))
        dd = mjw.dof_damping.numpy()
        df = mjw.dof_frictionloss.numpy()
        dd[:] = 0.0
        df[:] = 0.0
        wp.copy(mjw.dof_damping, wp.array(dd, dtype=wp.float32))
        wp.copy(mjw.dof_frictionloss, wp.array(df, dtype=wp.float32))
    elif mj is not None:
        mj.actuator_gainprm[:] = 0.0
        mj.actuator_biasprm[:] = 0.0
        mj.dof_damping[:] = 0.0
        mj.dof_frictionloss[:] = 0.0


def generate_torques(n_frames, scale=1.0, n_dofs=9):
    """Generate smooth sinusoidal torques for the arm joints."""
    t = np.linspace(0, 2 * np.pi, n_frames)
    torques = np.zeros((n_frames, n_dofs), dtype=np.float32)
    torques[:, 0] = scale * np.sin(t)
    torques[:, 1] = scale * 0.6 * np.sin(1.5 * t)
    torques[:, 3] = scale * 0.4 * np.cos(t)
    return torques


def main():
    N_FRAMES = 30
    DT = 1.0 / 1200.0
    DEC = 20
    SCALE = 1.0

    print("=" * 70)
    print("TEST: Raw torque forward/reverse (RK4 integrator, no PD)")
    print(f"  {N_FRAMES} frames, dt={DT:.6f}, decimation={DEC}, scale={SCALE} Nm")
    print("=" * 70)

    env = NewtonShuffleboardEnv(dt=DT, decimation=DEC, episode_length=0)
    env.reset(randomize_puck=False)

    # Move puck far away
    puck_jq_start = env._find_puck_joint_q_start()
    jq_full = env.state_0.joint_q.numpy()
    jq_full[puck_jq_start + 0] = 10.0
    jq_full[puck_jq_start + 1] = 0.0
    jq_full[puck_jq_start + 2] = 10.0
    wp.copy(env.state_0.joint_q, wp.array(jq_full, dtype=wp.float32))
    newton.eval_fk(env.model, env.state_0.joint_q, env.state_0.joint_qd, env.state_0)

    disable_pd_controller(env)

    if env.control.joint_f is None:
        env.control.joint_f = wp.zeros(env.model.joint_dof_count, dtype=wp.float32)

    jq_init = env.state_0.joint_q.numpy()[:9].copy()
    print(f"Initial joint_q[:7]: {jq_init[:7]}")

    torques = generate_torques(N_FRAMES, scale=SCALE)

    # ---- FORWARD ----
    print(f"\n--- FORWARD ({N_FRAMES} frames) ---")
    fwd_jq = [jq_init.copy()]

    for i in range(N_FRAMES):
        jf = np.zeros(env.model.joint_dof_count, dtype=np.float32)
        jf[:9] = torques[i]
        wp.copy(env.control.joint_f, wp.array(jf, dtype=wp.float32))

        for _ in range(env.decimation):
            env.state_0.clear_forces()
            env.solver.step(env.state_0, env.state_1, env.control, env._contacts, env.dt)
            env.state_0, env.state_1 = env.state_1, env.state_0

        jq = env.state_0.joint_q.numpy()[:9].copy()
        fwd_jq.append(jq)

        if i % 5 == 0 or i == N_FRAMES - 1:
            delta = np.linalg.norm(jq[:7] - jq_init[:7])
            print(f"  fwd[{i+1:2d}] Δq_arm={delta:.6f} | q0={jq[0]:.6f} q1={jq[1]:.6f} q3={jq[3]:.6f}")

    total_motion = np.linalg.norm(fwd_jq[-1][:7] - jq_init[:7])
    print(f"  Total arm motion: {total_motion:.6f} rad")

    # ---- REVERSE ----
    print(f"\n--- REVERSE ({N_FRAMES} frames, -dt, reversed torques) ---")
    neg_dt = -abs(env.dt)

    for i in range(N_FRAMES):
        torque_idx = N_FRAMES - 1 - i
        jf = np.zeros(env.model.joint_dof_count, dtype=np.float32)
        jf[:9] = torques[torque_idx]
        wp.copy(env.control.joint_f, wp.array(jf, dtype=wp.float32))

        for _ in range(env.decimation):
            env.state_0.clear_forces()
            env.solver.step(env.state_0, env.state_1, env.control, env._contacts, neg_dt)
            env.state_0, env.state_1 = env.state_1, env.state_0

        jq = env.state_0.joint_q.numpy()[:9].copy()
        fwd_match = fwd_jq[N_FRAMES - 1 - i]
        err_a = np.linalg.norm(jq[:7] - fwd_match[:7])
        err_f = np.linalg.norm(jq[7:9] - fwd_match[7:9])

        if i % 5 == 0 or i == N_FRAMES - 1:
            print(f"  rev[{i+1:2d}]→fwd[{N_FRAMES-1-i:2d}] arm_err={err_a:.8f} finger_err={err_f:.8f}")

    # ---- SUMMARY ----
    jq_after = env.state_0.joint_q.numpy()[:9].copy()
    return_err_arm = np.linalg.norm(jq_after[:7] - jq_init[:7])
    return_err_finger = np.linalg.norm(jq_after[7:9] - jq_init[7:9])
    per_joint = np.abs(jq_after[:7] - jq_init[:7])

    print(f"\n--- SUMMARY ---")
    print(f"  Total forward motion:   {total_motion:.6f} rad")
    print(f"  Arm return error:       {return_err_arm:.8f} rad ({np.degrees(return_err_arm):.4f} deg)")
    print(f"  Finger return error:    {return_err_finger:.8f} rad ({np.degrees(return_err_finger):.4f} deg)")
    print(f"  Per-joint arm error:    {per_joint}")
    print(f"  Final q[:7]:  {jq_after[:7]}")
    print(f"  Init  q[:7]:  {jq_init[:7]}")

    if total_motion > 0.01:
        if return_err_arm < 1e-4:
            print("  PASS (arm < 0.1 mrad)")
        elif return_err_arm < 1e-3:
            print("  GOOD (arm < 1 mrad)")
        elif return_err_arm < 0.01:
            print("  MARGINAL (arm < 10 mrad)")
        else:
            print(f"  FAIL (arm err = {return_err_arm:.4f} rad)")
    else:
        print("  INCONCLUSIVE (robot didn't move enough)")

    env.control.joint_f.zero_()
    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
