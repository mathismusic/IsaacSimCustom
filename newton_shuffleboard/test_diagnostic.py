"""Diagnostic test: kd-correction reversal.

During reverse with negated kd:
  MuJoCo computes: τ_rev = kp*(T-q) + kd*v   (sign flip due to -kd)
  We want:         τ_fwd = kp*(T-q) - kd*v
  Correction:      -2*kd*v via joint_f

System matrix M + h*kd stays PD. Total torque matches forward exactly.

Usage:
    conda run -n env_isaacsim python test_diagnostic.py
"""

import sys
import numpy as np
import warp as wp

sys.path.insert(0, "/home/krishna/Documents/sim2real/IsaacSimCustom")

from newton_shuffleboard.env import NewtonShuffleboardEnv


def snapshot(env):
    bq = env.state_0.body_q.numpy()
    bqd = env.state_0.body_qd.numpy()
    jq = env.state_0.joint_q.numpy()
    jqd = env.state_0.joint_qd.numpy()
    return {
        "joint_q": jq[:9].copy(),
        "joint_qd": jqd[:9].copy(),
        "eef_pos": bq[env._ee_body_idx, :3].copy(),
        "puck_pos": bq[env._puck_body_idx, :3].copy(),
        "puck_vel": bqd[env._puck_body_idx, 3:6].copy(),
    }


def run_forward(env, push_action, hold, n_push, n_slide):
    """Run forward, recording snapshots."""
    n_total = n_push + n_slide
    fwd = [snapshot(env)]

    for frame in range(n_total):
        action = push_action if frame < n_push else hold
        action = np.asarray(action, dtype=np.float32).flatten()

        target_pos = action[:3]
        target_quat_wxyz = action[3:7]
        arm_q = env._solve_ik(target_pos, target_quat_wxyz)
        joint_targets = np.zeros(9, dtype=np.float32)
        joint_targets[:7] = arm_q
        wp.copy(env.control.joint_target_pos[:9], wp.array(joint_targets, dtype=wp.float32))
        env._target_history.append(joint_targets.copy())

        for _ in range(env.decimation):
            env.state_0.clear_forces()
            env.solver.step(env.state_0, env.state_1, env.control, env._contacts, env.dt)
            env.state_0, env.state_1 = env.state_1, env.state_0
        env.solver.update_contacts(env._contacts, env.state_0)
        fwd.append(snapshot(env))

    return fwd


def main():
    print("=" * 80)
    print("DIAGNOSTIC: kd-correction reversal")
    print("=" * 80)

    N_PUSH = 5
    N_SLIDE = 3
    N_TOTAL = N_PUSH + N_SLIDE
    DT = 1.0 / 1200.0
    DEC = 20

    # =====================================================================
    # FORWARD
    # =====================================================================
    print(f"\n--- FORWARD ({N_PUSH} push + {N_SLIDE} slide) ---")
    env = NewtonShuffleboardEnv(dt=DT, decimation=DEC, episode_length=0)
    env.reset(randomize_puck=False)

    hold = env.compute_hold_action()
    push_action = hold.copy()
    push_action[0] += 0.15

    # Extract kd values
    if env._mjw_model is not None:
        bp = env._mjw_model.actuator_biasprm.numpy()
        kd_vals = -bp[0, :9, 2]  # biasprm stores -kd
    else:
        kd_vals = -env.solver.mj_model.actuator_biasprm[:9, 2]
    print(f"  kd: {kd_vals}")

    fwd = run_forward(env, push_action, hold, N_PUSH, N_SLIDE)

    for i in range(1, N_TOTAL + 1):
        s = fwd[i]
        phase = "push" if i <= N_PUSH else "slide"
        print(f"  fwd[{i:2d}] {phase:5s} | eef_x={s['eef_pos'][0]:.6f} puck_x={s['puck_pos'][0]:.6f}")
    fwd_dist = np.linalg.norm(fwd[-1]["puck_pos"][:2] - fwd[0]["puck_pos"][:2])
    print(f"  Puck displacement: {fwd_dist:.6f}m")

    # =====================================================================
    # REVERSE C: kd negation + reversed targets + kd correction force
    # =====================================================================
    print(f"\n--- REVERSE C: kd negation + targets + correction force ---")
    print(f"  Correction: inject -2*kd*v via joint_f at each substep")

    env._set_mujoco_kd(-env._original_biasprm_kd)
    neg_dt = -abs(env.dt)

    if env.control.joint_f is None:
        env.control.joint_f = wp.zeros(env.model.joint_dof_count, dtype=wp.float32)

    revC = [snapshot(env)]

    for i in range(N_TOTAL):
        targets = env._target_history[-(i + 1)]
        wp.copy(env.control.joint_target_pos[:9], wp.array(targets, dtype=wp.float32))

        for sub in range(env.decimation):
            # Compute correction: -2 * kd * v
            jqd = env.state_0.joint_qd.numpy()[:9]
            correction = -2.0 * kd_vals * jqd

            joint_f = np.zeros(env.model.joint_dof_count, dtype=np.float32)
            joint_f[:9] = correction
            wp.copy(env.control.joint_f, wp.array(joint_f, dtype=wp.float32))

            env.state_0.clear_forces()
            env.solver.step(env.state_0, env.state_1, env.control, env._contacts, neg_dt)
            env.state_0, env.state_1 = env.state_1, env.state_0

        env.solver.update_contacts(env._contacts, env.state_0)

        s = snapshot(env)
        revC.append(s)
        fwd_match = fwd[N_TOTAL - (i + 1)]
        puck_err = np.linalg.norm(s["puck_pos"] - fwd_match["puck_pos"])
        eef_err = np.linalg.norm(s["eef_pos"] - fwd_match["eef_pos"])
        print(f"  revC[{i+1:2d}]→fwd[{N_TOTAL-(i+1):2d}] | puck_err={puck_err:.6f} eef_err={eef_err:.6f} | rev_eef_x={s['eef_pos'][0]:.6f} fwd_eef_x={fwd_match['eef_pos'][0]:.6f}")

    env._set_mujoco_kd(env._original_biasprm_kd)
    env.control.joint_f.zero_()

    puck_errC = np.linalg.norm(revC[-1]["puck_pos"][:2] - fwd[0]["puck_pos"][:2])
    eef_errC = np.linalg.norm(revC[-1]["eef_pos"][:2] - fwd[0]["eef_pos"][:2])

    print(f"\n--- SUMMARY ---")
    print(f"  Forward puck displacement: {fwd_dist:.6f}m")
    print(f"  RevC (kd negation + targets + correction): puck={puck_errC:.6f}m  eef={eef_errC:.6f}m")
    if puck_errC < 0.001:
        print(f"  PASS")
    elif puck_errC < 0.01:
        print(f"  GOOD")
    elif puck_errC < 0.05:
        print(f"  MARGINAL")
    else:
        print(f"  FAIL")

    env.close()


if __name__ == "__main__":
    main()
