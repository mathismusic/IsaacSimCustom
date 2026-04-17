"""Test time-reversal after robot pushes the puck.

1. Reset env with puck at default position (near robot fingers)
2. Command EEF to move forward (+x), pushing the puck
3. Let puck slide freely for M steps
4. Reverse (N+M) steps with -dt and -kd (physics reverses itself)
5. Check that puck returns to pre-push position

The torque commands (joint targets) are frozen during reversal.
The kd-negation trick makes the implicit solver reverse the dynamics:
  forward:  (M + dt·kd)·Δv = dt·F
  reverse:  (M + dt·kd)·Δv = -dt·F   (same system matrix, flipped RHS)

Usage:
    conda run -n env_isaacsim python test_push_reverse.py
"""

import sys
import numpy as np

sys.path.insert(0, "/home/krishna/Documents/sim2real/IsaacSimCustom")

from newton_shuffleboard.env import NewtonShuffleboardEnv


def main():
    print("Creating NewtonShuffleboardEnv...")
    env = NewtonShuffleboardEnv(
        dt=1.0 / 1200.0,
        decimation=20,
        episode_length=0,
    )

    obs = env.reset(randomize_puck=False)

    # Record initial state
    body_q = env.state_0.body_q.numpy()
    puck_start = body_q[env._puck_body_idx, :3].copy()
    eef_start = body_q[env._ee_body_idx, :3].copy()
    print(f"Initial EEF pos:  {eef_start}")
    print(f"Initial puck pos: {puck_start}")

    # Compute a push action: move EEF forward in +x by 0.15m
    hold = env.compute_hold_action()
    push_action = hold.copy()
    push_action[0] += 0.15  # +x

    # ---- Phase 1: Push the puck ----
    N_PUSH = 30  # 30 frames at 60Hz = 0.5s of pushing
    print(f"\nPhase 1: Pushing puck ({N_PUSH} steps)...")
    for i in range(N_PUSH):
        obs, rew, term, trunc, info = env.step(push_action)
        if i % 10 == 0:
            bq = env.state_0.body_q.numpy()
            p = bq[env._puck_body_idx, :3]
            e = bq[env._ee_body_idx, :3]
            print(f"  Step {i}: eef={e}, puck={p}")

    body_q = env.state_0.body_q.numpy()
    puck_after_push = body_q[env._puck_body_idx, :3].copy()
    eef_after_push = body_q[env._ee_body_idx, :3].copy()
    print(f"After push: eef={eef_after_push}, puck={puck_after_push}")

    push_distance = np.linalg.norm(puck_after_push[:2] - puck_start[:2])
    print(f"Puck pushed:      {push_distance:.4f}m")

    # ---- Phase 2: Free slide (hold position, let puck coast) ----
    N_SLIDE = 20  # 20 frames
    print(f"\nPhase 2: Free slide ({N_SLIDE} steps, hold action)...")
    for i in range(N_SLIDE):
        obs, rew, term, trunc, info = env.step(hold)

    body_q = env.state_0.body_q.numpy()
    puck_after_slide = body_q[env._puck_body_idx, :3].copy()
    total_distance = np.linalg.norm(puck_after_slide[:2] - puck_start[:2])
    print(f"Puck pos after slide: {puck_after_slide}")
    print(f"Total puck distance:  {total_distance:.4f}m")

    # ---- Phase 3: Reverse everything ----
    N_TOTAL = N_PUSH + N_SLIDE
    print(f"\nPhase 3: Reversing {N_TOTAL} steps (-dt, -kd, frozen targets)...")
    reversed_obs = env.step_reverse(n_steps=N_TOTAL)

    body_q = env.state_0.body_q.numpy()
    puck_after_rev = body_q[env._puck_body_idx, :3].copy()
    eef_after_rev = body_q[env._ee_body_idx, :3].copy()

    puck_error = np.linalg.norm(puck_after_rev[:2] - puck_start[:2])
    eef_error = np.linalg.norm(eef_after_rev[:2] - eef_start[:2])

    print(f"Puck after reversal:  {puck_after_rev}")
    print(f"EEF after reversal:   {eef_after_rev}")
    print(f"Puck reversal error:  {puck_error:.6f}m")
    print(f"EEF reversal error:   {eef_error:.6f}m")

    # ---- Summary ----
    print("\n=== Summary ===")
    print(f"  Push distance:      {push_distance:.4f}m")
    print(f"  Total distance:     {total_distance:.4f}m")
    print(f"  Puck reversal err:  {puck_error:.6f}m")
    print(f"  EEF reversal err:   {eef_error:.6f}m")

    if total_distance > 0.02:
        if puck_error < 0.01:
            print("  PASS")
        elif puck_error < 0.05:
            print("  MARGINAL")
        else:
            print("  FAIL")
    else:
        print("  INCONCLUSIVE (puck didn't move enough)")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
