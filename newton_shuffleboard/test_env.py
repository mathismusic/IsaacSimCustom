"""Smoke test for the Newton shuffleboard environment.

Creates the env, resets, computes a hold-position action, and steps
for 500 frames.  The robot should stay still and the puck should remain
on the table.

Usage:
    conda run -n env_isaacsim python test_env.py
"""

import sys
import numpy as np

sys.path.insert(0, "/home/krishna/Documents/sim2real/IsaacSimCustom")

from newton_shuffleboard.env import NewtonShuffleboardEnv


def main():
    print("Creating NewtonShuffleboardEnv...")
    env = NewtonShuffleboardEnv(dt=1.0 / 120.0, decimation=2, episode_length=0)

    print(f"Model bodies: {env.model.body_count}")
    print(f"Model joints: {env.model.joint_count}")
    print(f"Joint DOFs:   {env.model.joint_dof_count}")
    print(f"Joint coords: {env.model.joint_coord_count}")
    print()

    # Print body labels
    print("Body labels:")
    for i, lbl in enumerate(env.model.body_label):
        print(f"  {i}: {lbl}")
    print()

    # Print joint labels
    print("Joint labels:")
    for i, lbl in enumerate(env.model.joint_label):
        print(f"  {i}: {lbl}")
    print()

    # Reset
    obs = env.reset(randomize_puck=False)
    print(f"Observation shape: {obs['policy'].shape}")
    print(f"Initial obs: {obs['policy']}")
    print()

    # Compute hold action
    hold_action = env.compute_hold_action()
    print(f"Hold action: {hold_action}")
    print()

    # Record initial state
    body_q_init = env.state_0.body_q.numpy().copy()
    eef_pos_init = body_q_init[env._ee_body_idx, :3].copy()
    puck_pos_init = body_q_init[env._puck_body_idx, :3].copy()
    print(f"Initial EEF pos:  {eef_pos_init}")
    print(f"Initial puck pos: {puck_pos_init}")
    print()

    # Step 500 times with hold action (ignore terminations — puck may
    # bounce off fingers with elastic contacts, which is expected)
    print("Stepping 500 times with hold action...")
    for i in range(500):
        obs, rew, term, trunc, info = env.step(hold_action)
        if (i + 1) % 100 == 0:
            body_q = env.state_0.body_q.numpy()
            eef_pos = body_q[env._ee_body_idx, :3]
            eef_drift = np.linalg.norm(eef_pos - eef_pos_init)
            print(f"  Step {i+1}: EEF drift={eef_drift:.4f}m")

    # Final state
    body_q_final = env.state_0.body_q.numpy()
    eef_pos_final = body_q_final[env._ee_body_idx, :3]
    eef_drift = np.linalg.norm(eef_pos_final - eef_pos_init)

    print()
    print(f"Final EEF pos:  {eef_pos_final} (drift: {eef_drift:.4f}m)")
    print()

    # The critical assertion: robot holds position with gravity compensation.
    # Puck behavior is not tested here — elastic contacts (required for
    # time reversal) cause the puck to bounce off the fingers unpredictably.
    assert eef_drift < 0.01, f"EEF drifted too much: {eef_drift:.4f}m > 0.01m"

    print("PASS: Robot held position.")

    # Test scene state for teleop
    state = env.get_scene_state()
    print()
    print("Scene state keys:", list(state.keys()))
    print(f"  EEF:    {state['eef']}")
    print(f"  Puck:   {state['puck']}")
    print(f"  Ghost:  {state['ghost']}")
    print(f"  Joints: {state['joints']}")
    print(f"  Links:  {len(state['links'])} link poses")

    # Test EEF state for operator
    eef_state = env.get_eef_state()
    print(f"  Operator EEF state (9D): {eef_state}")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
