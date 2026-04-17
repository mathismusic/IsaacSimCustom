"""Keyboard teleop for NewtonShuffleboardReverseEnv.

Opens a viewer and lets you drive the EEF target pose in env-local XYZ
with the keyboard. The arm PD controller tracks the target under
reversed-dt dynamics; the puck starts at the forward target and slides
back toward a sampled point in the start box.

Controls (no conflict with the viewer's WASD+arrow camera bindings)
    I / K : +X / -X  (toward / away from far end of table)
    J / L : +Y / -Y  (lateral — J=+Y=left side, L=-Y=right side)
    U / O : +Z / -Z  (up / down)
    R     : reset episode (discards any in-progress recording)
    P     : save recorded demo and run forward-time playback (USD output)
    SPACE : pause / resume physics
    ESC   : quit
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np

from newton_shuffleboard.env import NewtonShuffleboardReverseEnv
from newton_shuffleboard.playback import DEMO_DIR, playback, save_demo


# Translation step per frame while a key is held (m). Small enough for
# smooth PD tracking; large enough to traverse the workspace in seconds.
TRANSLATION_STEP = 0.003


def main():
    env = NewtonShuffleboardReverseEnv(
        dt=1.0 / 1200.0,
        decimation=20,
        render=True,
        episode_length=0,          # no timeout — keep session open
        initial_puck_speed=0.25,
        seed=0,
    )
    obs = env.reset()
    env.start_recording()

    viewer = env.viewer
    assert viewer is not None, "Viewer failed to initialize — cannot teleop."

    # Seed the target EEF pose with the current EEF pose (hold-action).
    target_pose = env.compute_hold_action()  # [x, y, z, qx, qy, qz, qw]
    paused = False
    pause_latch = False
    reset_latch = False
    playback_latch = False
    frame = 0
    wall_start = time.perf_counter()

    print("=" * 60)
    print("Teleop: I/K=X  J/L=Y  U/O=Z  R=reset  P=save+playback  SPACE=pause  ESC=quit")
    print(f"Sampled puck goal (xy) = {env._sampled_goal_xy.tolist()}")
    print("=" * 60)

    while viewer.is_running():
        # ---- Read keys ----
        if viewer.is_key_down("escape"):
            break

        # Reset (edge-triggered)
        if viewer.is_key_down("r"):
            if not reset_latch:
                env.reset()
                env.start_recording()
                target_pose = env.compute_hold_action()
                frame = 0
                print(f"[reset] new goal xy = {env._sampled_goal_xy.tolist()}")
                reset_latch = True
        else:
            reset_latch = False

        # Playback (edge-triggered): stop recording, save demo, run forward
        # rollout with USD output, then reset and continue.
        if viewer.is_key_down("p"):
            if not playback_latch:
                playback_latch = True
                demo = env.stop_recording()
                if demo["torques"].shape[0] == 0:
                    print("[playback] no torques recorded — nothing to replay")
                else:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    npz_path = DEMO_DIR / f"demo_{stamp}.npz"
                    usd_path = DEMO_DIR / f"demo_{stamp}_forward.usd"
                    mp4_path = DEMO_DIR / f"demo_{stamp}_forward.mp4"
                    save_demo(demo, npz_path)
                    print(f"[playback] saved demo npz: {npz_path}")
                    try:
                        playback(demo, usd_path=usd_path,
                                 mp4_path=mp4_path, render_gl=False)
                    except Exception as e:
                        print(f"[playback] failed: {e}")
                env.reset()
                env.start_recording()
                target_pose = env.compute_hold_action()
                frame = 0
                print(f"[reset] new goal xy = {env._sampled_goal_xy.tolist()}")
        else:
            playback_latch = False

        # Pause (edge-triggered)
        if viewer.is_key_down("space"):
            if not pause_latch:
                paused = not paused
                print(f"[pause]={paused}")
                pause_latch = True
        else:
            pause_latch = False

        # Translation (level-triggered while held). IJKL+UO chosen to avoid
        # the viewer's built-in WASD+arrow camera bindings.
        dx = dy = dz = 0.0
        if viewer.is_key_down("i"):
            dx += TRANSLATION_STEP
        if viewer.is_key_down("k"):
            dx -= TRANSLATION_STEP
        if viewer.is_key_down("j"):
            dy += TRANSLATION_STEP
        if viewer.is_key_down("l"):
            dy -= TRANSLATION_STEP
        if viewer.is_key_down("u"):
            dz += TRANSLATION_STEP
        if viewer.is_key_down("o"):
            dz -= TRANSLATION_STEP

        target_pose[0] += dx
        target_pose[1] += dy
        target_pose[2] += dz

        # ---- Step physics ----
        if not paused:
            obs, reward, terminated, truncated, info = env.step(target_pose)
            frame += 1

            if terminated or truncated:
                reason = info.get("termination", "?")
                print(f"[episode end] reason={reason} reward={reward:.2f} "
                      f"frames={frame}")
                env.reset()
                target_pose = env.compute_hold_action()
                frame = 0
                print(f"[reset] new goal xy = {env._sampled_goal_xy.tolist()}")

        env.render()

        # HUD every ~30 frames
        if frame % 30 == 0 and not paused:
            puck_xy = obs["policy"][7:9]
            speed = env._puck_speed_xy()
            print(f"  frame={frame:4d} target=({target_pose[0]:+.3f},"
                  f"{target_pose[1]:+.3f},{target_pose[2]:+.3f}) "
                  f"puck_xy=({puck_xy[0]:+.3f},{puck_xy[1]:+.3f}) "
                  f"|v|={speed:.3f} m/s")

    env.close()
    elapsed = time.perf_counter() - wall_start
    print(f"Session ended after {elapsed:.1f} s wall time.")


if __name__ == "__main__":
    main()
