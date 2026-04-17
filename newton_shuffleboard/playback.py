"""Forward-time playback of a reverse-physics demo.

Builds a forward twin of NewtonShuffleboardReverseEnv, loads the recorded
torques + final state, and replays them in reverse order under dt > 0 to
visualize that the demo reverses cleanly.

Time-reversal plumbing (semi-implicit Euler, integrator="euler"):
  - Start state for forward rollout = reverse run's FINAL state, verbatim.
    No velocity flip: the integrator state (q, qd) at the end of the reverse
    run is also the correct starting state for forward replay. (Continuous-
    time intuition says qd should flip; the discrete semi-implicit update
    rule retraces without it, because Δq per step uses the post-update qd
    in both directions.)
  - Torques replayed in reverse order of recording (last substep first).
  - No sign flip on the torques: our reverse-PD law tau_rev = -kp*e + kd*qd
    is already the correct forward-time torque at the current state.

Outputs a time-sampled USD under newton_shuffleboard/demos/ which can be
opened in Omniverse / usdview.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import warp as wp
import newton

from newton_shuffleboard.env import NewtonShuffleboardReverseEnv


DEMO_DIR = Path(__file__).parent / "demos"
DEMO_DIR.mkdir(exist_ok=True)


def save_demo(demo: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        torques=demo["torques"],
        initial_joint_q=demo["initial_state"]["joint_q"],
        initial_joint_qd=demo["initial_state"]["joint_qd"],
        final_joint_q=demo["final_state"]["joint_q"],
        final_joint_qd=demo["final_state"]["joint_qd"],
        sampled_goal_xy=demo["sampled_goal_xy"],
        dt=demo["dt"],
        decimation=demo["decimation"],
    )
    return path


def load_demo(path: str | Path) -> dict:
    data = np.load(path)
    return {
        "torques": data["torques"],
        "initial_state": {
            "joint_q": data["initial_joint_q"],
            "joint_qd": data["initial_joint_qd"],
        },
        "final_state": {
            "joint_q": data["final_joint_q"],
            "joint_qd": data["final_joint_qd"],
        },
        "sampled_goal_xy": data["sampled_goal_xy"],
        "dt": float(data["dt"]),
        "decimation": int(data["decimation"]),
    }


# Close-up side-view camera framing the puck path + Franka.
DEFAULT_CAM_POS = wp.vec3(1.3, -2.2, 1.35)
DEFAULT_CAM_PITCH = -20.0
DEFAULT_CAM_YAW = 90.0


def playback(
    demo: dict,
    usd_path: str | Path | None = None,
    mp4_path: str | Path | None = None,
    render_gl: bool = False,
    refresh_contacts_every_substep: bool = True,
    log_jsonl: bool = True,
    cam_pos: wp.vec3 | None = None,
    cam_pitch: float | None = None,
    cam_yaw: float | None = None,
) -> dict:
    """Replay a reverse-physics demo forward-in-time.

    Args:
        demo: dict as returned by NewtonShuffleboardReverseEnv.stop_recording().
        usd_path: if not None, write a time-sampled USD to this path.
        render_gl: if True, also open a live GL viewer.

    Returns:
        Dict with final state + reversal-error diagnostics.
    """
    dt = float(demo["dt"])
    decimation = int(demo["decimation"])
    torques = np.asarray(demo["torques"])
    n_substeps = torques.shape[0]
    if n_substeps == 0:
        raise ValueError("demo has no recorded substeps")

    env = NewtonShuffleboardReverseEnv(
        dt=dt,
        decimation=decimation,
        render=False,
        episode_length=0,
        initial_puck_speed=0.0,
        seed=0,
        forward_mode=True,
    )

    # Seed forward rollout from reverse's FINAL state, verbatim.
    env.set_state_snapshot({
        "joint_q": demo["final_state"]["joint_q"].copy(),
        "joint_qd": demo["final_state"]["joint_qd"].copy(),
    })

    cam_pos = cam_pos if cam_pos is not None else DEFAULT_CAM_POS
    cam_pitch = cam_pitch if cam_pitch is not None else DEFAULT_CAM_PITCH
    cam_yaw = cam_yaw if cam_yaw is not None else DEFAULT_CAM_YAW

    usd_viewer = None
    if usd_path is not None:
        usd_path = Path(usd_path)
        usd_path.parent.mkdir(parents=True, exist_ok=True)
        frame_dt = dt * decimation
        fps = max(1, int(round(1.0 / frame_dt)))
        usd_viewer = newton.viewer.ViewerUSD(
            output_path=str(usd_path),
            fps=fps,
            up_axis="Z",
        )
        usd_viewer.set_model(env.model)
        try:
            usd_viewer.set_camera(cam_pos, cam_pitch, cam_yaw)
        except Exception:
            pass  # USD viewer may not honor runtime camera

    gl_viewer = None
    if render_gl:
        try:
            gl_viewer = newton.viewer.ViewerGL()
            gl_viewer.set_model(env.model)
            gl_viewer.set_camera(cam_pos, cam_pitch, cam_yaw)
        except Exception as e:
            print(f"[playback] GL viewer unavailable: {e}")

    # MP4 capture: headless GL viewer + imageio-ffmpeg writer.
    mp4_viewer = None
    mp4_writer = None
    if mp4_path is not None:
        import imageio
        mp4_path = Path(mp4_path)
        mp4_path.parent.mkdir(parents=True, exist_ok=True)
        frame_dt = dt * decimation
        fps = max(1, int(round(1.0 / frame_dt)))
        try:
            mp4_viewer = newton.viewer.ViewerGL(headless=True, width=1280, height=720)
            mp4_viewer.set_model(env.model)
            mp4_viewer.set_camera(cam_pos, cam_pitch, cam_yaw)
            mp4_writer = imageio.get_writer(
                str(mp4_path), fps=fps, codec="libx264",
                quality=8, macro_block_size=1,
            )
        except Exception as e:
            print(f"[playback] MP4 setup failed ({e}); skipping MP4")
            if mp4_writer is not None:
                mp4_writer.close()
            mp4_viewer = None
            mp4_writer = None

    print(
        f"[playback] replaying {n_substeps} substeps "
        f"(~{n_substeps // decimation} control steps) under dt=+{dt:.5f}"
    )

    sim_time = 0.0
    frame_dt = dt * decimation
    t_start = time.perf_counter()

    puck_idx = env._puck_body_idx
    puck_jq_start = env._find_puck_joint_q_start()
    puck_jqd_start = env._find_puck_joint_qd_start()

    log_path = None
    log_file = None
    if log_jsonl and (usd_path is not None or mp4_path is not None):
        stem = (Path(usd_path).stem if usd_path is not None
                else Path(mp4_path).stem)
        log_path = DEMO_DIR / f"{stem}.jsonl"
        log_file = log_path.open("w")

    max_puck_omega = 0.0

    for sub_idx in range(n_substeps):
        # Refresh contacts. Per-substep refresh is the most faithful choice
        # (reverse env refreshes once per control step, but under dt<0 vs
        # dt>0 the contact pairs at each substep can differ; staler contacts
        # leak into puck angular velocity -> visible tumbling).
        if refresh_contacts_every_substep or sub_idx % decimation == 0:
            env.solver.update_contacts(env._contacts, env.state_0)

        tau = torques[n_substeps - 1 - sub_idx]
        env.step_substep_with_torque(tau)

        # Track puck angular speed across the whole rollout.
        jqd = env.state_0.joint_qd.numpy()
        omega = float(np.linalg.norm(jqd[puck_jqd_start + 3: puck_jqd_start + 6]))
        max_puck_omega = max(max_puck_omega, omega)

        # Render one frame per control step (decimation boundary).
        if (sub_idx + 1) % decimation == 0:
            sim_time += frame_dt
            if log_file is not None:
                bq = env.state_0.body_q.numpy()
                jqd_all = env.state_0.joint_qd.numpy()
                rec = {
                    "sub_idx": sub_idx,
                    "sim_time": sim_time,
                    "puck_pos": bq[puck_idx, :3].tolist(),
                    "puck_quat_xyzw": bq[puck_idx, 3:7].tolist(),
                    "puck_lin_vel": jqd_all[puck_jqd_start:puck_jqd_start + 3].tolist(),
                    "puck_ang_vel": jqd_all[puck_jqd_start + 3:puck_jqd_start + 6].tolist(),
                }
                log_file.write(json.dumps(rec) + "\n")
            if usd_viewer is not None:
                usd_viewer.begin_frame(sim_time)
                usd_viewer.log_state(env.state_0)
                usd_viewer.end_frame()
            if gl_viewer is not None and gl_viewer.is_running():
                gl_viewer.begin_frame(sim_time)
                gl_viewer.log_state(env.state_0)
                gl_viewer.end_frame()
            if mp4_viewer is not None:
                mp4_viewer.begin_frame(sim_time)
                mp4_viewer.log_state(env.state_0)
                mp4_viewer.end_frame()
                frame_wp = mp4_viewer.get_frame()
                # Origin top-left per get_frame docs; ffmpeg expects top-left.
                mp4_writer.append_data(frame_wp.numpy())

    env.solver.update_contacts(env._contacts, env.state_0)

    if usd_viewer is not None:
        usd_viewer.close()
        print(f"[playback] USD written: {usd_path}")
    if gl_viewer is not None:
        gl_viewer.close()
    if mp4_writer is not None:
        mp4_writer.close()
        print(f"[playback] MP4 written: {mp4_path}")
    if mp4_viewer is not None:
        mp4_viewer.close()

    # Reversal error: forward-final should match reverse-initial (with qd flip).
    final_q = env.state_0.joint_q.numpy()
    final_qd = env.state_0.joint_qd.numpy()
    init_q = demo["initial_state"]["joint_q"]
    init_qd = demo["initial_state"]["joint_qd"]
    q_err = float(np.max(np.abs(final_q - init_q)))
    qd_err = float(np.max(np.abs(final_qd - init_qd)))

    # Puck-specific reversal diagnostics (catches tumbling that averages out
    # in joint-wise max). Reverse initial puck had zero angular velocity;
    # any nonzero forward-final puck_omega is integrated contact asymmetry.
    puck_final_omega = float(np.linalg.norm(
        final_qd[puck_jqd_start + 3: puck_jqd_start + 6]))
    puck_init_omega = float(np.linalg.norm(
        init_qd[puck_jqd_start + 3: puck_jqd_start + 6]))
    puck_quat_err = float(np.max(np.abs(
        final_q[puck_jq_start + 3: puck_jq_start + 7]
        - init_q[puck_jq_start + 3: puck_jq_start + 7])))

    elapsed = time.perf_counter() - t_start
    print(
        f"[playback] done in {elapsed:.2f}s  "
        f"max|Δq|={q_err:.3e}  max|Δqd|={qd_err:.3e}"
    )
    print(
        f"[playback] puck tumbling: max|ω|_during_rollout={max_puck_omega:.3e}  "
        f"|ω|_final={puck_final_omega:.3e} (init={puck_init_omega:.3e})  "
        f"max|Δquat|={puck_quat_err:.3e}"
    )
    if log_path is not None:
        print(f"[playback] per-step puck log: {log_path}")
        log_file.close()

    return {
        "q_err": q_err, "qd_err": qd_err,
        "puck_final_omega": puck_final_omega,
        "puck_max_omega": max_puck_omega,
        "puck_quat_err": puck_quat_err,
        "final_q": final_q, "final_qd": final_qd,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m newton_shuffleboard.playback <demo.npz> [--gl] [--no-mp4] [--no-usd]")
        sys.exit(1)

    demo_path = Path(sys.argv[1])
    use_gl = "--gl" in sys.argv
    want_mp4 = "--no-mp4" not in sys.argv
    want_usd = "--no-usd" not in sys.argv
    demo = load_demo(demo_path)
    usd_out = DEMO_DIR / (demo_path.stem + "_forward.usd") if want_usd else None
    mp4_out = DEMO_DIR / (demo_path.stem + "_forward.mp4") if want_mp4 else None
    playback(demo, usd_path=usd_out, mp4_path=mp4_out, render_gl=use_gl)
