"""Smoke test: record a short reverse rollout, replay forward, check reversal."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from newton_shuffleboard.env import NewtonShuffleboardReverseEnv
from newton_shuffleboard.playback import DEMO_DIR, load_demo, playback, save_demo


def test_playback_reversal(n_steps: int = 60):
    env = NewtonShuffleboardReverseEnv(
        dt=1.0 / 1200.0,
        decimation=20,
        render=False,
        episode_length=0,
        initial_puck_speed=0.25,
        seed=0,
    )
    env.reset()
    env.start_recording()

    hold = env.compute_hold_action()
    for _ in range(n_steps):
        env.step(hold)

    demo = env.stop_recording()
    assert demo["torques"].shape[0] == n_steps * env.decimation

    npz_path = DEMO_DIR / "smoke_demo.npz"
    save_demo(demo, npz_path)
    loaded = load_demo(npz_path)
    assert loaded["torques"].shape == demo["torques"].shape

    usd_path = DEMO_DIR / "smoke_demo_forward.usd"
    result = playback(loaded, usd_path=usd_path, render_gl=False)

    # Under hold-action the arm barely moves, so reversal should be ~exact.
    print(f"[test_playback] q_err={result['q_err']:.3e}  qd_err={result['qd_err']:.3e}")
    assert result["q_err"] < 1e-2, f"position reversal too large: {result['q_err']}"
    assert result["qd_err"] < 1e-2, f"velocity reversal too large: {result['qd_err']}"
    assert usd_path.exists(), f"USD not written at {usd_path}"


if __name__ == "__main__":
    print("=" * 60)
    print("test_playback_reversal")
    print("=" * 60)
    test_playback_reversal()
    print("PASSED")
