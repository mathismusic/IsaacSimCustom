# Newton Shuffleboard — Reverse-Time Environment

Reversed Newtonian dynamics version of the IsaacLab shuffleboard task,
running on Newton's MuJoCo solver backend with `dt = -h < 0`.

- Forward (reference) env: `IsaacLab/.../manager_based/manipulation/shuffleboard/`
  — PhysX, unchanged, kept as geometry/reward/termination reference.
- Frozen prior Newton attempt: `../newton_shuffleboard_forwards/` (snapshot
  of the earlier torque-replay design; not imported by this package).
- This package: `NewtonShuffleboardReverseEnv` — a gym-style env where the
  puck starts at the (forward) target with a small velocity, and the goal
  is to intercept it inside the (forward) start box.

## Dynamics spec (implemented)

- Integrator: `SolverMuJoCo(integrator="euler")`, stepped with `dt = -|h|`.
- Non-conservative effects zeroed: `dof_damping`, `dof_frictionloss`,
  joint friction, contact friction, URDF mimic-finger equality constraint
  (`mjw_data.eq_active = False`), PD `kd`.
- Contacts elastic: `geom_solref = [-1e4, 0]` (direct spring, zero damping).
- Gravity compensation on Franka links and on the puck (puck floats at
  EEF TCP height and slides in 2D).
- PD active with `kp = 650`, `kd = 0`. Position-only spring is conservative
  under reversed-dt; any `kd > 0` would pump energy.

## Initial condition (reset)

- Franka at `INIT_JOINT_Q = [0, -0.3, 0, -2.2, 0, 2.0, 0.7854, 0, 0]`,
  zero joint velocities.
- Sampled goal `g ~ Uniform(start_box)`; start box is centered at
  `(0.51, 0)` with half-extents `(0.15, 0.45)`.
- Puck at forward-target `(3.11, 0, 0.41)` with velocity pointing from
  `g` toward the target, magnitude `initial_puck_speed` (default 0.5 m/s).
  Sign is the forward-time velocity at arrival; reversed-dt Euler
  `x += dt·v` with `dt<0` then plays the trajectory back to `g`.

## Observations, reward, terminations

- Obs (10-D): `eef_pos(3) + eef_quat_xyzw(4) + puck_pos(3)`.
- Reward: sparse `+10` on success, else `0`.
- Success: puck inside start box AND `|v_xy| < 0.01 m/s`.
- Fail: puck off table (`x ∉ [0.21, 3.21]` or `|y| > 0.5`).
- Truncate: timeout at `episode_length` control steps (default 400).

## Running

All commands below assume the `env_isaacsim` conda env (has Newton + Warp
+ MuJoCo-Warp). Run from the *parent* of this directory (so
`newton_shuffleboard` is importable).

```bash
cd /home/krishna/Documents/sim2real/IsaacSimCustom

# Smoke tests (writes diagnostic logs to newton_shuffleboard/logs/*.jsonl)
/home/krishna/miniconda3/envs/env_isaacsim/bin/python \
    -m newton_shuffleboard.test_reverse_env
```

Expected: `test_obs_shape` and `test_free_flight` both print `PASSED`.
Free-flight test verifies the puck reaches the sampled start-box point
(~6e-4 m closest approach) and the Franka stays stationary under PD hold
(drift ~1.5e-6 rad) over ~800 reverse-time steps.

## Keyboard teleop + visualization

```bash
cd /home/krishna/Documents/sim2real/IsaacSimCustom
/home/krishna/miniconda3/envs/env_isaacsim/bin/python \
    -m newton_shuffleboard.teleop_keyboard
```

Controls:

| Key                     | Action                            |
| ----------------------- | --------------------------------- |
| Arrow Up / Down         | EEF +X / −X                       |
| Arrow Left / Right      | EEF +Y / −Y                       |
| W / S                   | EEF +Z / −Z                       |
| R                       | Reset episode (re-sample goal)    |
| SPACE                   | Pause / resume physics            |
| ESC                     | Quit                              |

Caveat: Newton's GL viewer also binds WASD+arrows to camera motion, so
these keys pan the camera as well. Mouse drag re-aims the view. If this
is too disruptive, switch to IJKL+UO (TODO).

## Using the env in code

```python
from newton_shuffleboard import NewtonShuffleboardReverseEnv
import numpy as np

env = NewtonShuffleboardReverseEnv(
    dt=1.0 / 120.0,
    decimation=2,
    initial_puck_speed=0.5,
    episode_length=400,
    seed=0,
)
obs = env.reset()

while True:
    # action = 7-D [x, y, z, qx, qy, qz, qw] target EEF pose (xyzw quat)
    action = env.compute_hold_action()          # e.g. passive hold
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Diagnostic logs

Tests write `logs/<name>.jsonl` — one JSON object per line. Structured
records include per-step puck position, puck free-joint velocity
(6-D linear+angular, world frame), Franka joint_q, distance-to-goal, and
termination events. Inspect with:

```bash
python -c "import json; [print(l.strip()) for l in
 open('newton_shuffleboard/logs/free_flight.jsonl')]"
```

## Open items

1. **Contact-interception test** — not written yet. Scripted action that
   puts the gripper in the puck's reverse path; verify the elastic contact
   reduces puck `|v_xy|` below the success threshold without energy
   blow-up. This is the core success case the policy must eventually
   solve, so we want a hand-rolled version working first.
2. **Policy integration** — env is a plain gym-style single-env. No
   RL training loop / vectorized env wrapper / rsl_rl config yet.
3. **PD gains** — currently uniform `ke=650, kd=0`. IsaacLab forward
   uses per-group gains (shoulder 100/20, forearm 25/10, hand 2000/100)
   but those require `kd > 0`, which pumps energy under reversed-dt
   integration. If tracking error becomes a problem under policy control,
   we'll need either (a) per-group kp retuning, or (b) a different
   controller (e.g. torque-space policy + explicit conservative gravcomp).
4. **Initial puck speed** — default `0.5 m/s`. Episode budget 400 steps
   × `decimation=2` × `1/120 s = 6.67 s` so the puck travels up to ~3.3 m
   within an episode (enough to cross the table). Tune if needed.
5. **Free-flight test assertion window** — test uses `episode_length=0`
   (no timeout) so the puck can pass the goal and exit the table; success
   termination fires correctly the moment the gripper (absent here) would
   stop it. If we later want the test to check "reaches goal and stops"
   rather than "reaches goal", we'll need a contact-stop test (open item 1).
6. **Quat convention** — action and obs both use xyzw (Warp/Newton
   native). If the eventual policy expects wxyz (IsaacLab convention),
   we'll need a wrapper.
7. **Body-qd spatial-vector layout** — the per-body `state.body_qd`
   slice returned zero for the puck's linear velocity in early testing
   even though `state.joint_qd` was correct. `_puck_speed_xy()` now
   reads `joint_qd` directly. If downstream code needs body-space
   velocities, we need to nail down the `[lin, ang]` vs `[ang, lin]`
   ordering convention for the MuJoCo solver specifically (the IK env
   in `newton/` uses `[wx,wy,wz,vx,vy,vz]` in comments, but our puck
   showed zeros in both halves — worth revisiting).
