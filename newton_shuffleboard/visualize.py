"""Visualize the shuffleboard environment with forward/reverse cycling.

Kicks the puck at vx=5 m/s, runs forward for N frames, then reverses
for N frames, looping indefinitely. The puck should return to its
starting position each cycle.

Usage:
    conda run -n env_isaacsim python visualize.py
    conda run -n env_isaacsim python visualize.py --n_cycle 60
"""

import sys
import numpy as np
import warp as wp

sys.path.insert(0, "/home/krishna/Documents/sim2real/IsaacSimCustom")

import newton
import newton.examples
from newton_shuffleboard.env import NewtonShuffleboardEnv, PUCK_Z


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        # Fine substeps for elastic contact stability
        sim_dt = 1.0 / 1200.0
        decimation = 20  # 20 substeps × (1/1200)s = 1/60s per frame

        self.env = NewtonShuffleboardEnv(
            dt=sim_dt,
            decimation=decimation,
            episode_length=0,
        )

        self.viewer = viewer
        self.viewer.set_model(self.env.model)

        # Forward/reverse cycle
        self.n_cycle = getattr(args, "n_cycle", 60)
        self.cycle_step = 0
        self.forward = True

        self._reset_puck()

    def _reset_puck(self):
        """Place puck far from robot and give it a kick."""
        self.env.reset(randomize_puck=False)

        puck_jq_start = self.env._find_puck_joint_q_start()
        jq = self.env.state_0.joint_q.numpy()
        jq[puck_jq_start + 0] = 1.5
        jq[puck_jq_start + 1] = 0.0
        jq[puck_jq_start + 2] = PUCK_Z  # At EEF height (gravity-compensated)
        jq[puck_jq_start + 3:puck_jq_start + 7] = [0.0, 0.0, 0.0, 1.0]
        wp.copy(self.env.state_0.joint_q, wp.array(jq, dtype=wp.float32))

        # Kick puck: vx = 5 m/s
        # MuJoCo free joint DOF order: [vx, vy, vz, wx, wy, wz]
        jqd = self.env.state_0.joint_qd.numpy()
        jqd[9 + 0] = 5.0
        wp.copy(self.env.state_0.joint_qd, wp.array(jqd, dtype=wp.float32))
        newton.eval_fk(
            self.env.model,
            self.env.state_0.joint_q,
            self.env.state_0.joint_qd,
            self.env.state_0,
        )

        self.cycle_step = 0
        self.forward = True

    def step(self):
        if self.cycle_step >= self.n_cycle:
            direction = "FORWARD" if self.forward else "REVERSE"
            print(f"{direction} phase done ({self.n_cycle} frames). Switching.")
            self.forward = not self.forward
            self.cycle_step = 0

            if self.forward:
                # Negate kd back (reverse just ended, kd was restored by step_reverse)
                # Re-kick the puck for next forward phase
                pass  # kd already restored by step_reverse's finally block

        if self.forward:
            self.env._step_physics()
        else:
            # Single-frame reverse: negate kd, step with -dt, restore kd
            self.env._set_mujoco_kd(-self.env._original_biasprm_kd)
            self.env._step_physics(dt_override=-abs(self.env.dt))
            self.env._set_mujoco_kd(self.env._original_biasprm_kd)

        self.cycle_step += 1
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.env.state_0)
        self.viewer.log_contacts(self.env._contacts, self.env.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--n_cycle",
        type=int,
        default=60,
        help="Number of frames per forward/reverse phase",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
