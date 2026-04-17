"""Visualize Franka forward/reverse with raw torques (no PD controller).

Applies sinusoidal torques for N frames, then reverses them with -dt.
The robot should move and come back, looping indefinitely.

Usage:
    conda run -n env_isaacsim python visualize_torque_reverse.py
    conda run -n env_isaacsim python visualize_torque_reverse.py --n_cycle 60 --scale 2.0
"""

import sys
import numpy as np
import warp as wp

sys.path.insert(0, "/home/krishna/Documents/sim2real/IsaacSimCustom")

import newton
import newton.examples
from newton_shuffleboard.env import NewtonShuffleboardEnv


def disable_pd_controller(env):
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
    t = np.linspace(0, 2 * np.pi, n_frames)
    torques = np.zeros((n_frames, n_dofs), dtype=np.float32)
    torques[:, 0] = scale * np.sin(t)
    torques[:, 1] = scale * 0.6 * np.sin(1.5 * t)
    torques[:, 3] = scale * 0.4 * np.cos(t)
    return torques


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        sim_dt = 1.0 / 1200.0
        decimation = 20

        self.env = NewtonShuffleboardEnv(dt=sim_dt, decimation=decimation, episode_length=0)
        self.env.reset(randomize_puck=False)

        # Move puck far away
        puck_jq_start = self.env._find_puck_joint_q_start()
        jq_full = self.env.state_0.joint_q.numpy()
        jq_full[puck_jq_start + 0] = 10.0
        jq_full[puck_jq_start + 1] = 0.0
        jq_full[puck_jq_start + 2] = 10.0
        wp.copy(self.env.state_0.joint_q, wp.array(jq_full, dtype=wp.float32))
        newton.eval_fk(self.env.model, self.env.state_0.joint_q, self.env.state_0.joint_qd, self.env.state_0)

        disable_pd_controller(self.env)

        if self.env.control.joint_f is None:
            self.env.control.joint_f = wp.zeros(self.env.model.joint_dof_count, dtype=wp.float32)

        # Verify integrator is RK4 (id=1)
        mj = self.env.solver.mj_model
        if mj is not None:
            print(f"Integrator: {mj.opt.integrator} (expect 1=RK4)")

        self.viewer = viewer
        self.viewer.set_model(self.env.model)

        self.n_cycle = getattr(args, "n_cycle", 30)
        self.scale = getattr(args, "scale", 50.0)
        self.torques = generate_torques(self.n_cycle, scale=self.scale)

        self.cycle_step = 0
        self.forward = True
        self.cycle_count = 0

        self._jq_init = self.env.state_0.joint_q.numpy()[:9].copy()
        print(f"Torque scale: {self.scale} Nm, cycle length: {self.n_cycle} frames")

    def step(self):
        if self.cycle_step >= self.n_cycle:
            direction = "FORWARD" if self.forward else "REVERSE"
            jq = self.env.state_0.joint_q.numpy()[:7]
            err = np.linalg.norm(jq - self._jq_init[:7])
            print(f"{direction} done (cycle {self.cycle_count}). Arm err from init: {err:.6f} rad")
            self.forward = not self.forward
            self.cycle_step = 0
            if self.forward:
                self.cycle_count += 1

        if self.forward:
            torque_idx = self.cycle_step
            dt = self.env.dt
        else:
            torque_idx = self.n_cycle - 1 - self.cycle_step
            dt = -abs(self.env.dt)

        jf = np.zeros(self.env.model.joint_dof_count, dtype=np.float32)
        jf[:9] = self.torques[torque_idx]
        wp.copy(self.env.control.joint_f, wp.array(jf, dtype=wp.float32))

        for _ in range(self.env.decimation):
            self.env.state_0.clear_forces()
            self.env.solver.step(
                self.env.state_0, self.env.state_1, self.env.control,
                self.env._contacts, dt,
            )
            self.env.state_0, self.env.state_1 = self.env.state_1, self.env.state_0

        jq = self.env.state_0.joint_q.numpy()[:7]
        delta = np.linalg.norm(jq - self._jq_init[:7])
        phase = "FWD" if self.forward else "REV"
        print(f"  {phase}[{self.cycle_step:2d}] Δq={delta:.6f} q0={jq[0]:.4f} q1={jq[1]:.4f} q3={jq[3]:.4f}")

        self.cycle_step += 1
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.env.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--n_cycle", type=int, default=30, help="Frames per forward/reverse phase")
    parser.add_argument("--scale", type=float, default=50.0, help="Torque scale in Nm")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
