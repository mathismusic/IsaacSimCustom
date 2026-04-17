"""Newton/MuJoCo shuffleboard REVERSE-TIME environment.

Reversed Newtonian dynamics: v_{t+h} = v_t - h*a(t+h), obtained by passing
dt = -|h| to SolverMuJoCo.step().

Puck starts at the (forward) target location with a small velocity directed
toward a uniformly sampled point inside the (forward) start box. Since the
table is frictionless and contact is elastic with zero friction, the puck
travels in a straight line toward that sampled point unless intercepted by
the Franka gripper. Success = puck enters the start box AND speed drops
below a threshold (i.e. the gripper has stopped it).

Matches scene geometry of the IsaacLab shuffleboard env. Forward reference
lives at isaaclab_tasks/manager_based/manipulation/shuffleboard/.
"""

import copy

import numpy as np
import warp as wp

import newton
import newton.ik as ik

# ---------------------------------------------------------------------------
# Geometry constants (env-local Z-up, Franka base at origin)
# ---------------------------------------------------------------------------

TABLE_LENGTH = 3.0
TABLE_WIDTH = 1.0
TABLE_THICKNESS = 0.04
TABLE_NEAR_EDGE_X = 0.21
TABLE_CENTER_X = TABLE_NEAR_EDGE_X + TABLE_LENGTH / 2.0  # 1.71
TABLE_CENTER_Z = 0.035
TABLE_SURFACE_Z = TABLE_CENTER_Z + TABLE_THICKNESS / 2.0  # 0.055

PUCK_RADIUS = 0.05
PUCK_HEIGHT = 0.03
PUCK_MASS = 0.1
PUCK_Z = 0.41  # puck floats at EEF TCP height (gravity-compensated)

# Forward start-box (puck success region in reverse env)
START_BOX_CENTER_X = 0.51
START_BOX_CENTER_Y = 0.0
PUCK_SPAWN_X_HALF = 0.15
PUCK_SPAWN_Y_HALF = 0.45

# Forward target (puck initial position in reverse env)
TARGET_X = TABLE_NEAR_EDGE_X + TABLE_LENGTH - 0.10  # 3.11
TARGET_Y = 0.0
TARGET_Z = PUCK_Z

# Success thresholds
SUCCESS_SPEED_THRESHOLD = 0.01  # m/s, puck considered "stopped"
# Puck is "in start box" iff |x - center_x| <= half_x AND |y - center_y| <= half_y

# Initial Franka joint config (push-ready pose, gripper closed)
INIT_JOINT_Q = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.7854, 0.0, 0.0]


def _quat_to_vec4(q):
    """Convert wp.quat to wp.vec4 (Newton IK convention, xyzw)."""
    return wp.vec4(q[0], q[1], q[2], q[3])


class NewtonShuffleboardReverseEnv:
    """Reversed-time shuffleboard environment on Newton/MuJoCo.

    Args:
        dt: physics timestep magnitude in seconds. Actual integration uses
            -|dt| (negative-dt for reversed Newtonian dynamics).
        decimation: physics substeps per control step.
        render: if True, create a Newton Viewer.
        episode_length: max control steps per episode (0 disables timeout).
        initial_puck_speed: magnitude of the initial puck velocity [m/s]
            pointing from the target back toward the sampled start-box point.
        seed: optional np.random seed for reset sampling.
    """

    def __init__(
        self,
        dt: float = 1.0 / 120.0,
        decimation: int = 4,
        render: bool = False,
        episode_length: int = 400,
        initial_puck_speed: float = 0.25,
        seed: int | None = None,
        forward_mode: bool = False,
    ):
        self.dt = abs(float(dt))
        self.forward_mode = bool(forward_mode)
        self._dt_signed = +self.dt if self.forward_mode else -self.dt
        self.decimation = int(decimation)
        # Torque recording (filled in _apply_reverse_pd when enabled).
        self._record_torques = False
        self._recorded_torques: list[np.ndarray] = []
        self._demo_initial_snapshot: dict | None = None
        self.episode_length = int(episode_length)
        self.initial_puck_speed = float(initial_puck_speed)
        self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._sampled_goal_xy = np.zeros(2, dtype=np.float32)

        # ---- Build scene --------------------------------------------------
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.0)

        franka_urdf = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
        builder.add_urdf(
            str(franka_urdf),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=True,
        )

        builder.joint_q[:9] = INIT_JOINT_Q
        builder.joint_target_pos[:9] = INIT_JOINT_Q
        # Disable MuJoCo's built-in PD actuators. We apply reverse-time PD
        # torque manually through control.joint_f each step (see _apply_pd).
        # MuJoCo rejects negative actuator gains as unstable, so the sign
        # flip required for reversed-dt stability has to live outside the
        # actuator path.
        builder.joint_target_ke[:9] = [0.0] * 9
        builder.joint_target_kd[:9] = [0.0] * 9

        # PD gains applied manually (stored in float32 for wp.array writes).
        # Under reversed-dt integration we apply tau = +kp*(q - target) + kd*qd
        # so the integrator's negative dt flips it back to stable forward PD.
        # Sign-flipped PD (tau = +kp*e + kd*qd) under dt<0 integration. Start
        # at low gains to stay within explicit-Euler stability band
        # (kp*dt^2/m_eff < 1 at dt=1/120). Tune up only if tracking is too
        # soft and divergence doesn't return.
        self._pd_kp = np.array([650.0] * 7 + [650.0, 650.0], dtype=np.float32)
        self._pd_kd = np.array([20.0] * 7 + [20.0, 20.0], dtype=np.float32)
        # Latest PD diagnostic snapshot (populated in _apply_reverse_pd;
        # used by tests/teleop to log without re-reading state arrays).
        self._pd_last = {
            "max_abs_err": 0.0, "argmax_err_dof": -1,
            "max_abs_qd": 0.0,  "argmax_qd_dof": -1,
            "max_abs_tau": 0.0, "argmax_tau_dof": -1,
            "nan": False,
        }
        builder.joint_armature[:9] = [0.3] * 4 + [0.11] * 3 + [0.15] * 2
        builder.joint_effort_limit[:8] = [80.0] * 8
        builder.joint_effort_limit[8:9] = [20.0] * 1
        builder.joint_friction[:9] = [0.0] * 9

        # Actuator-level gravcomp on arm DOFs 0..7
        gravcomp_attr = builder.custom_attributes["mujoco:jnt_actgravcomp"]
        if gravcomp_attr.values is None:
            gravcomp_attr.values = {}
        for dof_idx in range(8):
            gravcomp_attr.values[dof_idx] = True

        # Body-level gravcomp on arm links (2..13: link1..link7, link8, hand,
        # hand_tcp, leftfinger, rightfinger)
        gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp_body.values is None:
            gravcomp_body.values = {}
        for body_idx in range(2, 14):
            gravcomp_body.values[body_idx] = 1.0

        # Table (static, frictionless)
        frictionless = newton.ModelBuilder.ShapeConfig(
            mu=0.0, mu_torsional=0.0, mu_rolling=0.0,
        )
        builder.add_shape_box(
            body=-1,
            hx=TABLE_LENGTH / 2.0,
            hy=TABLE_WIDTH / 2.0,
            hz=TABLE_THICKNESS / 2.0,
            xform=wp.transform(
                wp.vec3(TABLE_CENTER_X, 0.0, TABLE_CENTER_Z),
                wp.quat_identity(),
            ),
            cfg=frictionless,
            label="table",
        )

        self._puck_body_local = builder.add_body(
            xform=wp.transform(
                wp.vec3(TARGET_X, TARGET_Y, PUCK_Z),
                wp.quat_identity(),
            ),
            mass=PUCK_MASS,
            label="puck",
        )
        builder.add_shape_cylinder(
            body=self._puck_body_local,
            radius=PUCK_RADIUS,
            half_height=PUCK_HEIGHT / 2.0,
            cfg=frictionless,
            label="puck_shape",
        )
        # Gravcomp the puck so it floats at EEF height.
        gravcomp_body.values[self._puck_body_local] = 1.0

        # Ghost target at start-box center (purely visual)
        no_collide = newton.ModelBuilder.ShapeConfig(collision_group=0, mu=0.0)
        self._target_body_local = builder.add_body(
            xform=wp.transform(
                wp.vec3(START_BOX_CENTER_X, START_BOX_CENTER_Y, PUCK_Z),
                wp.quat_identity(),
            ),
            is_kinematic=True,
            label="target",
        )
        builder.add_shape_cylinder(
            body=self._target_body_local,
            radius=PUCK_RADIUS,
            half_height=PUCK_HEIGHT / 2.0,
            cfg=no_collide,
            color=(0.1, 0.9, 0.1),
            label="target_shape",
        )

        # Second ghost at the puck's reverse-physics ORIGIN (forward target).
        # Purely visual — marks the spot where the puck spawns in reverse /
        # where it should arrive in forward playback.
        self._origin_body_local = builder.add_body(
            xform=wp.transform(
                wp.vec3(TARGET_X, TARGET_Y, PUCK_Z),
                wp.quat_identity(),
            ),
            is_kinematic=True,
            label="origin_ghost",
        )
        builder.add_shape_cylinder(
            body=self._origin_body_local,
            radius=PUCK_RADIUS,
            half_height=PUCK_HEIGHT / 2.0,
            cfg=no_collide,
            color=(0.9, 0.2, 0.2),
            label="origin_ghost_shape",
        )

        builder.add_ground_plane(height=-1.0)

        # Finalize
        self._model_ik = copy.deepcopy(builder).finalize()
        self.model = builder.finalize()

        self._puck_body_idx = next(
            i for i, lbl in enumerate(self.model.body_label) if "puck" in lbl
        )
        self._target_body_idx = next(
            i for i, lbl in enumerate(self.model.body_label) if "target" in lbl
        )
        self._ee_body_idx = next(
            i for i, lbl in enumerate(self.model.body_label)
            if lbl.endswith("/fr3_hand_tcp")
        )
        self._hand_body_idx = next(
            i for i, lbl in enumerate(self.model.body_label)
            if lbl.endswith("/fr3_hand")
        )
        self._link_body_indices = []
        for i in range(8):
            idx = next(
                j for j, lbl in enumerate(self.model.body_label)
                if lbl.endswith(f"/fr3_link{i}")
            )
            self._link_body_indices.append(idx)

        # State & control
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.control = self.model.control()
        wp.copy(self.control.joint_target_pos[:9], self.model.joint_q[:9])

        # ---- MuJoCo solver ------------------------------------------------
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="euler",
            cone="elliptic",
            njmax=500,
            nconmax=500,
            iterations=20,
            ls_iterations=100,
            impratio=1000.0,
        )
        self._contacts = newton.Contacts(
            self.solver.get_max_contact_count(), 0
        )

        # Elastic, frictionless contacts: negative solref = [-stiffness, 0]
        # gives direct spring (zero damping); zero friction everywhere.
        contact_stiffness = 1e4
        target_solref = wp.vec2(-contact_stiffness, 0.0)
        if self.solver.mjw_model is not None:
            self.solver.mjw_model.geom_solref.fill_(target_solref)
            self.solver.mjw_model.geom_friction.fill_(wp.vec3(0.0, 0.0, 0.0))
        elif self.solver.mj_model is not None:
            self.solver.mj_model.geom_solref[:] = [-contact_stiffness, 0.0]
            self.solver.mj_model.geom_friction[:] = [0.0, 0.0, 0.0]

        # Kill all non-conservative dof-level dissipation and the mimic
        # finger equality constraint (see example_ik_cube_stacking_rvrs.py).
        if self.solver.mjw_model is not None:
            self.solver.mjw_model.dof_damping.zero_()
            self.solver.mjw_model.dof_frictionloss.zero_()
        if self.solver.mjw_data is not None:
            self.solver.mjw_data.eq_active.fill_(False)

        # IK
        self._setup_ik()

        self.viewer = None
        if render:
            self._setup_viewer()

        self._init_joint_q = np.array(self.model.joint_q.numpy(), copy=True)

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    def _setup_ik(self):
        state_ik = self._model_ik.state()
        newton.eval_fk(
            self._model_ik, self._model_ik.joint_q,
            self._model_ik.joint_qd, state_ik,
        )
        body_q_np = state_ik.body_q.numpy()
        ee_tf = wp.transform(*body_q_np[self._ee_body_idx])
        self._pos_obj = ik.IKObjectivePosition(
            link_index=self._ee_body_idx,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array(
                [wp.transform_get_translation(ee_tf)], dtype=wp.vec3
            ),
        )
        self._rot_obj = ik.IKObjectiveRotation(
            link_index=self._ee_body_idx,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array(
                [_quat_to_vec4(wp.transform_get_rotation(ee_tf))], dtype=wp.vec4
            ),
        )
        self._joint_limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self._model_ik.joint_limit_lower,
            joint_limit_upper=self._model_ik.joint_limit_upper,
        )
        self._joint_q_ik = wp.array(
            self._model_ik.joint_q,
            shape=(1, self._model_ik.joint_coord_count),
        )
        self._ik_solver = ik.IKSolver(
            model=self._model_ik,
            n_problems=1,
            objectives=[self._pos_obj, self._rot_obj, self._joint_limit_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _solve_ik(self, target_pos, target_quat_xyzw):
        """Action uses xyzw quaternion (Warp/Newton native)."""
        self._pos_obj.set_target_position(
            0, wp.vec3(float(target_pos[0]), float(target_pos[1]), float(target_pos[2]))
        )
        x, y, z, w = target_quat_xyzw
        self._rot_obj.set_target_rotation(
            0, wp.vec4(float(x), float(y), float(z), float(w)),
        )
        self._ik_solver.step(self._joint_q_ik, self._joint_q_ik, iterations=24)
        return self._joint_q_ik.numpy()[0, :7]

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def _apply_reverse_pd(self, joint_target_9):
        """Compute reverse-time PD torque and write it into control.joint_f.

        tau = +kp*(q - target) + kd*qd for arm (0..6) and fingers (7..8).
        With dt=-h<0 in the integrator, this gives the same force update as
        forward-time PD with positive gains.
        """
        q = self.state_0.joint_q.numpy()
        qd = self.state_0.joint_qd.numpy()
        tau = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        e = joint_target_9 - q[:9]
        # Reverse-dt PD. Newton's law m*q.. = tau is invariant under t -> -t
        # (second derivative is even); velocity reverses sign. Evaluating
        # forward PD at the time-reversed state (q, -qd) gives the physical
        # torque to apply for forward-PD-like tracking under dt=-h:
        #   tau = -kp*(q - q*) - kd*(-qd) = -kp*e + kd*qd
        # Semi-implicit Euler with dt=-h then reproduces forward PD in s=-t.
        tau[:9] = self._pd_kp * e + self._pd_kd * qd[:9]
        abs_e, abs_qd, abs_tau = np.abs(e), np.abs(qd[:9]), np.abs(tau[:9])
        self._pd_last = {
            "max_abs_err": float(abs_e.max()), "argmax_err_dof": int(abs_e.argmax()),
            "max_abs_qd":  float(abs_qd.max()), "argmax_qd_dof":  int(abs_qd.argmax()),
            "max_abs_tau": float(abs_tau.max()), "argmax_tau_dof": int(abs_tau.argmax()),
            "nan": bool(np.isnan(tau).any() or np.isnan(q).any() or np.isnan(qd).any()),
        }
        wp.copy(self.control.joint_f, wp.array(tau, dtype=wp.float32))
        if self._record_torques:
            self._recorded_torques.append(tau.copy())

    def _step_physics(self, joint_target_9):
        for _ in range(self.decimation):
            self._apply_reverse_pd(joint_target_9)
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0, self.state_1, self.control,
                self._contacts, self._dt_signed,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.solver.update_contacts(self._contacts, self.state_0)

    # ------------------------------------------------------------------
    # Env interface
    # ------------------------------------------------------------------

    def reset(self):
        """Reset the reverse-time env. Returns observation dict.

        - Franka to INIT_JOINT_Q, zero velocity.
        - Puck to target pose (3.11, 0, PUCK_Z).
        - Puck velocity: initial_puck_speed toward a uniformly sampled
          point inside the start box.
        """
        self._step_count = 0

        # Sample goal point in start box
        gx = START_BOX_CENTER_X + 0.5 * self._rng.uniform(-PUCK_SPAWN_X_HALF, PUCK_SPAWN_X_HALF)
        gy = START_BOX_CENTER_Y + 0.5 * self._rng.uniform(-PUCK_SPAWN_Y_HALF, PUCK_SPAWN_Y_HALF)
        self._sampled_goal_xy = np.array([gx, gy], dtype=np.float32)

        # Build joint_q: Franka at init, puck at target, ghost at sampled goal
        joint_q_np = self._init_joint_q.copy()
        puck_jq_start = self._find_puck_joint_q_start()
        joint_q_np[puck_jq_start + 0] = TARGET_X
        joint_q_np[puck_jq_start + 1] = TARGET_Y
        joint_q_np[puck_jq_start + 2] = PUCK_Z
        joint_q_np[puck_jq_start + 3] = 0.0  # qx
        joint_q_np[puck_jq_start + 4] = 0.0  # qy
        joint_q_np[puck_jq_start + 5] = 0.0  # qz
        joint_q_np[puck_jq_start + 6] = 1.0  # qw

        tgt_jq_start, tgt_has_joint = self._find_target_joint_q_start()
        if tgt_has_joint:
            joint_q_np[tgt_jq_start + 0] = gx
            joint_q_np[tgt_jq_start + 1] = gy
            joint_q_np[tgt_jq_start + 2] = PUCK_Z
            joint_q_np[tgt_jq_start + 3] = 0.0
            joint_q_np[tgt_jq_start + 4] = 0.0
            joint_q_np[tgt_jq_start + 5] = 0.0
            joint_q_np[tgt_jq_start + 6] = 1.0

        # Build joint_qd: zero except puck linear vel.
        # Free-joint qd layout in Newton: [vx, vy, vz, wx, wy, wz].
        #
        # Sign convention: MuJoCo Euler with dt=-h<0 integrates as
        # x_{t+h} = x_t + dt*v = x_t - h*v. To move the puck FROM the target
        # TO the sampled start-box point (Δx = goal - target, negative in x),
        # we need v such that -h*v has the same sign as (goal - target), i.e.
        # v points in direction (target - goal). Physically: v is the forward-
        # time velocity the puck had when it arrived at the target; reversed-dt
        # integration then plays the trajectory back to the start-box.
        joint_qd_np = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        puck_qd_start = self._find_puck_joint_qd_start()
        direction = np.array([TARGET_X - gx, TARGET_Y - gy], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-9:
            direction = direction / norm
        joint_qd_np[puck_qd_start + 0] = self.initial_puck_speed * float(direction[0])
        joint_qd_np[puck_qd_start + 1] = self.initial_puck_speed * float(direction[1])

        # SolverMuJoCo._update_mjc_data reads state.joint_q / state.joint_qd
        # (not model.*), so write the ICs into state_0 directly. Seed state_1
        # identically so the ping-pong swap in _step_physics is consistent.
        joint_q_wp = wp.array(joint_q_np, dtype=wp.float32)
        joint_qd_wp = wp.array(joint_qd_np, dtype=wp.float32)
        for st in (self.state_0, self.state_1):
            if st.joint_q is None:
                st.joint_q = wp.empty_like(self.model.joint_q)
            if st.joint_qd is None:
                st.joint_qd = wp.empty_like(self.model.joint_qd)
            wp.copy(st.joint_q, joint_q_wp)
            wp.copy(st.joint_qd, joint_qd_wp)

        # Also update model.joint_q/qd (used by IK and any diagnostics).
        wp.copy(self.model.joint_q, joint_q_wp)
        wp.copy(self.model.joint_qd, joint_qd_wp)

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

        # Ensure ghost pose matches sampled goal even if it has no free joint
        # (kinematic body with no articulation — eval_fk wouldn't touch it).
        for st in (self.state_0, self.state_1):
            bq = st.body_q.numpy()
            bq[self._target_body_idx, 0] = gx
            bq[self._target_body_idx, 1] = gy
            bq[self._target_body_idx, 2] = PUCK_Z
            bq[self._target_body_idx, 3:7] = [0.0, 0.0, 0.0, 1.0]
            wp.copy(st.body_q, wp.array(bq, dtype=wp.float32))

        wp.copy(self.control.joint_target_pos[:9], self.model.joint_q[:9])

        ik_q = self._joint_q_ik.numpy()
        ik_q[0, :9] = joint_q_np[:9]
        wp.copy(self._joint_q_ik, wp.array(ik_q, dtype=wp.float32))

        return self._get_obs()

    def step(self, action):
        """One control step with reversed Newtonian dynamics.

        Args:
            action: np.ndarray shape (7,) = [x, y, z, qx, qy, qz, qw] — target
                EEF pose in env-local frame (xyzw quaternion).

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        action = np.asarray(action, dtype=np.float32).flatten()
        assert action.shape == (7,), f"Expected action (7,), got {action.shape}"

        arm_q = self._solve_ik(action[:3], action[3:7])
        joint_targets = np.zeros(9, dtype=np.float32)
        joint_targets[:7] = arm_q
        joint_targets[7:9] = 0.0  # gripper held closed

        self._step_physics(joint_targets)
        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated, truncated, info = self._check_terminations()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_scene_state(self):
        body_q = self.state_0.body_q.numpy()

        def pose7(body_idx):
            return body_q[body_idx, :7].tolist()

        return {
            "eef": pose7(self._hand_body_idx),
            "puck": pose7(self._puck_body_idx),
            "ghost": pose7(self._target_body_idx),
            "joints": self.model.joint_q.numpy()[:9].tolist(),
            "links": [pose7(i) for i in self._link_body_indices],
            "sampled_goal_xy": self._sampled_goal_xy.tolist(),
        }

    def compute_hold_action(self):
        """7D action [x,y,z,qx,qy,qz,qw] that holds the current EEF pose."""
        body_q = self.state_0.body_q.numpy()
        t = body_q[self._ee_body_idx]  # [px,py,pz, qx,qy,qz,qw]
        return np.array(
            [t[0], t[1], t[2], t[3], t[4], t[5], t[6]], dtype=np.float32
        )

    def render(self, sim_time: float | None = None):
        """Draw one frame to the viewer (no-op if render=False at ctor)."""
        if self.viewer is None:
            return
        t = sim_time if sim_time is not None else float(self._step_count) * self.dt * self.decimation
        self.viewer.begin_frame(t)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def close(self):
        if self.viewer is not None:
            self.viewer = None

    # ------------------------------------------------------------------
    # Demo recording / playback
    # ------------------------------------------------------------------

    def get_state_snapshot(self) -> dict:
        return {
            "joint_q": self.state_0.joint_q.numpy().copy(),
            "joint_qd": self.state_0.joint_qd.numpy().copy(),
        }

    def set_state_snapshot(self, snap: dict) -> None:
        jq = wp.array(snap["joint_q"].astype(np.float32), dtype=wp.float32)
        jqd = wp.array(snap["joint_qd"].astype(np.float32), dtype=wp.float32)
        for st in (self.state_0, self.state_1):
            if st.joint_q is None:
                st.joint_q = wp.empty_like(self.model.joint_q)
            if st.joint_qd is None:
                st.joint_qd = wp.empty_like(self.model.joint_qd)
            wp.copy(st.joint_q, jq)
            wp.copy(st.joint_qd, jqd)
        wp.copy(self.model.joint_q, jq)
        wp.copy(self.model.joint_qd, jqd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

    def start_recording(self) -> None:
        """Begin torque recording (call just after reset, before stepping)."""
        self._recorded_torques = []
        self._record_torques = True
        self._demo_initial_snapshot = self.get_state_snapshot()

    def stop_recording(self) -> dict:
        """Return the recorded demo: torques per substep + bracketing states."""
        self._record_torques = False
        n_dof = self.model.joint_dof_count
        torques = (
            np.stack(self._recorded_torques).astype(np.float32)
            if self._recorded_torques
            else np.zeros((0, n_dof), dtype=np.float32)
        )
        return {
            "torques": torques,
            "initial_state": self._demo_initial_snapshot,
            "final_state": self.get_state_snapshot(),
            "sampled_goal_xy": self._sampled_goal_xy.copy(),
            "dt": float(self.dt),
            "decimation": int(self.decimation),
        }

    def step_substep_with_torque(self, tau: np.ndarray) -> None:
        """Apply a pre-recorded torque and integrate one substep (forward_mode only)."""
        assert self.forward_mode, "step_substep_with_torque requires forward_mode=True"
        wp.copy(self.control.joint_f, wp.array(tau.astype(np.float32), dtype=wp.float32))
        self.state_0.clear_forces()
        self.solver.step(
            self.state_0, self.state_1, self.control,
            self._contacts, self._dt_signed,
        )
        self.state_0, self.state_1 = self.state_1, self.state_0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_puck_joint_q_start(self):
        q_start = self.model.joint_q_start.numpy()
        for j_idx, lbl in enumerate(self.model.joint_label):
            if "puck" in lbl:
                return int(q_start[j_idx])
        raise RuntimeError("puck free joint not found in model")

    def _find_puck_joint_qd_start(self):
        qd_start = self.model.joint_qd_start.numpy()
        for j_idx, lbl in enumerate(self.model.joint_label):
            if "puck" in lbl:
                return int(qd_start[j_idx])
        raise RuntimeError("puck free joint not found in model")

    def _find_target_joint_q_start(self):
        """Return (q_start, found) for the target ghost's free joint, or (-1, False)."""
        q_start = self.model.joint_q_start.numpy()
        for j_idx, lbl in enumerate(self.model.joint_label):
            if "target" in lbl:
                return int(q_start[j_idx]), True
        return -1, False

    def _get_obs(self):
        """10-D: eef_pos(3) + eef_quat_xyzw(4) + puck_pos(3)."""
        body_q = self.state_0.body_q.numpy()
        eef = body_q[self._ee_body_idx]  # [px,py,pz, qx,qy,qz,qw]
        eef_pos = eef[:3]
        eef_quat_xyzw = eef[3:7]
        puck_pos = body_q[self._puck_body_idx, :3]
        return {
            "policy": np.concatenate([eef_pos, eef_quat_xyzw, puck_pos]).astype(np.float32),
        }

    def _puck_in_start_box(self, puck_xy):
        return (
            abs(puck_xy[0] - START_BOX_CENTER_X) <= PUCK_SPAWN_X_HALF
            and abs(puck_xy[1] - START_BOX_CENTER_Y) <= PUCK_SPAWN_Y_HALF
        )

    def _puck_speed_xy(self):
        # Read directly from the puck's free-joint qd (world-frame linear
        # components), which is the integrator's own state. Avoids ambiguity
        # in body_qd spatial-vector layout (origin vs COM, linear vs angular
        # ordering) between solvers.
        jqd = self.state_0.joint_qd.numpy()
        qd0 = self._find_puck_joint_qd_start()
        vx, vy = float(jqd[qd0 + 0]), float(jqd[qd0 + 1])
        return float(np.sqrt(vx * vx + vy * vy))

    def _compute_reward(self):
        body_q = self.state_0.body_q.numpy()
        puck_xy = body_q[self._puck_body_idx, :2]
        speed = self._puck_speed_xy()
        if self._puck_in_start_box(puck_xy) and speed < SUCCESS_SPEED_THRESHOLD:
            return 10.0
        return 0.0

    def _check_terminations(self):
        body_q = self.state_0.body_q.numpy()
        puck_pos = body_q[self._puck_body_idx, :3]
        puck_xy = puck_pos[:2]
        speed = self._puck_speed_xy()
        info = {}

        if self._puck_in_start_box(puck_xy) and speed < SUCCESS_SPEED_THRESHOLD:
            info["termination"] = "success"
            return True, False, info

        if (
            puck_pos[0] < TABLE_NEAR_EDGE_X
            or puck_pos[0] > TABLE_NEAR_EDGE_X + TABLE_LENGTH
            or puck_pos[1] < -TABLE_WIDTH / 2.0
            or puck_pos[1] > TABLE_WIDTH / 2.0
        ):
            info["termination"] = "puck_off_table"
            return True, False, info

        if self.episode_length > 0 and self._step_count >= self.episode_length:
            info["termination"] = "timeout"
            return False, True, info

        return False, False, info

    def _setup_viewer(self):
        try:
            self.viewer = newton.viewer.ViewerGL()
            self.viewer.set_model(self.model)
        except Exception as e:
            print(f"[NewtonShuffleboardReverseEnv] Could not create viewer: {e}")
            self.viewer = None


# Backwards-compatible alias
NewtonShuffleboardEnv = NewtonShuffleboardReverseEnv
