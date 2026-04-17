"""
Shuffleboard Newton environment for VR teleoperation.

Drop-in replacement for the IsaacLab-based shuffleboard_env.py in Open-Teach.
Uses Newton/MuJoCo backend instead of IsaacLab/PhysX, enabling negative dt.

Scene state published via HTTP JSON on port 10512:
  {"eef":[7], "puck":[7], "ghost":[7], "joints":[9], "links":[[7]x8]}

All poses are [x, y, z, qx, qy, qz, qw] in Z-up frame.

Action received from operator (ZMQ, 13-D):
  [delta_pos(3), delta_rot(3), hand_pos(3), hand_quat_xyzw(4)]
We use hand_pos + hand_quat (absolute target) and convert xyzw -> wxyz
for the IK action [x, y, z, qw, qx, qy, qz].
"""

import json
import sys
import threading

import numpy as np
import zmq

sys.path.insert(0, "/home/krishna/Documents/sim2real/IsaacSimCustom")

from openteach.components import Component
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQKeypointPublisher, ZMQKeypointSubscriber


class ShuffleboardEnv(Component):
    """OpenTeach environment component wrapping Newton Shuffleboard."""

    def __init__(
        self,
        host,
        endeff_publish_port,
        endeffpossubscribeport,
        robotposepublishport,
        reset_subscribe_port=None,
    ):
        self._timer = FrequencyTimer(60)
        self.host = host
        self.name = "Shuffleboard_Sim"

        # -- ZMQ sockets -----------------------------------------------

        self.endeff_publisher = ZMQKeypointPublisher(
            host=host, port=endeff_publish_port
        )
        self.endeff_pos_subscriber = ZMQKeypointSubscriber(
            host=host, port=endeffpossubscribeport, topic="endeff_coords"
        )
        self.robot_pose_publisher = ZMQKeypointPublisher(
            host=host, port=robotposepublishport
        )

        # -- HTTP state server (Quest polls) ----------------------------

        from http.server import HTTPServer, BaseHTTPRequestHandler

        self._latest_state_json = "{}"
        self._state_lock = threading.Lock()
        env_ref = self

        class StateHandler(BaseHTTPRequestHandler):
            def do_GET(self_handler):
                with env_ref._state_lock:
                    data = env_ref._latest_state_json.encode()
                self_handler.send_response(200)
                self_handler.send_header("Content-Type", "application/json")
                self_handler.send_header("Content-Length", str(len(data)))
                self_handler.end_headers()
                self_handler.wfile.write(data)

            def log_message(self_handler, format, *args):
                pass

        self._http_server = HTTPServer((host, 10512), StateHandler)
        threading.Thread(target=self._http_server.serve_forever, daemon=True).start()
        print(f"[ShuffleboardEnv] HTTP state server listening on {host}:10512")

        # Reset subscriber
        if reset_subscribe_port is not None:
            self._reset_subscriber = ZMQKeypointSubscriber(
                host=host, port=reset_subscribe_port, topic="reset"
            )
        else:
            self._reset_subscriber = None

        # -- Initialize Newton env (no AppLauncher needed!) -------------

        from newton_shuffleboard.env import NewtonShuffleboardEnv

        self._env = NewtonShuffleboardEnv(
            dt=1.0 / 120.0,
            decimation=2,
            episode_length=0,  # infinite for teleop
        )

        obs = self._env.reset()
        self._hold_action = self._env.compute_hold_action()

        print(f"[ShuffleboardEnv] Ready. Newton/MuJoCo backend.")

    @property
    def timer(self):
        return self._timer

    # -- Main loop ------------------------------------------------------

    def stream(self):
        """Main loop: receive VR poses, step physics, publish state."""
        self.notify_component_start(f"{self.name} environment")
        print("[ShuffleboardEnv] Streaming started. Waiting for operator actions...")

        step_count = 0
        _last_hand_target = None
        _prev_target_quat = None

        while True:
            self.timer.start_loop()

            # 1. Receive VR action (drain queue, keep latest)
            _got_new = False
            try:
                while True:
                    action = self.endeff_pos_subscriber.recv_keypoints(
                        flags=zmq.NOBLOCK
                    )
                    if action is None:
                        break
                    action_np = np.array(action, dtype=np.float32)
                    if len(action_np) >= 13:
                        _last_hand_target = action_np[6:13]
                        _got_new = True
            except (zmq.Again, zmq.ZMQError):
                pass

            # 2. Build IK action [x, y, z, qw, qx, qy, qz]
            if _last_hand_target is not None:
                pos = _last_hand_target[:3]
                quat_xyzw = _last_hand_target[3:7].copy()

                # Hemisphere correction
                if _prev_target_quat is not None:
                    if np.dot(quat_xyzw, _prev_target_quat) < 0:
                        quat_xyzw = -quat_xyzw
                _prev_target_quat = quat_xyzw.copy()

                # Convert xyzw -> wxyz for action
                quat_wxyz = np.array(
                    [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
                )
                action_7d = np.concatenate([pos, quat_wxyz])
            else:
                action_7d = self._hold_action

            # 3. Step environment
            obs, rew, terminated, truncated, info = self._env.step(action_7d)

            if terminated or truncated:
                reason = info.get("termination", "unknown")
                print(f"[ShuffleboardEnv] Episode ended ({reason}) — auto-reset")
                obs = self._env.reset()
                self._hold_action = self._env.compute_hold_action()
                _last_hand_target = None
                _prev_target_quat = None

            # 4. Publish EEF state (for operator)
            position = self._env.get_eef_state()
            self.endeff_publisher.pub_keypoints(position, "endeff_coords")
            self.robot_pose_publisher.pub_keypoints(position, "robot_pose")

            # 5. Update HTTP state (Quest polls)
            scene_state = self._env.get_scene_state()
            with self._state_lock:
                self._latest_state_json = json.dumps(scene_state)

            # 6. Check manual reset signal
            if self._reset_subscriber is not None:
                try:
                    sig = self._reset_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
                    if sig is not None:
                        print(">>> RESETTING SHUFFLEBOARD ENVIRONMENT <<<")
                        obs = self._env.reset()
                        self._hold_action = self._env.compute_hold_action()
                        _last_hand_target = None
                        _prev_target_quat = None
                except (zmq.Again, zmq.ZMQError):
                    pass

            step_count += 1
            if step_count == 1:
                print(f"[ShuffleboardEnv] First step complete. Scene state:")
                print(f"  EEF:   {scene_state['eef']}")
                print(f"  Puck:  {scene_state['puck']}")
                print(f"  Ghost: {scene_state['ghost']}")

            self.timer.end_loop()
