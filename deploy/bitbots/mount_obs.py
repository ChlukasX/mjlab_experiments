"""Observation and action handling of the ball-mount policy, without ROS.

Shared by the robot node (`mount_ball_node.py`) and the simulation check
(`scripts/sim2sim_onnx.py`), so the robot builds exactly the input the policy was
trained on (task `Mjlab-Piplus-Ball-MountMin-BlindOnBall`):

  per term, a 5-step history (oldest first; a fresh history is filled with the first
  value), terms concatenated in this order, 69 x 5 = 345 values:
    joint_pos           20  joint position minus the home pose (rad), action order
    joint_vel           20  joint velocity (rad/s)
    projected_gravity    3  unit gravity vector in the IMU (base) frame
    base_ang_vel         3  gyro, IMU frame (rad/s)
    actions             20  the policy's previous raw output (before scaling and clipping)
    ball_pos             3  ball centre in the IMU frame (m) until "on the ball"; after it zeros,
                            or with `kin_ball_radius` set the kinematic estimate (`kinematic_ball`)

Action: joint target = home pose + 0.5 * action, clipped to the joint ranges.
No observation scaling or clipping; the normalizer is inside the ONNX model.
"""

import numpy as np

TERM_DIMS = (("joint_pos", 20), ("joint_vel", 20), ("projected_gravity", 3), ("base_ang_vel", 3), ("actions", 20), ("ball_pos", 3))
HISTORY = 5


class MountObs:
    def __init__(self, home_pose, joint_lower, joint_upper, action_scale=0.5,
                 on_ball_height=0.42, on_ball_max_tilt=0.35, kin_ball_radius=None,
                 ankle_above_contact=0.045):
        self.home = np.asarray(home_pose, dtype=np.float32)
        self.lo = np.asarray(joint_lower, dtype=np.float32)
        self.hi = np.asarray(joint_upper, dtype=np.float32)
        self.scale = action_scale
        self.on_ball_height = on_ball_height
        self.cos_max_tilt = float(np.cos(on_ball_max_tilt))
        # Radius of the ball actually used (m); None: zeros on the ball (tasks before -KinBall).
        self.kin_ball_radius = kin_ball_radius
        self.ankle_above_contact = ankle_above_contact
        self.reset()

    def reset(self) -> None:
        """Call when the policy is (re)activated: empty histories, not on the ball."""
        self.hist = {name: None for name, _ in TERM_DIMS}
        self.on_ball = False
        self.last_action = np.zeros(len(self.home), dtype=np.float32)
        self.last_ball = None

    def update_on_ball(self, base_height: float, gravity_b) -> bool:
        """Latch "on the ball": base (IMU) height above the threshold while upright."""
        upright = -float(gravity_b[2]) > self.cos_max_tilt
        self.on_ball |= (base_height > self.on_ball_height) and upright
        return self.on_ball

    def kinematic_ball(self, ankle_pos_b, sole_normal_b, gravity_b) -> np.ndarray:
        """Ball centre in the IMU frame from the support foot (as `kinematic_ball_pos` in the
        task): below the higher ankle (along -gravity), along its sole normal, by
        ankle_above_contact + radius. `ankle_pos_b`, `sole_normal_b`: [2, 3], the ankle roll
        links' origins and z axes in the IMU frame (forward kinematics / tf)."""
        p = np.asarray(ankle_pos_b, dtype=np.float32)
        up = -np.asarray(gravity_b, dtype=np.float32)
        i = int(np.argmax(p @ up))
        return p[i] - (self.kin_ball_radius + self.ankle_above_contact) * np.asarray(sole_normal_b[i], dtype=np.float32)

    def build(self, joint_pos, joint_vel, gravity_b, gyro_b, ball_b, base_height,
              ankle_pos_b=None, sole_normal_b=None) -> np.ndarray:
        """One control step's observation [345]. `ball_b`: ball centre in the IMU frame, or
        None when no estimate is available (the last one is held; the ball is stationary).
        `ankle_pos_b`, `sole_normal_b`: needed on the ball with `kin_ball_radius` set."""
        self.update_on_ball(base_height, gravity_b)
        if ball_b is not None:
            self.last_ball = np.asarray(ball_b, dtype=np.float32)
        if self.on_ball and self.kin_ball_radius is not None:
            ball = self.kinematic_ball(ankle_pos_b, sole_normal_b, gravity_b)
        elif self.on_ball:
            ball = np.zeros(3, dtype=np.float32)
        elif self.last_ball is None:
            raise RuntimeError("no ball estimate yet: the policy needs the ball position before the mount")
        else:
            ball = self.last_ball
        values = {
            "joint_pos": np.asarray(joint_pos, dtype=np.float32) - self.home,
            "joint_vel": np.asarray(joint_vel, dtype=np.float32),
            "projected_gravity": np.asarray(gravity_b, dtype=np.float32),
            "base_ang_vel": np.asarray(gyro_b, dtype=np.float32),
            "actions": self.last_action,
            "ball_pos": ball,
        }
        parts = []
        for name, dim in TERM_DIMS:
            v = values[name]
            assert v.shape == (dim,), (name, v.shape)
            h = self.hist[name]
            self.hist[name] = np.repeat(v[None], HISTORY, axis=0) if h is None else np.concatenate([h[1:], v[None]])
            parts.append(self.hist[name].reshape(-1))
        return np.concatenate(parts).astype(np.float32)

    def targets(self, action) -> np.ndarray:
        """Joint position targets for the policy output; remembers it as the next "last action"."""
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        return np.clip(self.home + self.scale * self.last_action, self.lo, self.hi)
