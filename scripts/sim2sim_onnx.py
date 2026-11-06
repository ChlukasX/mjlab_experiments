"""Run the exported ONNX policy in the simulation with the robot-side observation code.

PARKED: `deploy/` is not tracked in git. This script works only on a
machine that has the local `deploy/bitbots/` directory, including `mount_obs.py`. It stays until a
deploy strategy exists.

Each step the observation is rebuilt by `deploy/bitbots/mount_obs.py` from raw signals
(encoder angles and velocities, IMU orientation and gyro, ball position in the IMU frame,
base height), exactly as the robot node does, and fed to onnxruntime. Checks that it
equals the training observation and that the targets equal the env's, then reports the
success like `eval_mountbalance.py`. Catches ordering, history and latch mistakes.

    uv run --with onnxruntime python scripts/sim2sim_onnx.py --onnx deploy/bitbots/mount_ball.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

import mjlab_piplus  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from mjlab_piplus.tasks.ball_mount.piplus_ball_mount_env_cfg import (
    COM_OVER_BALL, STAND_FOOT_Z, _ball_radius, ball_pos_base_obs, single_leg_stance_lifted,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "bitbots"))
from mount_obs import MountObs  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--onnx", default="deploy/bitbots/mount_ball.onnx")
parser.add_argument("--task", default="Mjlab-Piplus-Ball-MountMin-BlindOnBall")
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--seconds", type=float, default=20.0)
args = parser.parse_args()

p = yaml.safe_load(Path(args.onnx).with_suffix(".yaml").read_text())["mount_ball_node"]["ros__parameters"]
j = p["joints"]
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
cfg = load_env_cfg(args.task, play=True)
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = args.seconds
kin = cfg.observations["actor"].terms["ball_pos"].params.get("kinematic") is not None
if kin:  # the node's estimate has no error model; compare against the env's without it
    cfg.observations["actor"].terms["ball_pos"].params["kin_bias"] = 0.0
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=dev))
u = env.unwrapped
robot, n = u.scene["robot"], args.num_envs
sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
builders = [MountObs(j["walkready_state"], j["lower_limits"], j["upper_limits"], j["action_scales"][0],
                     p["on_ball"]["height"], p["on_ball"]["max_tilt"],
                     # true per-env radius here; on the robot the YAML's (the ball in use)
                     None, p["on_ball"].get("ankle_above_contact", 0.045)) for _ in range(n)]
radius = _ball_radius(u, SceneEntityCfg("ball"), None).cpu().numpy()
if kin:
    for b, r in zip(builders, radius):
        b.kin_ball_radius = float(r)
foot_cfg = SceneEntityCfg("robot", body_names=("r_ankle_roll_link", "l_ankle_roll_link"))
foot_cfg.resolve(u.scene)
term = u.action_manager.get_term("joint_pos")

def quat_to_gravity(q):  # (w, x, y, z) -> gravity in the body frame, as GravityHandler
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R.T @ np.array([0.0, 0.0, -1.0])

obs_env = env.get_observations()
steps, last = int(args.seconds / u.step_dt) - 1, int(3.0 / u.step_dt)
failed = np.zeros(n, bool); hold = np.zeros(n); obs_err = tgt_err = ball_err = 0.0
for step in range(steps):
    q = robot.data.joint_pos_biased.cpu().numpy(); dq = robot.data.joint_vel.cpu().numpy()
    quat = robot.data.root_link_quat_w.cpu().numpy(); gyro = robot.data.root_link_ang_vel_b.cpu().numpy()
    height = robot.data.root_link_pos_w[:, 2].cpu().numpy()
    ball = (ball_pos_base_obs(u, SceneEntityCfg("robot"), SceneEntityCfg("ball")) + u._ball_obs_bias).cpu().numpy()
    pose = robot.data.body_link_pose_w[:, foot_cfg.body_ids]  # ankle roll links, world
    rel = quat_apply_inverse(robot.data.root_link_quat_w[:, None].expand(-1, 2, -1), pose[..., :3] - robot.data.root_link_pos_w[:, None])
    nrm = quat_apply_inverse(robot.data.root_link_quat_w[:, None].expand(-1, 2, -1),
                             quat_apply(pose[..., 3:7], torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 2, 3)))
    ank, nrm = rel.cpu().numpy(), nrm.cpu().numpy()
    obs = np.stack([b.build(q[i], dq[i], quat_to_gravity(quat[i]), gyro[i], ball[i], height[i], ank[i], nrm[i]) for i, b in enumerate(builders)])
    # The env applied +-2 cm noise to the zeros after the latch only in training; play has no noise.
    diff = np.abs(obs - obs_env["actor"].cpu().numpy())
    obs_err = max(obs_err, float(diff.max())); ball_err = max(ball_err, float(diff[:, -15:].max()))  # ball_pos: last 3 x 5
    act = np.concatenate([sess.run(None, {inp: o[None]})[0] for o in obs])  # batch 1, as on the robot
    tg = np.stack([b.targets(a) for b, a in zip(builders, act)])
    obs_env, _, dones, _ = env.step(torch.as_tensor(act, device=dev))
    tgt_err = max(tgt_err, float(np.abs(tg - term._processed_actions.cpu().numpy()).max())) if step == 0 else tgt_err
    d = dones.cpu().numpy().astype(bool)
    failed |= d
    for i in np.flatnonzero(d):
        builders[i].reset()
    stance = single_leg_stance_lifted(u, foot_cfg, SceneEntityCfg("ball"), None, True, STAND_FOOT_Z, COM_OVER_BALL, 0.10)
    if step >= steps - last:
        hold += stance.cpu().numpy()
ok = ~failed & (hold / last >= 0.9)
print(f"max |robot-side obs - training obs| {obs_err:.2e} (ball_pos {ball_err:.2e}); first-step target diff {tgt_err:.2e}")
print(f"ONNX + robot-side observation: success {ok.mean():.0%}, survived {(~failed).mean():.0%} ({n} envs, {args.seconds:.0f} s)")
