"""Record the mount policy's hand-off states: the spawn states for the balance policy.

Runs a mount policy on mount spawns (training randomization and pushes on, observation
noise off) and saves the simulation state at every moment the robot has just held the
single-leg stance (free sole `--clearance` off the floor, CoM over the ball, flat sole)
for `--hold` seconds. After a tap of the free foot the next such moment is recorded
again, so one episode gives several states. `reset_from_mount_states`
(tasks/ball_mount/events.py) spawns from the file.

    uv run python scripts/record_mount_states.py \
        --checkpoint logs/rsl_rl/piplus_ball_mountmin_clip/<run>/model_N.pt \
        --out logs/mount_states/<name>.pt
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab_piplus  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from mjlab_piplus.tasks.ball_mount.piplus_ball_mount_env_cfg import (
    COM_OVER_BALL,
    STAND_FOOT_Z,
    _ball_radius,
    single_leg_stance_lifted,
)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--task", default="Mjlab-Piplus-Ball-MountMin-Clip")
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--seconds", type=float, default=60.0)
parser.add_argument("--clearance", type=float, default=0.06)
parser.add_argument("--hold", type=float, default=0.2)
args = parser.parse_args()

dev = "cuda:0"
cfg = load_env_cfg(args.task)  # training config: randomization, pushes, resets
cfg.scene.num_envs = args.num_envs
cfg.observations["actor"].enable_corruption = False
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=dev))
u = env.unwrapped
runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(load_rl_cfg(args.task)), device=dev)
runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=dev)
policy = runner.get_inference_policy(device=dev)

foot_cfg = SceneEntityCfg("robot", body_names=("r_ankle_roll_link", "l_ankle_roll_link"))
foot_cfg.resolve(u.scene)
ball_cfg = SceneEntityCfg("ball")
robot, ball = u.scene["robot"], u.scene["ball"]
radius = _ball_radius(u, ball_cfg, None)
hold_steps = max(1, int(args.hold / u.step_dt))
counter = torch.zeros(u.num_envs, device=dev)
rows = []

obs = env.get_observations()
for _ in range(int(args.seconds / u.step_dt)):
    with torch.no_grad():
        action = policy(obs)
    obs, _, dones, _ = env.step(action)
    stance = single_leg_stance_lifted(
        u, foot_cfg, ball_cfg, None, True, STAND_FOOT_Z, COM_OVER_BALL, args.clearance
    ) > 0
    counter = torch.where(stance & ~dones.bool(), counter + 1, torch.zeros_like(counter))
    m = counter == hold_steps
    if m.any():
        origin = torch.zeros(int(m.sum()), 13, device=dev)
        origin[:, :3] = u.scene.env_origins[m]
        rows.append(
            torch.cat(
                [
                    radius[m, None],
                    torch.cat([robot.data.root_link_pose_w[m], robot.data.root_link_vel_w[m]], -1) - origin,
                    robot.data.joint_pos[m],
                    robot.data.joint_vel[m],
                    torch.cat([ball.data.root_link_pose_w[m], ball.data.root_link_vel_w[m]], -1) - origin,
                    u.action_manager.action[m],  # the mount policy's last (raw) action
                ],
                dim=-1,
            ).cpu()
        )

states = torch.cat(rows)
states = states[states[:, 0].argsort()]  # sorted by ball radius for the lookup at reset
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
# Columns: radius 1, robot root 13, joint pos nj, joint vel nj, ball root 13, last action na.
torch.save(
    {
        "states": states,
        "num_joints": robot.data.joint_pos.shape[1],
        "num_actions": u.action_manager.action.shape[1],
    },
    args.out,
)
print(f"saved {len(states)} states to {args.out}; radius {states[0, 0]:.3f} to {states[-1, 0]:.3f} m")
