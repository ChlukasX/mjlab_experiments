"""Model-based search for a mount trajectory: CEM over open-loop joint-target splines.

All envs start from the same pose (mount spawn, ball size 5, no randomization or noise)
and replay a sampled trajectory open loop for `--horizon` seconds; the reward is the
task's own (progress potential with terminal potential 0, plus the stance reward).
The cross-entropy method refits the sampling distribution to the best `--elite-frac`
trajectories. It answers "does a mount exist" without relying on PPO exploration, and the
best trajectory (actions plus the joint, base and ball states along it) can seed a
reverse curriculum or a tracking reward.

    uv run python scripts/mount_cem.py --out logs/mount_cem/free.pt
    uv run python scripts/mount_cem.py --lock --out logs/mount_cem/locked.pt

`--lock` uses the task with the ball's rolling resistance raised (a ball that cannot roll).
Success = some sample holds single-leg stance (CoM over the ball, flat sole, free foot
10 cm clear) for at least `--hold` seconds before the end.
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F

import mjlab_piplus  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg

from mjlab_piplus.tasks.ball_mount.piplus_ball_mount_env_cfg import (
    COM_OVER_BALL,
    FREE_FOOT_LIFT,
    STAND_FOOT_Z,
    single_leg_stance_lifted,
)

parser = argparse.ArgumentParser()
parser.add_argument("--lock", action="store_true", help="ball with large rolling resistance")
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--iters", type=int, default=100)
parser.add_argument("--knots", type=int, default=16)
parser.add_argument("--horizon", type=float, default=4.0, help="seconds")
parser.add_argument("--elite-frac", type=float, default=0.05)
parser.add_argument("--init-std", type=float, default=1.0, help="action units (1 = 0.5 rad)")
parser.add_argument("--settle", type=float, default=1.5, help="seconds the last action is held after the motion: a dive at the end of the horizon must survive this or it is penalised")
parser.add_argument("--init-theta", default=None, help="saved best trajectory to start the search from")
parser.add_argument("--hold", type=float, default=1.0, help="seconds of stance that count as a mount")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", required=True)
args = parser.parse_args()

torch.manual_seed(args.seed)
dev = "cuda:0"
task = "Mjlab-Piplus-Ball-MountCurr-" + ("Locked" if args.lock else "Pot")
cfg = load_env_cfg(task, play=False)
cfg.scene.num_envs = args.num_envs
# Same start for every sample: keep only the reset and the spawn (no noise); the ball lock
# event stays when requested. Drop all randomization and the pushes.
keep = {"reset_scene_to_default", "spawn_mode", "ball_lock"}
cfg.events = {k: v for k, v in cfg.events.items() if k in keep}
cfg.events["spawn_mode"].params["ball_xy_noise"] = (0.0, 0.0)
cfg.episode_length_s = args.horizon + args.settle + 5.0  # no time-out inside the rollout
env = ManagerBasedRlEnv(cfg, device=dev)
T = int(args.horizon / env.step_dt)
T_total = T + int(args.settle / env.step_dt)
D = env.action_manager.total_action_dim
N = args.num_envs
robot, ball = env.scene["robot"], env.scene["ball"]
foot_cfg = SceneEntityCfg("robot", body_names=("r_ankle_roll_link", "l_ankle_roll_link"))
foot_cfg.resolve(env.scene)
ball_cfg = SceneEntityCfg("ball")
lifted_z = STAND_FOOT_Z + FREE_FOOT_LIFT
hold_steps = int(args.hold / env.step_dt)
k_elite = max(2, int(args.elite_frac * N))

mean = torch.zeros(args.knots, D, device=dev)
std = torch.full((args.knots, D), args.init_std, device=dev)
if args.init_theta:
    prev = torch.load(args.init_theta, map_location=dev, weights_only=False)["theta"]
    assert prev.shape == mean.shape, f"--knots must match the saved trajectory {tuple(prev.shape)}"
    mean = prev.to(dev)
best = {"score": -float("inf")}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
print(f"task {task}, {N} envs, horizon {args.horizon}s = {T} steps + {T_total - T} settling, {args.knots} knots, elite {k_elite}")

for it in range(args.iters):
    t0 = time.time()
    theta = mean + std * torch.randn(N, args.knots, D, device=dev)
    theta[0] = mean  # keep the current mean in the population
    actions = F.interpolate(theta.permute(0, 2, 1), size=T, mode="linear", align_corners=True).permute(0, 2, 1)
    actions = torch.cat([actions, actions[:, -1:].expand(-1, T_total - T, -1)], dim=1)  # hold the last action

    env.reset()
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    score = torch.zeros(N, device=dev)
    run = torch.zeros(N, device=dev)  # current consecutive stance steps
    longest = torch.zeros(N, device=dev)
    traj = torch.zeros(T_total, N, 20 + 7 + 3, device=dev)
    for t in range(T_total):
        _, rew, term, trunc, _ = env.step(actions[:, t])
        score += rew * alive
        stance = (single_leg_stance_lifted(env, foot_cfg, ball_cfg, None, True, lifted_z, COM_OVER_BALL) > 0) & alive
        run = torch.where(stance, run + 1, torch.zeros_like(run))
        longest = torch.maximum(longest, run)
        traj[t] = torch.cat([robot.data.joint_pos, robot.data.root_link_pose_w[:, :7], ball.data.root_link_pos_w], dim=-1)
        alive &= ~(term | trunc)

    elite = score.topk(k_elite).indices
    mean = 0.5 * mean + 0.5 * theta[elite].mean(0)
    std = 0.5 * std + 0.5 * theta[elite].std(0).clamp_min(0.05)

    i = int(score.argmax())
    success = longest >= hold_steps
    if score[i] > best["score"]:
        best = {
            "score": float(score[i]), "iter": it, "theta": theta[i].cpu(), "actions": actions[i].cpu(),
            "states": traj[:, i].cpu(), "longest_stance_s": float(longest[i] * env.step_dt),
            "args": vars(args), "task": task,
        }
        torch.save(best, args.out)
    print(
        f"it {it:3d} | best {score.max():7.2f} mean {score.mean():7.2f} elite {score[elite].mean():7.2f}"
        f" | alive at end {alive.float().mean():.0%} | stance>0 {(longest > 0).float().mean():.1%}"
        f" | longest stance {longest.max() * env.step_dt:.2f}s | mounts (>={args.hold}s) {int(success.sum())}"
        f" | std {std.mean():.2f} | {time.time() - t0:.0f}s",
        flush=True,
    )
    if success.any() and it >= 10:
        print(f"MOUNT FOUND: {int(success.sum())} samples held the stance >= {args.hold}s; best saved to {args.out}")
        break
print(f"best score {best['score']:.2f} at it {best['iter']}, longest stance {best['longest_stance_s']:.2f}s -> {args.out}")
