"""Evaluate a Mjlab-Piplus-Ball-MountBalance checkpoint by ball size and spawn mode.

Runs one 20 s episode per env (pushes on, observation noise off). An env counts
as a success if it never terminated early and had single-leg stance (sole flat
on the ball, other foot lifted) for at least `--hold-frac` of the last 3 s.
Envs are binned by their own randomized ball radius and by spawn mode.

    uv run python scripts/eval_mountbalance.py \
        --checkpoint logs/rsl_rl/piplus_ball_mountbalance/<run>/model_N.pt
"""

import argparse
from dataclasses import asdict

import torch

import mjlab_piplus  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from mjlab_piplus.tasks.ball_mount.piplus_ball_mount_env_cfg import (
    FREE_FOOT_LIFT,
    STAND_FOOT_Z,
    _ball_radius,
    single_leg_stance_lifted,
)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Mjlab-Piplus-Ball-MountBalance")
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--hold-frac", type=float, default=0.9)
args = parser.parse_args()

dev = "cuda:0"
cfg = load_env_cfg(args.task, play=True)
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = 20.0  # one finite episode per env
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=dev))
u = env.unwrapped
runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(load_rl_cfg(args.task)), device=dev)
runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=dev)
policy = runner.get_inference_policy(device=dev)

foot_cfg = SceneEntityCfg("robot", body_names=("r_ankle_roll_link", "l_ankle_roll_link"))
foot_cfg.resolve(u.scene)
ball_cfg = SceneEntityCfg("ball")

obs = env.get_observations()
mount = u.spawn_is_mount.clone()  # spawn mode of each env's first episode
radius = _ball_radius(u, ball_cfg, None).clone()
n_steps = int(20.0 / (u.step_dt)) - 1  # stop before the time-out reset
last = int(3.0 / u.step_dt)
failed = torch.zeros(u.num_envs, dtype=torch.bool, device=dev)
hold = torch.zeros(u.num_envs, device=dev)
first_stance = torch.full((u.num_envs,), float("nan"), device=dev)

for step in range(n_steps):
    with torch.no_grad():
        action = policy(obs)
    obs, _, dones, _ = env.step(action)
    failed |= dones.bool()
    stance = single_leg_stance_lifted(u, foot_cfg, ball_cfg, None, True, STAND_FOOT_Z + FREE_FOOT_LIFT) > 0
    new = stance & first_stance.isnan() & ~failed
    first_stance[new] = step * u.step_dt
    if step >= n_steps - last:
        hold += stance.float()

success = ~failed & (hold / last >= args.hold_frac)
edges = [0.0, 0.083, 0.097, 1.0]
names = ["size 1-2 (<0.083 m)", "size 3 (0.083-0.097)", "size 4-5 (>0.097)"]
print(f"\n{args.task}  {args.checkpoint}  ({u.num_envs} envs)")
print(f"{'spawn':<6} {'ball radius':<22} {'envs':>5} {'success':>8} {'survived':>9} {'t to stance':>12}")
for is_mount, label in ((True, "mount"), (False, "drop")):
    for lo, hi, name in zip(edges[:-1], edges[1:], names):
        m = (mount == is_mount) & (radius >= lo) & (radius < hi)
        if not m.any():
            continue
        t = first_stance[m]
        t_str = f"{t[~t.isnan()].mean().item():.1f}s" if (~t.isnan()).any() else "-"
        print(
            f"{label:<6} {name:<22} {int(m.sum()):>5} {success[m].float().mean().item():>8.0%}"
            f" {(~failed[m]).float().mean().item():>9.0%} {t_str:>12}"
        )
print(f"overall success {success.float().mean().item():.0%}, survived {(~failed).float().mean().item():.0%}")
