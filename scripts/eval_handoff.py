"""Hand-off test: does the balance policy finish the mount from the mount policy's end state?

Runs a mount policy (bounded joint targets, as trained) on mount spawns. Once the
robot has been in the straddle for `--hold` seconds (foot flat on the ball, centre of
mass within `COM_OVER_BALL` of the ball centre) control switches to a balance policy
(trained on drops, unbounded targets) for the rest of the 20 s episode.

An env counts as a success if it never fell and had single-leg stance (CoM over the
ball, flat sole, free foot 10 cm clear) for `--hold-frac` of the last 3 s.

    uv run python scripts/eval_handoff.py \
        --mount-ckpt logs/rsl_rl/piplus_ball_mountbalance_mix100_clip/<run>/model_N.pt \
        --balance-ckpt logs/rsl_rl/piplus_ball_mountbalance_mix0/<run>/model_N.pt

`--mode mount_only` and `--mode balance_only` give the two baselines without a switch.
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
    COM_OVER_BALL,
    FREE_FOOT_LIFT,
    STAND_FOOT_Z,
    _ball_radius,
    single_leg_stance_lifted,
    support_on_ball,
)

MOUNT_TASK = "Mjlab-Piplus-Ball-MountBalance-Mix100-Clip"  # for the agent config only
BALANCE_TASK = "Mjlab-Piplus-Ball-MountBalance-Mix0"
ENV_TASK = "Mjlab-Piplus-Ball-MountBalance-Mix100-Clip"  # bounded targets; lifted per env after the switch

parser = argparse.ArgumentParser()
parser.add_argument("--mount-ckpt", required=True)
parser.add_argument("--balance-ckpt", required=True)
parser.add_argument("--mode", choices=["handoff", "mount_only", "balance_only"], default="handoff")
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--hold", type=float, default=1.0, help="seconds in the straddle before the switch")
parser.add_argument("--hold-frac", type=float, default=0.9)
args = parser.parse_args()

dev = "cuda:0"
cfg = load_env_cfg(ENV_TASK, play=True)
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = 20.0
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=dev))
u = env.unwrapped


def load_policy(task: str, ckpt: str):
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(load_rl_cfg(task)), device=dev)
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=dev)
    return runner.get_inference_policy(device=dev)


mount_policy = load_policy(MOUNT_TASK, args.mount_ckpt)
balance_policy = load_policy(BALANCE_TASK, args.balance_ckpt)

# The mount policy was trained with joint targets clipped to the joint limits (the env
# above), the balance policy without. After the switch the clip is lifted for that env
# only, inside the action term, so the "last action" observation stays the raw action
# for both (clipping the action before the step would change that observation).
term = u.action_manager.get_term("joint_pos")
assert term.cfg.clip is not None
UNBOUNDED = torch.tensor([-float("inf"), float("inf")], device=dev)

foot_cfg = SceneEntityCfg("robot", body_names=("r_ankle_roll_link", "l_ankle_roll_link"))
foot_cfg.resolve(u.scene)
ball_cfg = SceneEntityCfg("ball")
lifted_z = STAND_FOOT_Z + FREE_FOOT_LIFT

obs = env.get_observations()
radius = _ball_radius(u, ball_cfg, None).clone()
n_steps = int(20.0 / u.step_dt) - 1  # stop before the time-out reset
last = int(3.0 / u.step_dt)
hold_steps = int(args.hold / u.step_dt)
failed = torch.zeros(u.num_envs, dtype=torch.bool, device=dev)
handed = torch.zeros(u.num_envs, dtype=torch.bool, device=dev)
counter = torch.zeros(u.num_envs, device=dev)
handoff_t = torch.full((u.num_envs,), float("nan"), device=dev)
hold = torch.zeros(u.num_envs, device=dev)

for step in range(n_steps):
    with torch.no_grad():
        a_mount = mount_policy(obs)
        a_bal = balance_policy(obs)
    if args.mode == "mount_only":
        action = a_mount
    elif args.mode == "balance_only":
        action = a_bal
    else:
        action = torch.where(handed.unsqueeze(-1), a_bal, a_mount)
    obs, _, dones, _ = env.step(action)
    failed |= dones.bool()
    sup = support_on_ball(u, foot_cfg, ball_cfg, None, True, COM_OVER_BALL) > 0
    counter = torch.where(sup & ~failed, counter + 1, torch.zeros_like(counter))
    new = (counter >= hold_steps) & ~handed
    handoff_t[new] = step * u.step_dt
    handed |= new
    if args.mode == "handoff" and new.any():
        term._clip[new] = UNBOUNDED  # this env's balance policy runs unbounded from now on
    if step >= n_steps - last:
        stance = single_leg_stance_lifted(u, foot_cfg, ball_cfg, None, True, lifted_z, COM_OVER_BALL) > 0
        hold += stance.float()

success = ~failed & (hold / last >= args.hold_frac)
edges = [0.0, 0.083, 0.097, 1.0]
names = ["size 1-2 (<0.083 m)", "size 3 (0.083-0.097)", "size 4-5 (>0.097)"]
print(f"\nmode={args.mode}  mount={args.mount_ckpt}\n      balance={args.balance_ckpt}  ({u.num_envs} envs)")
print(f"{'ball radius':<22} {'envs':>5} {'reached straddle':>17} {'success':>8} {'survived':>9}")
for lo_r, hi_r, name in zip(edges[:-1], edges[1:], names):
    m = (radius >= lo_r) & (radius < hi_r)
    if m.any():
        print(
            f"{name:<22} {int(m.sum()):>5} {handed[m].float().mean().item():>17.0%}"
            f" {success[m].float().mean().item():>8.0%} {(~failed[m]).float().mean().item():>9.0%}"
        )
t = handoff_t[~handoff_t.isnan()]
print(
    f"overall: reached straddle {handed.float().mean().item():.0%}"
    + (f" (mean {t.mean().item():.1f} s)" if len(t) else "")
    + f", success {success.float().mean().item():.0%}, survived {(~failed).float().mean().item():.0%}"
)
