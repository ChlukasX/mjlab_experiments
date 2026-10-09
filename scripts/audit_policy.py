"""Audit a policy's actions for hardware safety.

Runs a checkpoint in its task's play config (observation noise off, pushes on) and reports
how hard it drives the actuators: size of the actions, how often the commanded joint
targets lie outside the joint limits, how often the PD torque saturates, and how fast the
actions change. A policy with mean |action| of several units commands targets far beyond
the joint range and would be bang-bang on the real robot.

    uv run python scripts/audit_policy.py --task Mjlab-Piplus-Ball-Small \
        --checkpoint logs/rsl_rl/piplus_ball_small/<run>/model_N.pt

The proposed pass thresholds below are defaults to argue about, not measured limits.
"""

import argparse
from dataclasses import asdict

import torch

import mjlab_piplus  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

THRESHOLDS = {
    "mean |action|": 1.0,  # 1 unit = 0.5 rad of target offset
    "targets outside joint limits": 0.02,
    "mean |action change| per step": 0.30,
}

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--seconds", type=float, default=10.0)
args = parser.parse_args()

dev = "cuda:0"
cfg = load_env_cfg(args.task, play=True)
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = 20.0
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=dev))
u = env.unwrapped
runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(load_rl_cfg(args.task)), device=dev)
runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=dev)
policy = runner.get_inference_policy(device=dev)

# Per-joint actuator gains, limits and joint ranges, in the action order.
term = u.action_manager.get_term("joint_pos")
robot = u.scene["robot"]
mjm = u.sim.mj_model
names = term.target_names
jids = [int(robot.indexing.joint_ids[robot.joint_names.index(n)]) for n in names]  # scene-level ids
act_of_joint = {int(mjm.actuator_trnid[i, 0]): i for i in range(mjm.nu)}
aids = [act_of_joint[j] for j in jids]
kp = torch.tensor([mjm.actuator_gainprm[a, 0] for a in aids], device=dev)
kv = torch.tensor([-mjm.actuator_biasprm[a, 2] for a in aids], device=dev)
fmax = torch.tensor([mjm.actuator_forcerange[a, 1] for a in aids], device=dev)
lo = torch.tensor([mjm.jnt_range[j, 0] for j in jids], device=dev)
hi = torch.tensor([mjm.jnt_range[j, 1] for j in jids], device=dev)
q_idx = [robot.joint_names.index(n) for n in names]
is_leg = torch.tensor([("hip" in n or "thigh" in n or "calf" in n or "ankle" in n) for n in names], device=dev)
scale, offset = term._scale, term._offset

obs = env.get_observations()
n_steps = int(args.seconds / u.step_dt)
acc = {k: 0.0 for k in ("abs", "outside", "sat_all", "sat_leg", "rate")}
abs_all, prev = [], None
alive_end = torch.ones(u.num_envs, dtype=torch.bool, device=dev)
count = 0
for step in range(n_steps):
    with torch.no_grad():
        a = policy(obs)
    target = a * scale + offset  # what the policy asks for, before any clipping
    obs, _, dones, _ = env.step(a)
    alive_end &= ~dones.bool()
    applied = term._processed_actions
    q = robot.data.joint_pos[:, q_idx]
    dq = robot.data.joint_vel[:, q_idx]
    torque = kp * (applied - q) - kv * dq
    acc["abs"] += a.abs().mean().item()
    abs_all.append(a.abs().flatten())
    acc["outside"] += ((target < lo) | (target > hi)).float().mean().item()
    sat = torque.abs() >= fmax
    acc["sat_all"] += sat.float().mean().item()
    acc["sat_leg"] += sat[:, is_leg].float().mean().item()
    if prev is not None:
        acc["rate"] += (a - prev).abs().mean().item()
    prev = a.clone()
    count += 1

allabs = torch.cat(abs_all)
res = {
    "mean |action|": acc["abs"] / count,
    "targets outside joint limits": acc["outside"] / count,
    "mean |action change| per step": acc["rate"] / max(count - 1, 1),
}
print(f"\n{args.task}  {args.checkpoint}  ({u.num_envs} envs, {args.seconds:.0f} s)")
print(f"  survived the full {args.seconds:.0f} s : {alive_end.float().mean().item():.0%}")
print(f"  |action| mean {res['mean |action|']:.2f}, 95th pct {allabs.quantile(0.95).item() if allabs.numel() < 16_000_000 else float('nan'):.2f}, max {allabs.max().item():.1f}")
print(f"  PD torque saturated: legs {acc['sat_leg'] / count:.0%}, all joints {acc['sat_all'] / count:.0%} of joint-steps")
for k, v in res.items():
    print(f"  {k:<32} {v:6.3f}   proposed limit {THRESHOLDS[k]:.2f}   {'ok' if v <= THRESHOLDS[k] else 'ABOVE'}")
