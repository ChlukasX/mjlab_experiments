"""Evaluate a Mjlab-Piplus-Ball-MountBalance checkpoint by ball size and spawn mode.

Runs one 20 s episode per env (pushes on, observation noise off). An env counts
as a success if it never terminated early and had single-leg stance (sole flat
on the ball, the other sole's lowest corner `--free-foot-clearance` off the floor,
centre of mass over the ball) for at least `--hold-frac` of the last 3 s.
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
    COM_OVER_BALL,
    FREE_FOOT_LIFT,
    STAND_FOOT_Z,
    _ball_radius,
    _com_to_ball,
    _sole_clearance,
    _support_foot,
    single_leg_stance_lifted,
)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Mjlab-Piplus-Ball-MountBalance")
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--hold-frac", type=float, default=0.9)
parser.add_argument("--free-foot-clearance", type=float, default=FREE_FOOT_LIFT,
                    help="m the free foot must clear the floor in the success test")
parser.add_argument("--probe", action="store_true",
                    help="also print the posture and actuator saturation over the last 3 s")
parser.add_argument("--torsional", type=float, default=None,
                    help="feet get condim 6 with this torsional friction coefficient (m), a check of the spin")
parser.add_argument("--cone", choices=["pyramidal", "elliptic"], default=None, help="override the friction cone")
args = parser.parse_args()

dev = "cuda:0"
cfg = load_env_cfg(args.task, play=True)
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = 20.0  # one finite episode per env
if args.cone:
    cfg.sim.mujoco.cone = args.cone
if args.torsional is not None:
    from dataclasses import replace

    foot = r".*_ankle_roll_link_collision[0-9]$"
    robot_cfg = cfg.scene.entities["robot"]
    robot_cfg.collisions = tuple(
        replace(c, condim={foot: 6, ".*": c.condim}, friction={foot: (0.6, args.torsional, 0.0001), ".*": c.friction or (0.6,)})
        for c in robot_cfg.collisions
    )
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
# Spawn mode of each env's first episode (the plain Ball-Mount tasks only have mounts).
mount = getattr(u, "spawn_is_mount", torch.ones(u.num_envs, dtype=torch.bool, device=dev)).clone()
radius = _ball_radius(u, ball_cfg, None).clone()
n_steps = int(20.0 / (u.step_dt)) - 1  # stop before the time-out reset
last = int(3.0 / u.step_dt)
failed = torch.zeros(u.num_envs, dtype=torch.bool, device=dev)
hold = torch.zeros(u.num_envs, device=dev)
first_stance = torch.full((u.num_envs,), float("nan"), device=dev)

# Probe: sums over the last 3 s. Free foot = the one not on the ball (the lower one).
robot = u.scene["robot"]
mj = u.sim.mj_model
force_max = torch.tensor(
    [mj.actuator(f"robot/{n}").forcerange[1] for n in robot.actuator_names], device=dev
)
probe = {k: torch.zeros(u.num_envs, device=dev) for k in ("com", "base", "free_z", "sole", "on_floor", "on_ball", "yaw", "ball_yaw")}
saturated = torch.zeros(u.num_envs, len(robot.actuator_names), device=dev)

for step in range(n_steps):
    with torch.no_grad():
        action = policy(obs)
    obs, _, dones, _ = env.step(action)
    failed |= dones.bool()
    stance = single_leg_stance_lifted(u, foot_cfg, ball_cfg, None, True, STAND_FOOT_Z, COM_OVER_BALL, args.free_foot_clearance) > 0
    new = stance & first_stance.isnan() & ~failed
    first_stance[new] = step * u.step_dt
    if step >= n_steps - last:
        hold += stance.float()
        free_z = robot.data.body_link_pose_w[:, foot_cfg.body_ids, 2].min(dim=-1).values
        top_z = u.scene["ball"].data.root_link_pos_w[:, 2] + radius
        probe["com"] += _com_to_ball(u, foot_cfg, ball_cfg).norm(dim=-1)
        probe["base"] += robot.data.root_link_pos_w[:, 2] - top_z
        probe["free_z"] += free_z - STAND_FOOT_Z
        sole = _sole_clearance(u, foot_cfg).min(dim=-1).values  # lower sole's lowest corner
        probe["sole"] += sole
        probe["on_floor"] += (sole < 0.01).float()
        probe["on_ball"] += _support_foot(u, foot_cfg, ball_cfg, None, True).any(dim=-1).float()
        probe["yaw"] += robot.data.root_link_vel_w[:, 5]  # rad/s about the world vertical
        probe["ball_yaw"] += u.scene["ball"].data.root_link_vel_w[:, 5]
        saturated += (robot.data.actuator_force.abs() > 0.95 * force_max).float()

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

if args.probe:
    print("\nProbe, mean over the last 3 s, envs that survived:")
    print(f"{'spawn':<6} {'envs':>5} {'CoM-ball m':>11} {'base-top m':>11} {'free ankle m':>12} {'free sole m':>12} {'on floor':>9} {'on ball':>8}")
    for is_mount, label in ((True, "mount"), (False, "drop")):
        m = (mount == is_mount) & ~failed
        if not m.any():
            continue
        v = {k: (x[m] / last).mean().item() for k, x in probe.items()}
        print(
            f"{label:<6} {int(m.sum()):>5} {v['com']:>11.3f} {v['base']:>11.3f} {v['free_z']:>12.3f} {v['sole']:>12.3f}"
            f" {v['on_floor']:>9.0%} {v['on_ball']:>8.0%}"
        )
        yaw = probe["yaw"][m] / last
        print(f"  base yaw rate: mean {yaw.mean().item():+.2f} rad/s, median |.| {yaw.abs().median().item():.2f}, "
              f"envs turning clockwise (<0) {(yaw < 0).float().mean().item():.0%}; ball yaw rate mean "
              f"{(probe['ball_yaw'][m] / last).mean().item():+.2f} rad/s")
        h = hold[m] / last
        print(f"  stance held: mean {h.mean().item():.0%} of the last 3 s; envs above 50%: {(h > 0.5).float().mean().item():.0%}")
        sat = (saturated[m] / last).mean(dim=0)
        print("  share of steps at the force limit (>95%): " + ", ".join(
            f"{n} {x:.0%}" for n, x in zip(robot.actuator_names, sat.tolist()) if x > 0.02
        ) or "  none")
