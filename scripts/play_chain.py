"""Play the ball chain in the viewer: mount policy, switch, balance policy.

The robot starts standing beside a randomized ball. The mount policy runs until the robot
has held the single-leg stance (free sole `--switch-clearance` off the floor, CoM over the
ball, flat sole) for `--hold` seconds; then the balance policy takes over for that env
until it resets. The switch is printed for env 0.

    uv run python scripts/play_chain.py                 # best checkpoints so far, viser
    uv run python scripts/play_chain.py --num-envs 16 --mount-ckpt <model.pt> --balance-ckpt <model.pt>

`--steps N` runs N control steps without a viewer and prints how many envs switched and fell.
"""

import argparse
import os
from dataclasses import asdict
from glob import glob

import torch

import mjlab_piplus  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from mjlab_piplus.tasks.ball_mount.piplus_ball_mount_env_cfg import (
    COM_OVER_BALL,
    STAND_FOOT_Z,
    single_leg_stance_lifted,
)

TASK = "Mjlab-Piplus-Ball-Chain"
MOUNT_TASK = "Mjlab-Piplus-Ball-MountMin-Clip"  # for the agent configs only
BALANCE_TASK = "Mjlab-Piplus-Ball-BalanceFromMount"
MOUNT_CKPT = "logs/rsl_rl/piplus_ball_mountmin_clip/*_mm3-c10-s16/model_11997.pt"
BALANCE_CKPT = "logs/rsl_rl/piplus_ball_balancefrommount/*_bfm1-warm/model_12998.pt"

parser = argparse.ArgumentParser()
parser.add_argument("--mount-ckpt", default=MOUNT_CKPT)
parser.add_argument("--balance-ckpt", default=BALANCE_CKPT)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--hold", type=float, default=0.2, help="seconds in the stance before the switch")
parser.add_argument("--switch-clearance", type=float, default=0.06)
parser.add_argument("--balance-task", default=BALANCE_TASK, help="task of the balance checkpoint (agent config)")
parser.add_argument("--fit-head", action="store_true",
                    help="head collision box resized to the head mesh (not what the policies trained on)")
parser.add_argument("--balance-bounded", action="store_true",
                    help="the balance policy was trained with clipped joint targets: keep the clip")
parser.add_argument("--viewer", choices=["viser", "native"], default="viser")
parser.add_argument("--steps", type=int, default=0, help="headless check: run this many steps, no viewer")
parser.add_argument("--dump", default=None,
                    help="with --steps: save sampled simulator states and contact statistics here "
                    "(input of scripts/check_collisions.py)")
args = parser.parse_args()

dev = "cuda:0"
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = args.num_envs
if args.fit_head:
    from functools import partial

    from mjlab_piplus.robot.piplus_constants import get_spec_with_arms

    cfg.scene.entities["robot"].spec_fn = partial(get_spec_with_arms, fit_head=True)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=dev))
u = env.unwrapped


def load_policy(task: str, pattern: str):
    (ckpt,) = glob(pattern) or [pattern]
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(load_rl_cfg(task)), device=dev)
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=dev)
    print(f"[chain] {task}: {os.path.relpath(ckpt)}")
    return runner.get_inference_policy(device=dev)


class ChainPolicy:
    """Mount policy until the stance has been held, then the balance policy, per env.

    The mount policy was trained with joint targets clipped to the joint limits, the
    balance policy without, so the clip is lifted for an env at its switch and restored
    when it resets (inside the action term: the "last action" observation stays raw).
    """

    def __init__(self):
        self.mount = load_policy(MOUNT_TASK, args.mount_ckpt)
        self.balance = load_policy(args.balance_task, args.balance_ckpt)
        self.term = u.action_manager.get_term("joint_pos")
        self.bounded = self.term._clip.clone()
        self.foot_cfg = SceneEntityCfg("robot", body_names=("r_ankle_roll_link", "l_ankle_roll_link"))
        self.foot_cfg.resolve(u.scene)
        self.hold_steps = max(1, int(args.hold / u.step_dt))
        self.handed = torch.zeros(u.num_envs, dtype=torch.bool, device=dev)
        self.counter = torch.zeros(u.num_envs, device=dev)
        self.switches = self.falls = 0

    def reset(self) -> None:
        self._restart(torch.ones_like(self.handed))

    def _restart(self, envs: torch.Tensor) -> None:
        self.handed[envs] = False
        self.counter[envs] = 0
        self.term._clip[envs] = self.bounded[envs]

    def __call__(self, obs) -> torch.Tensor:
        fresh = u.episode_length_buf <= 1  # the env reset since the last call (fall or viewer reset)
        self.falls += int((fresh & self.handed).sum())
        if fresh[0] and self.handed[0]:
            print("[chain] env 0: reset, mount policy", flush=True)
        self._restart(fresh)
        stance = single_leg_stance_lifted(
            u, self.foot_cfg, SceneEntityCfg("ball"), None, True, STAND_FOOT_Z, COM_OVER_BALL,
            args.switch_clearance,
        ) > 0
        self.counter = torch.where(stance, self.counter + 1, torch.zeros_like(self.counter))
        new = (self.counter >= self.hold_steps) & ~self.handed
        if new.any():
            self.handed |= new
            if not args.balance_bounded:
                self.term._clip[new] = torch.tensor([-float("inf"), float("inf")], device=dev)
            self.switches += int(new.sum())
            if new[0]:
                print(f"[chain] env 0: balance policy at {u.episode_length_buf[0].item() * u.step_dt:.2f} s", flush=True)
        with torch.no_grad():
            return torch.where(self.handed.unsqueeze(-1), self.balance(obs), self.mount(obs))


policy = ChainPolicy()
if args.steps:
    obs = env.get_observations()
    d, radius = u.sim.data, u.sim.model.geom_size[:, u.sim.mj_model.geom("ball/ball_geom").id, 0]
    qpos, rad, phase, nefc_max, nacon_max = [], [], [], 0, 0
    for step in range(args.steps):
        obs, _, _, _ = env.step(policy(obs))
        if args.dump:
            nefc_max = max(nefc_max, int(d.nefc.max()))
            nacon_max = max(nacon_max, int(d.nacon.max()))
            if step % 5 == 0:  # ponytail: every 0.1 s, all envs; thin out if the file gets big
                qpos.append(d.qpos.clone().cpu())
                rad.append(radius.expand(u.num_envs).clone().cpu())
                phase.append(policy.handed.clone().cpu())
    if args.dump:
        torch.save(
            {
                "qpos": torch.cat(qpos), "radius": torch.cat(rad), "balance_phase": torch.cat(phase),
                "nefc_max": nefc_max, "njmax": cfg.sim.njmax,
                "nacon_max": nacon_max, "naconmax": cfg.sim.nconmax * u.num_envs,
            },
            args.dump,
        )
        import mujoco

        mujoco.mj_saveModel(u.sim.mj_model, args.dump + ".mjb")
        print(f"[chain] max constraint rows in one env {nefc_max} (njmax {cfg.sim.njmax}); "
              f"max contacts in all envs {nacon_max} (limit {cfg.sim.nconmax * u.num_envs})")
    print(
        f"[chain] {args.steps} steps, {u.num_envs} envs: {policy.switches} switches, "
        f"{policy.falls} falls after a switch, {int(policy.handed.sum())} envs on the balance policy now"
    )
elif args.viewer == "native":
    NativeMujocoViewer(env, policy).run()
else:
    ViserPlayViewer(env, policy).run()
env.close()
