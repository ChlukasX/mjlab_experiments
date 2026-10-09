"""Replay the best trajectory saved by scripts/mount_cem.py and print how far the mount gets.

    uv run python scripts/replay_cem.py logs/mount_cem/free_s5.pt
"""

import sys

import torch

import mjlab_piplus  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg

from mjlab_piplus.tasks.ball_mount.piplus_ball_mount_env_cfg import (
    COM_OVER_BALL,
    FREE_FOOT_LIFT,
    STAND_FOOT_Z,
    _com_to_ball,
    _mount_phi,
    _support_foot,
    single_leg_stance_lifted,
)

best = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
dev = "cuda:0"
cfg = load_env_cfg(best["task"], play=False)
cfg.scene.num_envs = 4
keep = {"reset_scene_to_default", "spawn_mode", "ball_lock"}
cfg.events = {k: v for k, v in cfg.events.items() if k in keep}
cfg.events["spawn_mode"].params["ball_xy_noise"] = (0.0, 0.0)
cfg.episode_length_s = 20.0
env = ManagerBasedRlEnv(cfg, device=dev)
env.reset()
foot_cfg = SceneEntityCfg("robot", body_names=("r_ankle_roll_link", "l_ankle_roll_link"))
foot_cfg.resolve(env.scene)
ball_cfg = SceneEntityCfg("ball")
robot, ball = env.scene["robot"], env.scene["ball"]
actions = best["actions"].to(dev)
print(f"{best['task']}  score {best['score']:.1f} (iteration {best['iter']})")
print("   t     alive  foot_on_ball(R,L)  CoM-ball  base-top  ankle z (R, L)   phi    stance")
for t in range(actions.shape[0]):
    out = env.step(actions[t].expand(4, -1))
    term = out[2] | out[3]
    if t % 25 == 24 or term[0]:
        on = _support_foot(env, foot_cfg, ball_cfg, None, True)[0].tolist()
        d = _com_to_ball(env, foot_cfg, ball_cfg)[0].norm().item()
        top = (ball.data.root_link_pos_w[0, 2] + 0.11).item()
        fz = robot.data.body_link_pose_w[0, foot_cfg.body_ids, 2].tolist()
        st = single_leg_stance_lifted(env, foot_cfg, ball_cfg, None, True, STAND_FOOT_Z + FREE_FOOT_LIFT, COM_OVER_BALL)[0].item()
        print(f"{(t + 1) * env.step_dt:5.2f}   {not bool(term[0])!s:5}  {on}   {d:6.3f}   {robot.data.root_link_pos_w[0, 2].item() - top:6.3f}   ({fz[0]:.3f}, {fz[1]:.3f})  {_mount_phi(env, foot_cfg, ball_cfg, None)[0].item():.2f}   {st:.0f}")
        if term[0]:
            print("   -> terminated")
            break
