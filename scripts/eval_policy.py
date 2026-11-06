"""Evaluate a Mjlab-Piplus-Ball-MountBalance checkpoint by ball size and spawn mode.

Runs one 20 s episode per env (pushes on, observation noise off). An env counts
as a success if it never terminated early and had single-leg stance (sole flat
on the ball, the other sole's lowest corner `--free-foot-clearance` off the floor,
centre of mass over the ball) for at least `--hold-frac` of the last 3 s. Strict
success also requires that the free foot does not touch the ball; level success also that
the support sole is near horizontal (foot near the ball top, not on its flank).
Envs are binned by their own randomized ball radius and by spawn mode.

    uv run python scripts/eval_policy.py \
        --checkpoint logs/rsl_rl/piplus_ball_mountbalance/<run>/model_N.pt

`--probe` (posture, saturation), `--audit` (action size against the clip and the joint limits)
and `--gait` (free-foot placement, floor taps, stance length, foot heading) add diagnostics.
"""

import argparse
from dataclasses import asdict

import math

import torch

import mjlab_piplus  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply

from mjlab_piplus.common.ball_stance import (
    COM_OVER_BALL,
    FOOT_BALL_SENSORS,
    FREE_FOOT_LIFT,
    STAND_FOOT_Z,
    ball_contact,
    ball_contact_sensors,
    ball_radii,
    com_to_ball,
    sole_clearance,
    support_foot,
)
from mjlab_piplus.common.rewards.stance import free_foot_contact, single_leg_stance_lifted

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Mjlab-Piplus-Ball-MountBalance")
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=0, help="env seed (same DR draws across runs; the default was random)")
parser.add_argument("--hold-frac", type=float, default=0.9)
parser.add_argument("--free-foot-clearance", type=float, default=FREE_FOOT_LIFT,
                    help="m the free foot must clear the floor in the success test")
parser.add_argument("--probe", action="store_true",
                    help="also print the posture and actuator saturation over the last 3 s")
parser.add_argument("--audit", action="store_true",
                    help="also print the action audit: raw output size against the runner's clip, targets outside "
                    "the joint limits, action change per step (limits: mean |a| 1.0, outside 2%%, change 0.30)")
parser.add_argument("--gait", action="store_true",
                    help="also print how the stance holds on the ball (after 3 s): free foot relative to the support "
                    "foot and CoM relative to the ball in the torso heading frame, free-sole floor taps, stance "
                    "segment lengths, support foot heading against the torso")
parser.add_argument("--torsional", type=float, default=None,
                    help="feet get condim 6 with this torsional friction coefficient (m), a check of the spin")
parser.add_argument("--cone", choices=["pyramidal", "elliptic"], default=None, help="override the friction cone")
parser.add_argument("--ball-obs-freeze", type=float, default=None,
                    help="from this time (s) on, the actor's ball position stops updating (held at its last value; "
                    "a camera that loses the ball under the foot)")
parser.add_argument("--ball-obs-zero", action="store_true", help="with --ball-obs-freeze: zeros instead of holding")
parser.add_argument("--ball-obs-bias", type=float, nargs=3, default=None,
                    help="BlindOnBall tasks: per-episode ball-position error, uniform +- per axis (m)")
# Robustness sweep (deployment gaps); see docs/deploy_mount/deploy_mount.md.
parser.add_argument("--obs-noise", action="store_true", help="keep the training observation noise (play turns it off)")
parser.add_argument("--ball-noise-clean-zeros", action="store_true",
                    help="with --obs-noise: the ball input's per-step noise only before the on-ball latch "
                    "(the robot sends exact zeros after it; the term's noise config also hits the zeros)")
parser.add_argument("--noise-terms", nargs="*", default=None, help="with --obs-noise: only these actor terms keep their noise")
parser.add_argument("--obs-delay", type=int, default=0, help="actor observations this many control steps old (20 ms each)")
parser.add_argument("--obs-delay-terms", nargs="+", default=None, help="with --obs-delay: only these actor terms")
parser.add_argument("--act-lag", type=int, nargs=2, default=None, help="actuator lag range in physics steps (5 ms; trained 0 3)")
parser.add_argument("--encoder-bias", type=float, default=None,
                    help="per-joint encoder bias +- this (rad), and the actor reads the biased angles as on the robot")
parser.add_argument("--latch-height-bias", type=float, default=0.0,
                    help="BlindOnBall: base-height estimate reads this much too high (m; > 0 latches early, < 0 late)")
parser.add_argument("--kp-range", type=float, nargs=2, default=None, help="kp and kd scale range (trained 0.8 1.2)")
parser.add_argument("--ball-radius", type=float, default=None, help="fixed ball radius (m) instead of the size 1-5 range")
parser.add_argument("--ball-mass", type=float, default=None, help="fixed ball mass (kg), no jitter")
parser.add_argument("--ball-solid", action="store_true", help="solid-sphere ball inertia (0.4 m r^2) instead of a shell")
parser.add_argument("--floor-history", action="store_true",
                    help="floor-contact termination checks all physics steps of a control step, not only the last")
args = parser.parse_args()

dev = "cuda:0"
cfg = load_env_cfg(args.task, play=True)
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.episode_length_s = 20.0  # one finite episode per env
if args.ball_obs_freeze is not None:
    term = cfg.observations["actor"].terms["ball_pos"]
    live = term.func

    def frozen_ball_pos(env, **kw):
        val = live(env, **kw)
        t = env.episode_length_buf * env.step_dt
        if not hasattr(env, "_held_ball"):
            env._held_ball = val.clone()
        keep = t < args.ball_obs_freeze
        env._held_ball[keep] = val[keep]
        return torch.zeros_like(val) if args.ball_obs_zero and not keep.all() else env._held_ball.clone()

    term.func = frozen_ball_pos
if args.ball_obs_bias is not None:
    cfg.observations["actor"].terms["ball_pos"].params["bias"] = tuple(args.ball_obs_bias)
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
actor = cfg.observations["actor"]
actor.enable_corruption = args.obs_noise
if args.obs_noise and args.ball_noise_clean_zeros:
    term = actor.terms["ball_pos"]
    term.params["step_noise"], term.noise = term.noise.n_max, None
if args.noise_terms is not None:
    for name, t in actor.terms.items():
        if name not in args.noise_terms:
            t.noise = None
for name, t in actor.terms.items():
    if name != "actions" and (args.obs_delay_terms is None or name in args.obs_delay_terms):  # the node knows its own previous action without delay
        t.delay_min_lag = t.delay_max_lag = args.obs_delay
if args.act_lag is not None:
    from dataclasses import replace

    robot_cfg = cfg.scene.entities["robot"]
    robot_cfg.articulation = replace(robot_cfg.articulation, actuators=tuple(
        replace(a, delay_min_lag=args.act_lag[0], delay_max_lag=args.act_lag[1]) for a in robot_cfg.articulation.actuators))
if args.encoder_bias is not None:
    cfg.events["encoder_bias"].params["bias_range"] = (-args.encoder_bias, args.encoder_bias)
    actor.terms["joint_pos"].params["biased"] = True
if args.latch_height_bias:
    actor.terms["ball_pos"].params["height_threshold"] -= args.latch_height_bias
if args.kp_range is not None:
    cfg.events["pd_gains"].params["kp_range"] = cfg.events["pd_gains"].params["kd_range"] = tuple(args.kp_range)
if args.ball_radius is not None:
    from mjlab_piplus.tasks.ball_mount.piplus_ball_mountbalance_env_cfg import RADIUS_RANGE

    cfg.events["ball_size"].params["ranges"] = (args.ball_radius / RADIUS_RANGE[1],) * 2
if args.ball_mass is not None:
    cfg.events["ball_mass_from_size"].params.update(mass_range=(args.ball_mass,) * 2, jitter=(1.0, 1.0))
if args.ball_solid:
    cfg.events["ball_mass_from_size"].params["inertia_factor"] = 0.4
if args.floor_history:
    from dataclasses import replace

    cfg.scene.sensors = tuple(replace(s, fields=("found", "force"), history_length=cfg.decimation)
                              if s.name == "body_floor_contact" else s for s in cfg.scene.sensors)
# Strict test: the free foot must also not touch the ball in any physics step. The stance
# reward without free_off_ball calls a foot free when it is not flat on the ball, which a foot
# resting tilted on the ball passes (two-foot stance; see docs/mount_latency/mount_latency_log.md).
if not any(s.name in FOOT_BALL_SENSORS for s in cfg.scene.sensors):
    cfg.scene.sensors += ball_contact_sensors(cfg.decimation)
clip = load_rl_cfg(args.task).clip_actions
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=dev), clip_actions=clip)
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
radius = ball_radii(u, ball_cfg, None).clone()
n_steps = int(20.0 / (u.step_dt)) - 1  # stop before the time-out reset
last = int(3.0 / u.step_dt)
failed = torch.zeros(u.num_envs, dtype=torch.bool, device=dev)
hold = torch.zeros(u.num_envs, device=dev)
hold_strict = torch.zeros(u.num_envs, device=dev)
hold_level = torch.zeros(u.num_envs, device=dev)
roll_ids = [j for j, n in enumerate(u.scene["robot"].joint_names) if n.endswith("ankle_roll_joint")]
roll_max = []  # largest |ankle roll| per stance step, rad
first_stance = torch.full((u.num_envs,), float("nan"), device=dev)

# Probe: sums over the last 3 s. Free foot = the one not on the ball (the lower one).
robot = u.scene["robot"]
mj = u.sim.mj_model
force_max = torch.tensor(
    [mj.actuator(f"robot/{n}").forcerange[1] for n in robot.actuator_names], device=dev
)
probe = {k: torch.zeros(u.num_envs, device=dev) for k in ("com", "base", "free_z", "sole", "on_floor", "on_ball", "yaw", "ball_yaw", "tilt")}
saturated = torch.zeros(u.num_envs, len(robot.actuator_names), device=dev)

# Audit: raw policy output (before the runner's clip) and the joint targets it asks for.
aterm = u.action_manager.get_term("joint_pos")
a_names = list(aterm.target_names)
a_jids = [int(robot.indexing.joint_ids[robot.joint_names.index(n)]) for n in a_names]
j_lo = torch.tensor([mj.jnt_range[j, 0] for j in a_jids], device=dev)
j_hi = torch.tensor([mj.jnt_range[j, 1] for j in a_jids], device=dev)
audit = {"abs": [], "outside": 0.0, "change": 0.0, "n": 0}
prev_action = None

# Gait: heading frame of the torso (x forward, y left).
torso_id = robot.body_names.index("torso_link")
ar = torch.arange(u.num_envs, device=dev)
gait = {"off": [], "com": [], "yaw": [], "seg": [], "taps": 0.0, "steps": 0.0}
seg_len = torch.zeros(u.num_envs, device=dev)
free_down_prev = torch.zeros(u.num_envs, dtype=torch.bool, device=dev)
# Free-foot state (--gait): steps by support state, free-foot ball and floor events, the training term.
fs = dict.fromkeys(("n", "both", "none", "ball", "floor", "air", "ball_ev", "floor_ev", "term", "any_floor"), 0.0)
fs_ball_prev = torch.zeros(u.num_envs, dtype=torch.bool, device=dev)
fs_floor_prev = fs_ball_prev.clone()


def heading(q):
    x = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(q.shape[:-1] + (3,)))
    return torch.atan2(x[..., 1], x[..., 0])


def to_heading(v, h):  # world xy -> torso heading frame
    c, s_ = torch.cos(h), torch.sin(h)
    return torch.stack([c * v[:, 0] + s_ * v[:, 1], -s_ * v[:, 0] + c * v[:, 1]], dim=-1)


for step in range(n_steps):
    with torch.no_grad():
        action = policy(obs)
    obs, _, dones, _ = env.step(action)
    failed |= dones.bool()
    if args.audit and step >= n_steps - last:
        a = action[~failed]
        target = a * aterm._scale + aterm._offset[~failed] if torch.is_tensor(aterm._offset) else a * aterm._scale + aterm._offset
        audit["abs"].append(a.abs().flatten())
        audit["outside"] += ((target < j_lo) | (target > j_hi)).float().mean().item()
        if prev_action is not None:
            audit["change"] += (action - prev_action)[~failed].abs().mean().item()
        audit["n"] += 1
    prev_action = action.clone()
    stance = single_leg_stance_lifted(u, foot_cfg, ball_cfg, None, True, STAND_FOOT_Z, COM_OVER_BALL, args.free_foot_clearance) > 0
    new = stance & first_stance.isnan() & ~failed
    first_stance[new] = step * u.step_dt
    if step >= n_steps - last:
        hold += stance.float()
        free = ~support_foot(u, foot_cfg, ball_cfg, None, True)  # [B, 2]; with stance, exactly one
        free_clear = ~(free & ball_contact(u)).any(dim=-1)  # contact only: a foot may hover beside the ball
        hold_strict += (stance & free_clear).float()
        level = single_leg_stance_lifted(u, foot_cfg, ball_cfg, None, "level", STAND_FOOT_Z, COM_OVER_BALL, args.free_foot_clearance) > 0
        hold_level += (level & free_clear).float()
        roll_max.append(u.scene["robot"].data.joint_pos[:, roll_ids].abs().amax(dim=-1)[stance & ~failed])
        free_z = robot.data.body_link_pose_w[:, foot_cfg.body_ids, 2].min(dim=-1).values
        top_z = u.scene["ball"].data.root_link_pos_w[:, 2] + radius
        probe["com"] += com_to_ball(u, foot_cfg, ball_cfg).norm(dim=-1)
        probe["base"] += robot.data.root_link_pos_w[:, 2] - top_z
        probe["free_z"] += free_z - STAND_FOOT_Z
        sole = sole_clearance(u, foot_cfg).min(dim=-1).values  # lower sole's lowest corner
        probe["sole"] += sole
        probe["on_floor"] += (sole < 0.01).float()
        probe["on_ball"] += support_foot(u, foot_cfg, ball_cfg, None, True).any(dim=-1).float()
        probe["yaw"] += robot.data.root_link_vel_w[:, 5]  # rad/s about the world vertical
        probe["ball_yaw"] += u.scene["ball"].data.root_link_vel_w[:, 5]
        probe["tilt"] += torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1)).rad2deg()
        saturated += (robot.data.actuator_force.abs() > 0.95 * force_max).float()
    if args.gait and step * u.step_dt > 3.0:
        on = ~failed & getattr(u, "_on_ball_latch", support_foot(u, foot_cfg, ball_cfg, None, True).any(dim=-1))
        sup = support_foot(u, foot_cfg, ball_cfg, None, "level")
        si = sup[:, 1].long(); fi = 1 - si; single = sup.sum(dim=-1) == 1
        held = single_leg_stance_lifted(u, foot_cfg, ball_cfg, None, "level", STAND_FOOT_Z, COM_OVER_BALL, args.free_foot_clearance) > 0
        pose = robot.data.body_link_pose_w[:, foot_cfg.body_ids]
        h = heading(robot.data.body_link_pose_w[:, torso_id, 3:7])
        k = on & single & held
        gait["off"].append(to_heading(pose[ar, fi, :2] - pose[ar, si, :2], h)[k])
        gait["com"].append(to_heading(-com_to_ball(u, foot_cfg, ball_cfg), h)[k])
        gait["yaw"].append(torch.remainder(heading(pose[ar, si, 3:7]) - h + math.pi, 2 * math.pi)[k] - math.pi)
        down = sole_clearance(u, foot_cfg)[ar, fi] < 0.01
        gait["taps"] += (on & down & ~free_down_prev).sum().item()
        free_down_prev = down
        gait["steps"] += on.sum().item()
        flat = support_foot(u, foot_cfg, ball_cfg, None, True)  # flat support, as the training term
        single_flat = on & (flat.sum(dim=-1) == 1)
        free_i = 1 - flat[:, 1].long()  # the foot that is not the support foot (valid where single_flat)
        free_ball = ball_contact(u)[ar, free_i]
        free_floor = sole_clearance(u, foot_cfg)[ar, free_i] < 0.01
        fs["n"] += single_flat.sum().item()
        fs["both"] += (on & (flat.sum(dim=-1) == 2)).sum().item()
        fs["none"] += (on & (flat.sum(dim=-1) == 0)).sum().item()
        fs["ball"] += (single_flat & free_ball).sum().item()
        fs["floor"] += (single_flat & ~free_ball & free_floor).sum().item()
        fs["air"] += (single_flat & ~free_ball & ~free_floor).sum().item()
        fs["ball_ev"] += (single_flat & free_ball & ~fs_ball_prev).sum().item()
        fs["floor_ev"] += (single_flat & free_floor & ~fs_floor_prev).sum().item()
        fs_ball_prev, fs_floor_prev = single_flat & free_ball, single_flat & free_floor
        fs["term"] += (on & (free_foot_contact(u, foot_cfg, ball_cfg) > 0)).sum().item()
        fs["any_floor"] += (on & (sole_clearance(u, foot_cfg) < 0.01).any(dim=-1)).sum().item()
        ended = on & ~held & (seg_len > 0)
        gait["seg"].append(seg_len[ended] * u.step_dt)
        seg_len = torch.where(on & held, seg_len + 1, torch.zeros_like(seg_len))

success = ~failed & (hold / last >= args.hold_frac)
strict = ~failed & (hold_strict / last >= args.hold_frac)
level = ~failed & (hold_level / last >= args.hold_frac)
edges = [0.0, 0.083, 0.097, 1.0]
names = ["size 1-2 (<0.083 m)", "size 3 (0.083-0.097)", "size 4-5 (>0.097)"]
print(f"\n{args.task}  {args.checkpoint}  ({u.num_envs} envs)")
print(f"{'spawn':<6} {'ball radius':<22} {'envs':>5} {'success':>8} {'strict':>7} {'level':>6} {'survived':>9} {'t to stance':>12}")
for is_mount, label in ((True, "mount"), (False, "drop")):
    for lo, hi, name in zip(edges[:-1], edges[1:], names):
        m = (mount == is_mount) & (radius >= lo) & (radius < hi)
        if not m.any():
            continue
        t = first_stance[m]
        t_str = f"{t[~t.isnan()].mean().item():.1f}s" if (~t.isnan()).any() else "-"
        print(
            f"{label:<6} {name:<22} {int(m.sum()):>5} {success[m].float().mean().item():>8.0%} {strict[m].float().mean().item():>7.0%} {level[m].float().mean().item():>6.0%}"
            f" {(~failed[m]).float().mean().item():>9.0%} {t_str:>12}"
        )
print(f"overall success {success.float().mean().item():.0%}, survived {(~failed).float().mean().item():.0%}, "
      f"strict success (free foot off the ball) {strict.float().mean().item():.0%}, level success {level.float().mean().item():.0%}")
roll = torch.cat(roll_max)
print(f"ankle roll in the stance: median max |q| {roll.median().item():.2f} rad, at the stop (> 0.62) {(roll > 0.62).float().mean().item():.0%}")

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
              f"{(probe['ball_yaw'][m] / last).mean().item():+.2f} rad/s; base tilt mean {(probe['tilt'][m] / last).mean().item():.1f} deg")
        h = hold[m] / last
        print(f"  stance held: mean {h.mean().item():.0%} of the last 3 s; envs above 50%: {(h > 0.5).float().mean().item():.0%}")
        sat = (saturated[m] / last).mean(dim=0)
        print("  share of steps at the force limit (>95%): " + ", ".join(
            f"{n} {x:.0%}" for n, x in zip(robot.actuator_names, sat.tolist()) if x > 0.02
        ) or "  none")

if args.audit:
    A = torch.cat(audit["abs"])
    q = A.quantile(0.95).item() if A.numel() < 16_000_000 else float("nan")
    print(f"\nAction audit, last 3 s, envs alive (limits: mean |a| 1.0, targets outside 2%, change 0.30):")
    print(f"  raw |a| mean {A.mean().item():.2f}, median {A.median().item():.2f}, p95 {q:.2f}"
          + (f", past the +-{clip:g} clip {(A > clip).float().mean().item():.0%}" if clip else ""))
    print(f"  targets outside the joint limits {audit['outside'] / max(audit['n'], 1):.1%}, "
          f"mean |action change| per step {audit['change'] / max(audit['n'] - 1, 1):.3f}")

if args.gait:
    off, com, yaw, seg = (torch.cat(gait[k]) for k in ("off", "com", "yaw", "seg"))
    cm = lambda x, p: x.quantile(p).item() * 100
    print(f"\nGait on the ball (after 3 s, envs alive and on the ball; heading frame x forward, y left):")
    if len(off):
        print(f"  in stance: free foot - support foot x median {cm(off[:, 0], .5):+.1f} cm (p10 {cm(off[:, 0], .1):+.1f}, p90 "
              f"{cm(off[:, 0], .9):+.1f}), y {cm(off[:, 1], .5):+.1f} cm; behind (< -3 cm) {(off[:, 0] < -0.03).float().mean().item():.0%}, "
              f"in front (> 3 cm) {(off[:, 0] > 0.03).float().mean().item():.0%}")
        print(f"  CoM - ball centre x {cm(com[:, 0], .5):+.1f} cm, y {cm(com[:, 1], .5):+.1f} cm; "
              f"support foot heading - torso median {yaw.median().rad2deg().item():+.0f} deg (p10 {yaw.quantile(.1).rad2deg().item():+.0f}, "
              f"p90 {yaw.quantile(.9).rad2deg().item():+.0f})")
    tot = gait["steps"] * u.step_dt
    print(f"  stance {len(off) / max(gait['steps'], 1):.0%} of the on-ball steps; free-sole floor taps "
          f"{gait['taps'] / max(tot, 1e-9) * 10:.1f} per 10 s; stance segments "
          + (f"median {seg.median().item():.2f} s, p90 {seg.quantile(.9).item():.2f} s (n {len(seg)})" if len(seg) else "none"))
    st, ns = max(gait["steps"], 1), max(fs["n"], 1)
    print(f"  free-foot state: single support {fs['n'] / st:.0%} of the on-ball steps, both feet on the ball {fs['both'] / st:.0%}, "
          f"neither {fs['none'] / st:.0%}")
    print(f"  in single support the free foot: touches the ball {fs['ball'] / ns:.1%}, on the floor (no ball) "
          f"{fs['floor'] / ns:.1%}, in the air {fs['air'] / ns:.1%}")
    print(f"  free-foot ball-contact events {fs['ball_ev'] / max(tot, 1e-9) * 10:.1f} per 10 s, floor events "
          f"{fs['floor_ev'] / max(tot, 1e-9) * 10:.1f} per 10 s")
    print(f"  training term (free_foot_contact) active on {fs['term'] / st:.1%} of the on-ball steps; any sole on the floor "
          f"{fs['any_floor'] / st:.1%}")
