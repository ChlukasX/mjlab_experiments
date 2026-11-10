"""Self-contact and close-call probe of a Pi Plus ball policy (docs/mount_selfcontact).

Rolls out one policy on its stage-8 play config (observation noise off, one 20 s episode per env,
each env counted until its first termination) and records, per body pair of the robot:
  1. every MuJoCo contact with both geoms on the robot (read after every physics substep:
     per control step a pair "touches" if any of the 4 substeps has a contact; depth and normal
     force are the largest over the substeps),
  2. the signed distance of every unfiltered body pair (mj_geomDistance on a CPU MjData, every
     --every control steps), once for the collision primitives and once for the visual meshes
     (convex hulls: an overestimate for concave parts).
Phases per env-step: A on the ball (latch) + gated single support; B on the ball, other;
C not latched, episode time >= 3 s; M not latched, episode time < 3 s (the mount);
E latched, episode time < 3 s. Shares "t>=3" use A+B+C only.

Run (one policy; 256 envs, seed 3 and about 10 minutes, most of it the CPU distance samples):

    uv run python scripts/eval_selfcontact.py --task <task> --checkpoint <model.pt> \
        --label <label> --out <dir>/<label>

Then `scripts/gen_selfcontact_report.py <dir> <label> [<label> ...]` prints the tables.
"""

import argparse
import json
import math
from dataclasses import asdict

import mujoco
import numpy as np
import torch
import warp as wp
from mujoco_warp._src.support import contact_force

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import mjlab_piplus  # noqa: F401  (registers tasks)
from mjlab_piplus.tasks.ball_mount.piplus_ball_mount_env_cfg import (
    COM_OVER_BALL,
    FOOT_BALL_SENSORS,
    FREE_FOOT_LIFT,
    STAND_FOOT_Z,
    _sole_clearance,
    _support_foot,
    ball_contact_sensors,
    single_leg_stance_lifted,
)
from mjlab_piplus.tasks.ball_mount_min.piplus_ball_mount_curriculum_cfg import free_foot_contact
from mjlab_piplus.tasks.ball_mount_min.piplus_ball_mountmin_env_cfg import _single_support, _yaw

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--num-envs", type=int, default=256)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument(
    "--every", type=int, default=5, help="control steps between primitive distance samples"
)
ap.add_argument(
    "--dist-envs",
    type=int,
    default=256,
    help="envs (the first N) in the primitive distance samples",
)
ap.add_argument(
    "--mesh-every", type=int, default=10, help="control steps between mesh distance samples (slow)"
)
ap.add_argument(
    "--mesh-envs", type=int, default=64, help="envs (the first N) in the mesh distance samples"
)
ap.add_argument("--out", required=True, help="output prefix (.json, _dist.npz)")
args = ap.parse_args()
torch.set_grad_enabled(False)
dev, DISTMAX, PH = "cuda:0", 0.06, ("A", "B", "C", "M", "E")
THR = (0.0, 0.005, 0.01, 0.02, 0.05)

# ---- environment, as scripts/eval_mountbalance.py (play config, noise off, 20 s episodes)
cfg = load_env_cfg(args.task, play=True)
cfg.scene.num_envs, cfg.seed, cfg.episode_length_s = args.num_envs, args.seed, 20.0
actor = cfg.observations["actor"]
actor.enable_corruption = False
for name, t in actor.terms.items():
    if name != "actions":
        t.delay_min_lag = t.delay_max_lag = 0
if not any(s.name in FOOT_BALL_SENSORS for s in cfg.scene.sensors):
    cfg.scene.sensors += ball_contact_sensors(cfg.decimation)
clip = load_rl_cfg(args.task).clip_actions
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=dev), clip_actions=clip)
u = env.unwrapped
runner = (load_runner_cls(args.task) or MjlabOnPolicyRunner)(
    env, asdict(load_rl_cfg(args.task)), device=dev
)
runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=dev)
policy = runner.get_inference_policy(device=dev)
B, dt = u.num_envs, u.step_dt
dt_sub = float(cfg.decimation)
ar = torch.arange(B, device=dev)
robot, m = u.scene["robot"], u.sim.mj_model
foot_cfg = SceneEntityCfg("robot", body_names=("r_ankle_roll_link", "l_ankle_roll_link"))
foot_cfg.resolve(u.scene)
fbody = torch.tensor(foot_cfg.body_ids, device=dev)
ball_cfg = SceneEntityCfg("ball")

# ---- body pairs: robot bodies, filtered as MuJoCo does (same weld body, or weld parent/child)
def bname(b):
    return m.body(b).name.removeprefix("robot/").removesuffix("_link")
rbodies = [b for b in range(m.nbody) if m.body(b).name.startswith("robot/")]
weld, par = m.body_weldid, m.body_parentid
def wpar(w):
    return weld[par[w]]
def filt(a, b):
    return (
    weld[a] == weld[b]
    or (weld[a] and weld[b] and (weld[a] == wpar(weld[b]) or weld[b] == wpar(weld[a])))
)
pairs = [(a, b) for i, a in enumerate(rbodies) for b in rbodies[i + 1 :] if not filt(a, b)]
P = len(pairs)
plabel = [f"{bname(a)}|{bname(b)}" for a, b in pairs]
pair_tab = torch.full((m.nbody, m.nbody), -1, dtype=torch.long, device=dev)
for p, (a, b) in enumerate(pairs):
    pair_tab[a, b] = pair_tab[b, a] = p
geom_body = torch.tensor(m.geom_bodyid, dtype=torch.long, device=dev)
coll = {
    b: [
        g
        for g in range(m.ngeom)
        if m.geom_bodyid[g] == b
        and m.geom(g).name.startswith("robot/")
        and "_collision" in m.geom(g).name
    ]
    for b in rbodies
}
mesh = {
    b: [
        g
        for g in range(m.ngeom)
        if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    ]
    for b in rbodies
}
no_prim = sorted(bname(b) for b in rbodies if not coll[b])
dpairs = {
    k: [(p, g1, g2) for p, (a, b) in enumerate(pairs) for g1 in src[a] for g2 in src[b]]
    for k, src in (("coll", coll), ("mesh", mesh))
}

# ---- per-substep contact hook
cnt, dep, frc = (torch.zeros(B * P, device=dev) for _ in range(3))
ext = torch.zeros(
    B, device=dev
)  # check: normal force of robot-vs-ball/floor contacts per env, summed over substeps
is_robot_geom = torch.tensor(
    [m.body(m.geom_bodyid[g]).name.startswith("robot/") for g in range(m.ngeom)], device=dev
)
weight = float(m.body_subtreemass[rbodies[0]]) * 9.81
ext_acc = [0.0, 0.0]
nmax = u.sim.wp_data.contact.dist.shape[0]
ids_wp = wp.from_torch(torch.arange(nmax, dtype=torch.int32, device=dev))
out_wp = wp.zeros(nmax, dtype=wp.spatial_vector, device=dev)
orig_step = u.sim.step


def hooked_step():
    orig_step()
    d = u.sim.wp_data
    n = int(wp.to_torch(d.nacon)[0].item())
    if n == 0:
        return
    g = wp.to_torch(d.contact.geom)[:n].long()
    pid = pair_tab[geom_body[g[:, 0]], geom_body[g[:, 1]]]
    sel = pid >= 0
    with wp.ScopedDevice(u.sim.wp_device):
        contact_force(u.sim.wp_model, d, ids_wp, False, out_wp)  # normal force = component 0
    one = is_robot_geom[g[:, 0]] ^ is_robot_geom[g[:, 1]]
    ext.index_add_(
        0, wp.to_torch(d.contact.worldid)[:n].long()[one], wp.to_torch(out_wp)[:n, 0][one]
    )
    if not sel.any():
        return
    idx = (wp.to_torch(d.contact.worldid)[:n].long() * P + pid)[sel]
    cnt.index_add_(0, idx, torch.ones(idx.shape, device=dev))
    dep.scatter_reduce_(0, idx, (-wp.to_torch(d.contact.dist)[:n][sel]).clamp_min(0), "amax")
    frc.scatter_reduce_(0, idx, wp.to_torch(out_wp)[:n, 0][sel], "amax")


u.sim.step = hooked_step

# ---- accumulators
def Z(*s):
    return torch.zeros(*s, device=dev)
acc = {
    k: Z(5, P)
    for k in (
        "cnt",
        "dsum",
        "fsum",
        "dmax",
        "fmax",
        "f10",
        "f50",
        "fw",
        "tap",
        "tapn",
        "hipsat",
        "anysat",
        "hiplim",
    )
}
base_ph = Z(
    5, 5
)  # per phase: steps with tap, tap within 0.1 s, r_hip_pitch at the force limit, any actuator at it, r_hip_pitch joint at its limit
nst = Z(5)
fmax_act = torch.tensor(
    [m.actuator(f"robot/{n}").forcerange[1] for n in robot.actuator_names], device=dev
)
r_hip = robot.actuator_names.index("r_hip_pitch")
sat_ph = Z(5, len(robot.actuator_names))
env_n, env_any, env_f10, env_taps, env_on = (Z(B) for _ in range(5))
env_pair_f10 = Z(B, P)
failed = torch.zeros(B, dtype=torch.bool, device=dev)
tap_age, age_any, age_sig = (torch.full((B,), 10_000, device=dev) for _ in range(3))
falls = {"n": 0, "any": 0, "sig": 0, "terms": {}}
base = {"n": 0, "any": 0, "sig": 0}
age_p = torch.full((B, P), 10_000, device=dev)  # steps since the last contact of each pair
fall_p, live_p = (
    Z(P),
    Z(P),
)  # envs with a contact of the pair in the last 25 steps: at a fall / at all live t>=3 s steps
gait = {"on": 0.0, "held": 0.0, "taps": 0.0, "yaw": []}
down_prev = torch.zeros(B, dtype=torch.bool, device=dev)
last = int(3.0 / dt)
n_steps = int(20.0 / dt) - 1
hold = Z(B)
jn = list(robot.joint_names)
jr = torch.tensor(
    np.array([m.joint(f"robot/{n}").range for n in jn]), device=dev, dtype=torch.float32
)
jstat = {"n": 0.0, "s": Z(len(jn)), "ss": Z(len(jn)), "lim": Z(len(jn)), "tgt": Z(len(jn))}
aterm = u.action_manager.get_term("joint_pos")
samples = {
    "coll": [],
    "ph": [],
    "env": [],
    "t": [],
    "mesh": [],
    "mph": [],
    "menv": [],
    "mt": [],
    "jq": [],
    "af": [],
    "pres": [],
    "pf": [],
    "tap": [],
    "si": [],
}  # joints, actuator force, contact flags per primitive sample
dd = np.zeros(1)
cpu = mujoco.MjData(m)
fromto = np.zeros(6)
counter = 0


DG = {
    k: (
        np.array([x[1] for x in lst]),
        np.array([x[2] for x in lst]),
        np.array([m.geom_rbound[x[1]] + m.geom_rbound[x[2]] + DISTMAX for x in lst]),
        np.r_[0, 1 + np.flatnonzero(np.diff([x[0] for x in lst]))],
        np.array([lst[i][0] for i in np.r_[0, 1 + np.flatnonzero(np.diff([x[0] for x in lst]))]]),
    )
    for k, lst in dpairs.items()
}


def dist_rows(kind):
    """Minimum signed distance per body pair [P] (DISTMAX if farther). Geom pairs whose bounding spheres are
    more than DISTMAX apart are skipped: their distance is above DISTMAX."""
    g1, g2, rs, starts, pidx = DG[kind]
    vals = np.full(len(g1), DISTMAX)
    near = np.flatnonzero(np.linalg.norm(cpu.geom_xpos[g1] - cpu.geom_xpos[g2], axis=1) < rs)
    for i in near:
        vals[i] = mujoco.mj_geomDistance(m, cpu, g1[i], g2[i], DISTMAX, fromto)
    row = np.full(P, DISTMAX, dtype=np.float32)
    row[pidx] = np.minimum.reduceat(vals, starts)
    return row


def finalize():
    global counter, tap_age, age_any, age_sig, age_p, down_prev, failed
    t = u.episode_length_buf.float() * dt
    valid = ~failed
    latch = getattr(u, "_on_ball_latch", torch.zeros(B, dtype=torch.bool, device=dev))
    single_g = _single_support(u, foot_cfg, ball_cfg, True)[0]
    late = t >= 3.0
    ph = torch.where(
        late, torch.where(latch, torch.where(single_g, 0, 1), 2), torch.where(latch, 4, 3)
    )
    oh = torch.nn.functional.one_hot(ph, 5).float() * valid.unsqueeze(-1).float()  # [B, 5]
    present, depth, force = (cnt > 0).view(B, P), dep.view(B, P), frc.view(B, P)
    tap = free_foot_contact(u, foot_cfg, ball_cfg) > 0
    tap_age = torch.where(tap, torch.zeros_like(tap_age), tap_age + 1)
    sat = robot.data.actuator_force.abs() > 0.95 * fmax_act
    nst.add_(oh.sum(0))
    sat_ph.add_(oh.T @ sat.float())
    q_hip = robot.data.joint_pos[:, jn.index("r_hip_pitch_joint")]
    hip_lim = (q_hip - jr[jn.index("r_hip_pitch_joint"), 0] < 0.03) | (
        jr[jn.index("r_hip_pitch_joint"), 1] - q_hip < 0.03
    )
    base_ph.add_(
        oh.T @ torch.stack([tap, tap_age <= 5, sat[:, r_hip], sat.any(-1), hip_lim], -1).float()
    )
    for k, x in (
        ("cnt", present),
        ("dsum", depth),
        ("fsum", force),
        ("f10", present & (force > 10)),
        ("f50", present & (force > 50)),
        ("fw", present & (force > weight)),
        ("hiplim", present & hip_lim[:, None]),
        ("tap", present & tap[:, None]),
        ("tapn", present & (tap_age <= 5)[:, None]),
        ("hipsat", present & sat[:, r_hip, None]),
        ("anysat", present & sat.any(-1)[:, None]),
    ):
        acc[k].add_(oh.T @ x.float())
    for k in range(5):
        mk = (oh[:, k] > 0).unsqueeze(-1)
        acc["dmax"][k] = torch.maximum(acc["dmax"][k], (depth * mk).amax(0))
        acc["fmax"][k] = torch.maximum(acc["fmax"][k], (force * mk).amax(0))
    # per-env counters (t >= 3 s, until the first termination) for the correlation with the taps
    cnt3 = (valid & late).float()
    env_n.add_(cnt3)
    env_any.add_(cnt3 * present.any(-1).float())
    env_f10.add_(cnt3 * (present & (force > 10)).any(-1).float())
    env_pair_f10.add_(cnt3.unsqueeze(-1) * (present & (force > 10)).float())
    # falls: self-contact in the last 0.5 s (25 steps) before a termination, against all steps
    any_p, any_s = present.any(-1), (present & (force > 10)).any(-1)
    age_any = torch.where(any_p, torch.zeros_like(age_any), age_any + 1)
    age_sig = torch.where(any_s, torch.zeros_like(age_sig), age_sig + 1)
    term, done = u.termination_manager.terminated & valid, u.reset_buf.bool() & valid
    age_p = torch.where(present, torch.zeros_like(age_p), age_p + 1)
    fall_p.add_(((age_p <= 25) & term[:, None]).float().sum(0))
    live_p.add_(((age_p <= 25) & (valid & late & ~done)[:, None]).float().sum(0))
    live = valid & late & ~done
    base["n"] += int(live.sum())
    base["any"] += int((live & (age_any <= 25)).sum())
    base["sig"] += int((live & (age_sig <= 25)).sum())
    falls["n"] += int(term.sum())
    falls["any"] += int((term & (age_any <= 25)).sum())
    falls["sig"] += int((term & (age_sig <= 25)).sum())
    for n in u.termination_manager.active_terms:
        if n != "time_out":
            falls["terms"][n] = falls["terms"].get(n, 0) + int(
                (u.termination_manager.get_term(n).bool() & valid).sum()
            )
    # gait metrics of scripts/eval_mountbalance.py --gait, per env from t >= 3 s
    on = valid & latch & late
    sup = _support_foot(u, foot_cfg, ball_cfg, None, "level")
    si = sup[:, 1].long()
    fi = 1 - si
    single = sup.sum(-1) == 1
    held = (
        single_leg_stance_lifted(
            u, foot_cfg, ball_cfg, None, "level", STAND_FOOT_Z, COM_OVER_BALL, FREE_FOOT_LIFT
        )
        > 0
    )
    down = _sole_clearance(u, foot_cfg)[ar, fi] < 0.01
    ev = on & down & ~down_prev
    down_prev = down
    env_taps.add_(ev.float())
    env_on.add_(on.float())
    gait["on"] += on.sum().item()
    gait["held"] += (on & held).sum().item()
    gait["taps"] += ev.sum().item()
    poses = robot.data.body_link_pose_w
    k = on & single & held
    d_yaw = _yaw(poses[ar, fbody[si], 3:7]) - _yaw(
        poses[:, robot.body_names.index("torso_link"), 3:7]
    )
    gait["yaw"].append((torch.remainder(d_yaw + math.pi, 2 * math.pi) - math.pi)[k].cpu())
    # joints in phase A (arms and legs): mean, spread, share at the joint limit, mean target
    ka = (ph == 0) & valid
    if ka.any():
        q = robot.data.joint_pos[ka]
        jstat["n"] += q.shape[0]
        jstat["s"] += q.sum(0)
        jstat["ss"] += (q * q).sum(0)
        jstat["lim"] += ((q - jr[:, 0] < 0.03) | (jr[:, 1] - q < 0.03)).float().sum(0)
        jstat["tgt"] += aterm._processed_actions[ka].sum(0)
    # distance samples: primitives (fast) for dist-envs envs, meshes (slow) for mesh-envs envs
    for kind, ev, ne, ph_k in (
        ("coll", args.every, args.dist_envs, ""),
        ("mesh", args.mesh_every, args.mesh_envs, "m"),
    ):
        if counter % ev == 0:
            sel = (valid & (ar < ne)).nonzero().squeeze(-1)
            qpos = wp.to_torch(u.sim.wp_data.qpos)[sel].cpu().numpy()
            for e, q in zip(sel.tolist(), qpos):
                cpu.qpos[:] = q
                mujoco.mj_kinematics(m, cpu)
                samples[kind].append(dist_rows(kind))
                if kind == "coll":
                    samples["jq"].append(robot.data.joint_pos[e].cpu().numpy())
                    samples["af"].append(robot.data.actuator_force[e].cpu().numpy())
                    samples["pres"].append(present[e].cpu().numpy())
                    samples["pf"].append(force[e].cpu().numpy().astype(np.float16))
                    samples["tap"].append(bool(tap[e]))
                    samples["si"].append(int(si[e]))
                samples[ph_k + "ph"].append(int(ph[e]))
                samples[ph_k + "env" if ph_k else "env"].append(e)
                samples[ph_k + "t"].append(float(t[e]))
    counter += 1
    ok3 = valid & late
    ext_acc[0] += float((ext / dt_sub)[ok3].sum())
    ext_acc[1] += float(ok3.sum())
    ext.zero_()
    cnt.zero_()
    dep.zero_()
    frc.zero_()
    failed = failed | u.reset_buf.bool()


orig_compute = u.reward_manager.compute


def hooked_compute(*a, **kw):
    r = orig_compute(*a, **kw)
    finalize()
    return r


u.reward_manager.compute = hooked_compute

# ---- rollout
obs = env.get_observations()
for step in range(n_steps):
    action = policy(obs)
    obs, _, _, _ = env.step(action)
    if step >= n_steps - last:  # held-stance success, as the eval (last 3 s)
        hold += (
            single_leg_stance_lifted(
                u, foot_cfg, ball_cfg, None, True, STAND_FOOT_Z, COM_OVER_BALL, FREE_FOOT_LIFT
            )
            > 0
        ).float()
success = (~failed) & (hold / last >= 0.9)

# ---- results
res = {
    "label": args.label,
    "task": args.task,
    "checkpoint": args.checkpoint,
    "num_envs": B,
    "seed": args.seed,
    "ext_force_over_weight": ext_acc[0] / max(ext_acc[1], 1) / weight,
    "weight_N": weight,
    "pairs": plabel,
    "phases": PH,
    "no_primitive": no_prim,
    "steps_per_phase": nst.tolist(),
    "acc": {k: v.tolist() for k, v in acc.items()},
    "base_ph": base_ph.tolist(),
    "survived": float((~failed).float().mean()),
    "success": float(success.float().mean()),
    "falls": falls,
    "base": base,
    "fall_p": fall_p.tolist(),
    "live_p": live_p.tolist(),
    "sat_ph": (sat_ph / nst.clamp_min(1).unsqueeze(-1)).tolist(),
    "actuators": list(robot.actuator_names),
    "gait": {
        "on_steps": gait["on"],
        "stance_share": gait["held"] / max(gait["on"], 1),
        "taps_per_10s": gait["taps"] / max(gait["on"] * dt, 1e-9) * 10,
        "yaw_deg_median": float(torch.cat(gait["yaw"]).rad2deg().median()) if gait["yaw"] else None,
    },
    "joints": {
        "names": jn,
        "n": jstat["n"],
        "mean": (jstat["s"] / max(jstat["n"], 1)).tolist(),
        "std": (
            (jstat["ss"] / max(jstat["n"], 1) - (jstat["s"] / max(jstat["n"], 1)) ** 2)
            .clamp_min(0)
            .sqrt()
        ).tolist(),
        "at_limit": (jstat["lim"] / max(jstat["n"], 1)).tolist(),
        "target": (jstat["tgt"] / max(jstat["n"], 1)).tolist(),
        "range": jr.tolist(),
    },
}
def rank(x):
    return x.argsort().argsort().float()
def corr(a, b):
    return float(torch.corrcoef(torch.stack([rank(a), rank(b)]))[0, 1])
ok = env_n > 100
tap_rate = env_taps / env_on.clamp_min(1) / dt * 10
res["corr"] = {
    "n_envs": int(ok.sum()),
    "taps_vs_any_contact": corr(tap_rate[ok], (env_any / env_n.clamp_min(1))[ok]),
    "taps_vs_f10_contact": corr(tap_rate[ok], (env_f10 / env_n.clamp_min(1))[ok]),
    "by_pair_f10": {
        plabel[p]: corr(tap_rate[ok], (env_pair_f10[:, p] / env_n.clamp_min(1))[ok])
        for p in env_pair_f10.sum(0).argsort(descending=True)[:5].tolist()
    },
}
json.dump(res, open(args.out + ".json", "w"))
np.savez_compressed(
    args.out + "_dist.npz",
    coll=np.array(samples["coll"]),
    mesh=np.array(samples["mesh"]),
    ph=np.array(samples["ph"]),
    env=np.array(samples["env"]),
    t=np.array(samples["t"]),
    jq=np.array(samples["jq"]),
    af=np.array(samples["af"]),
    pres=np.array(samples["pres"]),
    pf=np.array(samples["pf"]),
    tap=np.array(samples["tap"]),
    si=np.array(samples["si"]),
    joints=np.array(jn),
    actuators=np.array(robot.actuator_names),
    mph=np.array(samples["mph"]),
    menv=np.array(samples["menv"]),
    mt=np.array(samples["mt"]),
    pairs=np.array(plabel),
)
print(
    f"done {args.label}: envs {B}, steps/phase {dict(zip(PH, nst.tolist()))}, distance samples {len(samples['ph'])} (mesh {len(samples['mph'])}), "
    f"survived {res['survived']:.0%}, success {res['success']:.0%}"
)
