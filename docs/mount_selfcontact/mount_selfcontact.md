# mount_selfcontact: penalty for a right leg that is close to the left leg

Status: finished (Nov 14 2026).
Code: branch `feat/mount-selfcontact` (off `feat/mount-curriculum-turn`). Files: `tasks/ball_mount_min/piplus_ball_mount_selfcontact_cfg.py`, `scripts/eval_selfcontact.py`, `scripts/gen_selfcontact_report.py`, `tests/test_leggap.py`.
Tasks: `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap`.
Log: [mount_selfcontact_log.md](mount_selfcontact_log.md). Run ID prefix: `sg`.

## Why

Question: does the robot hit itself while it balances on the ball, and which part hits which? The robot (legs and arms, `FULL_COLLISION`) must run on the HighTorque Pi Plus, so a contact that the policy uses in the simulation can break on the real robot.

**Measurement (Nov 9 2026).** Script `scripts/eval_selfcontact.py` (tables: `scripts/gen_selfcontact_report.py`). Five policies on the stage-8 play config of their own task (blind on the ball, ball sizes 3-5, observation noise off, 256 envs, seed 3, one 20 s episode per env, steps from 3 s on). Each env counts until its first termination. The probe reads every MuJoCo contact with both geoms on the robot after each of the 4 physics steps, and samples the signed distance between all body pairs (`mj_geomDistance`) every 5th control step. Check of the force: the summed normal force of all robot-ball and robot-floor contacts is 1.02 to 1.04 times the weight of the robot (128 N).

Gait phases: A = latched on the ball and gated single support; B = latched, other (taps, transitions); M = not latched, episode time below 3 s (the mount). Shares are shares of env-steps. Env-steps in A / B / M: `cur8` 141929 / 66048 / 7999, `cur6` 136800 / 71059 / 8068, `cur7` 137659 / 74066 / 8029, `kb1` 172226 / 32518 / 20633, `cur4c` 131679 / 77649 / 7850. There are no unlatched steps after 3 s.

| Policy | Taps per 10 s | Stance share | Support foot − torso | `r_hip_pitch` at force limit (A+B) | Survived | Held-stance success |
|---|---|---|---|---|---|---|
| `cur8` | 11.9 | 64% | −13° | 48% | 93% | 0% |
| `cur6` | 12.1 | 63% | −20° | 44% | 93% | 0% |
| `cur7` | 12.6 | 61% | −23° | 48% | 95% | 0% |
| `cur4c` | 13.1 | 58% | −16° | 55% | 94% | 0% |
| `kb1` | 4.4 | 80% | +109° | 5% | 91% | 18% |

Self-contact between body pairs (share of env-steps with contact from 3 s on; normal force mean / max):

| Pair | `cur8` | `cur6` | `cur7` | `cur4c` | `kb1` |
|---|---|---|---|---|---|
| `torso\|r_thigh` | 30.4% (919 / 3235 N) | 30.1% (927 / 3393 N) | 31.5% (907 / 3273 N) | 36.5% (932 / 3392 N) | 0.09% (491 / 1629 N) |
| `r_ankle_roll\|l_ankle_roll` | 29.1% (47 / 3297 N) | 24.5% (40 / 2133 N) | 36.0% (49 / 2427 N) | 32.3% (52 / 3294 N) | 3.6% (39 / 1474 N) |
| `r_ankle_pitch\|l_ankle_pitch` | 9.2% | 15.0% | 0.04% | 5.7% | 0.2% |
| `r_ankle_roll\|l_ankle_pitch` | 4.2% | 7.4% | 2.8% | 10.5% | 4.9% |
| `r_ankle_roll\|l_calf` | — | — | — | — | 3.7% (62 / 488 N) |
| hand against torso or head (largest pair) | `torso\|l_wrist` 9.4% | `l_wrist\|head_pitch` 7.5% | `torso\|l_wrist` 2.8% | `r_wrist\|head_pitch` 1.2% | 0% |
| any contact with F > 10 N | 65.8% | 68.6% | 68.9% | 73.0% | 11.7% |
| the same, without `torso\|r_thigh` | 36.8% | 39.9% | 39.7% | 37.3% | 11.6% |
| any contact with F > weight (128 N) | 30.8% | 30.5% | 32.1% | 37.6% | 1.3% |

Findings:

- **`torso|r_thigh` is a stiff hip-flexion stop.** In 99% of its contact steps the force is above the weight of the robot (mean about 910 N at a depth of 1.5 mm). The right hip pitch is at about +1.54 rad and the calf at about +1.96 rad. With all other joints at zero, the thigh cylinder reaches the torso box at a hip pitch of 1.42 rad (the joint range is ±2.84 rad). The contact occurs in 86-93% of the phase-B steps, 1-3% of the phase-A steps, and 65% of the mount steps. 76-79% of its contact steps coincide with a free-foot tap (baseline 23-29%). Across envs, the Spearman correlation of the tap rate with the share of this contact is +0.77 to +0.83. The contact does not predict a fall. The force does not follow the `r_hip_pitch` actuator force (correlation −0.13 to −0.24). The torso box is a little too big: in 1500 sampled contact states of `cur8`, the thigh mesh is a median 12 mm clear of the torso mesh. The hip-roll link mesh (no primitive) reaches the torso mesh at about the same angle (median 0.2 mm; convex hulls overestimate). A hard stop near 1.5 rad is therefore plausible on the robot.
- **Foot against foot is a brush during the stance.** `r_ankle_roll|l_ankle_roll` touches in 37-55% of the phase-A steps of the straight-foot policies and 4% in `kb1`. The mean force is 40-52 N, and the force is above the weight in only 2-3% of its steps. It never coincides with a tap (0.01-0.04%). Median distance in phase A, collision primitives: `cur8` foot to foot 6.6 mm (36% overlap), any right-left pair of calf and ankle 2.7 mm; `kb1` 20.9 mm and 10.0 mm. In phase B the feet are far apart (median 60 mm, the cap of the probe). The contact is a prop that the policy uses during the stance. The real feet would not give this support.
- **`kb1` (the support foot turned by +109°) has the least self-contact.** The right thigh is yawed by about −1.95 rad, so the hip flexion no longer folds the thigh into the torso. The remaining contacts are crossed-leg contacts (`r_ankle_roll` against `l_ankle_pitch` and `l_calf`, 4-5%) and `base|r_ankle_roll` (2.4%, only during taps, mean 105 N).
- **Arms and head:** only the hand touches (torso, head). The forces are small (mean 12-25 N). The arm joints never reach their limits. Contact of a wrist or an elbow is more frequent before a fall (for example `cur4c` `r_wrist|head_pitch` 9 of 15 falls against 7.6% at live steps); there are only 12-24 falls per policy.
- **Not visible in the simulation:** the camera, the hip pitch, hip roll, shoulder pitch, and shoulder roll links have no collision primitive. Their meshes nest at their joints (static overlaps up to 55 mm; convex hulls).

**Why a penalty and not larger collision primitives.** A larger primitive moves the contact outward, and the policy can still lean on it. It also changes the contact with the ball and the floor, and every checkpoint would need a new model. A reward term leaves the model, the observations, and the network unchanged. The `cur4c` actor and critic load.

**Sim2real.** The term is a reward only. The actor never reads it, so the policy needs no new sensor on the robot. The term uses the distance with a margin (2 cm) and not only the contact. Reason: training already randomizes 10-40 ms loop latency and ±0.03 rad encoder bias (about 1.5 cm of foot position over a leg). A policy that keeps a 3-7 mm gap in the simulation touches on the robot. Only the leg pairs are penalized, because the feet and calves are modelled with primitives that agree with the meshes (`cur8` foot to foot, median 6.3 mm on meshes, 6.6 mm on primitives). The torso box is too big and some links have no primitive, so a penalty on `torso|r_thigh` would train against the model and not against the robot. Two numbers are needed from the real robot: the least clearance between the feet when it stands, and the angle at which the right hip stops against the torso (the simulation stops at about 1.4 rad). Neither is measured.

## Design

Task `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap` (flag `LegGap`, experiment `piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap`). It builds on `-NoTap` (full thigh range, `cur6`). It adds one reward term, `leg_gap`, and the curriculum term `leg_gap_curriculum`. The model, the observations, the actions, the network, and the other rewards are the same, so the test `tests/test_leggap.py` compares the config with `-NoTap` term by term.

**Term `leg_gap`** (module `piplus_ball_mount_selfcontact_cfg.py`). Per step, d is the smallest signed distance (m) between a collision primitive of the right leg and a collision primitive of the left leg. The primitives are the boxes and capsules of the calf, the ankle pitch link, and the ankle roll link (8 per leg; the thighs are far apart). The value is (0.02 − d) / 0.02, limited to the range 0 to 2. It is 0 from a gap of 2 cm, 1 at touch, and 2 at an overlap of 2 cm. The weight is −4 per second at the full ramp.

**Distance on the GPU.** The code samples points on one leg (8 corners, 12 edge centers, and 6 face centers of a box; 5 points on the axis of a capsule) and takes the exact signed distance of each point to each primitive of the other leg (box or capsule), both ways. Check against `mj_geomDistance` on 176 states of `cur6` (script in section Re-run): mean error 0.17 mm, 0.31 mm where the gap is below 3 cm, 95th percentile 1.5 mm, one outlier of 10 mm. The cost is not measurable: 25 ms per control step at 4096 envs, with and without the term.

**Two ways that failed.**
- A copy of the leg primitives with `margin` = `gap` = 2 cm (no force, but contacts up to 2 cm). `mujoco_warp` does not support a non-zero margin together with `MULTICCD` and box-box contacts (`NotImplementedError`). Disabling `MULTICCD` would change the dynamics of the other contacts.
- MuJoCo `geomdist` sensors, one for each pair of the calf, ankle pitch, and ankle roll links (9 sensors, cutoff 2 cm). They need no new geom, but a control step took 3655 ms at 512 envs (22 ms without the sensors), and the default 100 contacts per env overflowed (169 contacts per env at the start pose; illegal memory access at 4 envs).

**Weight.** The term is linear in the weight. On `cur6` `model_27449` (play config, 64 envs, the first 6 s of the episodes, from `reset`) the value is 0.41 per step on average; it is above 0 in 45% of the steps and at least 1 in 29%. At weight −4 this is −1.6 per second, in the same range as the tap penalty (−2.3 per second in `cur6`) and 10% of the stance weight (16). The weight is a first guess. It is measured again in the run log.

**Curriculum.** The weight ramps in with the stage-8 ramp (500 iterations; the ramp factor is the ramped weight of `support_foot_heading`, as in `turn_curriculum`), after `stage_curriculum`. `stage_curriculum` stays unchanged, so the fingerprints of the old tasks stay equal (check C2).

**Run.** Warm start from `cur4c` `model_24450` (a byte-identical copy as `2026-11-09_00-00-00_init-from-cur4c-24450` in the log group of the new task), `--env.curriculum.stage.params.start_stage 8`, 3000 iterations, seed 42, 4096 envs. All terms that the policy needs exist. A cold start through all stages is only needed if the warm start does not move the posture (`cur7` and `cur8` did not leave the straight basin).

**What is not penalized.** `torso|r_thigh` (the hip stop; a penalty could stop the mount, because 65% of the mount steps touch it; it needs a decision on a hip limit or a smaller torso box first), the hands and the head (small forces), and the thighs.

**What to check after the run.** With `scripts/eval_selfcontact.py` on the new policy and the same conditions as the measurement: the share of phase-A steps with foot-foot contact (`cur6` 37%), the median foot-foot distance in phase A (`cur8` 6.6 mm), and with `scripts/eval_mountbalance.py --gait --probe --audit` the taps, the stance share, the survival, and the held-stance success. The term must not cost survival.

## Result

`sg1-leggap` removed the foot-on-foot prop and left the gait unchanged (log). Foot-foot contact in phase A fell from 37% to 0.7%, and the median foot-foot distance rose from 4 mm to 51 mm. Survival (94%), taps (12.8 per 10 s) and stance share (62%) equal `cur6`. The taps and the 0% held-stance success are not changed by this penalty. Not changed: `torso`\|`r_thigh` (31%, by design). Side effect: `torso`\|`l_wrist` rose from 0.8% to 4.5% (mean force 24 N).

## Next steps

- Decide on the hip stop (`torso|r_thigh`): a hip pitch limit near 1.4 rad, or a smaller torso box. Needs the angle of the real robot.
- A penalty for the hands against the head, if they touch more after this run (the camera has no collision primitive).

## Re-run

Training (about 3 h for 3000 iterations at 4096 envs):

```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap --name sg1-leggap --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-09_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 3000 --agent.seed 42
```

Probe (one policy; 256 envs, about 10 min) and tables:

```
uv run python scripts/eval_selfcontact.py --task <task> --checkpoint <model.pt> --label <label> --num-envs 256 --seed 3 --every 5 --dist-envs 256 --mesh-every 10 --mesh-envs 64 --out <dir>/<label>
uv run python scripts/gen_selfcontact_report.py <dir> cur8 cur6 cur7 kb1 cur4c
```

Policies of the measurement: `cur8` `model_27449` (task `-NoTap-Turn`), `cur6` `model_27449` (`-NoTap`), `cur7` `model_27449` (`-NoTap-MidThigh`), `cur4c` `model_24450` (`-FullThigh-Ramp-Excess`), `kb1` `model_59946` (`Mjlab-Piplus-Ball-MountMin-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp-Latch45-Hold-KinBall`). Run directories: see [runs_index.md](../runs_index.md).

The one-off checks below are not in `scripts/`. Their full text is here because this document cites their results.

Hip stop: the contact angle with the other joints at zero, and the actuator limits (result: contact at 1.42 rad; hip pitch range ±2.84 rad; force limit 20 N·m):

```python
"""Geometry check (not tracked): hip pitch sweep against torso/base boxes, actuator limits of the hips."""
import mujoco, numpy as np, torch, warp as wp
import mjlab_piplus
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg("Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-Turn", play=True)
cfg.scene.num_envs = 2
u = ManagerBasedRlEnv(cfg, device="cuda:0")
m = u.sim.mj_model
d = mujoco.MjData(m)
g = lambda n: m.geom(f"robot/{n}").id
ft = np.zeros(6)
q0 = wp.to_torch(u.sim.wp_data.qpos)[0].cpu().numpy().copy()
jq = lambda n: m.joint(f"robot/{n}")
print("start joints (reset state):", {n: round(float(q0[jq(n).qposadr[0]]), 2) for n in ("r_hip_pitch_joint", "l_hip_pitch_joint", "r_calf_joint", "l_calf_joint", "r_thigh_joint", "r_hip_roll_joint")})
for n in ("r_hip_pitch", "l_hip_pitch", "r_thigh", "r_hip_roll"):
    a = m.actuator(f"robot/{n}")
    j = jq(n + "_joint")
    print(f"{n}: joint range {np.round(j.range, 3)}, forcerange {a.forcerange}, kp {a.gainprm[0]:.1f}, ctrlrange {a.ctrlrange}")
def dist(a, b):
    mujoco.mj_kinematics(m, d)
    return mujoco.mj_geomDistance(m, d, g(a), g(b), 0.2, ft)
pairs = [("torso_link_collision0", "r_thigh_link_collision0"), ("base_link_collision0", "r_thigh_link_collision0"),
         ("torso_link_collision0", "l_thigh_link_collision0"), ("base_link_collision0", "l_thigh_link_collision0")]
d.qpos[:] = q0
print("\nstart pose distances mm:", {f"{a[:5]}|{b[:7]}": round(1000 * dist(a, b), 1) for a, b in pairs})
for side in ("r", "l"):
    j = jq(f"{side}_hip_pitch_joint"); adr = j.qposadr[0]
    print(f"\nsweep {side}_hip_pitch (range {np.round(j.range, 2)}), other joints at the start pose; distance mm (negative = penetration)")
    print("  q    " + "  ".join(f"{a[:5]}|{b[:2]}{b[5]}" for a, b in pairs[(0 if side == 'r' else 2):(2 if side == 'r' else 4)]))
    for q in np.linspace(j.range[0], j.range[1], 13):
        d.qpos[:] = q0; d.qpos[adr] = q
        print(f"  {q:+.2f} " + "  ".join(f"{1000 * dist(a, b):8.1f}" for a, b in pairs[(0 if side == 'r' else 2):(2 if side == 'r' else 4)]))
print("\nbox sizes: torso", m.geom(g("torso_link_collision0")).size, "pos", m.geom(g("torso_link_collision0")).pos, "base", m.geom(g("base_link_collision0")).size, m.geom(g("base_link_collision0")).pos)
print("thigh cyl size", m.geom(g("r_thigh_link_collision0")).size, "pos", m.geom(g("r_thigh_link_collision0")).pos)
```

Mesh distances in the states where the primitives of `torso|r_thigh` touch (`python contact_meshes.py cur8 <task> <dir>`; result: thigh mesh median 12 mm clear, hip-roll link mesh median 0.2 mm):

```python
"""At the sampled joint states where torso|r_thigh touches (primitives), the mesh distances (not tracked)."""
import sys, mujoco, numpy as np
import mjlab_piplus
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
label, task, n_use = sys.argv[1], sys.argv[2], 1500
z = np.load(f"{sys.argv[3]}/{label}_dist.npz")
cfg = load_env_cfg(task, play=True); cfg.scene.num_envs = 2
u = ManagerBasedRlEnv(cfg, device="cuda:0"); m = u.sim.mj_model; d = mujoco.MjData(m); ft = np.zeros(6)
pairs = list(z["pairs"]); p = pairs.index("torso|r_thigh")
rng = np.random.default_rng(0)
idx = np.flatnonzero(z["pres"][:, p] & np.isin(z["ph"], (0, 1)))
idx = rng.choice(idx, min(n_use, len(idx)), replace=False)
adr = {n: m.joint(f"robot/{n}").qposadr[0] for n in z["joints"]}
bid = lambda n: m.body_name2id(f"robot/{n}_link") if hasattr(m, "body_name2id") else m.body(f"robot/{n}_link").id
mesh = lambda n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid(n) and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH][0]
coll = lambda n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid(n) and m.geom(g).name.startswith("robot/") and "_collision" in m.geom(g).name]
checks = [("torso", "r_thigh"), ("torso", "r_hip_roll"), ("torso", "r_hip_pitch"), ("base", "r_thigh"), ("base", "r_hip_roll"), ("r_hip_roll", "r_calf")]
res = {("prim", "torso", "r_thigh"): []}
for a, b in checks:
    res[("mesh", a, b)] = []
for i in idx:
    d.qpos[:] = m.qpos0; d.qpos[3:7] = [1, 0, 0, 0]
    for n, q in zip(z["joints"], z["jq"][i]):
        d.qpos[adr[n]] = q
    mujoco.mj_kinematics(m, d)
    res[("prim", "torso", "r_thigh")].append(min(mujoco.mj_geomDistance(m, d, g1, g2, 0.2, ft) for g1 in coll("torso") for g2 in coll("r_thigh")))
    for a, b in checks:
        res[("mesh", a, b)].append(mujoco.mj_geomDistance(m, d, mesh(a), mesh(b), 0.2, ft))
print(f"{label}: {len(idx)} sampled A+B states with the torso|r_thigh primitive contact; distance mm (negative = overlap)")
print("| geoms | median | p10 | min | share < 0 | share < 5 mm |\n|---|---|---|---|---|---|")
for k, v in res.items():
    v = 1000 * np.array(v)
    print(f"| {k[0]} {k[1]}|{k[2]} | {np.median(v):.1f} | {np.percentile(v, 10):.1f} | {v.min():.1f} | {(v < 0).mean():.0%} | {(v < 5).mean():.0%} |")
```

Check of the distance on the GPU against `mj_geomDistance`, and of the model (same number of geoms and sensors as `-NoTap`) (result: mean error 0.17 mm):

```python
"""Validate the LegGap shells and sensor (not tracked): dynamics, mass, sensor sign and agreement with mj_geomDistance."""
import numpy as np, torch, mujoco, warp as wp
from dataclasses import asdict
import mjlab_piplus
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
R = "logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap/2026-11-07_18-42-03_cur6-notap/model_27449.pt"
T0 = "Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap"; T1 = T0 + "-LegGap"
def build(task):
    cfg = load_env_cfg(task, play=True); cfg.scene.num_envs = 64; cfg.seed = 3; cfg.episode_length_s = 20.0
    cfg.observations["actor"].enable_corruption = False
    clip = load_rl_cfg(task).clip_actions
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cuda:0"), clip_actions=clip)
    r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(load_rl_cfg(task)), device="cuda:0")
    r.load(R, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    return env, r.get_inference_policy(device="cuda:0")
e0, p0 = build(T0); e1, p1 = build(T1); e2, p2 = build(T0)
u0, u1, u2 = e0.unwrapped, e1.unwrapped, e2.unwrapped
m0, m1 = u0.sim.mj_model, u1.sim.mj_model
print("ngeom", m0.ngeom, m1.ngeom, "nsensor", m0.nsensor, m1.nsensor, "total mass", m0.body_subtreemass[2], m1.body_subtreemass[2], "nacon alloc", u1.sim.wp_data.contact.dist.shape)
o0, o1, o2 = e0.get_observations(), e1.get_observations(), e2.get_observations()
dmax = 0; dist_all = []; found_all = []; agree = []
cpu = mujoco.MjData(m1); ft = np.zeros(6)
rg = lambda s: [g for g in range(m1.ngeom) if m1.geom(g).name.startswith(f"robot/{s}_") and "_collision" in m1.geom(g).name and any(k in m1.geom(g).name for k in ("calf", "ankle"))]
RG, LG = rg("r"), rg("l")
for step in range(300):
    with torch.no_grad():
        a0 = p0(o0); a1 = p1(o1); a2 = p2(o2)
    o0, *_ = e0.step(a0); o1, *_ = e1.step(a1); o2, *_ = e2.step(a2)
    q0 = u0.scene["robot"].data.joint_pos; q1 = u1.scene["robot"].data.joint_pos; q2 = u2.scene["robot"].data.joint_pos
    if step in (0, 4, 19, 49):
        print(f"step {step}: |joint_pos diff| NoTap vs LegGap max {float((q0 - q1).abs().max()):.2e} mean {float((q0 - q1).abs().mean()):.2e}; NoTap vs NoTap (second copy) max {float((q0 - q2).abs().max()):.2e} mean {float((q0 - q2).abs().mean()):.2e}")
    from mjlab_piplus.tasks.ball_mount_min.piplus_ball_mount_selfcontact_cfg import leg_gap, leg_distance, LEG_GAP_MARGIN
    val = leg_gap(u1, LEG_GAP_MARGIN)  # [B]
    dist_now = leg_distance(u1)
    dist_all.append(val.cpu().numpy())
    if step % 20 == 0 and step > 60:
        qpos = wp.to_torch(u1.sim.wp_data.qpos).cpu().numpy()
        for e in range(0, 64, 4):
            cpu.qpos[:] = qpos[e]; mujoco.mj_kinematics(m1, cpu)
            true = min(mujoco.mj_geomDistance(m1, cpu, a, b, 0.06, ft) for a in RG for b in LG)
            agree.append((true, float(dist_now[e])))
x = np.concatenate(dist_all)
print("term value: mean", x.mean(), "share > 0", (x > 0).mean(), "share >= 1 (touch/overlap)", (x >= 1).mean(), "max", x.max())
a = np.array(agree); tr = np.minimum(a[:, 0], 0.06); se = np.minimum(a[:, 1], 0.06)
print("samples", len(a), "; mean |torch - geomDistance| (mm), both clipped at 6 cm:", 1000 * np.abs(tr - se).mean(), "; p95", 1000 * np.percentile(np.abs(tr - se), 95), "; max", 1000 * np.abs(tr - se).max())
print("true < 0 in", (a[:, 0] < 0).sum(), "samples; torch < 0 in", (a[:, 1] < 0).sum()); near = a[:, 0] < 0.03; print("where true < 3 cm (n %d): mean error mm %.2f, p95 %.2f, max %.2f; signed mean (torch - true) %.2f" % (near.sum(), 1000*np.abs(a[near,1]-a[near,0]).mean(), 1000*np.percentile(np.abs(a[near,1]-a[near,0]),95), 1000*np.abs(a[near,1]-a[near,0]).max(), 1000*(a[near,1]-a[near,0]).mean()))
t = u1.reward_manager
cfgt = t.get_term_cfg("leg_gap"); print("leg_gap weight", cfgt.weight, "value mean over envs (last step):", float(cfgt.func(u1, **cfgt.params).mean()))
```
