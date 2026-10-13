"""Where do the robot's meshes intersect although the simulation sees no collision?

Reads simulator states recorded from the chain (`scripts/play_chain.py --steps N --dump
FILE`) and, for every pair of robot bodies that are not parent and child, and for each
foot against the ball and the floor, compares two distances:
  visual     between the convex hulls of the visual meshes (what the viewer shows)
  collision  between the collision primitives (what the simulation enforces)
A pair whose meshes intersect while its collision primitives are apart (or missing) is a
gap in the collision model. Pairs whose mesh hulls already intersect at the home pose are
listed separately (hulls of neighbouring links overlap by construction).

    uv run python scripts/play_chain.py --num-envs 256 --steps 1000 --dump logs/mount_states/chain_dump.pt
    uv run python scripts/check_collisions.py logs/mount_states/chain_dump.pt
"""

import argparse
from itertools import combinations

import mujoco
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("dump")
parser.add_argument("--samples", type=int, default=2000)
parser.add_argument("--tol", type=float, default=0.002, help="m of overlap that counts as intersecting")
args = parser.parse_args()

dump = torch.load(args.dump)
m = mujoco.MjModel.from_binary_path(args.dump + ".mjb")
d = mujoco.MjData(m)
print(f"constraint rows: max {dump['nefc_max']} of njmax {dump['njmax']}; "
      f"contacts: max {dump['nacon_max']} of {dump['naconmax']}")

name = lambda t, i: mujoco.mj_id2name(m, t, i) or ""
bodies = [b for b in range(m.nbody) if name(mujoco.mjtObj.mjOBJ_BODY, b).startswith("robot/")]
geoms = {b: {"vis": [], "col": []} for b in bodies}
for g in range(m.ngeom):
    b = m.geom_bodyid[g]
    if b in geoms:
        geoms[b]["col" if "_collision" in name(mujoco.mjtObj.mjOBJ_GEOM, g) else "vis"].append(g)
ball = m.geom("ball/ball_geom").id
floor = m.geom("terrain").id if name(mujoco.mjtObj.mjOBJ_GEOM, 0) == "terrain" else next(
    g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
)
short = lambda b: name(mujoco.mjtObj.mjOBJ_BODY, b).removeprefix("robot/").removesuffix("_link")
pairs = [(a, b) for a, b in combinations(bodies, 2) if m.body_parentid[a] != b and m.body_parentid[b] != a]
feet = [b for b in bodies if short(b).endswith("ankle_roll")]


def dist(ga, gb) -> float:
    """Smallest signed distance between two geom lists (inf if either is empty)."""
    return min(
        (mujoco.mj_geomDistance(m, d, a, b, 0.05, None) for a in ga for b in gb), default=np.inf
    )


def measure():
    out = {}
    for a, b in pairs:
        out[(short(a), short(b))] = (dist(geoms[a]["vis"], geoms[b]["vis"]), dist(geoms[a]["col"], geoms[b]["col"]))
    for f in feet:
        for other, label in ((ball, "BALL"), (floor, "FLOOR")):
            out[(short(f), label)] = (dist(geoms[f]["vis"], [other]), dist(geoms[f]["col"], [other]))
    return out


mujoco.mj_resetDataKeyframe(m, d, 0) if m.nkey else mujoco.mj_resetData(m, d)
mujoco.mj_kinematics(m, d)
home = measure()

idx = torch.randperm(len(dump["qpos"]), generator=torch.Generator().manual_seed(0))[: args.samples]
keys = list(home)
vis = np.empty((len(idx), len(keys)))
col = np.empty_like(vis)
for n, i in enumerate(idx.tolist()):
    d.qpos[:] = dump["qpos"][i].numpy()
    m.geom_size[ball, 0] = m.geom_rbound[ball] = float(dump["radius"][i])
    mujoco.mj_kinematics(m, d)
    for k, (v, c) in enumerate(measure().values()):
        vis[n, k], col[n, k] = v, c

print(f"\n{len(idx)} states; a pair intersects when the distance is below -{args.tol * 1000:.0f} mm.")
print("Bodies without any collision primitive: "
      + ", ".join(short(b) for b in bodies if not geoms[b]["col"]))
header = f"{'pair':<40} {'meshes intersect':>17} {'deepest':>9} {'primitives intersect':>21} {'deepest':>9}  note"
for title, at_home in (("Pairs apart at the home pose", False), ("Pairs whose mesh hulls already overlap at the home pose", True)):
    rows = []
    for k, key in enumerate(keys):
        if (home[key][0] < -args.tol) != at_home:
            continue
        v_share = (vis[:, k] < -args.tol).mean()
        if v_share < 0.005 and not (col[:, k] < -args.tol).mean() >= 0.005:
            continue
        has_col = np.isfinite(col[:, k]).all()
        c_share = (col[:, k] < -args.tol).mean() if has_col else 0.0
        note = "no collision primitive on one body" if not has_col else (
            "meshes larger than primitives" if v_share > c_share + 0.01 else "")
        rows.append((v_share, f"{key[0] + ' - ' + key[1]:<40} {v_share:>17.1%} {vis[:, k].min() * 1000:>7.0f}mm "
                              f"{c_share:>21.1%} {(col[:, k].min() * 1000 if has_col else float('nan')):>7.0f}mm  {note}"))
    print(f"\n{title}\n{header}")
    for _, r in sorted(rows, reverse=True):
        print(r)
