"""`-LegGap` task: a penalty for a right leg that is close to the left leg (docs/mount_selfcontact).

Base: `piplus_ball_mount_curriculum_env_cfg` with the `-NoTap` settings (full thigh range). The
observations, the actions, the model and the network are the same, so the `cur4c` actor and critic
load and the dynamics do not change. The penalty is a reward term only. The actor never reads it,
so the policy runs on the robot without it.

The probe `scripts/eval_selfcontact.py` showed that the feet and calves of the right and left leg
are in contact in about half of the steps of the stance (mean normal force 47 N, `cur8`). The
contact is a brush, but the policy uses it as a prop. A contact penalty alone gives no signal
before the touch, so the term uses the distance up to LEG_GAP_MARGIN.

The distance is computed on the GPU from the poses of the collision primitives of the calf, the
ankle pitch link and the ankle roll link (boxes and capsules). Two other ways failed: a margin on
copies of the primitives is not supported by mujoco_warp with box-box contacts (MULTICCD), and
`geomdist` sensors made a control step 170 times slower (docs/mount_selfcontact). Here, points
are sampled on the surface of each shape of one leg (the 8 corners, 12 edge centers and 6 face
centers of a box, 5 points on the axis of a capsule). The term takes the exact signed distance of
each point to each shape of the other leg (box or capsule, less the capsule radius for points on a
capsule axis), both ways. The result is the distance between the shapes, to within the point
spacing (about 1 cm at the box edges, not above the true distance).
"""

import mujoco
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from mjlab_piplus.tasks.ball_mount_min.piplus_ball_mount_curriculum_cfg import (
    EARLY_ACTION_EXCESS,
    STEPS_PER_ITER,
    piplus_ball_mount_curriculum_env_cfg,
    ramped_stage_settings,
)

LEG_GAP_MARGIN = 0.02  # m. About 1.5 cm of foot position error from the encoder bias, plus latency
LEG_GAP_WEIGHT = -4.0  # per second at a gap of 0; see the section "Design" of the design doc
LEG_GAP_CAP = 2.0  # largest value of the term per step (overlap up to one margin)
LEG_LINKS = ("calf", "ankle_pitch", "ankle_roll")  # the thighs are far apart (> 6 cm, probe)
CAPSULE_POINTS = 5  # points along the axis of a capsule


def _box_points(half: torch.Tensor) -> torch.Tensor:
    """[26, 3] corners, edge centers and face centers of a box with half sizes `half`."""
    grid = torch.cartesian_prod(*[torch.tensor([-1.0, 0.0, 1.0])] * 3)
    return grid[grid.abs().sum(-1) > 0] * half


class LegShapes:
    """Collision primitives of one leg: geom ids, types, sizes, and the sample points in the local
    frame of their geom. Built once from the compiled model."""

    def __init__(self, model: mujoco.MjModel, side: str, device: str):
        ids = [
            g
            for g in range(model.ngeom)
            if model.geom(g).name.startswith(f"robot/{side}_")
            and "_collision" in model.geom(g).name
            and any(f"_{link}_" in model.geom(g).name for link in LEG_LINKS)
        ]
        self.ids = torch.tensor(ids, device=device)
        self.is_box = [bool(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX) for g in ids]
        self.size = torch.tensor(
            [model.geom_size[g] for g in ids], dtype=torch.float32, device=device
        )
        pts, owner, offset = [], [], []
        for i, g in enumerate(ids):
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                p, r = _box_points(torch.tensor(model.geom_size[g], dtype=torch.float32)), 0.0
            else:  # capsule: axis along the local z, half length size[1]
                z = torch.linspace(-1.0, 1.0, CAPSULE_POINTS) * float(model.geom_size[g][1])
                p, r = (
                    torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], -1),
                    float(model.geom_size[g][0]),
                )
            pts.append(p)
            owner += [i] * len(p)
            offset += [r] * len(p)
        self.points = torch.cat(pts).to(device)  # [K, 3] local to the owner geom
        self.owner = torch.tensor(owner, device=device)  # [K]
        self.offset = torch.tensor(
            offset, dtype=torch.float32, device=device
        )  # [K] radius to subtract


def _signed_distance(local: torch.Tensor, is_box: bool, size: torch.Tensor) -> torch.Tensor:
    """Signed distance of points `local` [B, K, 3] (in the frame of one shape) to that shape: a box
    with half sizes `size[:3]` or a capsule with radius `size[0]` and half length `size[1]`."""
    if is_box:
        q = local.abs() - size
        return q.clamp_min(0).norm(dim=-1) + q.amax(dim=-1).clamp_max(0)
    axis = torch.zeros_like(local)
    axis[..., 2] = local[..., 2].clamp(-size[1], size[1])
    return (local - axis).norm(dim=-1) - size[0]


def _one_way(pose_a, shapes_a: LegShapes, pose_b, shapes_b: LegShapes) -> torch.Tensor:
    """[B] smallest distance from the sample points of leg A to the shapes of leg B."""
    xpos_a, xmat_a = pose_a
    xpos_b, xmat_b = pose_b
    world = (
        torch.einsum("bkij,kj->bki", xmat_a[:, shapes_a.owner], shapes_a.points)
        + xpos_a[:, shapes_a.owner]
    )
    best = torch.full(world.shape[:2], 1e3, device=world.device)
    for j in range(len(shapes_b.ids)):
        local = torch.einsum("bji,bkj->bki", xmat_b[:, j], world - xpos_b[:, j : j + 1])
        d = _signed_distance(local, shapes_b.is_box[j], shapes_b.size[j])
        best = torch.minimum(best, d)
    return (best - shapes_a.offset).amin(dim=-1)


def leg_distance(env) -> torch.Tensor:
    """[B] smallest signed distance (m) between a collision primitive of the right leg and one of
    the left leg (calf, ankle pitch, ankle roll). Negative: the primitives overlap."""
    if not hasattr(env, "_leg_shapes"):
        env._leg_shapes = tuple(LegShapes(env.sim.mj_model, s, env.device) for s in ("r", "l"))
    right, left = env._leg_shapes
    data = env.sim.data
    poses = []
    for legs in (right, left):
        xpos = data.geom_xpos[:, legs.ids]
        xmat = data.geom_xmat[:, legs.ids].reshape(env.num_envs, len(legs.ids), 3, 3)
        poses.append((xpos, xmat))
    return torch.minimum(
        _one_way(poses[0], right, poses[1], left), _one_way(poses[1], left, poses[0], right)
    )


def leg_gap(env, margin: float) -> torch.Tensor:
    """Right leg close to the left leg [B] per step (give it a negative weight, per second). With
    d = `leg_distance` (m), the term is (margin - d) / margin: 0 from `margin` on, 1 at touch, up
    to LEG_GAP_CAP when the primitives overlap."""
    return ((margin - leg_distance(env)) / margin).clamp(0.0, LEG_GAP_CAP)


def leg_gap_curriculum(env, env_ids) -> dict[str, torch.Tensor]:
    """Set the weight of `leg_gap` with the ramp of stage 8 (the ramp of the other stage-8 terms).
    It runs after `stage_curriculum`, which stays unchanged: a change in its source changes the
    fingerprints of the old tasks. The ramp factor is the ramped weight of `support_foot_heading`
    (0 before stage 8, 1.0 at its end), as in `turn_curriculum`."""
    t0 = env.common_step_counter if getattr(env, "_cur_t0", None) is None else env._cur_t0
    iters = (env.common_step_counter - t0) / STEPS_PER_ITER
    stage = getattr(env, "_cur_stage", 0)
    f = ramped_stage_settings(stage, iters)["weights"]["support_foot_heading"]
    env.reward_manager.get_term_cfg("leg_gap").weight = LEG_GAP_WEIGHT * f
    return {"leg_gap_weight": torch.tensor(LEG_GAP_WEIGHT * f)}


def piplus_ball_mount_selfcontact_env_cfg(play: bool = False, **kwargs) -> ManagerBasedRlEnvCfg:
    """`-NoTap` task (full thigh range) plus the reward term `leg_gap` (weight 0 before stage 8)
    and the curriculum term `leg_gap_curriculum`. `kwargs` go to
    `piplus_ball_mount_curriculum_env_cfg` (`start_stage` and so on)."""
    cfg = piplus_ball_mount_curriculum_env_cfg(
        play=play, straight=False, ramp=True, excess0=EARLY_ACTION_EXCESS, notap=True, **kwargs
    )
    cfg.rewards["leg_gap"] = RewardTermCfg(
        func=leg_gap,
        weight=LEG_GAP_WEIGHT if play else 0.0,
        params={"margin": LEG_GAP_MARGIN},
    )
    if not play:
        cfg.curriculum["leg_gap"] = CurriculumTermCfg(func=leg_gap_curriculum)
    return cfg
