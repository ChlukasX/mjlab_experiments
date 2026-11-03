"""Mount with a pruned reward set and a staged stance target.

Mounts only, randomized ball (0.07 to 0.11 m). The reward terms that earlier rounds showed
to have no effect are removed, the free foot's lift is measured at the sole (the ankle
height is passed on tiptoe), and the stance reward starts at a small clearance that is
raised in later stages. See docs/ball_mount_rewards/ball_mount_rewards.md.
"""

from dataclasses import dataclass, fields, replace

import torch
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_piplus.tasks.ball_mount.events import ball_rolling_resistance
from mjlab_piplus.tasks.ball_mount.piplus_ball_mount_env_cfg import (
    BALL_SIZES,
    _ball_radius,
    _support_foot,
    ball_contact_sensors,
    ball_pos_until_on_ball,
)
from mjlab_piplus.tasks.ball_mount.piplus_ball_mountbalance_env_cfg import (
    piplus_ball_mountbalance_env_cfg,
    piplus_ball_mountbalance_ppo_runner_cfg,
)
from mjlab_piplus.tasks.ball_mount_curriculum.piplus_ball_mountcurr_env_cfg import (
    MU_ROLL_LOCKED,
    piplus_ball_mountcurr_ppo_runner_cfg,
)

# No effect in mb3 (momentum), mb5 (penalties); foot_on_ball is contained in
# support_on_ball; xy_centering had weight 0.
_REMOVED = (
    "foot_on_ball", "com_toward_ball_velocity", "push_up_velocity",
    "straddle_penalty", "termination_penalty", "xy_centering",
)
COM_LINEAR_RANGE = 0.30  # m: com_over_ball falls linearly to 0 at the mount spawn distance
# Blind on the ball: base height above which the robot counts as on the ball (standing on
# the floor 0.39 m, up to ~0.43 on tiptoe; on a size 1 ball from ~0.42 crouched, 0.46 tall).
ON_BALL_BASE_HEIGHT = 0.42
ON_BALL_MAX_TILT = 0.35  # rad (20 deg): only an upright robot counts as on the ball
BALL_OBS_BIAS = (0.03, 0.03, 0.01)  # m, per-episode error of the ball filter's estimate
# Latency: sensor-to-motor loop delay as actuator lag in physics steps (5 ms). Ramped from the
# earlier 0-15 ms to 10-40 ms; the real robot's latency is not measured (docs/mount_latency).
LAG_START = (0, 3)
LAG_FINAL = (2, 8)
LAG_RAMP_STEPS = 1500 * 64  # iterations x steps per env (bounded runner)
LAG_HOLD_PROB = 0.9  # per physics step: correlated lag that jumps now and then (jitter)
ENCODER_BIAS = 0.03  # rad; the actor reads the biased angles, as on the robot
# Big balls only (deployment ball size 4-5): size 3-5, the eval's size bins from 0.083 m on.
BIG_BALL_MIN_RADIUS = 0.083
# Raw policy output clamp (bounded tasks). action_rate squares the raw output, so an unbounded
# output made it explode (of1: -577k per episode, policy collapsed). 7 > the largest target
# offset that the joint clip lets through (hip pitch 6.7 at scale 0.5), so no target is lost.
RAW_ACTION_CLIP = 7.0
# Level: support ankle roll above this costs linearly (stop 0.648 rad; of2-clip sat at 0.60,
# where joint_pos_limits' soft limit 0.583 cost almost nothing).
ANKLE_ROLL_SOFT = 0.45  # rad


# Noise curriculum: the actor's observation noise is scaled from NOISE_START to 1 over
# NOISE_RAMP_STEPS (lv1: 85% without noise, 58% with the full noise).
NOISE_START = 0.25
NOISE_RAMP_STEPS = 1500 * 64


def noise_curriculum(env, env_ids, start, ramp_steps, t0):
    """Scale every uniform actor noise range from `start` to 1 over `ramp_steps`, counted from
    step `t0` (a resume restores the step counter: pass the checkpoint's)."""
    f = start + (1.0 - start) * min(max(env.common_step_counter - t0, 0) / ramp_steps, 1.0)
    for term in env.observation_manager._group_obs_term_cfgs["actor"]:
        n = term.noise
        if isinstance(n, UniformNoiseCfg):
            if not hasattr(n, "full_range"):
                n.full_range = (n.n_min, n.n_max)
            n.n_min, n.n_max = n.full_range[0] * f, n.full_range[1] * f
            n._tensor_cache.clear()  # apply() caches the range as tensors
    return {"noise_scale": torch.tensor(f)}


# Latch: "on the ball" (ball input zeroed) needs the base tilt below this. ON_BALL_MAX_TILT
# (0.35 rad) never latched: every policy balances with the base 28-32 deg over (0% of envs
# latched, bob2-up05 to nr1-up6), so all of them saw the ball through the whole balance.
LATCH_MAX_TILT = 0.79  # rad (45 deg); a robot falling sideways still does not count
# Blindness curriculum (latch_fix, training): share of episodes blind on the ball, ramped
# 0 -> 1 over BLIND_RAMP_STEPS from step t0. bl2 (blind from the start) stopped mounting.
BLIND_RAMP_STEPS = 2000 * 64


# Kinematic ball estimate (kin_ball): once latched, the actor gets the ball centre from the
# support foot's pose (kinematic_ball_pos) instead of zeros; bl2-bl4 never balanced blind.
KIN_BALL_BIAS = 0.015  # m per axis, per episode: encoder bias 0.03 rad over the ~0.5 m leg


def blind_curriculum(env, env_ids, ramp_steps, t0):
    env.blind_prob = min(max(env.common_step_counter - t0, 0) / ramp_steps, 1.0)
    return {"blind_prob": torch.tensor(env.blind_prob)}


# Mount deadline: an episode not latched "on the ball" this long after its start ends
# (bl2, bl3: parked on the floor once blind on the ball meant falling).
MOUNT_DEADLINE_S = 3.0  # 1 s start hold + 2 s; the policies are on the ball ~0.6 s after starting


def mount_deadline(env, deadline_s: float) -> torch.Tensor:
    latched = getattr(env, "_on_ball_latch", None)
    late = env.episode_length_buf >= round(deadline_s / env.step_dt)
    return late if latched is None else late & ~latched


# Start hold: after each reset the robot holds the home pose (action 0, also as the policy's
# last-action input) for this long, then the policy takes over; the robot node does the same
# after activation.
START_HOLD_S = 1.0


def _hold_steps(env) -> int:
    return round(START_HOLD_S / env.step_dt)


def start_hold_remaining(env) -> torch.Tensor:
    """Critic: seconds left in the start hold [B, 1]."""
    return ((_hold_steps(env) - env.episode_length_buf).clamp_min(0) * env.step_dt).float().unsqueeze(-1)


class HoldJointPositionAction(JointPositionAction):
    """Joint position action that holds the home pose (raw action 0) during the start hold.
    Zeroes the action manager's copy too, so the policy's last-action input reads 0 there."""

    def process_actions(self, actions: torch.Tensor):
        hold = self._env.episode_length_buf < _hold_steps(self._env)
        if hold.any():
            actions = torch.where(hold.unsqueeze(-1), torch.zeros_like(actions), actions)
            self._env.action_manager._action[hold] = 0.0  # single action term: whole row
        super().process_actions(actions)


@dataclass(kw_only=True)
class HoldJointPositionActionCfg(JointPositionActionCfg):
    def build(self, env) -> HoldJointPositionAction:
        return HoldJointPositionAction(self, env)


# Free leg (free_leg): kb1 holds the free foot tucked in over the ball's edge (ankle 0.99 r from
# the centre, 3 cm out from the support foot; posture's hip roll std 0.10 pins it) and balances
# with the whole support leg (ankle 32% of its joint motion). docs/mount_latency, round 8.
_FEET = ("r_ankle_roll_link", "l_ankle_roll_link")
_LEG = ("hip_roll", "hip_pitch", "thigh", "calf", "ankle_pitch", "ankle_roll")
FREE_CLEAR_NEAR, FREE_CLEAR_FAR = 0.03, 0.10  # m, free ankle to ball surface: 0 at NEAR, 1 from FAR


def _single_support(env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg):
    """([B] bool single support with a level sole, [B] index of the support foot, 0 = right)."""
    sup = _support_foot(env, foot_cfg, ball_cfg, None, "level")
    return sup.sum(-1) == 1, sup[:, 1].long()


class posture_free_leg(envs_mdp.posture):
    """`posture` without the free leg's joints in single support (support leg and arms only), so
    the free leg can swing out as a counterweight; both legs otherwise."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        _, names = env.scene[cfg.params["asset_cfg"].name].find_joints(cfg.params["asset_cfg"].joint_names)
        leg = lambda side: [n.startswith(side) and any(f"_{j}_joint" in n for j in _LEG) for n in names]
        self.free_mask = torch.tensor([leg("l_"), leg("r_")], device=env.device)  # row = support foot

    def __call__(self, env, std, asset_cfg: SceneEntityCfg, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg):
        robot = env.scene[asset_cfg.name]
        err = ((robot.data.joint_pos[:, asset_cfg.joint_ids] - self.default_joint_pos[:, asset_cfg.joint_ids]) / self.std) ** 2
        single, si = _single_support(env, foot_cfg, ball_cfg)
        w = (~(self.free_mask[si] & single.unsqueeze(-1))).float()
        return torch.exp(-(err * w).sum(-1) / w.sum(-1))


def free_foot_ball_clear(env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Single support: free ankle origin's distance to the ball surface, 0 at FREE_CLEAR_NEAR,
    1 from FREE_CLEAR_FAR (kb1: ~2.5 cm, hovering over the edge)."""
    single, si = _single_support(env, foot_cfg, ball_cfg)
    pos = env.scene[foot_cfg.name].data.body_link_pose_w[:, foot_cfg.body_ids, :3]
    free = pos[torch.arange(env.num_envs, device=env.device), 1 - si]
    gap = (free - env.scene[ball_cfg.name].data.root_link_pos_w).norm(dim=-1) - _ball_radius(env, ball_cfg, None)
    return ((gap - FREE_CLEAR_NEAR) / (FREE_CLEAR_FAR - FREE_CLEAR_NEAR)).clamp(0, 1) * single


def support_hip_vel(env, asset_cfg: SceneEntityCfg, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Single support: squared joint velocity of the support leg's hip roll, hip pitch, thigh and
    calf (ankle exempt), so the balance moves to the ankle. `asset_cfg` joints: right leg's four,
    then the left's."""
    single, si = _single_support(env, foot_cfg, ball_cfg)
    qd2 = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids].square().view(env.num_envs, 2, -1).sum(-1)
    return qd2[torch.arange(env.num_envs, device=env.device), si] * single


def ankle_roll_excess(env, asset_cfg: SceneEntityCfg, limit: float) -> torch.Tensor:
    """Sum over the ankle roll joints of |q| beyond `limit` (rad)."""
    q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return (q.abs() - limit).clamp_min(0.0).sum(dim=-1)


def latency_curriculum(
    env, env_ids: torch.Tensor, start: tuple[int, int], final: tuple[int, int], ramp_steps: int
) -> dict[str, torch.Tensor]:
    """Widen the actuator lag range linearly from `start` to `final` over `ramp_steps`.
    A resume restores the step counter from the checkpoint: zero it in the init copy."""
    f = min(env.common_step_counter / ramp_steps, 1.0)
    lo, hi = (round(a + f * (b - a)) for a, b in zip(start, final))
    for act in env.scene["robot"].actuators:
        buf = getattr(act, "_delay_buffer", None)
        if buf is not None:
            buf.min_lag, buf.max_lag = lo, hi
    return {"lag_min": torch.tensor(float(lo)), "lag_max": torch.tensor(float(hi))}


def piplus_ball_mountmin_env_cfg(
    play: bool = False,
    bounded_actions: bool = False,
    lifted_clearance: float = 0.03,
    locked: bool = False,
    blind_on_ball: bool = False,
    latency: bool = False,
    big_ball: bool = False,
    one_foot: bool = False,
    level: bool = False,
    noise_ramp: bool = False,
    latch_fix: bool = False,
    kin_ball: bool = False,
    free_leg: bool = False,
    start_hold: bool = False,
    deadline: bool = False,
) -> ManagerBasedRlEnvCfg:
    """`lifted_clearance`: m the free sole must clear the floor for the stance reward
    (stage 1: 0.03, final: 0.10). `locked`: ball held by rolling resistance all training
    (feasibility probe; play and eval use a free ball). `blind_on_ball`: the actor sees
    the ball position only until it is on the ball (base height), with a per-episode
    perception error, and balances on proprioception from then on (as on the robot).
    `latency`: 10-40 ms sensor-to-motor delay (ramped in from 0-15 ms during training) and
    the encoder bias widened and visible to the actor. `big_ball`: ball radius from
    BIG_BALL_MIN_RADIUS (size 3-5); mass and inertia keep following the radius.
    `one_foot`: the stance rewards count the free foot only while it does not touch the
    ball (contact sensors), against the two-foot stance with a tilted foot on the ball. `level`: the support sole must
    also be near horizontal (foot near the ball top, not on its flank), and the ankle roll
    pays beyond ANKLE_ROLL_SOFT. `noise_ramp`: the actor noise ramps in (noise_curriculum;
    set its `t0` to the resumed checkpoint's step counter). `latch_fix`: on-ball latch at
    LATCH_MAX_TILT, so the robot really goes blind on the ball. `start_hold`: home pose for
    START_HOLD_S after each reset before the policy acts. `deadline`: the episode ends if the
    robot is not latched on the ball MOUNT_DEADLINE_S after its start (training only).
    `kin_ball`: once latched, the kinematic ball estimate instead of zeros (no blindness
    curriculum: the input never goes away). `free_leg`: in single support the free leg is out of
    `posture` and paid for clearing the ball, and the support leg's hip and knee joints cost
    velocity (balance with the ankle)."""
    cfg = piplus_ball_mountbalance_env_cfg(
        play, mount_fraction=1.0, flat_sole=True, bounded_actions=bounded_actions
    )
    r = cfg.rewards
    for k in _REMOVED:
        del r[k]
    r["support_on_ball"].weight = 2.0
    r["com_over_ball"].params["linear_range"] = COM_LINEAR_RANGE
    r["single_leg_stance"].weight = 8.0
    r["single_leg_stance"].params["sole_clearance"] = lifted_clearance
    r["free_foot_lift"].weight = 3.0
    r["free_foot_lift"].params["sole"] = True

    if blind_on_ball:
        term = cfg.observations["actor"].terms["ball_pos"]
        term.func = ball_pos_until_on_ball
        term.params = {**term.params, "height_threshold": ON_BALL_BASE_HEIGHT,
            "bias": BALL_OBS_BIAS, "max_tilt": ON_BALL_MAX_TILT}
        # ponytail: the per-step noise (term.noise, +-2 cm) also hits the zeros once on the
        # ball; harmless as a "no ball" code, since a real value has z ~ -0.3 m.

    if latency:
        # Buffer sized for the final range; play evaluates the final range, training ramps it.
        lo = LAG_FINAL[0] if play else 0
        # New articulation object: the robot cfg's is a module-level constant shared by all tasks.
        robot = cfg.scene.entities["robot"]
        robot.articulation = replace(robot.articulation, actuators=tuple(
            replace(a, delay_min_lag=lo, delay_max_lag=LAG_FINAL[1], delay_hold_prob=LAG_HOLD_PROB)
            for a in robot.articulation.actuators
        ))
        if not play:
            cfg.curriculum["latency"] = CurriculumTermCfg(
                func=latency_curriculum,
                params={"start": LAG_START, "final": LAG_FINAL, "ramp_steps": LAG_RAMP_STEPS},
            )
        cfg.events["encoder_bias"].params["bias_range"] = (-ENCODER_BIAS, ENCODER_BIAS)
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True

    if one_foot:
        cfg.scene.sensors += ball_contact_sensors(cfg.decimation)
        r["single_leg_stance"].params["free_off_ball"] = True
        r["free_foot_lift"].params["free_off_ball"] = True

    if level:
        for term in r.values():
            if term.params.get("flat") is True:
                term.params["flat"] = "level"
        r["sole_flat"].params = {**r["sole_flat"].params, "level": True}  # dict shared with foot_lift
        r["ankle_roll_excess"] = RewardTermCfg(func=ankle_roll_excess, weight=-20.0, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_ankle_roll_joint",)), "limit": ANKLE_ROLL_SOFT})

    if noise_ramp and not play:
        cfg.curriculum["noise"] = CurriculumTermCfg(func=noise_curriculum,
            params={"start": NOISE_START, "ramp_steps": NOISE_RAMP_STEPS, "t0": 0})

    if latch_fix:
        cfg.observations["actor"].terms["ball_pos"].params["max_tilt"] = LATCH_MAX_TILT
        if kin_ball:
            cfg.observations["actor"].terms["ball_pos"].params.update(kin_bias=KIN_BALL_BIAS,
                kinematic=SceneEntityCfg("robot", body_names=("r_ankle_roll_link", "l_ankle_roll_link")))
        elif not play:
            cfg.curriculum["blind"] = CurriculumTermCfg(func=blind_curriculum, params={"ramp_steps": BLIND_RAMP_STEPS, "t0": 0})

    if start_hold:
        a = cfg.actions["joint_pos"]
        cfg.actions["joint_pos"] = HoldJointPositionActionCfg(**{f.name: getattr(a, f.name) for f in fields(a) if f.init})
        cfg.observations["critic"].terms["start_hold"] = ObservationTermCfg(func=start_hold_remaining)

    if deadline and not play:
        cfg.terminations["mount_deadline"] = TerminationTermCfg(func=mount_deadline, params={"deadline_s": MOUNT_DEADLINE_S})

    if free_leg:
        feet, ball = SceneEntityCfg("robot", body_names=_FEET), SceneEntityCfg("ball")
        r["posture"].func = posture_free_leg
        r["posture"].params = {**r["posture"].params, "foot_cfg": feet, "ball_cfg": ball}
        r["free_foot_ball_clear"] = RewardTermCfg(func=free_foot_ball_clear, weight=2.0,
            params={"foot_cfg": feet, "ball_cfg": ball})
        hips = tuple(f"{side}_{j}_joint" for side in ("r", "l") for j in _LEG[:4])
        r["support_hip_vel"] = RewardTermCfg(func=support_hip_vel, weight=-0.01, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=hips, preserve_order=True), "foot_cfg": feet, "ball_cfg": ball})

    if big_ball:
        hi = cfg.events["ball_size"].params["ranges"][1]
        cfg.events["ball_size"].params["ranges"] = (BIG_BALL_MIN_RADIUS / BALL_SIZES[5][0], hi)

    if locked and not play:
        cfg.events["ball_lock"] = EventTermCfg(
            func=ball_rolling_resistance,
            mode="reset",
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "mu_roll_max": MU_ROLL_LOCKED,
                "steps_per_iter": 1,
                "hold_iters": 0,
                "ramp_iters": 0,  # never released
            },
        )
    return cfg


def piplus_ball_mountmin_ppo_runner_cfg(
    suffix: str = "", bounded_actions: bool = False
) -> RslRlOnPolicyRunnerCfg:
    """Unbounded: the runner of the earlier mount runs (entropy 0.01, 24 steps), the only
    setting that learned to balance on the ball. Bounded: the mount curriculum's runner
    (entropy 0.003, 64 steps, gamma 0.995), raw outputs clamped to RAW_ACTION_CLIP."""
    if bounded_actions:
        cfg = piplus_ball_mountcurr_ppo_runner_cfg("")
        cfg.clip_actions = RAW_ACTION_CLIP
    else:
        cfg = piplus_ball_mountbalance_ppo_runner_cfg(100)
    cfg.experiment_name = "piplus_ball_mountmin" + suffix
    return cfg
