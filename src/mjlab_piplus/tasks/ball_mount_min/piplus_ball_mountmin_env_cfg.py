"""Mount with a pruned reward set and a staged stance target.

Mounts only, randomized ball (0.07 to 0.11 m). The reward terms that earlier rounds showed
to have no effect are removed, the free foot's lift is measured at the sole (the ankle
height is passed on tiptoe), and the stance reward starts at a small clearance that is
raised in later stages. See docs/ball_mount_rewards/ball_mount_rewards.md.
"""

from dataclasses import replace

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_piplus.tasks.ball_mount.events import ball_rolling_resistance
from mjlab_piplus.tasks.ball_mount.piplus_ball_mount_env_cfg import BALL_SIZES, ball_pos_until_on_ball
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
) -> ManagerBasedRlEnvCfg:
    """`lifted_clearance`: m the free sole must clear the floor for the stance reward
    (stage 1: 0.03, final: 0.10). `locked`: ball held by rolling resistance all training
    (feasibility probe; play and eval use a free ball). `blind_on_ball`: the actor sees
    the ball position only until it is on the ball (base height), with a per-episode
    perception error, and balances on proprioception from then on (as on the robot).
    `latency`: 10-40 ms sensor-to-motor delay (ramped in from 0-15 ms during training) and
    the encoder bias widened and visible to the actor. `big_ball`: ball radius from
    BIG_BALL_MIN_RADIUS (size 3-5); mass and inertia keep following the radius."""
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
    (entropy 0.003, 64 steps, gamma 0.995)."""
    if bounded_actions:
        cfg = piplus_ball_mountcurr_ppo_runner_cfg("")
    else:
        cfg = piplus_ball_mountbalance_ppo_runner_cfg(100)
    cfg.experiment_name = "piplus_ball_mountmin" + suffix
    return cfg
