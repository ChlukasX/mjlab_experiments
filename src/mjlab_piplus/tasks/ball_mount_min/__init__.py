from functools import partial

from mjlab.tasks.registry import register_mjlab_task

from .piplus_ball_mount_actfit_cfg import piplus_ball_mount_actfit_env_cfg
from .piplus_ball_mount_curriculum_cfg import (
    EARLY_ACTION_EXCESS,
    MAX_ACTION_STD,
    MID_THIGH_LIMIT,
    piplus_ball_mount_curriculum_env_cfg,
    piplus_ball_mount_curriculum_ppo_runner_cfg,
)
from .piplus_ball_mount_imunoise_cfg import piplus_ball_mount_imunoise_env_cfg
from .piplus_ball_mount_selfcontact_cfg import piplus_ball_mount_selfcontact_env_cfg
from .piplus_ball_mountmin_env_cfg import (
    piplus_ball_mountmin_env_cfg,
    piplus_ball_mountmin_ppo_runner_cfg,
)

# Each variant adds to the one before it (see piplus_ball_mountmin_env_cfg for the flags).
_BLIND = dict(bounded_actions=True, blind_on_ball=True)
_LAT = dict(_BLIND, latency=True)
_BIG = dict(_LAT, big_ball=True)
_ONE = dict(_BIG, one_foot=True)
_LEVEL = dict(_ONE, level=True)
_RAMP = dict(_LEVEL, noise_ramp=True)
_HOLD = dict(_RAMP, latch_fix=True, start_hold=True)
_KIN = dict(_HOLD, kin_ball=True)
_LEG = dict(_KIN, free_leg=True)
_STRAIGHT = dict(_LEG, straight=True)
_GATE = dict(_STRAIGHT, gate=True, deadline=True)
# Task suffix of the -Hold variant. The longer suffixes are built from it, so that no line of
# the table exceeds 100 characters.
_HOLD_SUFFIX = "-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp-Latch45-Hold"

# Task suffix -> flags of piplus_ball_mountmin_env_cfg.
VARIANTS = {
    "": {},
    "-Clip": dict(bounded_actions=True),
    "-Locked": dict(locked=True),
    "-BlindOnBall": _BLIND,
    "-BlindOnBall-Latency": _LAT,
    "-BlindOnBall-Latency-Big": _BIG,
    "-BlindOnBall-Latency-Big-OneFoot": _ONE,
    "-BlindOnBall-Latency-Big-OneFoot-Level": _LEVEL,
    "-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp": _RAMP,
    _HOLD_SUFFIX: _HOLD,
    _HOLD_SUFFIX + "-KinBall": _KIN,
    _HOLD_SUFFIX + "-KinBall-FreeLeg": _LEG,
    _HOLD_SUFFIX + "-KinBall-FreeLeg-Straight": _STRAIGHT,
    _HOLD_SUFFIX + "-KinBall-FreeLeg-Straight-Gate-Deadline": _GATE,
    _HOLD_SUFFIX + "-Deadline": dict(_HOLD, deadline=True),
}

for suffix, flags in VARIANTS.items():
    env_cfg = partial(piplus_ball_mountmin_env_cfg, **flags)
    register_mjlab_task(
        task_id=f"Mjlab-Piplus-Ball-MountMin{suffix}",
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=piplus_ball_mountmin_ppo_runner_cfg(
            suffix.lower().replace("-", "_"), flags.get("bounded_actions", False)
        ),
    )

# One cold start through all stages (mount first, then the robustness features);
# docs/mount_curriculum. Task suffix after "-Curriculum" -> (env factory, flags of the factory,
# cap of the action noise std). The suffix also names the runner config.
# -FullThigh: full thigh range (cur1 with +-0.6 rad never mounted).
# -Ramp: stage weights ramped in and the action noise std capped (cur2 collapsed at stage 3).
# -Excess: action_excess from stage 0 (cur2's outputs drifted past the clip, cur3 collapsed).
# -NoTap: the tap is penalised (free sole on the floor while on the ball) and a stance-duration
#   bonus is paid (cur4c taps). Reward change only; warm start from cur4c.
# -MidThigh: thigh limited to +-1.2 rad and no heading reward (cur6 stays straight and taps).
# -Turn: reward for a support foot turned by ~100 deg from the torso (kb1 holds the stance).
# -LegGap: penalty for a right leg close to the left leg (docs/mount_selfcontact).
# -ImuNoise: half the actor noise on the two IMU inputs (docs/mount_imu_noise).
# -ActFit: leg dynamics, PD gain scales and lag (10-25 ms) fit to rosbags (docs/mount_actuator_fit).
_CURRICULUM = piplus_ball_mount_curriculum_env_cfg
_FULL_THIGH = dict(straight=False)
_FULL_THIGH_RAMP = dict(_FULL_THIGH, ramp=True)
_FULL_THIGH_RAMP_EXCESS = dict(_FULL_THIGH_RAMP, excess0=EARLY_ACTION_EXCESS)
_NOTAP = dict(_FULL_THIGH_RAMP_EXCESS, notap=True)
_NOTAP_SUFFIX = "-FullThigh-Ramp-Excess-NoTap"
CURRICULUM_VARIANTS = {
    "": (_CURRICULUM, {}, None),
    "-FullThigh": (_CURRICULUM, _FULL_THIGH, None),
    "-FullThigh-Ramp": (_CURRICULUM, _FULL_THIGH_RAMP, MAX_ACTION_STD),
    "-FullThigh-Ramp-Excess": (_CURRICULUM, _FULL_THIGH_RAMP_EXCESS, MAX_ACTION_STD),
    _NOTAP_SUFFIX: (_CURRICULUM, _NOTAP, MAX_ACTION_STD),
    _NOTAP_SUFFIX + "-MidThigh": (
        _CURRICULUM,
        dict(_NOTAP, thigh_limit=MID_THIGH_LIMIT),
        MAX_ACTION_STD,
    ),
    _NOTAP_SUFFIX + "-Turn": (_CURRICULUM, dict(_NOTAP, turn=True), MAX_ACTION_STD),
    _NOTAP_SUFFIX + "-LegGap": (piplus_ball_mount_selfcontact_env_cfg, {}, MAX_ACTION_STD),
    _NOTAP_SUFFIX + "-LegGap-ImuNoise": (piplus_ball_mount_imunoise_env_cfg, {}, MAX_ACTION_STD),
    _NOTAP_SUFFIX + "-LegGap-ActFit": (piplus_ball_mount_actfit_env_cfg, {}, MAX_ACTION_STD),
}

for suffix, (factory, flags, max_std) in CURRICULUM_VARIANTS.items():
    env_cfg = partial(factory, **flags)
    register_mjlab_task(
        task_id=f"Mjlab-Piplus-Ball-MountMin-Curriculum{suffix}",
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=piplus_ball_mount_curriculum_ppo_runner_cfg(
            suffix.lower().replace("-", "_"), max_std=max_std
        ),
    )
