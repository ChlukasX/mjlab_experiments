"""`-ActFit` task: leg actuator dynamics and loop latency fit to the robot (docs/mount_actuator_fit).

Base: `piplus_ball_mount_selfcontact_env_cfg` (the `-LegGap` task). The observations, the
rewards, the actions and the network are the same, so the `cur4c` actor and critic load. The
values come from the RoboCup 2026 game rosbags (68 bags, 5 robots):

- Leg joints: passive damping, armature and Coulomb frictionloss from `LEG_DYNAMICS["fit_2026_11"]`
  (`get_spec_with_leg_dynamics`), each randomized per environment at startup.
- PD gains: the effective kp is 0.82-0.97 and the effective kd 0.69-0.92 of the commanded value,
  so the scale ranges are wider below 1.
- Loop latency: the rosbags show 12-18 ms from command to torque, never 40 ms. The curriculum
  lag range is capped at LAG_FIT[1] physics steps (25 ms); play uses LAG_FIT.

The torque-speed curve (full torque up to about 3 rad/s, about 0 at 8 rad/s) is not modelled here.
"""

from dataclasses import replace
from functools import partial

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_piplus.robot.piplus_constants import get_spec_with_leg_dynamics
from mjlab_piplus.tasks.ball_mount_min.piplus_ball_mount_selfcontact_cfg import (
    piplus_ball_mount_selfcontact_env_cfg,
)
from mjlab_piplus.tasks.ball_mount_min.piplus_ball_mountmin_env_cfg import set_lag

HIP_JOINTS = (".*_hip_pitch_joint", ".*_hip_roll_joint")
DISTAL_JOINTS = (".*_thigh_joint", ".*_calf_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint")
LEG_JOINTS = HIP_JOINTS + DISTAL_JOINTS
DAMPING_RANGE = (0.0, 0.15)  # Nm s/rad; fit 0.0-0.1
FRICTION_RANGE = (0.2, 0.5)  # Nm; fit 0.2-0.5
ARMATURE_HIP = (0.02, 0.044)  # kg m^2; fit 0.023-0.044
ARMATURE_DISTAL = (0.017, 0.025)  # kg m^2; fit 0.017-0.023
KP_SCALE = (0.75, 1.1)  # of the commanded kp; fit 0.82-0.97
KD_SCALE = (0.65, 1.1)  # of the commanded kd; fit 0.69-0.92
LAG_FIT = (2, 5)  # physics steps of 5 ms: 10-25 ms


def _dr(func, joints: tuple[str, ...], ranges: tuple[float, float]) -> EventTermCfg:
    """Startup randomization of one joint field on `joints` (absolute values)."""
    return EventTermCfg(
        func=func,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=joints), "ranges": ranges},
    )


def lag_cap(env, env_ids: torch.Tensor | None, hi: int) -> dict[str, torch.Tensor]:
    """Cap the lag range that the stage curriculum set at `hi` physics steps. Runs after "stage"."""
    for act in env.scene["robot"].actuators:
        buf = getattr(act, "_delay_buffer", None)
        if buf is not None:
            set_lag(env, min(buf.min_lag, hi), min(buf.max_lag, hi))
            return {"lag_max": torch.tensor(float(min(buf.max_lag, hi)))}
    return {}


def piplus_ball_mount_actfit_env_cfg(play: bool = False, **kwargs) -> ManagerBasedRlEnvCfg:
    """`-LegGap` task with the fitted leg dynamics, their randomization, the fitted PD gain
    scales and the lag capped at LAG_FIT. `kwargs` go to `piplus_ball_mount_selfcontact_env_cfg`."""
    cfg = piplus_ball_mount_selfcontact_env_cfg(play=play, **kwargs)
    robot = cfg.scene.entities["robot"]
    robot.spec_fn = partial(get_spec_with_leg_dynamics, robot.spec_fn)
    cfg.events["leg_damping"] = _dr(dr.joint_damping, LEG_JOINTS, DAMPING_RANGE)
    cfg.events["leg_friction"] = _dr(dr.joint_friction, LEG_JOINTS, FRICTION_RANGE)
    cfg.events["hip_armature"] = _dr(dr.joint_armature, HIP_JOINTS, ARMATURE_HIP)
    cfg.events["distal_armature"] = _dr(dr.joint_armature, DISTAL_JOINTS, ARMATURE_DISTAL)
    cfg.events["pd_gains"].params["kp_range"] = KP_SCALE
    cfg.events["pd_gains"].params["kd_range"] = KD_SCALE
    if play:
        # New articulation object: the base one may be shared with other tasks.
        robot.articulation = replace(
            robot.articulation,
            actuators=tuple(
                replace(a, delay_min_lag=LAG_FIT[0], delay_max_lag=LAG_FIT[1])
                for a in robot.articulation.actuators
            ),
        )
    else:
        # The delay buffer keeps its size of LAG_FINAL[1]; only the drawn range is capped.
        cfg.curriculum["lag_cap"] = CurriculumTermCfg(func=lag_cap, params={"hi": LAG_FIT[1]})
    return cfg
