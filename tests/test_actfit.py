"""The `-ActFit` task (docs/mount_actuator_fit): leg dynamics, PD gain scales and lag differ from
`-LegGap`; everything the actor sees and is paid for is the same.

GPU-free: the configs and the spec load without a simulation.
"""

import re

import pytest

from mjlab_piplus.robot.piplus_constants import (
    ACTUATOR_5036,
    LEG_DYNAMICS,
    get_spec_with_arms,
    get_spec_with_leg_dynamics,
)
from mjlab_piplus.tasks.ball_mount_min import piplus_ball_mount_actfit_cfg as af

PREFIX = "Mjlab-Piplus-"
LEGGAP = PREFIX + "Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap"
ACTFIT = LEGGAP + "-ActFit"
NEW_EVENTS = ["leg_damping", "leg_friction", "hip_armature", "distal_armature"]


def _cfg(task, play=False):
    from mjlab.tasks.registry import load_env_cfg

    import mjlab_piplus.tasks  # noqa: F401  (registers the tasks)

    return load_env_cfg(task, play=play)


def _lags(cfg):
    acts = cfg.scene.entities["robot"].articulation.actuators
    return {(a.delay_min_lag, a.delay_max_lag) for a in acts}


@pytest.mark.parametrize("play", [False, True])
def test_observations_rewards_and_actions_are_unchanged(play):
    a, b = _cfg(LEGGAP, play), _cfg(ACTFIT, play)
    for group in ("actor", "critic"):
        assert list(a.observations[group].terms) == list(b.observations[group].terms)
        for name, term in a.observations[group].terms.items():
            assert repr(term.noise) == repr(b.observations[group].terms[name].noise), name
    assert list(a.rewards) == list(b.rewards)
    for k in a.rewards:
        assert a.rewards[k].weight == b.rewards[k].weight, k
    assert list(a.terminations) == list(b.terminations)
    assert a.actions["joint_pos"].clip == b.actions["joint_pos"].clip
    assert list(b.events) == list(a.events) + NEW_EVENTS


def test_gain_scales_and_lag():
    for play in (False, True):
        a, b = _cfg(LEGGAP, play), _cfg(ACTFIT, play)
        assert b.events["pd_gains"].params["kp_range"] == af.KP_SCALE
        assert b.events["pd_gains"].params["kd_range"] == af.KD_SCALE
        assert a.events["pd_gains"].params["kp_range"] == (0.8, 1.2)  # base not mutated
    assert _lags(_cfg(ACTFIT, play=True)) == {af.LAG_FIT}
    train = _cfg(ACTFIT)
    assert list(train.curriculum) == list(_cfg(LEGGAP).curriculum) + ["lag_cap"]
    assert train.curriculum["lag_cap"].params == {"hi": af.LAG_FIT[1]}
    assert _lags(train) == _lags(_cfg(LEGGAP))  # buffer size unchanged; the cap acts at runtime
    assert af.LAG_FIT == (2, 5)  # 10-25 ms at 5 ms physics steps


def _dofs(model, legs: bool) -> list[int]:
    leg = re.compile("|".join(ACTUATOR_5036.target_names_expr))
    return [
        model.jnt_dofadr[i]
        for i in range(model.njnt)
        if model.joint(i).name and bool(leg.fullmatch(model.joint(i).name)) == legs
    ]


FIELDS = ("dof_damping", "dof_armature", "dof_frictionloss")


@pytest.mark.parametrize("dynamics", ["vendor", "fit_2026_11"])
def test_spec_has_the_leg_dynamics_and_unchanged_other_joints(dynamics):
    base = get_spec_with_arms().compile()
    model = get_spec_with_leg_dynamics(get_spec_with_arms, dynamics).compile()
    want = LEG_DYNAMICS[dynamics]
    legs = _dofs(model, legs=True)
    assert len(legs) == 12
    for field, key in zip(FIELDS, ("damping", "armature", "frictionloss")):
        assert getattr(model, field)[legs] == pytest.approx(want[key]), field
        assert getattr(base, field)[legs] == pytest.approx(LEG_DYNAMICS["vendor"][key]), field
        others = _dofs(model, legs=False)
        assert (getattr(model, field)[others] == getattr(base, field)[others]).all(), field


def test_no_other_task_uses_the_fitted_spec():
    from mjlab.tasks.registry import list_tasks

    for task in list_tasks():
        if task.startswith(PREFIX) and task != ACTFIT:
            fn = _cfg(task).scene.entities["robot"].spec_fn
            assert getattr(fn, "func", fn) is not get_spec_with_leg_dynamics, task
