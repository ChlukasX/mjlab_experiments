"""The `-Turn` task (docs/mount_curriculum): one new reward term, nothing else changes.

GPU-free: the configs load without a simulation, and the term runs on a stand-in env.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from mjlab_piplus.tasks.ball_mount_min import piplus_ball_mount_curriculum_cfg as cur

PREFIX = "Mjlab-Piplus-"
NOTAP = PREFIX + "Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap"
TURN = NOTAP + "-Turn"
TERM = "support_foot_turn"


def _tasks():
    from mjlab.tasks.registry import list_tasks

    import mjlab_piplus.tasks  # noqa: F401  (registers the tasks)

    return [t for t in list_tasks() if t.startswith(PREFIX)]


def _cfg(task, play=False):
    from mjlab.tasks.registry import load_env_cfg

    return load_env_cfg(task, play=play)


def test_only_the_turn_task_has_the_term_and_the_curriculum_term():
    assert TURN in _tasks()
    for task in _tasks():
        for play in (False, True):
            cfg = _cfg(task, play)
            assert (TERM in cfg.rewards) == (task == TURN), task
            assert ("turn" in cfg.curriculum) == (task == TURN and not play), task


def test_old_stage_settings_have_no_turn_weight():
    for s in range(cur.FINAL_STAGE + 1):
        for notap in (False, True):
            assert TERM not in cur.stage_settings(s, 0, notap=notap)["weights"]
            assert TERM not in cur.ramped_stage_settings(s, 0, notap=notap)["weights"]


def test_turn_task_differs_from_notap_only_in_the_term():
    for play in (False, True):
        a, b = _cfg(NOTAP, play), _cfg(TURN, play)
        assert set(b.rewards) - set(a.rewards) == {TERM}
        assert set(b.curriculum) - set(a.curriculum) == ({"turn"} if not play else set())
        for k in a.rewards:
            if k != "support_foot_heading":
                assert a.rewards[k].weight == b.rewards[k].weight, k
    # the thigh range is the full range of -NoTap: no thigh_limit, no spec change, no clip change
    a, b = _cfg(NOTAP), _cfg(TURN)
    assert a.scene.entities["robot"].spec_fn is b.scene.entities["robot"].spec_fn
    assert a.actions["joint_pos"].clip == b.actions["joint_pos"].clip
    assert set(a.observations["actor"].terms) == set(b.observations["actor"].terms)
    assert set(a.observations["critic"].terms) == set(b.observations["critic"].terms)


def test_heading_weight_is_zero_and_the_play_weight_is_the_full_weight():
    for play in (False, True):
        assert _cfg(TURN, play).rewards["support_foot_heading"].weight == 0.0
    assert _cfg(TURN, play=True).rewards[TERM].weight == cur.TURN_WEIGHT > 0
    assert _cfg(TURN).rewards[TERM].weight == 0.0  # the curriculum sets it
    assert _cfg(TURN).rewards[TERM].func is cur.support_foot_turn


def test_turn_needs_the_full_thigh_range_and_the_notap_terms():
    with pytest.raises(AssertionError):
        cur.piplus_ball_mount_curriculum_env_cfg(straight=True, notap=True, turn=True)
    with pytest.raises(AssertionError):
        cur.piplus_ball_mount_curriculum_env_cfg(straight=False, notap=False, turn=True)


def test_the_weight_is_below_the_stance_weight_and_the_target_is_about_100_deg():
    assert 0 < cur.TURN_WEIGHT < 16.0
    assert math.degrees(cur.TURN_TARGET) == pytest.approx(100.0)


def _yaw_quat(deg):
    """Quaternion (w, x, y, z) of a rotation by `deg` about the world z axis."""
    a = math.radians(deg) / 2
    return [math.cos(a), 0.0, 0.0, math.sin(a)]


def _turn_env(foot_deg, torso_deg=0.0, single=True, side=0):
    """Stand-in env: two feet (body 0 right, body 1 left) and a torso (body 2)."""
    pose = torch.zeros(1, 3, 7)
    pose[0, 0, 3:] = torch.tensor(_yaw_quat(foot_deg))
    pose[0, 1, 3:] = torch.tensor(_yaw_quat(foot_deg))
    pose[0, 2, 3:] = torch.tensor(_yaw_quat(torso_deg))
    robot = SimpleNamespace(
        data=SimpleNamespace(body_link_pose_w=pose), body_names=["r_foot", "l_foot", "torso_link"]
    )
    env = SimpleNamespace(num_envs=1, device="cpu", scene={"robot": robot})
    env._single = (torch.tensor([single]), torch.tensor([side]))
    return env


@pytest.fixture
def turn(monkeypatch):
    monkeypatch.setattr(cur, "_single_support", lambda env, f, b, gated: env._single)
    foot_cfg = SimpleNamespace(name="robot", body_ids=[0, 1])
    return lambda env, **kw: float(cur.support_foot_turn(env, foot_cfg, None, **kw)[0])


def test_term_is_one_at_the_target_and_falls_with_the_error(turn):
    assert turn(_turn_env(100.0)) == pytest.approx(1.0)
    assert turn(_turn_env(100.0 + 90.0)) == pytest.approx(0.5)
    assert turn(_turn_env(100.0 - 90.0)) == pytest.approx(0.5)
    assert turn(_turn_env(100.0 + 180.0)) == pytest.approx(0.0, abs=1e-6)
    values = [
        turn(_turn_env(d)) for d in range(-20, 101, 10)
    ]  # from the cur4c posture to the target
    assert all(b > a for a, b in zip(values, values[1:]))  # a gradient the whole way


def test_term_has_a_value_and_a_gradient_at_the_straight_foot_of_cur4c(turn):
    """A clamped cos around 100 deg is 0 at -15 deg (115 deg from the target)."""
    v = turn(_turn_env(-15.0))
    assert v == pytest.approx(0.5 * (1 + math.cos(math.radians(115.0))))
    assert 0.25 < v < 0.35
    assert turn(_turn_env(-5.0)) > v


def test_term_measures_the_foot_against_the_torso_and_wraps(turn):
    assert turn(_turn_env(130.0, torso_deg=30.0)) == pytest.approx(1.0)
    assert turn(_turn_env(100.0 + 360.0)) == pytest.approx(1.0, abs=1e-5)
    assert turn(_turn_env(-260.0)) == pytest.approx(1.0, abs=1e-5)


def test_left_foot_is_mirrored(turn):
    assert turn(_turn_env(-100.0, side=1)) == pytest.approx(1.0)
    assert turn(_turn_env(100.0, side=1)) < 0.1
    assert turn(_turn_env(100.0, side=0)) == pytest.approx(1.0)


def test_term_is_zero_outside_the_gated_single_support(turn):
    assert turn(_turn_env(100.0, single=False)) == 0.0


def test_term_range_is_zero_to_one(turn):
    for d in range(-180, 181, 15):
        for side in (0, 1):
            assert 0.0 <= turn(_turn_env(float(d), side=side)) <= 1.0


def _curriculum_env(stage, iters):
    heading = SimpleNamespace(weight=1.0)
    term = SimpleNamespace(weight=0.0)
    cfgs = {"support_foot_heading": heading, TERM: term}
    env = SimpleNamespace(
        common_step_counter=int(iters * cur.STEPS_PER_ITER) + 1000,
        _cur_t0=1000,
        _cur_stage=stage,
        reward_manager=SimpleNamespace(get_term_cfg=lambda k: cfgs[k]),
    )
    return env, heading, term


def test_turn_curriculum_ramps_the_weight_with_stage_8_and_zeroes_the_heading():
    ramp = cur.STAGE_RAMP_ITERS[cur.FINAL_STAGE]
    for stage in range(cur.FINAL_STAGE):
        env, heading, term = _curriculum_env(stage, 1e5)
        cur.turn_curriculum(env, None)
        assert term.weight == 0.0 and heading.weight == 0.0
    weights = []
    for iters in (0, ramp / 2, ramp, 2 * ramp):
        env, heading, term = _curriculum_env(cur.FINAL_STAGE, iters)
        out = cur.turn_curriculum(env, None)
        assert heading.weight == 0.0
        assert float(out["turn_weight"]) == pytest.approx(term.weight)
        weights.append(term.weight)
    assert weights[0] == 0.0
    assert 0.0 < weights[1] < weights[2]
    assert weights[2] == weights[3] == cur.TURN_WEIGHT


def test_turn_ramp_equals_the_ramp_of_the_other_stage_8_terms():
    """The weight follows the ramp of the `notap` terms (same stage clock, same shape)."""
    ramp = cur.STAGE_RAMP_ITERS[cur.FINAL_STAGE]
    for iters in (0, ramp / 4, ramp / 2, ramp * 0.9, ramp):
        env, _, term = _curriculum_env(cur.FINAL_STAGE, iters)
        cur.turn_curriculum(env, None)
        w = cur.ramped_stage_settings(cur.FINAL_STAGE, iters, notap=True)["weights"]
        assert term.weight / cur.TURN_WEIGHT == pytest.approx(
            w["stance_duration"] / cur.STANCE_BONUS
        )
