"""The `-ImuNoise` task (docs/mount_imu_noise): only the two IMU noise ranges differ from `-LegGap`.

GPU-free: the configs load without a simulation.
"""

import pytest

from mjlab_piplus.tasks.ball_mount_min import piplus_ball_mount_imunoise_cfg as im

PREFIX = "Mjlab-Piplus-"
LEGGAP = PREFIX + "Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap"
IMUNOISE = LEGGAP + "-ImuNoise"
IMU_TERMS = ("projected_gravity", "base_ang_vel")


def _cfg(task, play=False):
    from mjlab.tasks.registry import load_env_cfg

    import mjlab_piplus.tasks  # noqa: F401  (registers the tasks)

    return load_env_cfg(task, play=play)


def _range(cfg, name):
    noise = cfg.observations["actor"].terms[name].noise
    return noise.n_min, noise.n_max


@pytest.mark.parametrize("play", [False, True])
def test_only_the_imu_noise_ranges_differ(play):
    a, b = _cfg(LEGGAP, play), _cfg(IMUNOISE, play)
    for group in ("actor", "critic"):
        assert list(a.observations[group].terms) == list(b.observations[group].terms)
    for name, term in a.observations["actor"].terms.items():
        other = b.observations["actor"].terms[name]
        if name in IMU_TERMS:
            continue
        assert repr(term.noise) == repr(other.noise), name  # joint_pos, joint_vel, ball_pos
        assert term.params == other.params, name
    for name in IMU_TERMS:
        want = im.IMU_NOISE[name]
        assert _range(b, name) == (-want, want)
        assert _range(b, name)[1] < _range(a, name)[1]  # lower than the base task
    # the critic has no noise in either task, and the enable flag is the same
    for name, term in b.observations["critic"].terms.items():
        assert repr(term.noise) == repr(a.observations["critic"].terms[name].noise), name
    assert a.observations["actor"].enable_corruption == b.observations["actor"].enable_corruption
    assert a.observations["actor"].history_length == b.observations["actor"].history_length == 5


def test_rewards_curriculum_events_and_model_are_unchanged():
    for play in (False, True):
        a, b = _cfg(LEGGAP, play), _cfg(IMUNOISE, play)
        assert list(a.rewards) == list(b.rewards)
        for k in a.rewards:
            assert a.rewards[k].weight == b.rewards[k].weight, k
        assert list(a.curriculum) == list(b.curriculum)
        assert list(a.events) == list(b.events)
        assert list(a.terminations) == list(b.terminations)
        assert a.events["encoder_bias"].params == b.events["encoder_bias"].params
        assert a.actions["joint_pos"].clip == b.actions["joint_pos"].clip
        assert a.scene.entities["robot"].spec_fn is b.scene.entities["robot"].spec_fn
    assert (_cfg(LEGGAP).decimation, _cfg(LEGGAP).sim.mujoco.timestep) == (
        _cfg(IMUNOISE).decimation,
        _cfg(IMUNOISE).sim.mujoco.timestep,
    )


def test_amounts_are_below_the_base_values_and_positive():
    assert 0 < im.IMU_NOISE_GRAVITY < 0.05
    assert 0 < im.IMU_NOISE_ANG_VEL < 0.2


def test_no_other_task_has_the_lower_noise():
    from mjlab.tasks.registry import list_tasks

    for task in list_tasks():
        if task.startswith(PREFIX) and task != IMUNOISE:
            cfg = _cfg(task)
            terms = cfg.observations["actor"].terms
            if "projected_gravity" in terms and terms["projected_gravity"].noise is not None:
                assert _range(cfg, "projected_gravity")[1] != im.IMU_NOISE_GRAVITY, task
