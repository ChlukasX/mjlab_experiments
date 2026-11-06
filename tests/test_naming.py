"""Tests for `mjlab_piplus.naming.log_group` against the live task registry."""

import pytest

from mjlab_piplus.naming import LOG_GROUP_ALIASES, log_group


@pytest.fixture(scope="module")
def tasks():
    from mjlab_piplus.cli import piplus_tasks

    return piplus_tasks()


def test_log_group_pattern():
    assert log_group("Mjlab-Piplus-Upright") == "piplus_upright"
    assert log_group("Mjlab-Piplus-Ball-MountMin-Clip") == "piplus_ball_mountmin_clip"


def test_log_group_alias():
    assert log_group("Mjlab-Piplus-Ball-Small-Turf") == "piplus_ball_small"


def test_log_group_rejects_foreign_task():
    with pytest.raises(ValueError):
        log_group("Mjlab-Velocity-Flat-Unitree-G1")


def test_log_group_equals_experiment_name_of_every_task(tasks):
    from mjlab.tasks.registry import load_rl_cfg

    differs = {t: (log_group(t), load_rl_cfg(t).experiment_name) for t in tasks}
    differs = {t: v for t, v in differs.items() if v[0] != v[1]}
    assert differs == {}


def test_aliases_name_registered_tasks_only(tasks):
    assert set(LOG_GROUP_ALIASES) <= set(tasks)
