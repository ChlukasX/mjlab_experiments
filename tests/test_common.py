"""Old import paths of the names moved to `mjlab_piplus.common` still resolve (phase 8)."""

import importlib

import pytest

TASKS = "mjlab_piplus.tasks"
COMMON = "mjlab_piplus.common"

BALL_NAMES = (
    "BALL_RADIUS",
    "BALL_MASS",
    "BALL_TOP",
    "ROBOT_SPAWN_Z",
    "BASE_HEIGHT_TARGET",
    "get_ball_spec",
    "ball_state_obs",
    "ball_control",
    "ball_escaped",
    "robot_off_ball",
)

# (old module, common module, moved names)
MOVED = (
    ("ball.piplus_ball_env_cfg", "balls", BALL_NAMES),
    ("ball_arms.piplus_ball_arms_env_cfg", "balls", BALL_NAMES),
    ("ball.piplus_ball_env_cfg", "rewards", ("base_height_gauss", "xy_centering")),
    ("ball_arms.piplus_ball_arms_env_cfg", "rewards", ("base_height_gauss", "xy_centering")),
    (
        "noninertial.piplus_noninertial_env_cfg",
        "rewards",
        ("base_height_gauss", "foot_flatness", "xy_drift_gauss"),
    ),
    (
        "platform.piplus_platform_env_cfg",
        "rewards",
        ("base_height_gauss", "xy_drift_gauss"),
    ),
    ("upright.piplus_upright_env_cfg", "rewards", ("foot_flatness",)),
    (
        "locomotion.piplus_locomotion_env_cfg",
        "locomotion",
        ("FOOT_SENSOR_NAME", "VEL_CMD_NAME"),
    ),
    (
        "locomotion_arms.piplus_locomotion_arms_env_cfg",
        "locomotion",
        ("FOOT_SENSOR_NAME", "VEL_CMD_NAME"),
    ),
)

CASES = [(old, common, name) for old, common, names in MOVED for name in names]


@pytest.mark.parametrize(
    ("old", "common", "name"), CASES, ids=[f"{o.split('.')[0]}-{n}" for o, _, n in CASES]
)
def test_old_path_gives_the_common_object(old: str, common: str, name: str) -> None:
    from_old = getattr(importlib.import_module(f"{TASKS}.{old}"), name)
    from_common = getattr(importlib.import_module(f"{COMMON}.{common}"), name)
    assert from_old is from_common


def test_platform_keeps_its_own_variants() -> None:
    """Same name, different body or value: these stay in `platform` and are not merged."""
    platform = importlib.import_module(f"{TASKS}.platform.piplus_platform_env_cfg")
    balls = importlib.import_module(f"{COMMON}.balls")
    rewards = importlib.import_module(f"{COMMON}.rewards")
    assert platform.foot_flatness is not rewards.foot_flatness
    assert platform.ROBOT_SPAWN_Z != balls.ROBOT_SPAWN_Z
    assert platform.BASE_HEIGHT_TARGET != balls.BASE_HEIGHT_TARGET
