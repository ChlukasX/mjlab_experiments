"""The shared ball helpers live in `mjlab_piplus.common` (rules S7 and S15).

Every old name in the task modules must be the very same object as the function or constant in
`common/`, so check C2 (which hashes the callers), old scripts, and old `env.yaml` files see no
change. The first part checks the ball-stance helpers (old private names), the second part the
names moved by the rule of two.
"""

import ast
import importlib
from pathlib import Path

import pytest

COMMON = "mjlab_piplus.common.ball_stance"
OLD = "mjlab_piplus.tasks.ball_mount.piplus_ball_mount_env_cfg"

# (old name in the env module, public name in common)
FUNCTIONS = (
    ("_ball_radius", "ball_radii"),
    ("_foot_on_ball", "foot_on_ball"),
    ("_sole_tilt", "sole_tilt"),
    ("_sole_level_tilt", "sole_level_tilt"),
    ("_sole_clearance", "sole_clearance"),
    ("_ball_contact", "ball_contact"),
    ("_support_foot", "support_foot"),
    ("_com_to_ball", "com_to_ball"),
)
CONSTANTS = (
    "BALL_CONTACT_FORCE",
    "FOOT_BALL_SENSORS",
    "FOOT_ON_BALL_XY_MARGIN",
    "MAX_SOLE_TILT",
    "ON_BALL_Z_CENTER",
    "ON_BALL_Z_HALF",
)


@pytest.mark.parametrize(("old", "new"), FUNCTIONS, ids=[new for _, new in FUNCTIONS])
def test_old_private_name_is_the_common_function(old: str, new: str) -> None:
    common = importlib.import_module(COMMON)
    env_module = importlib.import_module(OLD)
    assert getattr(env_module, old) is getattr(common, new)
    assert not new.startswith("_")


@pytest.mark.parametrize("name", CONSTANTS)
def test_old_constant_is_the_common_constant(name: str) -> None:
    common = importlib.import_module(COMMON)
    env_module = importlib.import_module(OLD)
    assert getattr(env_module, name) is getattr(common, name)


def test_ball_small_margin_comes_from_common() -> None:
    common = importlib.import_module(COMMON)
    small = importlib.import_module("mjlab_piplus.tasks.ball_small.piplus_ball_small_env_cfg")
    assert small.FOOT_ON_BALL_XY_MARGIN is common.FOOT_ON_BALL_XY_MARGIN


def test_common_does_not_import_a_task_module() -> None:
    """`common/` is the bottom layer: a task import would make a cycle."""
    path = Path(importlib.import_module(COMMON).__file__)
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("mjlab_piplus.tasks"), node.module


BALL_MOUNT = "ball_mount.piplus_ball_mount_env_cfg"
MIN_ENV = "ball_mount_min.piplus_ball_mountmin_env_cfg"
MIN_CURRICULUM = "ball_mount_min.piplus_ball_mount_curriculum_cfg"
BALL_SMALL = "ball_small.piplus_ball_small_env_cfg"
CURRICULUM = "ball_mount_curriculum.piplus_ball_mountcurr_env_cfg"

# (old module under tasks/, old name, module under common/, name there)
MOVED = (
    (BALL_MOUNT, "mount_potential", "rewards.mounting", "mount_potential"),
    (BALL_MOUNT, "foot_to_ball_top", "rewards.mounting", "foot_to_ball_top"),
    (BALL_MOUNT, "single_leg_stance_lifted", "rewards.stance", "single_leg_stance_lifted"),
    (BALL_MOUNT, "ball_contact_sensors", "ball_stance", "ball_contact_sensors"),
    (BALL_MOUNT, "ball_pos_base_obs", "observations", "ball_pos_base_obs"),
    (BALL_MOUNT, "ball_pos_until_on_ball", "observations", "ball_pos_until_on_ball"),
    (BALL_MOUNT, "kinematic_ball_pos", "observations", "kinematic_ball_pos"),
    (BALL_MOUNT, "BALL_SIZES", "football", "BALL_SIZES"),
    (BALL_MOUNT, "get_ball_spec", "football", "get_ball_spec"),
    (BALL_MOUNT, "STAND_FOOT_Z", "ball_stance", "STAND_FOOT_Z"),
    (BALL_MOUNT, "LIFTED_FOOT_Z", "ball_stance", "LIFTED_FOOT_Z"),
    (BALL_MOUNT, "FREE_FOOT_LIFT", "ball_stance", "FREE_FOOT_LIFT"),
    (BALL_MOUNT, "COM_OVER_BALL", "ball_stance", "COM_OVER_BALL"),
    (BALL_MOUNT, "COM_SIGMA", "ball_stance", "COM_SIGMA"),
    (BALL_MOUNT, "BASE_ABOVE_TOP_TARGET", "ball_stance", "BASE_ABOVE_TOP_TARGET"),
    ("ball_mount.events", "ball_rolling_resistance", "events", "ball_rolling_resistance"),
    ("ball_mount.events", "reset_from_mount_states", "events", "reset_from_mount_states"),
    ("ball_mount.events", "last_action_restored", "events", "last_action_restored"),
    ("ball_mount.events", "_all_envs", "events", "all_envs"),
    (MIN_CURRICULUM, "free_foot_contact", "rewards.stance", "free_foot_contact"),
    (MIN_CURRICULUM, "FLOOR_TOUCH_CLEARANCE", "ball_stance", "FLOOR_TOUCH_CLEARANCE"),
    (MIN_ENV, "_single_support", "rewards.stance", "single_support"),
    (MIN_ENV, "_yaw", "rewards.stance", "heading"),
    (MIN_ENV, "GATE_FREE_CLEARANCE", "ball_stance", "GATE_FREE_CLEARANCE"),
    (BALL_SMALL, "get_ball_spec", "football", "get_ball_spec"),
    (BALL_SMALL, "BALL_RADIUS", "football", "BALL_RADIUS"),
    (BALL_SMALL, "BALL_MASS", "football", "BALL_MASS"),
    (BALL_SMALL, "ROBOT_SPAWN_X", "football", "ROBOT_SPAWN_X"),
    (BALL_SMALL, "ball_escaped", "balls", "ball_escaped"),
    (BALL_SMALL, "xy_centering", "rewards", "xy_centering"),
    (BALL_MOUNT, "ROBOT_SPAWN_X", "football", "ROBOT_SPAWN_X"),
    ("ball_mount.events", "ROBOT_SPAWN_X", "football", "ROBOT_SPAWN_X"),
    (CURRICULUM, "MU_ROLL_LOCKED", "football", "MU_ROLL_LOCKED"),
    (MIN_ENV, "MU_ROLL_LOCKED", "football", "MU_ROLL_LOCKED"),
)


@pytest.mark.parametrize(
    ("old_module", "old_name", "common_module", "name"),
    MOVED,
    ids=[f"{o.split('.')[0]}-{n}" for o, n, _, _ in MOVED],
)
def test_moved_name_is_the_common_object(
    old_module: str, old_name: str, common_module: str, name: str
) -> None:
    old = getattr(importlib.import_module(f"mjlab_piplus.tasks.{old_module}"), old_name)
    new = getattr(importlib.import_module(f"mjlab_piplus.common.{common_module}"), name)
    assert old is new


def test_rewards_package_reexports_every_category_function() -> None:
    package = importlib.import_module("mjlab_piplus.common.rewards")
    for name in package.__all__:
        found = [
            m
            for m in ("posture", "stance", "mounting")
            if hasattr(importlib.import_module(f"mjlab_piplus.common.rewards.{m}"), name)
        ]
        assert found, name
        assert getattr(package, name) is getattr(
            importlib.import_module(f"mjlab_piplus.common.rewards.{found[0]}"), name
        )


def test_rewards_docstring_lists_every_function() -> None:
    package = importlib.import_module("mjlab_piplus.common.rewards")
    for name in package.__all__:
        assert f"`{name}`" in package.__doc__, name
