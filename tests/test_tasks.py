"""Canaries for the registered tasks (see scripts/check_tasks.py).

    uv run pytest                   # C1, C2, and the AGENTS.md table check
    uv run pytest -m checkpoints    # C3 as well (needs logs/rsl_rl, about 1 minute on the CPU)
"""

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def ct():
    spec = importlib.util.spec_from_file_location("check_tasks", REPO / "scripts" / "check_tasks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def tasks(ct):
    return ct.piplus_tasks()


def test_c1_every_task_loads(ct, tasks):
    assert ct.check_load(tasks) == []


def test_c2_configs_equal_the_baseline(ct, tasks):
    assert ct.check_fingerprint(tasks) == []


@pytest.mark.checkpoints
@pytest.mark.skipif(not (REPO / "logs" / "rsl_rl").exists(), reason="no logs/rsl_rl on this machine")
def test_c3_checkpoints_fit_their_tasks(ct, tasks):
    assert ct.check_checkpoints(tasks) == []


@pytest.mark.xfail(
    reason="AGENTS.md lists tasks with bracket shorthand; a later cleanup phase "
    "rewrites the table to one task ID per row",
    strict=False,
)
def test_agents_task_table_lists_every_task(tasks):
    text = (REPO / "AGENTS.md").read_text()
    rows = [line for line in text.splitlines() if line.startswith("| `Mjlab-")]
    listed = {m for line in rows for m in re.findall(r"`(Mjlab-Piplus-[A-Za-z0-9-]+)`", line.split("|")[1])}
    assert set(tasks) <= listed, sorted(set(tasks) - listed)
