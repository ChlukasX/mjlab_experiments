"""Checks of the documentation structure: the experiment index and the relative links.

    uv run pytest                    # index and link checks
    uv run pytest -m runs_index      # also: docs/runs_index.md is up to date (needs logs/ and wandb/)
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
STATUSES = {"planned", "active", "finished", "parked", "superseded"}


def index_rows():
    text = (REPO / "docs" / "README.md").read_text()
    section = text.split("## Experiments", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        if not line.startswith("|") or line.startswith(("|---", "| Experiment")):
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        assert len(cells) == 6, f"index row needs 6 cells: {line[:80]}"
        rows.append(cells)
    return rows


def test_index_has_one_row_per_experiment_directory():
    linked = set()
    for cells in index_rows():
        linked |= set(re.findall(r"\]\(([^/)]+)/[^)]+\.md\)", cells[5]))
    directories = {p.name for p in (REPO / "docs").iterdir() if p.is_dir() and any(p.glob("*.md"))}
    assert directories == linked, (sorted(directories - linked), sorted(linked - directories))


def test_index_status_values_are_valid():
    for cells in index_rows():
        assert cells[1] in STATUSES, (cells[0], cells[1])


def test_index_code_packages_exist():
    packages = {p.name for p in (REPO / "src" / "mjlab_piplus" / "tasks").iterdir() if p.is_dir()}
    packages.add("robot")
    for cells in index_rows():
        if cells[3] == "—":
            continue
        for name in re.findall(r"`([^`]+)`", cells[3]):
            assert name in packages, (cells[0], name)


def tracked_markdown():
    out = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    return [REPO / p for p in out if (REPO / p).exists()]


def test_relative_markdown_links_resolve():
    broken = []
    for path in tracked_markdown():
        for target in re.findall(r"\]\(([^)\s]+)\)", path.read_text()):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            file_part = target.split("#", 1)[0]
            if file_part and not (path.parent / file_part).exists():
                broken.append(f"{path.relative_to(REPO)} -> {target}")
    assert broken == []


@pytest.mark.runs_index
@pytest.mark.skipif(not (REPO / "logs" / "rsl_rl").exists(), reason="no logs/rsl_rl on this machine")
def test_runs_index_is_up_to_date():
    result = subprocess.run(
        ["uv", "run", "python", "scripts/make_runs_index.py", "--check"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout[-400:]
