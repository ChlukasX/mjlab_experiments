"""Tests for the `mjx` command line and `scripts/train.sh` (no GPU, no training)."""

import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mjlab_piplus import cli

REPO = Path(__file__).resolve().parents[1]
TRAIN_SH = REPO / "scripts" / "train.sh"
FAKE_TASKS = ["Mjlab-Piplus-Aaa", "Mjlab-Piplus-Bbb"]


@pytest.fixture(scope="module")
def tasks():
    return cli.piplus_tasks()


def test_task_list_is_the_sorted_live_registry(tasks):
    from mjlab.tasks.registry import list_tasks

    assert tasks == sorted(tasks)
    assert tasks == [t for t in list_tasks() if t.startswith("Mjlab-Piplus-")]
    assert len(tasks) > 26  # the old hard-coded list had 26 tasks
    assert "Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess" in tasks


def test_entry_point_main_is_reexported():
    import mjlab_piplus

    assert mjlab_piplus.main is cli.main


def test_mjx_tasks_command(tasks):
    out = subprocess.run(
        [sys.executable, "-c", "from mjlab_piplus import main; main()", "tasks"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.split() == tasks


def test_mjx_rejects_foreign_task_id():
    result = subprocess.run(
        [sys.executable, "-c", "from mjlab_piplus import main; main()", "--task", "Foo", "runs"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--task must start with" in result.stderr


def test_play_timeout_is_15_minutes():
    assert cli.PLAY_TIMEOUT_S == 900


def test_run_with_timeout_returns_exit_code():
    assert cli.run_with_timeout(["true"], 5) == 0
    assert cli.run_with_timeout(["false"], 5) == 1


def test_run_with_timeout_stops_the_whole_process_group(tmp_path):
    # The shell starts a child (sleep). The limit must end both.
    pidfile = tmp_path / "child.pid"
    start = time.monotonic()
    code = cli.run_with_timeout(
        ["sh", "-c", f"sleep 60 & echo $! > {pidfile}; wait"], 0.5, kill_grace_s=2
    )
    assert code == 124
    assert time.monotonic() - start < 10
    pid = int(pidfile.read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.fixture
def fake_uv(tmp_path):
    """A `uv` stub: `uv run mjx tasks` lists FAKE_TASKS. Any other call is written to a file."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "uv_calls.txt"
    stub = bin_dir / "uv"
    lines = "\\n".join(FAKE_TASKS)
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'if [[ "$*" == "run mjx tasks" ]]; then printf "{lines}\\n"; exit 0; fi\n'
        f'echo "$*" >> {record}\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "WANDB_API_KEY": "x"}
    return env, record


def run_train(env, *args, stdin=""):
    return subprocess.run(
        ["bash", str(TRAIN_SH), *args],
        env=env,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def test_train_sh_requires_name(fake_uv):
    env, record = fake_uv
    result = run_train(env, "Mjlab-Piplus-Aaa")
    assert result.returncode == 1
    assert "--name is required" in result.stderr
    assert not record.exists()


def test_train_sh_name_without_value(fake_uv):
    env, record = fake_uv
    result = run_train(env, "Mjlab-Piplus-Aaa", "--name")
    assert result.returncode == 1
    assert "--name needs a value" in result.stderr
    assert not record.exists()


def test_train_sh_menu_comes_from_the_registry(fake_uv):
    env, record = fake_uv
    result = run_train(env, "--name", "t1", "--cpu", stdin="2\n")
    assert result.returncode == 0, result.stderr
    assert "1) Mjlab-Piplus-Aaa" in result.stderr
    assert "2) Mjlab-Piplus-Bbb" in result.stderr
    call = record.read_text()
    assert "run train Mjlab-Piplus-Bbb" in call
    assert "--agent.run-name t1" in call
    assert "--env.scene.num-envs 64" in call


def test_train_sh_stops_when_no_task_is_selected(fake_uv):
    env, record = fake_uv
    result = run_train(env, "--name", "t1", stdin="")
    assert result.returncode == 1
    assert "no task selected" in result.stderr
    assert not record.exists()


def test_train_sh_passes_name_and_extra_args(fake_uv):
    env, record = fake_uv
    result = run_train(env, "Mjlab-Piplus-Aaa", "--name=t2", "--cpu", "--agent.seed", "3")
    assert result.returncode == 0, result.stderr
    call = record.read_text()
    assert "run train Mjlab-Piplus-Aaa" in call
    assert "--agent.run-name t2" in call
    assert "--agent.seed 3" in call
