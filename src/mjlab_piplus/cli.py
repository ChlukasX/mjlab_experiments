"""The `mjx` command line: list tasks and runs, play a checkpoint, prune checkpoints.

Entry point `mjlab_piplus:main` (pyproject.toml) re-exports `main` from this module.
Commands are in AGENTS.md, section Commands.
"""

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

from mjlab_piplus.naming import TASK_PREFIX, log_group

LOGS_ROOT = Path(__file__).parents[2] / "logs" / "rsl_rl"

# A play session ends after 15 minutes, so no viewer stays on the GPU (AGENTS.md, Workflow Rules).
PLAY_TIMEOUT_S = 900
# Seconds between the stop signal (SIGTERM) and the kill signal (SIGKILL) of a play session.
PLAY_KILL_GRACE_S = 10


def piplus_tasks() -> list[str]:
    """Return the sorted IDs of the registered Pi Plus tasks, read from the live registry."""
    from mjlab.tasks.registry import list_tasks

    import mjlab_piplus.tasks  # noqa: F401  (registers the tasks)

    return [t for t in list_tasks() if t.startswith(TASK_PREFIX)]


def _pick_task() -> str:
    tasks = piplus_tasks()
    print("Select task:")
    for i, t in enumerate(tasks, 1):
        print(f"  {i}) {t}")
    while True:
        raw = input("Enter number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(tasks):
            return tasks[int(raw) - 1]
        print(f"Enter 1-{len(tasks)}.")


def _run_dirs(task: str) -> list[Path]:
    task_dir = LOGS_ROOT / log_group(task)
    if not task_dir.exists():
        return []
    return sorted(task_dir.iterdir(), reverse=True)


def _latest_checkpoint(run_dir: Path) -> Path | None:
    pts = list(run_dir.glob("model_*.pt"))
    if not pts:
        return None
    return max(pts, key=lambda p: int(p.stem.split("_")[1]))


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def run_with_timeout(
    cmd: list[str], timeout_s: float, kill_grace_s: float = PLAY_KILL_GRACE_S
) -> int:
    """Run `cmd` in its own process group. Stop the whole group after `timeout_s` seconds.

    Return the exit code, or 124 (as the `timeout` command does) when the time limit ended it.
    """
    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        return proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"Time limit of {timeout_s:.0f} s reached. Stopping {cmd[0]}.")
        _stop_group(proc, kill_grace_s)
        return 124
    except KeyboardInterrupt:
        _stop_group(proc, kill_grace_s)
        raise


def _stop_group(proc: subprocess.Popen, kill_grace_s: float) -> None:
    for sig, wait_s in ((signal.SIGTERM, kill_grace_s), (signal.SIGKILL, None)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


def cmd_tasks(args: argparse.Namespace) -> None:
    print("\n".join(piplus_tasks()))


def cmd_runs(args: argparse.Namespace) -> None:
    runs = _run_dirs(args.task)
    if not runs:
        print(f"No runs found for {args.task}")
        return
    now = time.time()
    print(f"{'':2} {'run':<22} {'iters':>6} {'size':>7}  age")
    print("-" * 50)
    for i, run_dir in enumerate(runs):
        ckpt = _latest_checkpoint(run_dir)
        iters = int(ckpt.stem.split("_")[1]) if ckpt else 0
        size_mb = _dir_size_mb(run_dir)
        age_h = (now - run_dir.stat().st_mtime) / 3600
        age_str = f"{age_h:.0f}h ago" if age_h >= 1 else f"{age_h * 60:.0f}m ago"
        marker = "* " if i == 0 else "  "
        print(f"{marker}{run_dir.name:<22} {iters:>6}  {size_mb:>5.1f}MB  {age_str}")


def cmd_play(args: argparse.Namespace) -> None:
    runs = _run_dirs(args.task)
    if not runs:
        print(f"No runs found for {args.task}")
        return
    if args.run and args.run != "latest":
        matches = [r for r in runs if r.name.startswith(args.run)]
        if not matches:
            print(f"No run matching '{args.run}'")
            return
        run_dir = matches[0]
    else:
        run_dir = runs[0]
    if args.ckpt is not None:
        ckpt = run_dir / f"model_{args.ckpt}.pt"
        if not ckpt.exists():
            print(f"No checkpoint {ckpt.name} in {run_dir}")
            return
    else:
        ckpt = _latest_checkpoint(run_dir)
    if not ckpt:
        print(f"No checkpoints in {run_dir}")
        return
    print(f"Playing {ckpt} (ends after {PLAY_TIMEOUT_S // 60} minutes)")
    run_with_timeout(
        [
            "uv",
            "run",
            "play",
            args.task,
            "--checkpoint-file",
            str(ckpt),
            "--viewer",
            "viser",
            "--num-envs",
            "1",
        ],
        PLAY_TIMEOUT_S,
    )


def cmd_clean(args: argparse.Namespace) -> None:
    runs = _run_dirs(args.task)
    if not runs:
        print(f"No runs found for {args.task}")
        return
    to_delete: list[Path] = []
    # ponytail: skip the latest run entirely
    for run_dir in runs[1:]:
        pts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if not pts:
            continue
        for i, pt in enumerate(pts[:-1]):
            if i % args.keep != 0:
                to_delete.append(pt)
    if not to_delete:
        print("Nothing to delete.")
        return
    total_mb = sum(p.stat().st_size for p in to_delete) / 1e6
    verb = "Would delete" if args.dry_run else "Deleting"
    print(f"{verb} {len(to_delete)} files ({total_mb:.1f} MB)")
    for p in to_delete:
        print(f"  {p.relative_to(LOGS_ROOT)}")
        if not args.dry_run:
            p.unlink()


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of `mjx`."""
    parser = argparse.ArgumentParser(prog="mjx")
    parser.add_argument("--task", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("tasks", help="List the registered Pi Plus tasks")
    sub.add_parser("runs", help="List training runs")

    p_play = sub.add_parser("play", help="Play latest checkpoint (ends after 15 minutes)")
    p_play.add_argument("--run", default="latest", help="Timestamp prefix or 'latest'")
    p_play.add_argument(
        "--ckpt", type=int, default=None, help="Iteration N of model_N.pt (default: latest)"
    )

    p_clean = sub.add_parser("clean", help="Prune old checkpoints")
    p_clean.add_argument("--keep", type=int, default=5, help="Keep every Nth checkpoint")
    p_clean.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd != "tasks":
        if args.task is None:
            args.task = _pick_task()
        elif not args.task.startswith(TASK_PREFIX):
            parser.error(f"--task must start with {TASK_PREFIX!r}: {args.task!r}")

    {"tasks": cmd_tasks, "runs": cmd_runs, "play": cmd_play, "clean": cmd_clean}[args.cmd](args)
