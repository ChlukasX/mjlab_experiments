import argparse
import os
import subprocess
import time
from pathlib import Path

LOGS_ROOT = Path(__file__).parents[2] / "logs" / "rsl_rl"

TASKS = [
    "Mjlab-Piplus-Upright",
    "Mjlab-Piplus-Locomotion",
    "Mjlab-Piplus-Locomotion-Arms",
    "Mjlab-Piplus-NonInertial",
    "Mjlab-Piplus-Platform",
    "Mjlab-Piplus-Ball",
    "Mjlab-Piplus-Ball-Arms",
    "Mjlab-Piplus-Ball-Small",
    "Mjlab-Piplus-Ball-Small-Turf",
    "Mjlab-Piplus-Ball-Mount",
    "Mjlab-Piplus-Ball-Mount-Flat",
    "Mjlab-Piplus-Ball-Mount-Size1",
    "Mjlab-Piplus-Ball-Mount-Size1-Flat",
    "Mjlab-Piplus-Ball-Balance-Size1",
    "Mjlab-Piplus-Ball-MountBalance",
    "Mjlab-Piplus-Ball-MountBalance-Mix30",
    "Mjlab-Piplus-Ball-MountBalance-Mix70",
    "Mjlab-Piplus-Ball-MountBalance-Mix0",
    "Mjlab-Piplus-Ball-MountBalance-Mix100",
    "Mjlab-Piplus-Ball-MountBalance-Clip",
    "Mjlab-Piplus-Ball-MountBalance-Mix100-Clip",
    "Mjlab-Piplus-Ball-MountBalance-Mix0-Clip",
]

# Tasks whose runner config logs into another task's experiment dir.
LOG_DIR_OVERRIDES = {"Mjlab-Piplus-Ball-Small-Turf": "piplus_ball_small"}


def _pick_task() -> str:
    print("Select task:")
    for i, t in enumerate(TASKS, 1):
        print(f"  {i}) {t}")
    while True:
        raw = input("Enter number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(TASKS):
            return TASKS[int(raw) - 1]
        print(f"Enter 1–{len(TASKS)}.")


def _task_slug(task: str) -> str:
    if task in LOG_DIR_OVERRIDES:
        return LOG_DIR_OVERRIDES[task]
    return task.replace("Mjlab-", "").replace("-", "_").lower()


def _run_dirs(task: str) -> list[Path]:
    slug = _task_slug(task)
    task_dir = LOGS_ROOT / slug
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
        age_str = f"{age_h:.0f}h ago" if age_h >= 1 else f"{age_h*60:.0f}m ago"
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
    print(f"Playing {ckpt}")
    subprocess.run([
        "uv", "run", "play", args.task,
        "--checkpoint-file", str(ckpt),
        "--viewer", "viser",
        "--num-envs", "1",
    ])


def cmd_clean(args: argparse.Namespace) -> None:
    runs = _run_dirs(args.task)
    if not runs:
        print(f"No runs found for {args.task}")
        return
    to_delete: list[Path] = []
    # ponytail: skip the latest run entirely
    for run_dir in runs[1:]:
        pts = sorted(
            run_dir.glob("model_*.pt"),
            key=lambda p: int(p.stem.split("_")[1])
        )
        if not pts:
            continue
        final = pts[-1]
        for i, pt in enumerate(pts[:-1]):
            if i % args.keep != 0:
                to_delete.append(pt)
    if not to_delete:
        print("Nothing to delete.")
        return
    total_mb = sum(p.stat().st_size for p in to_delete) / 1e6
    print(f"{'Would delete' if args.dry_run else 'Deleting'} {len(to_delete)} files ({total_mb:.1f} MB)")
    for p in to_delete:
        print(f"  {p.relative_to(LOGS_ROOT)}")
        if not args.dry_run:
            p.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(prog="mjx")
    parser.add_argument("--task", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("runs", help="List training runs")

    p_play = sub.add_parser("play", help="Play latest checkpoint")
    p_play.add_argument("--run", default="latest", help="Timestamp prefix or 'latest'")
    p_play.add_argument("--ckpt", type=int, default=None, help="Iteration N of model_N.pt (default: latest)")

    p_clean = sub.add_parser("clean", help="Prune old checkpoints")
    p_clean.add_argument("--keep", type=int, default=5, help="Keep every Nth checkpoint")
    p_clean.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.task is None:
        args.task = _pick_task()

    {"runs": cmd_runs, "play": cmd_play, "clean": cmd_clean}[args.cmd](args)
