"""Check that the registered tasks and their checkpoints still work (the "canaries").

    uv run python scripts/check_tasks.py                    # all three checks
    uv run python scripts/check_tasks.py --load             # C1: every task loads
    uv run python scripts/check_tasks.py --fingerprint      # C2: configs equal the baseline
    uv run python scripts/check_tasks.py --checkpoints      # C3: checkpoints fit their tasks
    uv run python scripts/check_tasks.py --fingerprint --update-baseline   # after an intended change
    uv run python scripts/check_tasks.py --dump /tmp/dump   # write normalized configs, to diff two checkouts

C1 loads the training, play and runner config of each `Mjlab-Piplus-*` task and checks that no
two tasks share a log group (`experiment_name`), except the pairs in `KNOWN_SHARED_LOG_GROUPS`.

C2 reduces each config to a SHA-256 hash and compares it with `docs/baseline/task_fingerprints.json`.
Functions defined in `mjlab_piplus` are hashed by the AST of their source, so reformatting, moving
a function to another module, or re-exporting it does not change the hash. A change of any value does.
The hash sorts dict keys, so C2 also compares the key order of the observation groups, actions,
rewards, terminations, events, and scene entities with `docs/baseline/task_key_order.json`.

C3 builds each task with one environment on the CPU and loads every checkpoint of its log group.
It compares the first-layer width of the actor and the critic and the action width with the task.
It proves shape compatibility, not behavior. Checkpoints in `KNOWN_UNLOADABLE` are history only.
"""

import argparse
import ast
import contextlib
import dataclasses
import enum
import functools
import hashlib
import inspect
import json
import re
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "docs" / "baseline" / "task_fingerprints.json"
KEY_ORDER_BASELINE = REPO / "docs" / "baseline" / "task_key_order.json"
LOGS = REPO / "logs" / "rsl_rl"
PREFIX = "Mjlab-Piplus-"

# Pairs of tasks that write into one log group. Ball-Small-Turf continues Ball-Small on turf.
KNOWN_SHARED_LOG_GROUPS = {
    "piplus_ball_small": {"Mjlab-Piplus-Ball-Small", "Mjlab-Piplus-Ball-Small-Turf"},
}

# Runs from before the Ball-Small v4 observation change (80-input actor). They do not load into the
# current task. docs/ball_small/ball_small.md documents this. History only.
KNOWN_UNLOADABLE = {
    ("piplus_ball_small", run)
    for run in (
        "2026-09-24_17-59-02",
        "2026-09-24_17-59-28",
        "2026-09-24_17-59-58",
        "2026-09-25_15-24-22",
        "2026-09-25_15-35-48",
        "2026-09-25_16-39-34",
        "2026-09-25_16-45-53",
        "2026-09-25_19-15-52",
        "2026-09-25_19-25-48",
        "2026-09-30_01-57-51_termination-rework",
    )
}


# ----------------------------------------------------------------------------------------
# Task access
# ----------------------------------------------------------------------------------------


def piplus_tasks(only=None):
    from mjlab.tasks.registry import list_tasks

    import mjlab_piplus.tasks  # noqa: F401  (registers the tasks)

    tasks = [t for t in list_tasks() if t.startswith(PREFIX)]
    if only:
        missing = [t for t in only if t not in tasks]
        if missing:
            raise SystemExit(f"unknown task(s): {missing}")
        tasks = [t for t in tasks if t in only]
    return tasks


def load_configs(task):
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    return {
        "env": load_env_cfg(task),
        "play": load_env_cfg(task, play=True),
        "rl": load_rl_cfg(task),
    }


# ----------------------------------------------------------------------------------------
# Normalization: a deterministic, formatting-independent dump of a config
# ----------------------------------------------------------------------------------------

_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")


def _source_hash(fn):
    """Hash of the AST of a function's source: equal for reformatted or moved code."""
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
        for node in ast.walk(tree):  # docstrings and comments do not change behavior
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                body = node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(getattr(body[0], "value", None), ast.Constant)
                    and isinstance(body[0].value.value, str)
                    and len(body) > 1
                ):
                    node.body = body[1:]
        return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]
    except (OSError, TypeError, SyntaxError):
        return "nosource"


def norm(obj, _stack=None):
    """Return a JSON-serializable, deterministic form of `obj`."""
    stack = _stack or set()
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return repr(obj)  # keeps nan and inf, and the exact digits
    if isinstance(obj, enum.Enum):
        return f"enum:{type(obj).__qualname__}.{obj.name}"
    if isinstance(obj, type):
        return f"type:{obj.__module__}.{obj.__qualname__}"
    oid = id(obj)
    if oid in stack:
        return "<cycle>"
    stack = stack | {oid}
    if isinstance(obj, functools.partial):
        return {
            "partial": norm(obj.func, stack),
            "args": norm(obj.args, stack),
            "keywords": norm(obj.keywords, stack),
        }
    if inspect.isroutine(obj):
        module = getattr(obj, "__module__", None) or ""
        qual = getattr(obj, "__qualname__", repr(obj))
        if module.startswith("mjlab_piplus") or module == "__main__":
            # Own code: name and AST hash, not the module, so a move or re-export keeps the hash.
            out = {"fn": qual, "ast": _source_hash(obj)}
            defaults = getattr(obj, "__defaults__", None)
            if defaults:
                out["defaults"] = norm(defaults, stack)
            kwdefaults = getattr(obj, "__kwdefaults__", None)
            if kwdefaults:
                out["kwdefaults"] = norm(kwdefaults, stack)
            closure = getattr(obj, "__closure__", None)
            if closure:
                cells = []
                for cell in closure:
                    try:
                        cells.append(norm(cell.cell_contents, stack))
                    except ValueError:  # empty cell
                        cells.append("<empty>")
                out["closure"] = cells
            return out
        return {"fn": f"{module}.{qual}"}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {"__class__": f"{type(obj).__module__}.{type(obj).__qualname__}"}
        for f in sorted(dataclasses.fields(obj), key=lambda f: f.name):
            out[f.name] = norm(getattr(obj, f.name, "<unset>"), stack)
        return out
    if isinstance(obj, dict):
        items = sorted(((str(k), norm(v, stack)) for k, v in obj.items()), key=lambda kv: kv[0])
        return {"dict": items}
    if isinstance(obj, (list, tuple)):
        return [norm(v, stack) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return {"set": sorted(json.dumps(norm(v, stack), sort_keys=True) for v in obj)}
    if hasattr(obj, "tolist"):  # numpy arrays, torch tensors
        with contextlib.suppress(Exception):
            return {"array": norm(obj.tolist(), stack), "dtype": str(getattr(obj, "dtype", ""))}
    if hasattr(obj, "__dict__") and not inspect.ismodule(obj):
        return {
            "__class__": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "attrs": norm({k: v for k, v in vars(obj).items() if not k.startswith("__")}, stack),
        }
    return _ADDR.sub("", repr(obj))


def fingerprint(value):
    text = json.dumps(norm(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def task_fingerprints(task):
    return {name: fingerprint(cfg) for name, cfg in load_configs(task).items()}


# ----------------------------------------------------------------------------------------
# C1: load
# ----------------------------------------------------------------------------------------


def check_load(tasks):
    problems = []
    groups = {}
    for task in tasks:
        try:
            cfgs = load_configs(task)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{task}: failed to load: {type(e).__name__}: {e}")
            continue
        groups.setdefault(cfgs["rl"].experiment_name, set()).add(task)
    for group, members in sorted(groups.items()):
        if len(members) > 1 and members != KNOWN_SHARED_LOG_GROUPS.get(group):
            problems.append(f"log group {group} is shared by {sorted(members)}")
    print(f"C1 load: {len(tasks)} tasks, {len(groups)} log groups, {len(problems)} problems")
    return problems


# ----------------------------------------------------------------------------------------
# C2: fingerprint
# ----------------------------------------------------------------------------------------


def read_baseline():
    if not BASELINE.exists():
        return {}
    return json.loads(BASELINE.read_text())


def check_fingerprint(tasks, update=False):
    from importlib.metadata import version

    prints = {t: task_fingerprints(t) for t in tasks}
    if update:
        data = read_baseline()
        data["mjlab_version"] = version("mjlab")
        merged = data.get("tasks", {}) if set(tasks) != set(piplus_tasks()) else {}
        merged.update(prints)
        data["tasks"] = dict(sorted(merged.items()))
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n")
        print(f"C2 fingerprint: baseline written for {len(prints)} tasks: {BASELINE}")
        return []
    base = read_baseline().get("tasks", {})
    problems = []
    for task, now in prints.items():
        if task not in base:
            problems.append(f"{task}: not in the baseline (new task? run with --update-baseline)")
            continue
        for part in ("env", "play", "rl"):
            if now[part] != base[task][part]:
                problems.append(f"{task}: {part} config differs from the baseline")
    for task in sorted(set(base) - set(prints)) if set(tasks) == set(piplus_tasks()) else []:
        problems.append(f"{task}: in the baseline but not registered (removed task?)")
    print(f"C2 fingerprint: {len(prints)} tasks compared, {len(problems)} differences")
    return problems


def task_key_order(task):
    """Ordered keys of the env config dicts. `norm` sorts dict keys, so C2 cannot see the order.

    The order of the terms of an observation group is the layout of the actor input, so a swap
    breaks every checkpoint of the task.
    """
    env = load_configs(task)["env"]
    order = {"observations": list(env.observations)}
    for group, cfg in env.observations.items():
        order[f"observations.{group}"] = list(cfg.terms)
    for name in ("actions", "rewards", "terminations", "events", "commands", "curriculum"):
        order[name] = list(getattr(env, name, None) or {})
    order["entities"] = list(env.scene.entities)
    return order


def _write_key_order(tasks):
    lines = []
    for task, order in sorted(tasks.items()):
        rows = [f"  {json.dumps(key)}: {json.dumps(keys)}" for key, keys in order.items()]
        lines.append(f" {json.dumps(task)}: {{\n" + ",\n".join(rows) + "\n }")
    KEY_ORDER_BASELINE.write_text("{\n" + ",\n".join(lines) + "\n}\n")


def check_key_order(tasks, update=False):
    orders = {t: task_key_order(t) for t in tasks}
    if update:
        merged = {}
        if KEY_ORDER_BASELINE.exists() and set(tasks) != set(piplus_tasks()):
            merged = json.loads(KEY_ORDER_BASELINE.read_text())
        merged.update(orders)
        _write_key_order(merged)
        print(f"C2 key order: baseline written for {len(orders)} tasks: {KEY_ORDER_BASELINE}")
        return []
    base = json.loads(KEY_ORDER_BASELINE.read_text()) if KEY_ORDER_BASELINE.exists() else {}
    problems = []
    for task, now in orders.items():
        if task not in base:
            problems.append(f"{task}: not in the key order baseline (run with --update-baseline)")
            continue
        for key in sorted(set(now) | set(base[task])):
            if now.get(key) != base[task].get(key):
                problems.append(f"{task}: order of {key} differs from the baseline")
    print(f"C2 key order: {len(orders)} tasks compared, {len(problems)} differences")
    return problems


def dump(tasks, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        for name, cfg in load_configs(task).items():
            path = out / f"{task[len(PREFIX) :]}.{name}.json"
            path.write_text(json.dumps(norm(cfg), sort_keys=True, indent=1))
    print(f"normalized configs written to {out}")


# ----------------------------------------------------------------------------------------
# C3: checkpoints
# ----------------------------------------------------------------------------------------


def _total(dim):
    return int(sum(dim)) if isinstance(dim, (tuple, list)) else int(dim)


def task_dims(task):
    """(actor width, critic width, action width) of a task, from one CPU environment."""
    from mjlab.envs import ManagerBasedRlEnv

    cfg = load_configs(task)["env"]
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        dims = env.observation_manager.group_obs_dim
        return _total(dims["actor"]), _total(dims["critic"]), env.action_manager.total_action_dim
    finally:
        env.close()


def check_checkpoints(tasks):
    import torch

    log_groups = {}
    for task in tasks:
        log_groups.setdefault(load_configs(task)["rl"].experiment_name, []).append(task)
    problems = []
    n_ckpt = n_known = 0
    for group, group_tasks in sorted(log_groups.items()):
        ckpts = sorted((LOGS / group).glob("*/model_*.pt"))
        if not ckpts:
            continue
        dims = {t: task_dims(t) for t in group_tasks}
        for ckpt in ckpts:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            actor, critic = state["actor_state_dict"], state["critic_state_dict"]
            weights = [k for k in actor if k.endswith(".weight") and actor[k].dim() == 2]
            shape = (
                actor["mlp.0.weight"].shape[1],
                critic["mlp.0.weight"].shape[1],
                actor[weights[-1]].shape[0],
            )
            n_ckpt += 1
            fits = [t for t, d in dims.items() if tuple(d) == tuple(int(x) for x in shape)]
            run = ckpt.parent.name
            if fits:
                continue
            if (group, run) in KNOWN_UNLOADABLE:
                n_known += 1
                continue
            problems.append(
                f"{group}/{run}/{ckpt.name}: widths {tuple(int(x) for x in shape)} "
                f"fit none of {[(t[len(PREFIX) :], d) for t, d in dims.items()]}"
            )
    for rel, digest in read_baseline().get("protected_checkpoints", {}).items():
        path = REPO / rel
        if not path.exists():
            problems.append(f"protected checkpoint missing: {rel}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"protected checkpoint changed: {rel}")
    print(
        f"C3 checkpoints: {n_ckpt} checked, {n_known} known history-only, {len(problems)} problems"
    )
    return problems


# ----------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--load", action="store_true", help="C1")
    parser.add_argument("--fingerprint", action="store_true", help="C2")
    parser.add_argument("--checkpoints", action="store_true", help="C3")
    parser.add_argument("--update-baseline", action="store_true", help="write the C2 baseline")
    parser.add_argument(
        "--protect",
        nargs="*",
        default=None,
        metavar="CKPT",
        help="record the SHA-256 of these checkpoints (repo-relative) in the baseline",
    )
    parser.add_argument("--dump", metavar="DIR", help="write normalized configs and exit")
    parser.add_argument("--tasks", nargs="*", help="limit to these task IDs")
    args = parser.parse_args()

    tasks = piplus_tasks(args.tasks)
    if args.dump:
        dump(tasks, args.dump)
        return 0
    if args.protect is not None:
        data = read_baseline()
        data["protected_checkpoints"] = {
            rel: hashlib.sha256((REPO / rel).read_bytes()).hexdigest() for rel in args.protect
        }
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(data, indent=1) + "\n")
        print(f"recorded SHA-256 of {len(args.protect)} checkpoint(s) in {BASELINE}")
        return 0
    run_all = not (args.load or args.fingerprint or args.checkpoints)
    problems = []
    if args.load or run_all:
        problems += check_load(tasks)
    if args.fingerprint or run_all:
        problems += check_fingerprint(tasks, update=args.update_baseline)
        problems += check_key_order(tasks, update=args.update_baseline)
    if args.checkpoints or run_all:
        problems += check_checkpoints(tasks)
    for p in problems:
        print("  PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
