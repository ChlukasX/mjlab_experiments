"""Shared code lives in `mjlab_piplus.common`, not behind a private name (rules S7 and S15).

Rule of two: a function stays in its task package until a second package or a script needs it.
Then it moves to `src/mjlab_piplus/common/`. A module must not import a name with a leading
underscore from a module of another package. Modules of one task package (`tasks/<name>/`)
may share private names. Scripts, `common/`, and tests are outside every task package.
"""

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def package_of(module: str) -> str | None:
    """The task package of a dotted module name, or None outside `mjlab_piplus.tasks`."""
    parts = module.split(".")
    if len(parts) >= 3 and parts[:2] == ["mjlab_piplus", "tasks"]:
        return parts[2]
    return None


def own_module(path: Path) -> str:
    """Dotted module name of a file under src/ (scripts and tests keep their folder name)."""
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[0] == "src":
        parts = parts[1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def private_imports() -> list[tuple[str, str, str]]:
    """Every `from mjlab_piplus.x import _name` that crosses a task package boundary."""
    found = []
    for folder in ("src", "scripts", "tests"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            if ".venv" in path.parts or path.name == Path(__file__).name:
                continue
            mine = package_of(own_module(path))
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.ImportFrom) or node.level:
                    continue  # a relative import stays inside one package
                module = node.module or ""
                if not module.startswith("mjlab_piplus"):
                    continue
                if mine is not None and mine == package_of(module):
                    continue
                for alias in node.names:
                    if alias.name.startswith("_") and not alias.name.startswith("__"):
                        found.append((path.relative_to(ROOT).as_posix(), module, alias.name))
    return found


def test_no_private_import_across_packages() -> None:
    assert not private_imports(), (
        "Private name imported from another package. Move the helper to src/mjlab_piplus/common/ "
        f"and import its public name: {private_imports()}"
    )


def test_importing_common_first_registers_the_tasks() -> None:
    """`mjlab` loads the task plugin on its first import; `common/__init__.py` imports it first.

    Without that import, `import mjlab_piplus.common.rewards` in a fresh interpreter fails
    inside the plugin load, because the tasks import the half-loaded `common.rewards`.
    """
    code = (
        "import mjlab_piplus.common.rewards\n"
        "from mjlab.tasks.registry import list_tasks\n"
        "print('Mjlab-Piplus-Upright' in list_tasks())"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert "circular" not in run.stdout + run.stderr, run.stderr[-600:]
    assert run.returncode == 0, run.stderr[-600:]
    assert run.stdout.strip().splitlines()[-1] == "True"
