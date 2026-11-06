"""Shim: the script moved to `scripts/eval_policy.py`. Old commands and logs keep working."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("eval_policy.py")), run_name="__main__")
