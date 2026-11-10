"""Helpers and constants that two or more task modules share.

Each name here was defined with an identical body in several task modules. The old
modules import the name from here, so `from <old module> import <name>` still works.
Do not change a body here: a change alters every task that uses it (rule S6, check C2).
See AGENTS.md (rule S7 and the repository map).
"""

# The first import of `mjlab` loads its plugins, and the plugin `mjlab_piplus.tasks` imports
# this package. Importing `mjlab` here, before any submodule runs, keeps that cycle away from a
# half-loaded `common` module. `mjlab_piplus/__init__.py` stays without it, so `mjx runs` and
# `mjx clean` start without the 10 s import of `mjlab`.
import mjlab  # noqa: F401
