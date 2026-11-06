"""Helpers and constants that two or more task modules share.

Each name here was defined with an identical body in several task modules. The old
modules import the name from here, so `from <old module> import <name>` still works.
Do not change a body here: a change alters every task that uses it (rule S6, check C2).
See AGENTS.md (rule S7 and the repository map).
"""
