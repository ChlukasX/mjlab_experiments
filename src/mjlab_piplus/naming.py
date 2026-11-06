"""Names derived from a task ID (see AGENTS.md, section Naming convention)."""

TASK_PREFIX = "Mjlab-Piplus-"

# Tasks whose runner config logs into the log group of another task.
# Ball-Small-Turf continues Ball-Small on turf. This is the only known exception.
LOG_GROUP_ALIASES = {
    "Mjlab-Piplus-Ball-Small-Turf": "piplus_ball_small",
}


def log_group(task_id: str) -> str:
    """Return the log group (`experiment_name`) of a task: `piplus_` + the lowercase ID."""
    if task_id in LOG_GROUP_ALIASES:
        return LOG_GROUP_ALIASES[task_id]
    if not task_id.startswith(TASK_PREFIX):
        raise ValueError(f"not a Pi Plus task ID (prefix {TASK_PREFIX!r}): {task_id!r}")
    return "piplus_" + task_id.removeprefix(TASK_PREFIX).replace("-", "_").lower()
