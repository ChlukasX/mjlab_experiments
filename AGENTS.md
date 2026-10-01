# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) and other coding agents when working with code in this repository.

## Commands

```bash
# Train — interactive task selection menu when no task given
bash scripts/train.sh
bash scripts/train.sh Mjlab-Piplus-Platform --num-envs 64
bash scripts/train.sh --cpu   # CPU mode, 64 envs

# Train directly
uv run train Mjlab-Piplus-Platform --env.scene.num-envs 4096

# Play back a checkpoint (interactive selection via mjx)
uv run mjx play
uv run mjx --task Mjlab-Piplus-Platform play

# Play directly
uv run play Mjlab-Piplus-Platform --checkpoint-file logs/rsl_rl/piplus_platform/<run>/model_N.pt --viewer viser --num-envs 1
```

`scripts/train.sh` requires wandb login (`uv run wandb login` or `WANDB_API_KEY` env var) and creates a tmux session `mjlab-train` with a spare viewer window.

## Registered Tasks

| Task ID | Description |
|---|---|
| `Mjlab-Piplus-Upright` | Baseline: stand upright on flat terrain |
| `Mjlab-Piplus-NonInertial` | Sinusoidal forces injected on root body (fictitious platform acceleration) |
| `Mjlab-Piplus-Platform` | Physical 3 m × 3 m mocap platform with roll, pitch, heave (boat motion) |
| `Mjlab-Piplus-Locomotion` | Velocity-tracked walking on flat terrain (legs only) |
| `Mjlab-Piplus-Locomotion-Arms` | Same, with arms actuated |
| `Mjlab-Piplus-Ball` | Balance on a large free-rolling ball (0.45 m radius) |
| `Mjlab-Piplus-Ball-Arms` | Same, with arms actuated |
| `Mjlab-Piplus-Ball-Small` | Single-leg balance on a football; sim2real setup |
| `Mjlab-Piplus-Ball-Small-Turf` | Same, on astroturf (soft two-plane terrain) |

## Architecture

This is an `mjlab` plugin package. `mjlab` is the RL framework (MuJoCo + RSL-RL + PPO). This repo contributes a robot description and tasks to it.

**Task registration flow:**
1. `src/mjlab_piplus/tasks/__init__.py` — calls `import_packages(__name__)` which auto-imports all subpackages
2. Each `src/mjlab_piplus/tasks/<name>/__init__.py` — imports env/runner configs and calls `register_mjlab_task()`

**Adding a new task:** create `src/mjlab_piplus/tasks/<task_name>/` with `__init__.py` (register) and a config file; auto-imported.

**Robot config** lives in `src/mjlab_piplus/robot/piplus_constants.py`:
- `get_piplus_robot_cfg()` — returns full `EntityCfg` (used by env configs)
- `get_spec()` — loads MuJoCo XML and repaints visual meshes dark gray (MJCF ships light gray; vendored file left unchanged)
- `ACTUATOR_5036` covers all leg joints; arm actuators (`ACTUATOR_4438`) are defined but commented out
- Collision presets: `FULL_COLLISION`, `LEG_ONLY_COLLISION`, `FEET_ONLY_COLLISION` — swap in env cfg to control what self-collides

**Env config** uses `mjlab`'s manager-based pattern: dicts of `ObservationTermCfg`, `ActionTermCfg`, `RewardTermCfg`, `TerminationTermCfg` passed to `ManagerBasedRlEnvCfg`. `play=True` disables observation corruption and sets episode length to infinity.

**Platform task specifics** (`tasks/platform/piplus_platform_env_cfg.py`):
- `get_platform_spec()` — builds a fixed-base mocap BOX body; auto-wrapped by mjlab as a mocap entity
- `reset_scene_to_default` must be first in the events dict — it writes `init_state.pos` to qpos (without it, robot resets to MJCF `qpos0` at z=0.385, below the platform)
- `ApplyPlatformMotion` (step event) — drives mocap pose each control step via `write_mocap_pose_to_sim`
- Platform motion: roll/pitch/heave sinusoids with per-env randomised amplitude and frequency; curriculum ramps amplitudes from 0 to 0.30 rad / 0.25 m
- `njmax=400, nconmax=160` — sized for BOX-BOX foot/platform contact at 4096 envs without OOM

**Sim timestep:** 0.005 s × decimation 4 = 0.02 s control step (50 Hz).
