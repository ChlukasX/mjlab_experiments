# mjlab-piplus

Pi+ humanoid experiments using [mjlab](https://mujocolab.github.io/mjlab/main/index.html) (MuJoCo + RSL-RL + PPO). These are mainly personal experiments, if you want to see more with what we do with the PI+ humanoid robot, and many other cool things, check out the [Hamburg bit-bots](https://github.com/bit-bots/) RoboCup team. 

## Setup

```bash
uv sync
uv run wandb login   # required for training
```

## Experiments

| Task | Description |
|---|---|
| `Mjlab-Piplus-Upright` | Baseline: stand upright on flat terrain |
| `Mjlab-Piplus-NonInertial` | Sinusoidal forces on root body simulating platform acceleration |
| `Mjlab-Piplus-Platform` | Physical 3 m × 3 m mocap platform with roll, pitch, heave (boat motion) |

## Train

```bash
bash scripts/train.sh                          # interactive task selection menu
bash scripts/train.sh Mjlab-Piplus-Platform    # skip menu
bash scripts/train.sh --cpu                    # CPU mode, 64 envs
bash scripts/train.sh --tmux                   # wrap in tmux session mjlab-train
bash scripts/train.sh --num-envs 2048          # custom env count (reduce if OOM)
```

Runs 4096 envs on GPU by default, logs to wandb under the task name.

## Play

```bash
uv run mjx play                                # interactive task + run selection
uv run mjx --task Mjlab-Piplus-Platform play   # skip menu
```

Or directly:

```bash
uv run play Mjlab-Piplus-Platform \
  --checkpoint-file logs/rsl_rl/piplus_platform/<run>/model_N.pt \
  --viewer viser --num-envs 1
```

## Manage runs

```bash
uv run mjx runs                                # list runs with iteration count and age
uv run mjx clean                               # prune old checkpoints (keep every 5th)
uv run mjx clean --dry-run                     # preview what would be deleted
```

## Repo structure

```
src/mjlab_piplus/
  robot/                  Pi+ MJCF + constants (actuators, collision presets)
  tasks/
    upright/              Mjlab-Piplus-Upright
    noninertial/          Mjlab-Piplus-NonInertial
    platform/             Mjlab-Piplus-Platform
scripts/
  train.sh                GPU training launcher with wandb + optional tmux
logs/rsl_rl/              Checkpoints (gitignored)
docs/                     Per-experiment design notes
```

See `AGENTS.md` for architecture details and notes relevant to further development.
