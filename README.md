# mjlab-piplus

Pi+ humanoid experiments using [mjlab](https://mujocolab.github.io/mjlab/main/index.html) (MuJoCo + RSL-RL + PPO). These are mainly personal experiments, if you want to see more with what we do with the PI+ humanoid robot, and many other cool things, check out the [Hamburg bit-bots](https://github.com/bit-bots/) RoboCup team. 

## Setup

```bash
uv sync
uv run wandb login   # required for training
```

## Where to find things

| What | Where |
|---|---|
| Tasks (one per experiment variant) and their descriptions | [AGENTS.md](AGENTS.md#registered-tasks) |
| Commands: train, resume, evaluate, play, manage runs | [AGENTS.md](AGENTS.md#commands) |
| Experiments: status, design notes, run logs | [docs/README.md](docs/README.md) |
| Every training run, with its commit and launch arguments | [docs/runs_index.md](docs/runs_index.md) |
| Exact map of the repository, for agents and for contributors | [AGENTS.md](AGENTS.md#architecture) |

## Repository structure

This is the overview. It changes only when the structure changes in a big way. The exact map is in [AGENTS.md](AGENTS.md#architecture).

```
src/mjlab_piplus/   Python package: the Pi+ robot model, one folder per group of tasks, terrains
scripts/            tools to train, evaluate, check, and export
tests/              automatic checks that the tasks still work
docs/               one folder per experiment (design notes and run log) and the experiment index
logs/  wandb/       training output (not in git)
deploy/             files for the robot (not in git)
data/               datasets made from trained policies (not in git)
```

The task folders in `src/mjlab_piplus/tasks/` fall into four groups: standing (`upright`, `noninertial`, `platform`), walking (`locomotion`, `locomotion_arms`), balancing on a ball (`ball`, `ball_arms`, `ball_small`), and mounting a ball and balancing on it (`ball_mount`, `ball_mount_min`, `ball_mount_curriculum`, `ball_balance_from_mount`).
