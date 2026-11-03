# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) and other coding agents when working with code in this repository.

## Workflow Rules

- **New features and policies always go on a new branch** (e.g. `feat/ball-approach`), never directly on `main`. The repo is public; `main` must stay clean and working.
- Branch before the first edit, not after. Keep the working tree clean: no stray debug files, no half-finished changes left on `main`.
- Commit/push only on explicit user request; merge to `main` only when the user says so.
- **Observation or network size changes make a new experiment:** a change that alters an actor or critic input (or the network) goes into a new task variant with its own experiment name, never into an existing task, so old checkpoints keep loading and playing. A warm start into it loads the actor where its inputs are unchanged; for new critic inputs, pad the old critic with zero-weight columns (`logs/probes/make_padded_init.py`) rather than starting it fresh: a fresh critic wrecked the warm-started actor within ~40 iterations (`bl3` first try, see docs/mount_latency). Note this in the run log.
- **Play sessions time out after 15 minutes:** start every play/viewer run (`uv run play`, `mjx play`, play wrappers) under `timeout 15m`, e.g. `timeout 15m uv run play ...`, so no viewer is left running on the GPU.
- **Docs are one directory per experiment** (`docs/<experiment>/<experiment>.md` for the design, `docs/<experiment>/<experiment>_log.md` for its run log). Never mix several experiments in one log; see `docs/README.md` for the index and conventions.

## Skills

- **`mjlab-skillkit`**: At session start, if this skill is not present in
  `.claude/skills/mjlab-skillkit/SKILL.md` (or the project's local skill
  directory for non-Claude agents), ask the user once whether to install
  it for this project only (never globally).

  - Before downloading, tell the user:
    - This is third-party content from `cmjang/mjlab-skillkit`.
    - They should review the skill themselves for anything malicious or
      unsafe.
    - You will not install it without an explicit "yes".

  - If the user confirms:
    - Download the raw `SKILL.md` from:
      https://raw.githubusercontent.com/cmjang/mjlab-skillkit/refs/heads/master/SKILL.md
    - For Claude Code:
      - Save it as `.claude/skills/mjlab-skillkit/SKILL.md` in the
        project root.
    - For other agents:
      - Save it in the agent's local project skill directory according
        to that agent's docs (e.g. Cursor/Copilot/etc.).
    - After installation, confirm where the file was written and that
      the skill is now available.

  - If the user declines or does not answer clearly, do not download or
    create the skill.

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
| `Mjlab-Piplus-Ball-Mount` | Step from the floor onto the ball and hold single-leg stance (feasibility spike) |
| `Mjlab-Piplus-Ball-Mount-Flat` | Same, sole must also be flat on the ball |
| `Mjlab-Piplus-Ball-Mount-Size1[-Flat]` | Same on a size 1 football (radius 0.07 m) |
| `Mjlab-Piplus-Ball-Balance-Size1` | Drop onto a size 1 ball and balance (flat sole) |
| `Mjlab-Piplus-Ball-MountCurr-{Release,ReleasePot,Pot,Locked}` | Mount-only curriculum on a randomized ball: ball locked by rolling resistance then released, and/or potential-based progress reward |
| `Mjlab-Piplus-Ball-MountMin[-Clip/-Locked]` | Mount only, pruned reward set, free-foot lift measured at the sole, staged stance target (unbounded / bounded actions / locked ball) |
| `Mjlab-Piplus-Ball-BalanceFromMount[-Mix50/-Clip]` | Balance on the randomized ball starting from recorded hand-off states of the mount policy (-Mix50: half drops; -Clip: joint targets clipped, for smoothing) |
| `Mjlab-Piplus-Ball-Chain` | Mount scene played by two policies with a switch (mount, then balance): `uv run python scripts/play_chain.py` |
| `Mjlab-Piplus-Ball-MountMin-BlindOnBall[-Latency[-Big[-OneFoot[-Level[-NoiseRamp[-Latch45-Hold[-Deadline/-KinBall[-FreeLeg[-Straight[-Gate-Deadline]]]]]]]]]]` | Mount and stance with the ball input only until on the ball (deployment candidate); -Latency: 10-40 ms loop latency curriculum, encoder bias seen by the actor; -Big: size 3-5 balls only; -OneFoot: stance rewards only while the free foot does not touch the ball; -Level: support sole near horizontal (foot on the ball top), ankle roll kept off its stop; -NoiseRamp: actor noise ramped in from 0.25×; -Latch45-Hold: on-ball latch at 45° base tilt (the 20° latch never fired), 1 s home-pose hold after reset; -Deadline: episode ends if not on the ball 3 s after start; -KinBall: on the ball the actor gets the ball centre computed from the support foot (kinematics) instead of zeros; -FreeLeg: in single support the free leg is out of `posture` and paid to clear the ball, the support hip and knee cost velocity (ankle balance); -Straight: thigh (yaw) joint limited to ±0.6 rad in the model, support foot paid for pointing like the torso (cold start, docs/mount_straight); -Gate-Deadline: those terms only in a real stance (CoM over the ball, free sole off the floor), and the mount deadline |
| `Mjlab-Piplus-Ball-MountBalance[-Mix30/-Mix70]` | One policy for mount + balance on a randomized ball (size 1-5, mass, friction); 50/50 mount/drop spawn mix, -Mix30/-Mix70 change the ratio |

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

## TODO: Ball chain (find ball → walk → set stance → mount → balance)

Hierarchical: Approach policy → `(vx, vy, ωz)` → frozen Locomotion policy; separate Mount and Balance policies with a switch on the single-leg stance; switcher on distance to ball for walk → mount. One branch per phase; docs in `docs/`.

- [x] 0. `feat/mount-feasibility` — spike. Its "mounting is feasible" result was a tiptoe exploit of the lift test; see the correction in the doc. ([docs](docs/mount_feasibility/mount_feasibility.md))
- [ ] 1. `feat/locomotion-s2r` — Locomotion-Arms v2: proprio-only actor + history, noise, DR, ball obstacle, yaw-rate penalty. ([docs](docs/locomotion_s2r/locomotion_s2r.md))
- [ ] 2. `feat/ball-perception` — noisy ball position in base frame (noise, dropout, latency, FOV). ([docs](docs/ball_perception/ball_perception.md))
- [ ] 3. `feat/ball-approach` — high-level policy driving the frozen locomotion policy to the stand-off pose. ([docs](docs/ball_approach/ball_approach.md))
- [x] 4a. `feat/ball-mount-balance`, `feat/ball-mount-curriculum` — one Mount+Balance policy, then a mount curriculum: no mount (0% success), drop balance 97%. ([docs](docs/ball_mount/ball_mount.md), [docs](docs/ball_mount_curriculum/ball_mount_curriculum.md))
- [x] 4b. `feat/ball-mount-rewards` — pruned mount rewards with the lift measured at the sole: first real mount (`MountMin-Clip`, on the ball in 0.5 s; does not hold the stance alone). ([docs](docs/ball_mount_rewards/ball_mount_rewards.md))
- [x] 4c. `feat/balance-from-mount` — balance policy trained from the mount policy's hand-off states: 70% mount → balance success. ([docs](docs/balance_from_mount/balance_from_mount.md))
- [x] 4d. `feat/balance-smooth` (parked; tag `two-policy-chain`) — actions a real actuator can follow (audit thresholds), on the corrected model (fitted head box, randomized contact stiffness; [collision_check](docs/collision_check/collision_check.md)). Balance `bs4-exc100-ar2`: at the limits on two of three measures, hand-off 96%. Mount `mm7-exc200-ar2`: 87% alone, close to the limits. Open: torque saturation (11 to 17%), small balls, one policy or two. ([docs](docs/balance_smooth/balance_smooth.md), [mount log](docs/ball_mount_rewards/ball_mount_rewards_log.md))
- [ ] 4e. `feat/mount-single` — one policy for mount and stance (two-policy state tagged `two-policy-chain`). `ms3-shell` (hollow-shell ball): 98% success, 93% on the smallest balls, spin removed, leg torque at the limit 5%, at the audit limits. Open: real actuator limits, real ball measurements. ([docs](docs/mount_single/mount_single.md))
- [ ] 5. `feat/ball-chain` — `Mjlab-Piplus-Ball-Chain`: mount → balance playable (`scripts/play_chain.py`); approach and locomotion not connected. ([docs](docs/ball_chain/ball_chain.md))
- [ ] 6. `docs/ball-chain` — final docs, task table, README.

Other points:
- [ ] Fix Ball-Small policy spin (~−1.4 rad/s) with a yaw-rate penalty before using it as a warm start.
- [ ] Train flat floor first, then continue on turf (Ball-Small recipe).
- [ ] Actor must not use `base_lin_vel` or ball velocity (no real sensor); critic only.
- [ ] Check whether mjlab supports an action term wrapping a frozen policy.
