# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) and other coding agents when working with code in this repository. It is the single source of truth for agent rules, the task list, the commands, and the detailed repository map. A person who only wants to use the repository needs [README.md](README.md).

## Workflow Rules

- **New features and policies always go on a new branch**, never directly on `main`. The repo is public; `main` must stay clean and working. Branch names: `feat/<experiment>` (hyphens, for example `feat/mount-curriculum`), `docs/<topic>`, `chore/<topic>`.
- Branch before the first edit, not after. Keep the working tree clean: no stray debug files, no half-finished changes left on `main`.
- Commit/push only on explicit user request; merge to `main` only when the user says so. Create, move, or delete a git tag only on request. Never move or delete the tag `two-policy-chain`.
- Commit subject: `<experiment or scope>: <summary>`, lowercase, 72 characters or fewer. The commit body names the run IDs and the doc file.
- **A change that alters how a registered task behaves makes a new task variant.** This covers observations, network, rewards, events, terminations, and curriculum. Put the change into a new task variant (own experiment name) or into a flag whose default keeps the old behavior. Never put it into an existing task, so old checkpoints keep loading and playing. Run the compatibility checks (section Compatibility).
- **Observation or network size changes** need extra care: a warm start into a new variant loads the actor where its inputs are unchanged. For new critic inputs, pad the old critic with zero-weight columns (`scripts/make_padded_init.py`) rather than starting it fresh: a fresh critic wrecked the warm-started actor within ~40 iterations (`bl3` first try, see docs/mount_latency). Note this in the run log.
- **Play sessions time out after 15 minutes:** every time you run a play or viewer command for the user (`uv run play`, a play wrapper, a script that opens a viewer), run it under `timeout 15m`, for example `timeout 15m uv run play ...`. This holds for each task and each checkpoint. Then no viewer stays on the GPU. `mjx play` applies the 15 minutes by itself, so do not wrap it.
- **Docs are one directory per experiment** (`docs/<experiment>/<experiment>.md` for the design, `docs/<experiment>/<experiment>_log.md` for its run log). Never mix several experiments in one log. Add each new experiment to the index table in `docs/README.md` in the same commit. See `docs/README.md` for the index and conventions.
- **One fact lives in one file.** Do not copy a fact into a second file. Do not move information from one file to another without the approval of the user. Fix an error in the file that holds the fact. Owners: this file owns the task list, the commands, the rules, and the detailed repository map; `docs/README.md` owns the experiment index (status, tasks, code package, run ID prefix); the experiment docs own status and results; `docs/platform/platform.md` owns the platform task details; `README.md` owns the short overview of the repository for people.
- **Artifacts:** `logs/`, `wandb/`, `data/`, and `deploy/` are never committed. Never delete a file under `logs/` or `wandb/` without approval. Never run `mjx clean` without approval. Never delete or overwrite a checkpoint that `docs/baseline/task_fingerprints.json` lists under `protected_checkpoints`.
- **Hot experiments:** the index table marks the active experiments (`Status` = `active`). Do not rename, reformat, or restructure their files without approval.
- **Before you ask for a commit:** run `uv run pytest`, run `ruff check` on the files you changed, and run `ruff format` on the files you created.

## Skills

- **`mjlab-skillkit`** (third-party content from `cmjang/mjlab-skillkit`): at session start, if `.claude/skills/mjlab-skillkit/SKILL.md` does not exist, ask the user once whether to install it for this project only (never globally). Tell the user that the content is third-party and that they must review it for anything malicious or unsafe. Do not install it without an explicit "yes". If the user declines or does not answer clearly, install nothing.
- Download only the pinned file `https://raw.githubusercontent.com/cmjang/mjlab-skillkit/50da882cac069223dbd30fc60c15b1fd37b75dfb/SKILL.md`. Its SHA-256 must be `e40448b63b82b472d8cb45580238f2e03d6319356d6d4fdc7449d34e76ed9c29`. If the hash differs, stop and tell the user. Move the pin only with the approval of the user.
- Save the file as `.claude/skills/mjlab-skillkit/SKILL.md` in the project root (Claude Code) or in the local project skill directory of the agent (other agents). Then confirm the path and that the skill is available.

## Commands

```bash
# Train. --name is required: the script stops with an error (exit code 1) without it.
# The task menu (when no task is given) lists the live task registry (uv run mjx tasks).
bash scripts/train.sh --name <run-name>
bash scripts/train.sh Mjlab-Piplus-Platform --name <run-name> --num-envs 64
bash scripts/train.sh <task> --name <run-name> [overrides]   # wandb project = task id, default 4096 envs
bash scripts/train.sh <task> --name <run-name> --tmux          # wrap in tmux session mjlab-train, with a spare viewer window
bash scripts/train.sh <task> --name <run-name> --cpu           # CPU mode, 64 envs

# Train directly
uv run train Mjlab-Piplus-Platform --env.scene.num-envs 4096

# Overrides: reward weight
--env.rewards.<term>.weight <value>

# Resume (the iteration counter continues from N)
bash scripts/train.sh <task> --name <run-name> --agent.resume True --agent.load-run <run dir name> --agent.load-checkpoint model_N.pt --agent.max-iterations <additional>

# Evaluate a checkpoint (success by spawn mode and ball size)
uv run python scripts/eval_mountbalance.py --task <task> --checkpoint <model.pt>
#   --probe: posture and saturation; --audit: action size against the clip and the joint limits;
#   --gait: free-foot placement, floor taps, stance length, foot heading

# Check that no experiment broke (run after every change in src/)
uv run pytest
uv run python scripts/check_tasks.py --load --fingerprint --checkpoints

# Play back a checkpoint (interactive selection via mjx). mjx play ends by itself after 15 minutes.
uv run mjx play
uv run mjx --task Mjlab-Piplus-Platform play [--run <prefix>] [--ckpt N]

# Play directly. Start it under timeout 15m.
timeout 15m uv run play Mjlab-Piplus-Platform --checkpoint-file logs/rsl_rl/piplus_platform/<run>/model_N.pt --viewer viser --num-envs 1

# Manage runs
uv run mjx tasks                   # list the registered Pi Plus tasks (live registry, sorted)
uv run mjx runs                    # list runs with iteration count and age
uv run mjx clean --dry-run         # preview which checkpoints would be deleted
uv run mjx clean                   # DELETES intermediate checkpoints (keeps every 5th and each run's last); ask first
```

`scripts/train.sh` requires wandb login (`uv run wandb login` or `WANDB_API_KEY` env var). Parallel runs: stagger the launches by about 20 s. `MountBalance` tasks have a Spawn mode dropdown in the viewer. Logs, checkpoints, and run metadata: see `docs/README.md` (Conventions) and `docs/runs_index.md`.

Run `git config blame.ignoreRevsFile .git-blame-ignore-revs` once per clone. Then `git blame` skips the format-only commits that the file lists. Add the SHA of each new format-only commit to the file.

## Registered Tasks

One task ID per row (52 tasks). The index in `docs/README.md` maps each experiment to its tasks. A task ID that carries a flag name (`BlindOnBall`, `Latency`, ...) adds that flag to the task above it. The legend below the table explains each flag. `uv run pytest` fails when a registered task has no row here.

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
| `Mjlab-Piplus-Ball-Small-Turf` | Same, on astroturf (soft two-plane terrain). Writes into the log group of `Ball-Small` |
| `Mjlab-Piplus-Ball-Mount` | Step from the floor onto the ball and hold single-leg stance (feasibility spike, size 5 ball) |
| `Mjlab-Piplus-Ball-Mount-Flat` | Same, the sole must also be flat on the ball |
| `Mjlab-Piplus-Ball-Mount-Size1` | Same, on a size 1 football (radius 0.07 m) |
| `Mjlab-Piplus-Ball-Mount-Size1-Flat` | Size 1 football, flat sole |
| `Mjlab-Piplus-Ball-Balance-Size1` | Drop onto a size 1 ball and balance (flat sole) |
| `Mjlab-Piplus-Ball-MountBalance` | One policy for mount and balance on a randomized ball (size 1-5, mass, friction); 50/50 mount/drop spawn mix |
| `Mjlab-Piplus-Ball-MountBalance-Mix30` | Mount share 30% |
| `Mjlab-Piplus-Ball-MountBalance-Mix70` | Mount share 70% |
| `Mjlab-Piplus-Ball-MountBalance-Mix0` | Mount share 0% (drops only) |
| `Mjlab-Piplus-Ball-MountBalance-Mix100` | Mount share 100% (mount only) |
| `Mjlab-Piplus-Ball-MountBalance-Clip` | `MountBalance` with `Clip` |
| `Mjlab-Piplus-Ball-MountBalance-Mix100-Clip` | `Mix100` with `Clip` |
| `Mjlab-Piplus-Ball-MountBalance-Mix0-Clip` | `Mix0` with `Clip` |
| `Mjlab-Piplus-Ball-MountCurr-Release` | Mount-only curriculum on a randomized ball: ball locked by rolling resistance, then released |
| `Mjlab-Piplus-Ball-MountCurr-ReleasePot` | Same, with the potential-based progress reward |
| `Mjlab-Piplus-Ball-MountCurr-Pot` | Potential-based progress reward, ball never locked |
| `Mjlab-Piplus-Ball-MountCurr-Locked` | Ball locked for the whole episode, potential-based reward |
| `Mjlab-Piplus-Ball-MountMin` | Mount only, pruned reward set, free-foot lift measured at the sole, staged stance target (unbounded actions) |
| `Mjlab-Piplus-Ball-MountMin-Clip` | Same, bounded actions (`Clip`) |
| `Mjlab-Piplus-Ball-MountMin-Locked` | Same, ball locked |
| `Mjlab-Piplus-Ball-BalanceFromMount` | Balance on the randomized ball, starting from recorded hand-off states of the mount policy. Needs the states file `data/mount_states/mm4-dr-exc.pt` (not in git): make it with `scripts/make_mount_states.py`, command in its docstring |
| `Mjlab-Piplus-Ball-BalanceFromMount-Mix50` | Half of the resets start from a hand-off state, half are drops |
| `Mjlab-Piplus-Ball-BalanceFromMount-Clip` | Joint targets clipped (`Clip`), for smoothing |
| `Mjlab-Piplus-Ball-MountMin-BlindOnBall` | Mount and stance with the ball input only until on the ball (deployment candidate), bounded actions |
| `Mjlab-Piplus-Ball-MountMin-BlindOnBall-Latency` | Adds `Latency` |
| `Mjlab-Piplus-Ball-MountMin-BlindOnBall-Latency-Big` | Adds `Big` |
| `Mjlab-Piplus-Ball-MountMin-BlindOnBall-Latency-Big-OneFoot` | Adds `OneFoot` |
| `Mjlab-Piplus-Ball-MountMin-BlindOnBall-Latency-Big-OneFoot-Level` | Adds `Level` |
| `Mjlab-Piplus-Ball-MountMin-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp` | Adds `NoiseRamp` |
| `Mjlab-Piplus-Ball-MountMin-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp-Latch45-Hold` | Adds `Latch45` and `Hold` |
| `Mjlab-Piplus-Ball-MountMin-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp-Latch45-Hold-Deadline` | Adds `Deadline` to the `Hold` task (without `KinBall`) |
| `Mjlab-Piplus-Ball-MountMin-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp-Latch45-Hold-KinBall` | Adds `KinBall` |
| `Mjlab-Piplus-Ball-MountMin-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp-Latch45-Hold-KinBall-FreeLeg` | Adds `FreeLeg` |
| `Mjlab-Piplus-Ball-MountMin-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp-Latch45-Hold-KinBall-FreeLeg-Straight` | Adds `Straight` (cold start, thigh limited) |
| `Mjlab-Piplus-Ball-MountMin-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp-Latch45-Hold-KinBall-FreeLeg-Straight-Gate-Deadline` | Adds `Gate` and `Deadline` |
| `Mjlab-Piplus-Ball-MountMin-Curriculum` | One cold start through a staged curriculum: mount (`mm1` recipe) first, then clearance, smoothing, latency and big balls, one-foot and level stance, noise, blind on the ball (kinematic estimate), free leg. A stage advances on the running stance share. Thigh limited to ±0.6 rad |
| `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh` | Adds `FullThigh` |
| `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp` | Adds `Ramp` |
| `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess` | Adds `Excess` |
| `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap` | Adds `NoTap` |
| `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-MidThigh` | Adds `MidThigh` |
| `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-Turn` | Adds `Turn` to `NoTap` (not to `MidThigh`) |
| `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap` | Adds `LegGap` to `NoTap` (not to `Turn` or `MidThigh`) |

Flag legend:

- `Clip`: joint position targets clipped to the joint limits (bounded actions). On `MountBalance`, also a lower entropy bonus.
- `BlindOnBall`: the ball input reaches the actor only until the robot is on the ball. After the latch it is zero.
- `Latency`: 10-40 ms loop latency curriculum. The actor sees the encoder bias.
- `Big`: size 3-5 balls only.
- `OneFoot`: stance rewards only while the free foot does not touch the ball.
- `Level`: the support sole is near horizontal (foot on the ball top). The ankle roll stays off its stop.
- `NoiseRamp`: actor noise ramped in from 0.25×.
- `Latch45`: the on-ball latch fires at 45° base tilt (the 20° latch never fired).
- `Hold`: 1 s home-pose hold after reset.
- `Deadline`: the episode ends if the robot is not on the ball 3 s after the start.
- `KinBall`: on the ball the actor gets the ball centre computed from the support foot (kinematics) instead of zeros.
- `FreeLeg`: in single support the free leg is out of `posture` and is paid to clear the ball. The support hip and knee cost velocity (ankle balance).
- `Straight`: thigh (yaw) joint limited to ±0.6 rad in the model. The support foot is paid for pointing like the torso.
- `Gate`: the free-leg and straight terms act only in a real stance (CoM over the ball, free sole off the floor).
- `FullThigh`: full thigh range instead of ±0.6 rad.
- `Ramp`: stage weights ramped in. Action noise std capped at 1.
- `Excess`: `action_excess` penalty −1 from stage 0.
- `NoTap`: from stage 8, a penalty on free-foot contact with the floor or the ball while on the ball (`free_foot_contact`) and a bonus that grows with the length of the current stance (`stance_duration`). Reward change only.
- `MidThigh`: thigh (yaw) joint limited to ±1.2 rad in the model and in the action clip. The heading reward `support_foot_heading` has weight 0, so a turned support foot is allowed and not paid.
- `Turn`: full thigh range. The heading reward `support_foot_heading` has weight 0. The reward `support_foot_turn` pays for a support foot turned by about 100° from the torso.
- `LegGap`: full thigh range. From stage 8, a penalty `leg_gap` for a right leg that is closer than 2 cm to the left leg (distance between the collision primitives of calf, ankle pitch, and ankle roll). Reward change only: the model, the observations, and the network are the same as in `NoTap`.

## Architecture

This is an `mjlab` plugin package. `mjlab` is the RL framework (MuJoCo + RSL-RL + PPO). This repo contributes a robot description and tasks to it. [README.md](README.md) holds the short overview of the layout. This section holds the exact map.

**Task registration flow:**
1. `src/mjlab_piplus/tasks/__init__.py` — calls `import_packages(__name__)` which auto-imports all subpackages
2. Each `src/mjlab_piplus/tasks/<name>/__init__.py` — imports env/runner configs and calls `register_mjlab_task()`

**Adding a new task:** create `src/mjlab_piplus/tasks/<task_name>/` with `__init__.py` (register) and a config file; auto-imported. Follow the MJlab coding standard below. Add the task to the table in section Registered Tasks and to the index in `docs/README.md` in the same commit.

### Repository map

Experiments that use a code package are listed in the column `Code package` of the index in `docs/README.md`.

| Path | Role |
|---|---|
| `src/mjlab_piplus/__init__.py` | Re-exports `main` from `cli.py` (entry point `mjlab_piplus:main`) |
| `src/mjlab_piplus/cli.py` | The `mjx` command line (`tasks`, `runs`, `play`, `clean`). The task list and the menu come from the live registry. `play` ends after 15 minutes |
| `src/mjlab_piplus/naming.py` | `log_group(task_id)`: the log group of a task, with the alias table for the one known exception |
| `src/mjlab_piplus/robot/piplus_constants.py` | Robot spec loaders (`get_spec`, `get_spec_with_arms`), actuators, keyframes, collision presets, robot config factories |
| `src/mjlab_piplus/robot/piplus/` | Vendored MJCF `piplus.xml` and the STL meshes. Leave the XML unchanged |
| `src/mjlab_piplus/common/` | Helpers and constants that several task modules share. `rewards.py`: reward functions (`base_height_gauss`, `foot_flatness`, `xy_centering`, `xy_drift_gauss`). `balls.py`: the big ball (radius 0.45 m) constants, `get_ball_spec`, and its observation, reward, and terminations. `locomotion.py`: sensor and command names. The old task modules import these names from `common/`, so the old import paths still work |
| `src/mjlab_piplus/terrains/astroturf.py` | Astroturf terrain and the ground-softness event (`Ball-Small-Turf`) |
| `src/mjlab_piplus/tasks/upright/`, `noninertial/`, `platform/` | Standing tasks (flat ground, injected force, moving platform) |
| `src/mjlab_piplus/tasks/locomotion/`, `locomotion_arms/` | Velocity-tracked walking, legs only and with arms |
| `src/mjlab_piplus/tasks/ball/`, `ball_arms/`, `ball_small/` | Balance on a ball: large ball, large ball with arms, football (`ball_small` is the base of the later ball families) |
| `src/mjlab_piplus/tasks/ball_mount/` | `piplus_ball_mount_env_cfg.py`: Mount and Balance-Size1 tasks and the shared reward and observation functions. `piplus_ball_mountbalance_env_cfg.py`: one policy for mount and balance. `events.py`: reset and randomization events, viewer spawn-mode dropdown |
| `src/mjlab_piplus/tasks/ball_mount_curriculum/` | `MountCurr` tasks: ball lock and potential-based reward |
| `src/mjlab_piplus/tasks/ball_mount_min/` | `piplus_ball_mountmin_env_cfg.py`: the `MountMin` family with its feature flags. `piplus_ball_mount_curriculum_cfg.py`: the staged curriculum. `piplus_ball_mount_selfcontact_cfg.py`: the `LegGap` task (leg-to-leg distance penalty). `__init__.py`: the variant table |
| `src/mjlab_piplus/tasks/ball_balance_from_mount/` | Balance from recorded hand-off states |
| `scripts/train.sh` | Launch training with wandb: task menu from the live registry, `--name` (required), `--tmux`, `--cpu` |
| `scripts/eval_mountbalance.py` | Evaluate a checkpoint by ball size and spawn mode. Flags `--probe`, `--audit`, `--gait` and others |
| `scripts/eval_selfcontact.py`, `scripts/gen_selfcontact_report.py` | Self-contact and close-call probe of a policy (contacts and distances by body pair and gait phase), and its tables |
| `scripts/check_tasks.py` | Compatibility checks C1, C2, C3 (section Compatibility) |
| `scripts/make_runs_index.py` | Writes `docs/runs_index.md` from `logs/` and the local `wandb/` metadata |
| `scripts/make_mount_states.py` | Records the hand-off states that `BalanceFromMount` needs |
| `scripts/make_padded_init.py` | Pads a critic with zero-weight columns for a warm start |
| `scripts/export_mount.py`, `scripts/sim2sim_onnx.py` | ONNX export of a mount policy and its simulation check. Parked: they need the local, untracked `deploy/bitbots/` files |
| `tests/test_tasks.py`, `tests/test_docs.py`, `tests/test_cli.py`, `tests/test_naming.py` | Checks for the `mjx` command line, `scripts/train.sh`, `log_group()`, the task registry, the task fingerprints, this task table, the experiment index, and the links in tracked markdown files |
| `docs/README.md` | Experiment index and conventions |
| `docs/<experiment>/` | Design doc and run log of one experiment |
| `docs/runs_index.md` | Every training run (generated) |
| `docs/baseline/task_fingerprints.json` | Baseline for check C2 and the protected checkpoint hashes for C3 |
| `logs/rsl_rl/<log group>/<run>/`, `wandb/` | Training output. Not in git |
| `data/` | Datasets made from trained policies (`data/mount_states/`). Not in git |
| `deploy/` | Export files for the robot. Not in git |

Task modules import from each other. `ball_mount` imports `ball_small`. `ball_mount_curriculum`, `ball_mount_min`, and `ball_balance_from_mount` import the one-policy module and the events of `ball_mount`. `ball_mount_min` also imports `ball_mount_curriculum`, and its staged curriculum module imports the `MountMin` factory. A change in a base module changes every descendant. Rule S6 and check C2 cover this. The other packages are independent of each other.

**Robot config** lives in `src/mjlab_piplus/robot/piplus_constants.py`:
- `get_piplus_robot_cfg()` — legs-only `EntityCfg` (used by env configs). Its articulation uses `ACTUATOR_5036` for all leg joints.
- `get_piplus_robot_with_arms_cfg()` — adds the arm joints with `ACTUATOR_4438` (`PIPLUS_ARTICULATION_WITH_ARMS`).
- `get_spec()` — loads MuJoCo XML and repaints visual meshes dark gray (MJCF ships light gray; vendored file left unchanged)
- Collision presets: `FULL_COLLISION`, `LEG_ONLY_COLLISION`, `FEET_ONLY_COLLISION` — swap in env cfg to control what self-collides

**Env config** uses `mjlab`'s manager-based pattern: dicts of `ObservationTermCfg`, `ActionTermCfg`, `RewardTermCfg`, `TerminationTermCfg` passed to `ManagerBasedRlEnvCfg`. `play=True` disables observation corruption and sets episode length to infinity. Rules S5 to S10 give the conventions.

**Platform task:** the details (geometry, motion, rewards, events, sim sizes) are in [docs/platform/platform.md](docs/platform/platform.md).

**Sim timestep:** 0.005 s × decimation 4 = 0.02 s control step (50 Hz). This holds for all 52 tasks.

## Documentation standard

Write every document and every rule in controlled technical English. The style follows ASD-STE100 (Simplified Technical English, from the AeroSpace and Defence Industries Association of Europe).

- Write short sentences. Give one instruction or one statement per sentence.
- Use the imperative, the active voice, and the present tense. Example: "Run the script. Do not modify the config."
- Use one term for one concept. Use it everywhere. Do not use a synonym.
- Define each acronym and abbreviation at its first use in a document.
- Do not use vague words: "some", "several", "as needed", "etc.", "and so on". Give a number or a list.
- Prefer the simple verb forms: present tense, imperative, past participle for finished work. Avoid continuous forms.
- Write code identifiers, config keys, and established domain terms (MJlab terms) exactly as they are. Do not rephrase them.
- When a rule makes a passage unclear or wrong, keep the clear text and mark the deviation: `<!-- STE-deviation: reason -->`.
- When you edit an existing document, do not rewrite the surrounding text into this style. Style migration is a separate task that needs explicit approval.
- Do not cite a file that git does not track. Do not cite a line number. Cite `file::symbol`.
- Write the full launch command of a run in the run log. Copy it; do not paraphrase it.

## Naming convention

These rules apply to **new** names. Never rename a task ID, an `experiment_name` (log group), a Python module path, a script that a log cites, or a run directory: saved run files (`params/env.yaml`) and the docs cite them. Remove a task only on an explicit decision of the user. Record a former name of a doc in the index table of `docs/README.md`.

| Item | Rule | Example |
|---|---|---|
| Experiment name (doc directory, index key) | `<domain>_<topic>` or `<domain>_<topic>_<qualifier>`. Lowercase ASCII letters, digits, underscores. 2 or 3 words, 30 characters or fewer. `<domain>` is one of `stand`, `locomotion`, `ball`, `mount`, `balance`, `deploy`, `robot`. No version words (`v2`, `new`, `final`), no phase numbers, no hyphens. Never reuse a name, even of a deleted experiment. | `mount_curriculum` |
| Task ID | `Mjlab-Piplus-<Domain>-<Topic>[-<Variant>]`, PascalCase. One `<Variant>` segment names one feature flag. Use 2 variant segments or fewer. The cumulative chains continue only in the families `Ball-MountMin-BlindOnBall-…` and `Ball-MountMin-Curriculum-…`, with at most one new suffix per experiment. | `Mjlab-Piplus-Mount-NoTap` |
| Log group (`experiment_name` of the runner) | `piplus_` + the task ID without `Mjlab-Piplus-`, lowercase, `-` replaced by `_`. Create it with `log_group()` in `src/mjlab_piplus/naming.py`. 100 characters or fewer. Two tasks never share a log group (one old pair does: `Ball-Small` and `Ball-Small-Turf`). | `piplus_ball_mountmin_clip` |
| Run ID and run name | `<prefix><n>[<letter>][-<slug>]`. `<prefix>` has 2 to 4 lowercase letters and is unique per experiment. List it in the design doc and in the column `Run ID prefix` of the index. `<n>` counts fresh starts (cold or warm). `<letter>` marks a resume of the same run. `<slug>` has 3 words or fewer and names the one setting that differs. Always pass `--name`. Copies for warm starts use `init-from-<run ID>[-<checkpoint number>]`. | `cur4c`, `mm7-exc200-ar2` |
| Design doc and log | `docs/<experiment>/<experiment>.md` and `docs/<experiment>/<experiment>_log.md`. No other file in the directory, except images that the design doc embeds. | |
| Title lines (new docs) | Design: `# <experiment>: <one-line summary>`. Log: `# <experiment> run log`. Second block of the design doc: `Status:` (`planned`, `active`, `finished`, `parked`, or `superseded`, with a date), `Code:` (first and last commit), `Tasks:`, `Log:`. | |
| Headings of a design doc (new docs) | `Why`, `Design`, `Result`, `Next steps`, `Re-run`. A planned experiment needs only `Why` and `Design`. `Re-run` holds the full launch command, the run IDs it produced, and the commit. In a log, one `##` heading per run ID with the launch command, the wandb ID, and the outcome. | |
| Script | `<verb>_<object>.py` or `.sh`. `<verb>` is one of `train`, `eval`, `export`, `check`, `make`, `record`, `gen`. No experiment name in a script that serves more than one experiment. Keep only scripts for training, evaluation, export, checks, and datasets that a task needs. A diagnostic with lasting value becomes a flag of the eval script. A script that a log cites in a result must be tracked in git, or the log quotes its full text. A renamed script leaves a shim at the old path. | `make_mount_states.py` |
| Dataset | `data/<kind>/<name>.pt`. Name the dataset and the command that makes it in the `Re-run` section of the design doc. A task must not depend on a dataset that no tracked script can recreate. | `data/mount_states/mm4-dr-exc.pt` |
| Training output | Training writes only into `logs/rsl_rl/<log group>/<timestamp>_<run name>/`. Do not edit this tree by hand. | |

## MJlab coding standard

These rules apply to new and edited code. Do not reformat or restructure an existing file in a feature commit. A format-only change is its own commit and must leave the fingerprint check (C2) equal. Where the mjlab upstream differs (it indents with 2 spaces), this repository follows the rules below. `ruff` skips the hot and warm files (`extend-exclude` in `pyproject.toml`) until they are formatted.

| ID | Rule |
|---|---|
| S1 | Indent with 4 spaces. |
| S2 | Limit lines to 100 characters. Run `ruff format` on files you create. `[tool.ruff]` in `pyproject.toml` sets 100 characters and 4 spaces. |
| S3 | Order imports: standard library, third party (`torch`, `mujoco`), `mjlab`, `mjlab_piplus`. Use absolute imports across packages. Use relative imports for the config module inside its own `__init__.py`. `ruff check` (rule `I`) enforces the order. |
| S4 | Order a task module as follows: module docstring, imports, constants (`UPPER_SNAKE_CASE`), scene and entity builders, observation / reward / event / termination functions, `piplus_<family>_env_cfg`, `piplus_<family>_ppo_runner_cfg`. |
| S5 | Name the factory `piplus_<family>_env_cfg(play: bool = False, **flags) -> ManagerBasedRlEnvCfg`. Make every flag a keyword argument. Give every flag a default that reproduces the oldest behavior. Name a flag for the feature it enables (`blind_on_ball`). Do not use negated names. |
| S6 | A new family derives from at most one existing family. The derived module states its base in the docstring. Do not edit the behavior of a base family. Add a flag instead. Do not make the chain of bases deeper than 2 for a new family. |
| S7 | Build the robot only with `get_piplus_robot_cfg()`, `get_piplus_robot_with_arms_cfg()`, or `get_spec_with_arms()`. Do not copy a ball builder or the ball sizes into a new module. Import the big-ball constants and `get_ball_spec` from `mjlab_piplus.common.balls`, and the shared reward functions from `mjlab_piplus.common.rewards`. Import `BALL_SIZES` and the football builder from the nearest existing family (they are not in `common/` yet). |
| S8 | Use these observation term names: `joint_pos`, `joint_vel`, `projected_gravity`, `base_ang_vel`, `actions`. Name the groups `actor` and `critic`. Use 5 steps of history for tasks meant for the real robot. The actor never reads `base_lin_vel` or ball velocity (no real sensor). Only the critic may. Put a table of term names, sizes, and order in the design doc. |
| S9 | Use bounded raw actions (runner `clip_actions`) in every new task. Reason: unbounded outputs drifted past the clip in `cur2` and `cur3`. |
| S10 | Use named constants for the timestep (0.005), decimation (4), episode length, and the play value (1e9). Write `njmax` and `nconmax` explicitly and say why when they differ from 200 and 100. |
| S11 | Define the hyperparameters of a runner as module constants at the top of the module. Set `experiment_name` with `log_group(task_id)` from `mjlab_piplus.naming`. Keep the CLI values (`--agent.max-iterations`, number of environments, seed) in the run log. |
| S12 | Register a family in its `__init__.py` from one table `VARIANTS` (task suffix → flags) and one loop. Write one entry per line. Use no line longer than 100 characters. |
| S13 | Give each module a docstring that names the design doc. Give each public function a one-line docstring. State units and the sign in the docstring of a reward. Comments say why, not what. |
| S14 | Annotate all function signatures. |
| S15 | Use `snake_case` for functions. Name a function that another module imports without a leading underscore. |
| S16 | Do not align dictionary values with spaces. |
| S17 | Add each new task to the fingerprint baseline in the same commit (`uv run python scripts/check_tasks.py --update-baseline`) and state in the commit message why the baseline changes. |

How a new experiment follows the standard: copy the module layout of S4 from the nearest family, apply S5 and S6, add the variants table of S12, run `ruff format`, run the checks C1 and C2, and add the task to the task table, the baseline, and the index.

## Compatibility

All experiments stay backwards compatible. That means three things: every registered task still loads, every checkpoint that loaded before still loads into its task, and the documented re-run of an experiment still works.

- **After each change in `src/`:** run `uv run python scripts/check_tasks.py --load --fingerprint`.
  - C1 (`--load`): every task ID loads (environment config, play config, runner config). No two tasks share a log group, except the known pair `Ball-Small` and `Ball-Small-Turf`.
  - C2 (`--fingerprint`): the config of each task equals the baseline in `docs/baseline/task_fingerprints.json`. A difference you did not intend is a bug. A difference you intended needs a new baseline (`--update-baseline`) in the same commit, with the reason in the commit message. Reformatting must not change the fingerprint.
- **After each change that can touch checkpoints** (network, observation size, action size): also run `--checkpoints` (C3). Every checkpoint must load into a task of its log group (shape check), except the 10 old `Ball-Small` runs listed as known unloadable in `scripts/check_tasks.py`.
- Report the results of these checks when you finish.
- The task table (section Registered Tasks) must list every registered task, and the index in `docs/README.md` must have one row per directory in `docs/`. `uv run pytest` checks both.
- `docs/runs_index.md` is generated. After new runs, run `uv run python scripts/make_runs_index.py`.

## Roadmap: ball chain (find ball → walk → set stance → mount → balance)

Hierarchical: Approach policy → `(vx, vy, ωz)` → frozen Locomotion policy; separate Mount and Balance policies with a switch on the single-leg stance; switcher on distance to ball for walk → mount. One branch per phase. Each line gives the goal and the doc. The doc holds the status and the results; `docs/README.md` holds the one-word status.

- 0. Mount feasibility spike: can the robot step from the floor onto the ball? ([docs](docs/mount_feasibility/mount_feasibility.md))
- 1. Locomotion-Arms v2: proprio-only actor + history, noise, DR, ball obstacle, yaw-rate penalty. ([docs](docs/locomotion_s2r/locomotion_s2r.md))
- 2. Ball perception: noisy ball position in base frame (noise, dropout, latency, FOV). ([docs](docs/ball_perception/ball_perception.md))
- 3. Ball approach: high-level policy driving the frozen locomotion policy to the stand-off pose. ([docs](docs/ball_approach/ball_approach.md))
- 4a. One Mount+Balance policy, then a mount curriculum. ([docs](docs/ball_mount/ball_mount.md), [docs](docs/mount_potential/mount_potential.md))
- 4b. Pruned mount rewards with the lift measured at the sole. ([docs](docs/ball_mount_rewards/ball_mount_rewards.md))
- 4c. Balance policy trained from the mount policy's hand-off states. ([docs](docs/balance_from_mount/balance_from_mount.md))
- 4d. Balance and mount actions that a real actuator can follow, on the corrected model (two-policy state in tag `two-policy-chain`). ([docs](docs/balance_smooth/balance_smooth.md), [collision check](docs/collision_check/collision_check.md))
- 4e. One policy for mount and stance. ([docs](docs/mount_single/mount_single.md))
- 4f. One cold start through a staged curriculum. ([docs](docs/mount_curriculum/mount_curriculum.md))
- 5. Connect perception, approach, locomotion, and mount/balance in one chain. The task `Mjlab-Piplus-Ball-Chain` is dropped; the mount → balance chain is in the tag `two-policy-chain`. ([docs](docs/ball_chain/ball_chain.md))
- 6. Final docs, task table, README.
- 7. MLOps and deploy strategy: decide how a policy gets to the robot (export, versioned deploy files, robot node). ([docs](docs/deploy_mount/deploy_mount.md))
