# Ball Mount + Balance (phase 4)

Branch: `feat/ball-mount-balance` (deleted). Commits: `git log --oneline --grep '^ball-mount-balance:'`. — **done**: the policy learned drop balance on a randomized ball (97%, `mb3-mix0`). No run produced a mount (the first run passed the stance test on tiptoe, see [ball_mount_rewards](../ball_mount_rewards/ball_mount_rewards.md)). The work continues in [mount_potential](../mount_potential/mount_potential.md).

Task IDs: `Mjlab-Piplus-Ball-MountBalance` (50/50 mix), `-Mix30`, `-Mix70`

This experiment trains one policy that covers both skills, step onto the ball and balance on it, on a domain-randomized ball. The later approach policy hands over to this skill ([ball_approach.md](../ball_approach/ball_approach.md)). The end goal is that the robot walks to a ball by itself, mounts it, and balances ([ball_chain.md](../ball_chain/ball_chain.md)). The experiment builds on [mount_feasibility.md](../mount_feasibility/mount_feasibility.md).

Config: `tasks/ball_mount/piplus_ball_mountbalance_env_cfg.py`, events in `tasks/ball_mount/events.py`. Run-by-run record and launch commands: [ball_mount_log.md](ball_mount_log.md).

## Spawn mix (experiment variable)

At every reset, each environment (env) starts in one of two modes:

| Mode | Start |
|------|-------|
| mount | Robot standing, ball 0.19 m + radius ahead of the base on the right-foot line (± 3 cm, ± 2 cm). Must step on, then balance. |
| drop | Robot at the home pose just above the ball (base at 2r + 0.37 m: sole 3 cm above the ball top) with the right foot over it. It settles onto the ball instead of falling onto it. Balance only. |

The **share of mount spawns is `mount_fraction`**. The baseline is **50/50**. Each different ratio is a separate task. This allows a comparison:

| Task | Mount share | Log dir | Run |
|------|-------------|---------|-----|
| `Mjlab-Piplus-Ball-MountBalance-Mix0` | 0% (drops only, diagnostic) | `piplus_ball_mountbalance_mix0` | `mb3-mix0` (97% drops) |
| `Mjlab-Piplus-Ball-MountBalance-Mix100` | 100% (mounts only, diagnostic) | `piplus_ball_mountbalance_mix100` | `mb3-mix100`, `mb4-w4`, `mb5-c/d` (0%) |
| `Mjlab-Piplus-Ball-MountBalance` | 50% | `piplus_ball_mountbalance` | `mountbalance-mix50-v1` |
| `Mjlab-Piplus-Ball-MountBalance-Mix30` | 30% | `piplus_ball_mountbalance_mix30` | `mb4-w5` (drops 93-100%, mounts 0%) |
| `Mjlab-Piplus-Ball-MountBalance-Mix70` | 70% | `piplus_ball_mountbalance_mix70` | `mb4-w6` (drops 97-100%, mounts 0%) |

More weight on drops trains balancing more (the harder skill in earlier runs). More weight on mounts trains the step-up and the hand-off. To find the best ratio, compare the evaluation below across these runs.

The drop used to start with the base 0.5 m above the ball top (sole about 0.16 m above its contact height, ~1.8 m/s on impact). In the first run (`mountbalance-mix50-v1`), every drop spawn failed within about 2 s on every ball size. The policy learned a fixed size 1 ball with that same drop (`balance-size1-flat-v1`, episode 950/1000). Therefore the failure comes from the harder mix (randomized size and mass, size not observed, mixed spawns). The lower drop is an easier curriculum choice, not a proven fix. The drop is now a gentle settle. `Ball-Balance-Size1` uses the same height.

**Choosing the spawn in play.** The play viewer (viser) has a **Spawn mode** dropdown under Commands. The options are *Random mix* (the task's ratio), *Mount*, and *Drop*. A change resets the env with that mode. Therefore a single `uv run mjx --task Mjlab-Piplus-Ball-MountBalance play` shows both modes. The dropdown exists only in play configs (`SpawnModeCommand` in `events.py`).

## Domain randomization

| What | Range |
|------|-------|
| Ball radius | 0.07 – 0.11 m (FIFA size 1 to 5), per env, fixed at startup (`dr.geom_size`) |
| Ball mass | Linear in radius from 0.14 kg to 0.43 kg, × U(0.8, 1.2); inertia 0.4·m·r² |
| Ball friction | 0.3 – 1.2 |
| Ground friction | 0.3 – 1.2 |
| Foot friction | 0.3 – 1.2 (existing) |
| Robot mass, COM, encoder bias, PD gains, pushes | As Ball-Small |

MuJoCo mixes the friction of two geoms with a max, unless the contact priorities differ. The geoms of the robot have priority 1. Therefore the friction of the foot governs foot-ball and foot-ground contact. As a result, ground friction mainly changes ball-ground rolling. There the effective friction is max(ball, ground) and skews high. The size 1 radius and mass (0.07 m, 0.14 kg) are assumptions for a mini ball. Check them against the real ball.

<!-- STE-deviation: the acronyms COM, PD, and FIFA first appear in the table above; the table stays unchanged. CoM is defined in the section "Rewards and terminations" -->

## Observations

- **Actor:** proprioception with 5-step history, plus the noisy ball position in the base frame (345 inputs). The ball size is **not** given. The policy has to infer the size. The cues are the height of the ball relative to the base in the position observation, plus its own dynamics. <!-- STE-deviation: "its own dynamics" kept from the source; the source does not say whose dynamics -->
- **Critic:** the actor terms plus base linear velocity, ball position and velocity, foot-ball offsets, and the exact values of ball radius, mass, ball friction, and ground friction (87 inputs).

## Rewards and terminations

The rewards and terminations are those of `Mjlab-Piplus-Ball-Mount-Flat`: foot on the ball with the sole flat, single-leg stance, `foot_to_ball_top`, `foot_lift`, `sole_flat`, upright, posture, and the smoothness penalties. The task reads the ball radius per env. Episodes last 20 s. See [mount_feasibility.md](../mount_feasibility/mount_feasibility.md) for the on-ball test. After the first run, two terms and one stricter test were added:

| Term | Weight | Purpose |
|------|--------|---------|
| `stand_tall` | +2.0 | With a foot on the ball: `exp(-((h − 0.30) / 0.10)²)`, h = base height above the ball top. |
| `free_foot_lift` | +1.0 | With one foot on the ball: lift of the other foot, 0 to 1 over 10 cm of clearance. |
| `single_leg_stance` test | | Other foot must now clear the floor by 10 cm (ankle above 0.15 m), was 3 cm (0.08 m). |

| `support_on_ball` | +4.0 | Foot flat on the ball and whole-robot CoM within 0.12 m of the ball centre (`foot_on_ball` is now only a +1.0 touch bonus). |
| `com_over_ball` | +2.0 | With a foot on the ball: `exp(-(d / 0.25)²)`, d = CoM-to-ball distance. |
| `com_toward_ball_velocity` | +1.0 | Momentum: CoM speed toward the ball while it is not yet over it. |
| `push_up_velocity` | +0.5 | Upward CoM speed while the base is below the standing-tall height. |

`stance` also requires the centre of mass (CoM) within 0.12 m of the ball centre. `xy_centering` is off (weight 0). The last four terms were added after the `mb2-*` runs. The runs showed the robot standing beside the ball with a foot propped on it (CoM 0.26 m away). See [ball_mount_log.md](ball_mount_log.md).

Why: In the first run, all mount spawns succeeded (100%, on the ball within 0.26 s and held for 20 s), but as a deep crouch. The base sat 0.19 m above the ball top (about 0.29–0.30 m when standing tall on the home-pose legs). The free foot hovered 4.5 cm above the floor, which passed the old 3 cm test. A separate hand-off reward is not needed: the mount-to-stance chain already completes immediately. Therefore the new terms shape the quality of the balance. On the mount spawns of that run, the new terms score 0.30 (`stand_tall`) and 0.43 (`free_foot_lift`). None of the mount spawns pass the 10 cm stance test.

## Training

The actor and the critic are 512 × 256 × 128 networks with ELU (exponential linear unit) activation. They are wider than the 256 × 128 networks of the single-size runs, because the policy covers two skills over a size range. The run uses 8000 iterations and 4096 envs. It starts from scratch, because the observation sizes differ from the older checkpoints.

```bash
bash scripts/train.sh Mjlab-Piplus-Ball-MountBalance --name mountbalance-mix50-v1
```

## Evaluation

The first run `mountbalance-mix50-v1` (8000 iterations, before the changes above) was evaluated with `scripts/eval_mountbalance.py` and the old 3 cm test. Mount spawns: 100% success on every ball size. Drop spawns: 0% (dead within ~2 s). For later runs (`mb2` to `mb7`), see [ball_mount_log.md](ball_mount_log.md). Planned: the success rate (single-leg stance with the sole flat, held over the last 3 s) by ball radius {0.07, 0.09, 0.11} and spawn mode, per ratio. The reference is the single-size baselines (size 5 flat: on the ball 99.7% of the time).

## Open

- Jerky actions (`action_rate`) are not addressed here.
- Later: seed the mount spawn from the approach policy's end states.
