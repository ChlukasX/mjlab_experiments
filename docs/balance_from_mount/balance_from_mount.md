# Balance from the mount policy's end states

Branch: `feat/balance-from-mount` (deleted; was off `feat/ball-mount-rewards`). Commits: `git log --oneline --grep '^balance-from-mount:'`. — **done**: 70% hand-off. [balance_smooth](../balance_smooth/balance_smooth.md) smoothed the policy and improved the hand-off to 96%.

Goal: make a balance policy that can take over from the mount policy. The chain is mount policy -> switch -> balance policy. Runs: [balance_from_mount_log.md](balance_from_mount_log.md).

## Why

The mount policy ([ball_mount_rewards](../ball_mount_rewards/ball_mount_rewards.md)) gets onto the ball in 0.5 s. It does not hold the stance (the free foot taps the floor 12% of the time). The drop-balance policy `mb3-mix0` holds the stance for 20 s (97%). It survives 0 to 2% of the hand-offs. It has only seen the drop spawn (standing tall, at rest). It has not seen the mount policy's stance (base 5 cm lower, moving). On the real robot the balance policy always starts from the mount, never from a drop. Therefore train it on that start.

## Design

- **Recorded hand-off states.** The script `scripts/record_mount_states.py` (removed; now `scripts/make_mount_states.py`; [git history](../README.md#conventions-shared-by-all-experiments)) runs the mount policy (`mm3-c10-s16` `model_11997`, 4096 environments (envs), 60 s, training randomization and pushes). It saves the full simulation state each time the robot has just held the stance for 0.2 s with the free sole 6 cm clear. The switch will use the same condition. The state holds the robot root, the joints, and the ball (with velocities), and the ball radius. The file `logs/mount_states/mm3-c10-s16.pt` holds 262 056 states (not in git; regenerate with the script).
- **Spawn.** `reset_from_mount_states` (`tasks/ball_mount/events.py`) starts an env in a random recorded state. The ball radius of that state is the closest to the ball radius of the env (64 neighbours either side; radius mismatch < 0.1 mm). Checked: the state read back after the reset equals the recorded state. No env terminates on the first step.
- **Everything else as `MountBalance-Mix0`** (the setup of `mb3-mix0`): randomized ball, same observations and rewards, unbounded actions, entropy 0.01. One change: `single_leg_stance` (10 cm) and `free_foot_lift` measure the lift of the free foot at the sole.
- The spawn does not reproduce two items: the last action of the mount policy (the "last action" observation starts at 0) and the observation history. Suspect these first if the policy balances from the recorded states but fails the real hand-off.

| Task | Start | Log dir |
|---|---|---|
| `Mjlab-Piplus-Ball-BalanceFromMount` | 100% mount hand-off states | `piplus_ball_balancefrommount` |
| `Mjlab-Piplus-Ball-BalanceFromMount-Mix50` | 50% hand-off states, 50% drops | `piplus_ball_balancefrommount_mix50` |

Config: `tasks/ball_balance_from_mount/piplus_ball_balancefrommount_env_cfg.py`.

## Result

The criterion is met. `bfm1-warm` (resume of `mb3-mix0`, 5000 iterations on hand-off states only) gives 70% success in the real hand-off (50 / 70 / 86% by ball size; `mb3-mix0`: 0%). A training from scratch gives 66%. Mixing in drops does not help. The [log](balance_from_mount_log.md) gives the details.

## Success criterion

Measure the real hand-off, not the recorded states. The command is `scripts/eval_handoff.py --mount-ckpt <mm3-c10-s16> --balance-ckpt <new> --balance-task Mjlab-Piplus-Ball-BalanceFromMount --hold 0.2 --switch-clearance 0.06` (removed; [git history](../README.md#conventions-shared-by-all-experiments)). Success means a stance with the free foot 10 cm clear for 90% of the last 3 s. The success must be at least 50%, by ball size. `mb3-mix0` scores 0% there.

## Re-run

The `BalanceFromMount` tasks start from a file of recorded hand-off states of a mount policy. The file is not in git. Since Nov 6 2026 a tracked script makes the file. The tasks read it from `data/mount_states/mm4-dr-exc.pt`.

1. **Make the state file.** Use a graphics processing unit (GPU). The run takes about 100 s with 4096 envs and 60 s of simulation. The source is the mount policy `mm4-dr-exc` `model_14996`:

   ```bash
   uv run python scripts/make_mount_states.py \
       --checkpoint logs/rsl_rl/piplus_ball_mountmin_clip/2026-10-13_14-18-55_mm4-dr-exc/model_14996.pt \
       --out data/mount_states/mm4-dr-exc.pt
   ```

   The regenerated file (Nov 6 2026) holds 109,244 states, with a ball radius from 0.070 to 0.110 m. The original held 108,049 states. The random draws differ. Therefore results match statistically, not bit for bit.
2. **Train.** The example is the best balance policy `bs4-exc100-ar2` (resume of `bs3-exc25-ar2` `model_24995`). The full chain of resumes is in [runs_index.md](../runs_index.md):

   ```bash
   bash scripts/train.sh Mjlab-Piplus-Ball-BalanceFromMount-Clip --name bs4-exc100-ar2 \
       --agent.resume True --agent.load-run 2026-10-13_17-50-56_bs3-exc25-ar2 --agent.load-checkpoint model_24995.pt \
       --agent.max-iterations 4000 --env.rewards.action-excess.weight -2.0 --env.rewards.action-rate.weight -0.02
   ```
3. **Evaluate** from the recorded states:

   ```bash
   uv run python scripts/eval_mountbalance.py --task Mjlab-Piplus-Ball-BalanceFromMount-Clip \
       --checkpoint logs/rsl_rl/piplus_ball_balancefrommount_clip/2026-10-13_21-12-00_bs4-exc100-ar2/model_28994.pt
   ```

   The check of the regenerated file (Nov 6 2026, 1024 envs, 20 s, seeds 0, 1, 2) gives these results. `bs4-exc100-ar2` succeeds in 90%, 89%, and 89% of the episodes (by ball size 1-2 / 3 / 4-5: 89 / 92 / 89% for seed 0). The 96% in this doc is the real hand-off with a mount policy. It is a different measure. The script that measured it (`eval_handoff.py`) is in the tag `two-policy-chain`.

Runs before the model correction of Oct 13 2026 (fitted head box, randomized contact) used states recorded on the old model (`mm3-c10-s16.pt`). The current code cannot reproduce these runs. Check out the commit that `docs/runs_index.md` lists for the run.
