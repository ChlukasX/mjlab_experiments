# Balance from the mount policy's end states

Branch: `feat/balance-from-mount` (off `feat/ball-mount-rewards`) — **done**: 70% hand-off; smoothed and improved to 96% in [balance_smooth](../balance_smooth/balance_smooth.md).

Goal: a balance policy that can take over from the mount policy, so the chain is mount policy -> switch -> balance policy. Runs: [balance_from_mount_log.md](balance_from_mount_log.md).

## Why

The mount policy ([ball_mount_rewards](../ball_mount_rewards/ball_mount_rewards.md)) gets onto the ball in 0.5 s but does not hold the stance (the free foot taps the floor 12% of the time). The drop-balance policy `mb3-mix0` holds the stance for 20 s (97%) but survives 0 to 2% of the hand-offs: it has only seen the drop spawn (standing tall, at rest), not the mount policy's stance (base 5 cm lower, moving). On the real robot the balance policy will always start from the mount, never from a drop, so that is the start it has to be trained on.

## Design

- **Recorded hand-off states.** `scripts/record_mount_states.py` runs the mount policy (`mm3-c10-s16` `model_11997`, 4096 envs, 60 s, training randomization and pushes) and saves the full simulation state (robot root, joints, ball, with velocities, and the ball radius) each time the robot has just held the stance for 0.2 s with the free sole 6 cm clear: the condition the switch will use. 262 056 states in `logs/mount_states/mm3-c10-s16.pt` (not in git; regenerate with the script).
- **Spawn.** `reset_from_mount_states` (`tasks/ball_mount/events.py`) starts an env in a random recorded state whose ball radius is closest to its own (64 neighbours either side; radius mismatch < 0.1 mm). Checked: the state read back after the reset equals the recorded one, and no env terminates on the first step.
- **Everything else as `MountBalance-Mix0`** (the setup of `mb3-mix0`): randomized ball, same observations and rewards, unbounded actions, entropy 0.01. One change: the free foot's lift is measured at the sole in `single_leg_stance` (10 cm) and `free_foot_lift`.
- Not reproduced at the spawn: the mount policy's last action (the "last action" observation starts at 0) and the observation history. If the policy balances from the recorded states but fails the real hand-off, this is the first suspect.

| Task | Start | Log dir |
|---|---|---|
| `Mjlab-Piplus-Ball-BalanceFromMount` | 100% mount hand-off states | `piplus_ball_balancefrommount` |
| `Mjlab-Piplus-Ball-BalanceFromMount-Mix50` | 50% hand-off states, 50% drops | `piplus_ball_balancefrommount_mix50` |

Config: `tasks/ball_balance_from_mount/piplus_ball_balancefrommount_env_cfg.py`.

## Result

Met: `bfm1-warm` (resume of `mb3-mix0`, 5000 iterations on hand-off states only) gives 70% success in the real hand-off (50 / 70 / 86% by ball size; `mb3-mix0`: 0%). From scratch: 66%. Mixing in drops does not help. Details in the [log](balance_from_mount_log.md).

## Success criterion

The real hand-off, not the recorded states: `scripts/eval_handoff.py --mount-ckpt <mm3-c10-s16> --balance-ckpt <new> --balance-task Mjlab-Piplus-Ball-BalanceFromMount --hold 0.2 --switch-clearance 0.06`, success (stance with the free foot 10 cm clear for 90% of the last 3 s) at least 50%, by ball size. `mb3-mix0` scores 0% there.
