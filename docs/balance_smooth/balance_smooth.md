# Smoothing the balance policy

Branch: `feat/balance-smooth` (deleted; it started from `feat/ball-chain`, plus the audit tool from `feat/balance-deploy`). Commits: `git log --oneline --grep '^balance-smooth:'`. — **done, parked**: the balance policy `bs4-exc100-ar2` is at the audit limits with a 96% hand-off. On Oct 14 2026 the project changed to one policy ([mount_single](../mount_single/mount_single.md)). The tag `two-policy-chain` marks this two-policy state.

Goal: make a balance policy for the chain (mount policy -> switch -> balance policy). A real actuator must be able to follow its actions. The hand-off must not get worse. Runs: [balance_smooth_log.md](balance_smooth_log.md).

## Starting point

The audit tool is `scripts/audit_policy.py` (removed; now `eval_mountbalance.py --audit`; [git history](../README.md#conventions-shared-by-all-experiments)). It ran with 256 environments (envs) for 10 s in the play config. The thresholds are the proposals from [balance_deploy](../balance_deploy/balance_deploy.md). The table gives the results:

| Policy | mean \\|action\\| (limit 1.0) | Targets outside joint limits (limit 2%) | Mean action change per step (limit 0.30) | Leg torque saturated |
|---|---|---|---|---|
| balance `bfm1-warm` `model_12998` | 22.6 | 58% | 0.95 | 64% |
| mount `mm3-c10-s16` `model_11997` | 7.9 | 57% (clipped by the env) | 0.98 | 18% |

1 action unit = 0.5 rad of joint-target offset. The balance policy asks for targets about 11 rad from the home pose. It runs the leg actuators saturated two thirds of the time. This is bang-bang control through the torque limit. The mount policy is also outside the limits. The env clips its targets to the joint ranges, so its torque saturates far less. Its raw actions and their rate are not small either.

## Design

Two earlier results decide the approach. A training from scratch with stronger smoothness penalties never learns the task (`mount-smooth-v1`). A training from scratch with bounded actions and a low entropy bonus cannot balance (`mb7-mix0-clip`). Therefore **resume the working policy** (`bfm1-warm`) and change its constraints.

- **Bounded targets:** use the task `Mjlab-Piplus-Ball-BalanceFromMount-Clip`. It clips the joint targets to the joint ranges. Set entropy 0.003. The value 0.01 grows the action noise.
- **`action_excess` (new, weight 0 by default):** the term is the amount, in rad, by which the commanded targets lie outside the joint ranges, summed over the joints. It is 0 for any in-range target. Therefore it removes the bang-bang commands and does not pull the posture toward the home pose.
- **`action_rate`:** make it 5 times larger (−0.05).

Acceptance: the audit is within the three thresholds. The success in the real hand-off is at least 50%. The command for the hand-off is `scripts/eval_handoff.py ... --balance-task Mjlab-Piplus-Ball-BalanceFromMount-Clip --balance-bounded` (removed; [git history](../README.md#conventions-shared-by-all-experiments)). `bfm1-warm` scores 70%. The flag `--balance-bounded` (also in `scripts/play_chain.py`) keeps the joint-target clip after the switch.

## Re-run

`BalanceFromMount-Clip` needs the file `data/mount_states/mm4-dr-exc.pt`. Make the file, then train and evaluate `bs4-exc100-ar2`. The steps are under Re-run in [balance_from_mount](../balance_from_mount/balance_from_mount.md). The launch arguments of every round (`bs1` to `bs4`) are in [runs_index.md](../runs_index.md).
