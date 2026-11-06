# Smoothing the balance policy

Branch: `feat/balance-smooth` (off `feat/ball-chain`, plus the audit tool from `feat/balance-deploy`) — **done, parked**: balance `bs4-exc100-ar2` at the audit limits with 96% hand-off. On Oct 14 2026 the project switched to one policy ([mount_single](../mount_single/mount_single.md)); this two-policy state is tagged `two-policy-chain`.

Goal: a balance policy for the chain (mount policy -> switch -> balance policy) whose actions a real actuator can follow, without losing the hand-off. Runs: [balance_smooth_log.md](balance_smooth_log.md).

## Starting point

`scripts/audit_policy.py` (256 envs, 10 s, play config; thresholds are the proposals from [balance_deploy](../balance_deploy/balance_deploy.md)):

| Policy | mean \\|action\\| (limit 1.0) | Targets outside joint limits (limit 2%) | Mean action change per step (limit 0.30) | Leg torque saturated |
|---|---|---|---|---|
| balance `bfm1-warm` `model_12998` | 22.6 | 58% | 0.95 | 64% |
| mount `mm3-c10-s16` `model_11997` | 7.9 | 57% (clipped by the env) | 0.98 | 18% |

1 action unit = 0.5 rad of joint-target offset. The balance policy asks for targets ~11 rad from the home pose and runs the leg actuators saturated two thirds of the time: bang-bang through the torque limit. The mount policy is also outside the limits; its targets are clipped to the joint ranges by the env, so its torque saturates far less, but its raw actions and their rate are not small either.

## Design

Earlier results that decide the approach: stronger smoothness penalties from scratch never learn the task (`mount-smooth-v1`), and bounded actions with a low entropy bonus from scratch cannot balance (`mb7-mix0-clip`). So: **resume the working policy** (`bfm1-warm`) and change its constraints.

- **Bounded targets:** task `Mjlab-Piplus-Ball-BalanceFromMount-Clip`, joint targets clipped to the joint ranges, entropy 0.003 (0.01 grows the action noise).
- **`action_excess` (new, weight 0 by default):** rad by which the commanded targets lie outside the joint ranges, summed over the joints. It is 0 for any in-range target, so it removes the bang-bang commands without pulling the posture toward the home pose.
- **`action_rate`** 5 times larger (−0.05).

Acceptance: the audit within the three thresholds, and at least 50% success in the real hand-off (`scripts/eval_handoff.py ... --balance-task Mjlab-Piplus-Ball-BalanceFromMount-Clip --balance-bounded`; `bfm1-warm`: 70%). `--balance-bounded` (also in `scripts/play_chain.py`) keeps the joint-target clip after the switch.

## Re-run

`BalanceFromMount-Clip` needs the file `data/mount_states/mm4-dr-exc.pt`. Make it, and train and evaluate `bs4-exc100-ar2`, as described under Re-run in [balance_from_mount](../balance_from_mount/balance_from_mount.md). The launch arguments of every round (`bs1` to `bs4`) are in [runs_index.md](../runs_index.md).
