# Smoothing the balance policy: run log

One row per run, newest last; add a row at launch and fill in the result when it finishes. Design: [balance_smooth.md](balance_smooth.md). Shared conventions: [../README.md](../README.md).

## Round 1 (`bs1-*`)

Launched Oct 13 2026 on the working tree of `feat/balance-smooth` (on top of `f17038a`), all on cl06, 4096 envs, hand-off states only. All resume `bfm1-warm` `model_12998` for 4000 iterations with entropy 0.003 (for the `-Clip` task the checkpoint is copied to `piplus_ball_balancefrommount_clip/2026-10-13_00-00-00_init-from-bfm1-warm`).

| Run | Task | wandb | Joint targets | `action_excess` | `action_rate` | Question | Outcome |
|-----|------|-------|---------------|-----------------|---------------|----------|---------|
| `bs1-clip` | `BalanceFromMount-Clip` | `rygeha3v` | clipped | 0 | −0.01 | What do the clip and the lower entropy do alone? | Reward 208, episode 700/1000. Raw actions grew (no penalty on them); hand-off 41%. |
| `bs1-clip-exc` | `BalanceFromMount-Clip` | `ceufyscu` | clipped | −0.02 | −0.01 | Does the excess penalty bring the commands inside the limits? | **Best.** Reward 234, episode 722, `action_rate` −1.30 (was −2.9). Hand-off **83%**. |
| `bs1-clip-exc-ar5` | `BalanceFromMount-Clip` | `btz25yy8` | clipped | −0.02 | −0.05 | Does a 5x rate penalty on top smooth it, and at what cost? | **Collapsed:** episode 20 steps, reward 0.1. Hand-off 0%. |
| `bs1-exc-ar5` | `BalanceFromMount` | `w9yny0st` | unbounded | −0.02 | −0.05 | Is the clip needed when the penalties are there? | **Collapsed:** episode 16 steps. Hand-off 0%. |

All four finished (`model_16997`). Audit: `scripts/audit_policy.py` (removed; now `eval_mountbalance.py --audit`; [git history](../README.md#conventions-shared-by-all-experiments)), 256 envs, 10 s, hand-off-state starts. Hand-off: `scripts/eval_handoff.py` (removed; [git history](../README.md#conventions-shared-by-all-experiments)), 1024 envs, 20 s, mount policy `mm3-c10-s16`, `--hold 0.2 --switch-clearance 0.06` (`--balance-bounded` for the clipped ones).

| Policy | mean \|action\| (limit 1.0) | Targets outside limits (limit 2%) | Action change per step (limit 0.30) | Leg torque saturated | Hand-off success | By ball size (1-2 / 3 / 4-5) |
|---|---|---|---|---|---|---|
| `bfm1-warm` (start) | 22.6 | 58% | 0.95 | 64% | 70% | 50 / 70 / 86% |
| `bs1-clip` | 39.5 | 50% | 0.66 | 20% | 41% | 15 / 42 / 67% |
| `bs1-clip-exc` | **3.98** | 47% | **0.47** | **15%** | **83%** | 64 / 91 / 92% |
| `bs1-clip-exc-ar5` | 6.4 | | 0.87 | 25% | 0% | 0 / 0 / 0% |
| `bs1-exc-ar5` | 5.5 | | 0.84 | 58% | 0% | 0 / 0 / 0% |

**Findings**
- **Clip + `action_excess` made the policy both gentler and better:** actions 6 times smaller, leg torque saturated 15% of the time instead of 64%, action change halved, and the hand-off went from 70% to 83% (small balls 50% -> 64%). Not within the proposed limits yet.
- **The share of targets outside the limits barely moved (47%)** although their size did: at weight −0.02 the penalty is negligible once the excess is small (−0.39 per episode against a reward of 234). It needs a much larger weight for the last stretch.
- **A 5 times larger `action_rate` applied in one step destroys the policy** (episodes of 16 to 20 steps), with or without the clip: ending the episode is cheaper than paying the penalty. Raise it in smaller steps, or with a termination penalty.
- **The clip alone is not enough** (`bs1-clip`): without a penalty the raw actions grow further (mean 39.5) and the hand-off drops to 41%.

## Round 2 (`bs2-*`)

Resume `bs1-clip-exc` `model_16997`, 4000 iterations, task `BalanceFromMount-Clip`, cl06. Launched Oct 13 2026.

| Run | wandb | `action_excess` | `action_rate` | Other | Question | Outcome |
|-----|-------|-----------------|---------------|-------|----------|---------|
| `bs2-exc10` | `6lstb2od` | −0.2 | −0.01 | | Does a 10x excess penalty bring the targets inside the limits? | Reward 243, episode 701. Hand-off 68%. |
| `bs2-exc50` | `6ikof7vq` | −1.0 | −0.01 | | 50x: how far can it go before the balance suffers? | Reward 277, episode 784. Almost within the audit limits, but hand-off **16%**. |
| `bs2-exc10-ar2` | `ai8zz1f9` | −0.2 | −0.02 | | Is a 2x rate penalty survivable? | **Best trade.** Reward 285, episode 812. Hand-off 73%. |
| `bs2-exc10-ar2-term` | `t7xxdw2v` | −0.2 | −0.02 | `termination_penalty` −200 | Does a termination penalty protect against the collapse? | Reward 285, episode 833. Hand-off 60%; no gain from the termination penalty. |

All four finished (`model_20996`), evaluated on the model they were trained on (a worktree of `dbd8667`: the head box and contact changes came in after the launch).

| Policy | mean \|action\| (limit 1.0) | Targets outside limits (limit 2%) | Action change per step (limit 0.30) | Leg torque saturated | Survive 10 s from recorded states | Hand-off success | By ball size (1-2 / 3 / 4-5) |
|---|---|---|---|---|---|---|---|
| `bs1-clip-exc` (start) | 3.98 | 47% | 0.47 | 15% | 87% | 83% | 64 / 91 / 92% |
| `bs2-exc10` | 1.31 | 14% | 0.34 | 14% | 92% | 68% | 40 / 78 / 86% |
| `bs2-exc50` | **1.05** | **2.6%** | **0.26** | 14% | 91% | 16% | 5 / 21 / 22% |
| `bs2-exc10-ar2` | 1.19 | 7.6% | 0.25 | **11%** | **94%** | **73%** | 49 / 82 / 90% |
| `bs2-exc10-ar2-term` | 1.44 | 14% | 0.29 | 12% | 91% | 60% | 41 / 69 / 70% |

**Findings**
- **The audit limits are within reach:** `bs2-exc50` is at 1.05 / 2.6% / 0.26 against limits of 1.0 / 2% / 0.30, starting from 22.6 / 58% / 0.95 two rounds ago. Leg torque saturation stays at 11 to 14%.
- **A doubled `action_rate` is survivable when the commands are already near the limits** (`bs2-exc10-ar2`), and gives the best balance from recorded states (94%).
- **The smoother the balance policy, the worse the real hand-off, while the balance from recorded states stays above 90%** (`bs2-exc50`: 91% and 16%). The recorded-state spawn starts with a zero "last action"; at the real hand-off the observation holds the mount policy's last action (mean |action| 7.9), far from anything a policy with actions around 1 has seen. This was the open gap of [balance_from_mount](../balance_from_mount/balance_from_mount.md); smoothing made it large.
- The termination penalty does not help.

**Changes for round 3**
- `scripts/record_mount_states.py` (removed; now `scripts/make_mount_states.py`; [git history](../README.md#conventions-shared-by-all-experiments)) also stores the mount policy's last action; `reset_from_mount_states` hands it to the new observation `last_action_restored`, which puts it into the action buffers after the spawn (the "last action" observation and the reference of the action-rate penalty).
- New robot and contact model (fitted head box, contact time constant 0.01 to 0.03 s; [collision_check](../collision_check/collision_check.md)). The mount policy is resumed on it first (`mm4-*` in [ball_mount_rewards_log](../ball_mount_rewards/ball_mount_rewards_log.md)); its hand-off states are then recorded again and the balance policy resumed on them.

## Round 3 (`bs3-*`): new model, new hand-off states with the last action

Hand-off states recorded again from `mm4-dr-exc` `model_14996` on the new model (108 049 states, `logs/mount_states/mm4-dr-exc.pt`, mean |last action| 4.2). Check: after a spawn the action buffers equal the recorded action exactly; no env terminates on the first step. Resume `bs2-exc10-ar2` `model_20996`, 4000 iterations, task `BalanceFromMount-Clip`, cl06. Launched Oct 13 2026 on the working tree of `feat/balance-smooth` (on top of `dbd8667`).

| Run | wandb | `action_excess` | `action_rate` | Question | Outcome |
|-----|-------|-----------------|---------------|----------|---------|
| `bs3-exc10-ar2` | `754lr8av` | −0.2 | −0.02 | Does the hand-off recover with the last action restored and the matching mount policy? | Reward 239, episode 715. Hand-off **96%**. |
| `bs3-exc25-ar2` | `fj9ttq3u` | −0.5 | −0.02 | Can the commands come inside the limits now without losing the hand-off (as `bs2-exc50` did)? | **Best.** Reward 223, episode 677. Hand-off **96%**, smallest commands. |
| `bs3-exc10-ar3` | `3j1ck8ys` | −0.2 | −0.03 | A further step in the rate penalty. | Reward 236, episode 711. Hand-off 94%. |

All three finished (`model_24995`). New model; hand-off with mount policy `mm4-dr-exc` `model_14996` (`--hold 0.2 --switch-clearance 0.06 --balance-bounded`).

| Policy | mean \|action\| (limit 1.0) | Targets outside limits (limit 2%) | Action change per step (limit 0.30) | Leg torque saturated | Hand-off success | By ball size (1-2 / 3 / 4-5) |
|---|---|---|---|---|---|---|
| `bs1-clip-exc` + `mm3-c10-s16` (old pair on the new model) | 3.98 | 47% | 0.47 | 15% | 77% | 58 / 84 / 87% |
| `bs3-exc10-ar2` | 1.62 | 17% | 0.32 | 13% | **96%** | 92 / 98 / 98% |
| `bs3-exc25-ar2` | **1.45** | **6.3%** | 0.29 | 13% | **96%** | 94 / 97 / 98% |
| `bs3-exc10-ar3` | 1.75 | 21% | **0.28** | 12% | 94% | 87 / 97 / 98% |

**Findings**
- **Hand-off 96%, also on the small balls (92 to 94%).** Restoring the mount policy's last action at the spawn, and recording the states from the mount policy that is actually used, closed the gap that smoothing had opened (`bs2`: 60 to 73%, and 16% for the smoothest).
- **Smoothness and hand-off no longer trade off** in this range: the run with the strongest excess penalty has the smallest commands and the same success.
- **Robust to the mount policy:** with the later mount policy `mm5-exc-ar2` (not the one the states were recorded from) `bs3-exc25-ar2` still gives 95%.
- Still above the limits on action size (1.45) and targets outside the range (6.3%); the action change is inside (0.29).

Best pair: mount `piplus_ball_mountmin_clip/*_mm5-exc-ar2/model_17995.pt`, balance `piplus_ball_balancefrommount_clip/*_bs3-exc25-ar2/model_24995.pt` (the defaults of `scripts/play_chain.py` (removed; [git history](../README.md#conventions-shared-by-all-experiments)); headless check, 512 envs, 20 s: 549 switches, 37 falls after a switch).

## Round 4 (`bs4-*`)

Resume `bs3-exc25-ar2` `model_24995`, 4000 iterations, same task and states, cl06. Launched Oct 13 2026.

| Run | wandb | `action_excess` | `action_rate` | Question | Outcome |
|-----|-------|-----------------|---------------|----------|---------|
| `bs4-exc100-ar2` | `md2qhg5v` | −2.0 | −0.02 | Do the commands come fully inside the joint limits? | **Best.** Reward 225, episode 655. Hand-off 96%; targets outside the limits 2.0%. |
| `bs4-exc50-ar3` | `0vb8qvp7` | −1.0 | −0.03 | Excess and rate one step further together. | Reward 238, episode 695. Hand-off 97%. |

Both finished (`model_28994`). Hand-off with mount policy `mm5-exc-ar2`.

| Policy | mean \|action\| (limit 1.0) | Targets outside limits (limit 2%) | Action change per step (limit 0.30) | Leg torque saturated | Hand-off success | By ball size (1-2 / 3 / 4-5) |
|---|---|---|---|---|---|---|
| `bs3-exc25-ar2` (start) | 1.45 | 6.3% | 0.29 | 13% | 95% | |
| `bs4-exc100-ar2` | **1.18** | **2.0%** | 0.24 | 11% | 96% | 92 / 98 / 97% |
| `bs4-exc50-ar3` | 1.35 | 3.4% | **0.23** | **10%** | 97% | 92 / 98 / 100% |

**Findings**
- **The balance policy is at the audit limits on two of three measures** (targets outside the joint limits 2.0% against 2%, action change 0.24 against 0.30) with the hand-off unchanged at 96%. From `bfm1-warm`: mean |action| 22.6 -> 1.18, targets outside 58% -> 2%, action change 0.95 -> 0.24, leg torque saturated 64% -> 11%, hand-off 70% -> 96%.
- **Mean |action| 1.18 against a proposed 1.0** is the one left. With all targets inside the joint ranges it measures how far the single-leg posture is from the home pose (1 unit = 0.5 rad), not bang-bang control, so that limit may not be the right one for this task.
- With the newest mount policy (`mm6-exc50-ar2`) the pair gives 94% (91 / 94 / 97%).

Balance smoothing stops here until the mount policy has settled; if its hand-off states move far, record them again and resume once more.
