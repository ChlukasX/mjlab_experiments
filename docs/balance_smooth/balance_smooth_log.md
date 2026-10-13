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

All four finished (`model_16997`). Audit: `scripts/audit_policy.py`, 256 envs, 10 s, hand-off-state starts. Hand-off: `scripts/eval_handoff.py`, 1024 envs, 20 s, mount policy `mm3-c10-s16`, `--hold 0.2 --switch-clearance 0.06` (`--balance-bounded` for the clipped ones).

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
| `bs2-exc10` | `6lstb2od` | −0.2 | −0.01 | | Does a 10x excess penalty bring the targets inside the limits? | |
| `bs2-exc50` | `6ikof7vq` | −1.0 | −0.01 | | 50x: how far can it go before the balance suffers? | |
| `bs2-exc10-ar2` | `ai8zz1f9` | −0.2 | −0.02 | | Is a 2x rate penalty survivable? | |
| `bs2-exc10-ar2-term` | `t7xxdw2v` | −0.2 | −0.02 | `termination_penalty` −200 | Does a termination penalty protect against the collapse? | |
