# Balance from the mount policy's end states: run log

One row per run, newest last; add a row at launch and fill in the result when it finishes. Design: [balance_from_mount.md](balance_from_mount.md). Shared conventions: [../README.md](../README.md).

## Round 1 (`bfm1-*`)

Launched Oct 12 2026 on the working tree of `feat/balance-from-mount` (on top of `9d512dd`), 4096 envs, all on cl06 (the local GPU was in use by another user; four jobs give ~2.4 s per iteration). Warm starts resume `mb3-mix0` `model_7999` (copied into the experiment folder as `2026-10-12_00-00-00_init-from-mix0`) for 5000 iterations; the others train 8000 iterations from scratch.

| Run | Task | Start | wandb | Question | Outcome |
|-----|------|-------|-------|----------|---------|
| `bfm1-warm` | `BalanceFromMount` | `mb3-mix0` | `qbf3m09b` | Does the drop-balance policy adapt to the hand-off states? | **Hand-off 70% success.** 5000 it: reward 97, episode 499/1000, `single_leg_stance` 1.50 of 4, `stand_tall` 0.73, `action_rate` −2.92. |
| `bfm1-warm-mix50` | `BalanceFromMount-Mix50` | `mb3-mix0` | `14qm0q7d` | Does keeping drops in the mix help it adapt (or keep the drop skill)? | Hand-off 0%. 5000 it: reward 52, episode 278, `single_leg_stance` 1.06. Kept the drops, never learned the hand-off states. |
| `bfm1-scratch` | `BalanceFromMount` | scratch | `9uk2bz7l` | Is the balance learnable from the hand-off states alone? | Hand-off 66% success. 8000 it: reward 24, episode 197/1000, `single_leg_stance` 0.50 of 4. |
| `bfm1-scratch-mix50` | `BalanceFromMount-Mix50` | scratch | `gw4slcjf` | Same with drops in the mix. | Hand-off 61% success, drops 59%. 8000 it: reward 19, episode 168, `single_leg_stance` 0.46. |

Eval of the warm starts (`model_12998`, 1024 envs, 20 s, run on cl06). Success = stance with the free sole 10 cm clear, CoM over the ball, flat sole, for 90% of the last 3 s. Hand-off: mount policy `mm3-c10-s16` `model_11997` from the standing start, switch after 0.2 s in the stance with the free sole 6 cm clear (`scripts/eval_handoff.py --hold 0.2 --switch-clearance 0.06 --balance-task Mjlab-Piplus-Ball-BalanceFromMount`).

| Balance policy | From recorded hand-off states | From drops | **Real hand-off** | Hand-off by ball size (1-2 / 3 / 4-5) |
|---|---|---|---|---|
| `mb3-mix0` (before) | | 97% | 0% | 0 / 0 / 0% |
| `bfm1-warm` | 90% | 0% | **70%** | 50 / 70 / 86% |
| `bfm1-warm-mix50` | 0% | 98% | 0% | 0 / 0 / 0% |

Posture of `bfm1-warm` over the last 3 s (recorded-state starts): CoM 0.011 m from the ball centre, base 0.315 m above the ball top, free sole 0.170 m above the floor, never on the floor, stance held 100% of the time. That is the drop policy's tall stance.

**Findings**
- **The chain works in simulation for the first time:** standing beside the ball -> mount policy -> switch at 0.7 s -> balance policy holds a tall single-leg stance to the end of the 20 s episode in 70% of the episodes (target was 50%). In the hand-off every env that survives also succeeds.
- **Training on the hand-off states is what did it:** the same policy before this training scores 0%.
- **With half drops in the mix the policy keeps the drops and ignores the hand-off states** (98% and 0%), the same one-skill outcome as the earlier mount/drop mixes. The balance policy for the chain should train on hand-off states only; it loses the drop skill (0%), which the chain does not need.
- **Gap between recorded states (90%) and the real hand-off (70%),** and small balls are the weak bin in both (80% and 50%). Candidates: the last action and observation history are not reproduced at the recorded-state spawn; the eval's switch happens at the first qualifying moment, the recording also took later ones.

Eval of the runs from scratch (`model_7999`, same procedure; success, in brackets survived):

| Balance policy | From recorded hand-off states | From drops | Real hand-off |
|---|---|---|---|
| `bfm1-scratch` | 57% (70%) | 0% (0%) | 66% (70%) |
| `bfm1-scratch-mix50` | 55% (70%) | 59% (75%) | 61% (73%) |

- **The balance is learnable from the hand-off states without the warm start** (66% in the hand-off, against 70% for the warm start), in 8000 iterations instead of 5000 on top of 8000. The warm start is still the best and the cheapest.
- **From scratch the 50/50 mix learns both starts** (drops 59%, hand-off 61%), unlike the warm-started mix, which stayed with the drops. It is worse at each than the single-start policies.
- The training numbers of the scratch runs (episode 170 to 200 of 1000) understate the eval: training has pushes, observation noise and exploration noise.

## Status (Oct 12 2026)

Best pair: mount `piplus_ball_mountmin_clip/*_mm3-c10-s16/model_11997.pt`, balance `piplus_ball_balancefrommount/*_bfm1-warm/model_12998.pt`: 70% of the episodes go from standing beside the ball to a tall single-leg stance held to 20 s (50 / 70 / 86% for ball size 1-2 / 3 / 4-5). Open: the falls after the switch (30%), mostly on small balls; the gap between recorded states (90%) and the real hand-off (70%); a play task for the two-policy chain.
