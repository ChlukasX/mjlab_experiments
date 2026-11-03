# mount_straight run log

Design: [mount_straight.md](mount_straight.md).

## Round 1 (`st1`): cold start with the thigh limited to ±0.6 rad

Task `...-KinBall-FreeLeg-Straight`, cold start (no resume), 6000 iterations, 2048 envs, seed 42, `single_leg_stance` sole clearance 0.03 m (staged as `mm1`), the round 5 weights of mount_latency. Local 4090, 2.7 s per iteration. Launched Nov 3 2026 in tmux `mjlab-st1` (`logs/probes/run_st1.sh`; window `sweep` runs `after_st1.sh` when it ends, into `logs/probes/sweep_st1.txt`).

| Run | wandb | Outcome |
|-----|-------|---------|
| `st1` | `lqqrsv3j` | **Parked in a straddle, never mounted** (finished, `model_5999`; sweep: 0% success and 0% latched in every condition, survival 100%, 76% soft motors). Episode length ~980 (timeouts), falls ~0.05 per log interval; `single_leg_stance` 0 and `support_on_ball` 0 for the whole run, `free_foot_lift` 0.001. From iteration ~2000 it stands on the floor foot with the other sole level on the ball top: `sole_flat` 1.83 (of 2), `foot_to_ball_top` 1.62, `foot_lift` 0.93, and the new terms pay the straddle: `free_foot_ball_clear` 1.79 (of 2), `support_foot_heading` 0.88 (of 1). Latency at 10-40 ms and noise 1× from iteration ~1500 as planned. |

**Finding (`st1`):** the `-FreeLeg` and `-Straight` terms are gated by `_single_support` = exactly one level sole on the ball, which does not check the weight: a straddle (one foot on the ball, the other on the floor, CoM off the ball) passes. In the cold start they pay ~2.7 per episode for parking in that straddle, on top of `sole_flat`, `foot_to_ball_top` and `foot_lift` (~4.4), against 0 for any mount attempt that fails. `fl1` did not show it because it was warm-started from a mounting policy. Fix before a rerun: gate these terms on the real stance (CoM within 0.12 m of the ball centre, as `support_on_ball`, and/or the free sole off the floor), and/or the mount deadline (`-Deadline`).

## Round 2 (`st2`): terms gated on the real stance, mount deadline

Task `...-KinBall-FreeLeg-Straight-Gate-Deadline` (`gate=True`, `deadline=True`), on top of `-Straight`:
- **Gate** (`_single_support(gated=True)`): `posture` (free leg out), `free_foot_ball_clear`, `support_hip_vel` and `support_foot_heading` count single support only with the CoM within 0.12 m of the ball centre (`COM_OVER_BALL`) and the free sole more than 3 cm above the floor (`GATE_FREE_CLEARANCE`). The `-FreeLeg` and `-Straight` tasks keep the plain test (`fl1`, `st1` reproducible).
- **Mount deadline** (`-Deadline`, training only): the episode ends if the robot is not latched on the ball (base height > 0.42 m, tilt < 45°) 3 s after its start.
- Check `logs/probes/test_gate.py` (256 envs, steps 250-400): `st1` (parked) single support plain 100%, gated **0%**; `fl1` (stance on the ball) plain 95%, gated 76%; gated params and deadline present only in the new task.

Cold start as `st1` (`logs/probes/run_st2.sh`: 6000 iterations, 2048 envs, seed 42, clearance 0.03 m, the round 5 weights). Local 4090, 2.8 s per iteration. Launched Nov 3 2026 in tmux `mjlab-st2` (window `sweep` runs `after_st2.sh` when it ends, into `logs/probes/sweep_st2.txt`). First 60 iterations: falls 30 → 1.6 per log interval, then episodes end at the deadline (length 57 → 148 = 3 s, `mount_deadline` 12.5 per log interval); stance and heading reward 0, `foot_lift` 0.015; latency 0-15 ms, noise 0.28×.

| Run | wandb | Outcome |
|-----|-------|---------|
| `st2` | `5avigpq9` | **Stopped** at iteration 1648 (`model_1600`): stands still until the deadline. Episode length 150 (all deadline ends) from iteration ~200, falls ~0; `single_leg_stance`, `support_on_ball` 0 throughout; `foot_lift` per step flat at 6e-5 (`st1` lifted from iteration ~400: 3.6e-4, ~1e-3 by 1600); action noise std flat at 0.69. Latency 10-40 ms and noise 1× reached. The 3 s deadline (1 s of it the start hold) leaves 2 s, too short to find the step-up from the dense pulls; `st1` needed ~400 iterations of long episodes to start lifting. |

## Round 3 (`st3`): `st2`'s task, warm-started from `st1`

`st1` already does the first half with the straight thigh (foot level on the ball top, pointing like the torso); the gate takes away the extra pay for parking there and the deadline forces the step up. Task `...-Straight-Gate-Deadline`, resume `st1` `model_5999` (actor and critic; same inputs and network; copied to `piplus_ball_mountmin_blindonball_latency_big_onefoot_level_noiseramp_latch45_hold_kinball_freeleg_straight_gate_deadline/2026-11-03_00-00-00_init-from-st1`), 4000 iterations, 2048 envs, seed 42, clearance 0.03 m, the round 5 weights (`logs/probes/run_st3.sh`). The step counter comes from `st1`, so latency (10-40 ms) and noise (1×) are at their final values from the start. Local 4090. Launched Nov 3 2026 in tmux `mjlab-st3` (window `sweep` runs `after_st3.sh` when it ends, into `logs/probes/sweep_st3.txt`). First 80 iterations: episode length 34 → ~149 (all deadline ends), falls ~0.1 per log interval, `foot_lift` 0.09 and `sole_flat` 0.17 per episode (the parked pose of `st1` within the 3 s), stance reward 0.

| Run | wandb | Outcome |
|-----|-------|---------|
| `st3` | `xmvcad00` | Running. |
