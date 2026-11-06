# Ball mount: pruned rewards, staged stance target

Branch: `feat/ball-mount-rewards` (deleted; was off `feat/ball-mount-curriculum`). Commits: `git log --oneline --grep '^ball-mount-rewards:'`. — **done**: first real mount; the policy line continues as `mm4` to `mm7` (log below) and then in [mount_single](../mount_single/mount_single.md).

Goal: the mount (standing beside the ball to a single-leg stance on top of it) with a smaller reward set. Review of all earlier mount results, what was wrong with the rewards, and the tasks that test the fix. Runs: [ball_mount_rewards_log.md](ball_mount_rewards_log.md). Earlier rounds: [mount_feasibility](../mount_feasibility/mount_feasibility.md), [ball_mount](../ball_mount/ball_mount_log.md), [mount_potential](../mount_potential/mount_potential_log.md).

## Review of the earlier results (Oct 10 2026)

Probe: `scripts/eval_mountbalance.py --probe` (1024 envs, 20 s, means over the last 3 s of the envs that survived). "Free sole" is the lowest corner of the lower foot's sole above the floor.

| Checkpoint | CoM to ball centre | Base above ball top | Free ankle above standing height | Free sole above floor | Free sole on floor (< 1 cm) |
|---|---|---|---|---|---|
| `mount-flat-v1` `model_4999` (size 5, "mounts 99.7%") | 0.296 m | 0.162 m | 0.047 m | 0.003 m | 98% |
| `mountbalance-mix50-v1` `model_7999`, mount spawns ("100% success", 3 cm test) | 0.239 m | 0.186 m | 0.044 m | −0.004 m | 100% |
| `mb7-clip-ent-final-x2` `model_11998` (straddle) | 0.087 m | 0.229 m | 0.040 m | 0.006 m | 69% |
| `mb3-mix0` `model_7999`, drop spawns (the real stance, 97% success) | **0.011 m** | 0.322 m | 0.175 m | 0.170 m | 0% |

**No policy has ever mounted the ball.** The two runs recorded as successful mounts stand on the floor on the toes of the left foot with the right foot resting on the ball: the centre of mass is 0.24 to 0.30 m from the ball (it starts 0.26 to 0.30 m away), and the left sole touches the floor. The stance test measured the height of the ankle link, and tiptoe raises that link by 4.5 cm, which passed the 3 cm test. The finding "mounting is feasible in sim" in [mount_feasibility](../mount_feasibility/mount_feasibility.md) rests on that test and is not established.

Causes of the failure, in order:

1. **The lift tests measure the ankle, not the sole.** `single_leg_stance`, `free_foot_lift`, `straddle_penalty` and the potential's lift term all use the ankle link height. Tiptoe collects the 3 cm stance reward and up to ~45% of `free_foot_lift`, and escapes the straddle penalty (it tests ankle < 8 cm), without taking any weight off the floor. That explains why raising the test to 10 cm "broke" the mount (`mb2`: there was no mount to break) and why the straddle penalty did nothing (`mb5`).
2. **The centre-of-mass terms are satisfied in the straddle.** `com_over_ball` is `exp(-(d / 0.25)^2)`: 0.89 at the straddle's 0.087 m, 1.0 at the real stance's 0.011 m. `support_on_ball` switches on at 0.12 m, more than the radius of every ball. The last 8 cm of weight transfer, which is the whole mount, pays about 0.2 per step.
3. **In the straddle the ball leg carries little load.** `mb7`: no actuator of the right (ball) leg is at its force limit more than 2% of the time, while the left (floor) leg is (ankle pitch 41%, knee 21%, hip roll 18%).
4. **Bounded actions with a low entropy bonus cannot balance on the ball even when spawned there** (`mb7-mix0-clip`: 0% of drops survive; `mb3-mix0`, unbounded, entropy 0.01: 97%). All mount-curriculum runs used the bounded setting, so their policies had no way to hold a single-leg stance.
5. **Open: the step up is close to the knee's torque limit.** Robot 13.0 kg, thigh and shank 0.14 m each, leg actuators ±20 N m. With the foot on the ball top and the base at standing height the hip is 0.08 m (size 5) to 0.16 m (size 1) above the ankle; carrying the full weight there needs about 17 N m (size 5) and 15 N m (size 1) at the knee, static, before any acceleration or randomized mass. Not measured in simulation yet: no policy has loaded the ball leg.

Terms with no measured effect: `com_toward_ball_velocity`, `push_up_velocity` (`mb3-mom` = `mb3-com`), `straddle_penalty`, `termination_penalty` (`mb5`), `mount_potential` (dive and die, or stand still: `mc1`, `mc1b`), the ball lock (`mc1-release`).

## Design

Tasks `Mjlab-Piplus-Ball-MountMin[-Clip|-Locked]`, config `tasks/ball_mount_min/piplus_ball_mountmin_env_cfg.py`. Mounts only, randomized ball, flat sole. Built on the MountBalance config with these changes:

- **Removed:** `foot_on_ball` (contained in `support_on_ball`), `com_toward_ball_velocity`, `push_up_velocity`, `straddle_penalty`, `termination_penalty`, `xy_centering`; no potential reward. 13 terms remain.
- **Lift measured at the sole:** `single_leg_stance` (`sole_clearance`) and `free_foot_lift` (`sole=True`) use the lowest sole corner (`_sole_clearance`; 0.000 m standing flat). The eval success test uses it too.
- **`com_over_ball` linear:** `1 - d / 0.30`, the same slope from the spawn distance to the ball centre (`linear_range`).
- **Staged stance target:** `single_leg_stance` pays from a free-sole clearance of 3 cm (`lifted_clearance`); later stages resume with 6 cm, then 10 cm. `free_foot_lift` (0 to 1 over 10 cm) and `stand_tall` grade the rest. The eval target stays 10 cm.

| Term | Weight |
|---|---|
| `single_leg_stance` (free sole 3 cm clear, CoM within 0.12 m, flat sole) | 8.0 |
| `free_foot_lift` (sole) | 3.0 |
| `support_on_ball`, `com_over_ball` (linear), `stand_tall`, `sole_flat`, `foot_to_ball_top`, `upright` | 2.0 each |
| `foot_lift` | 1.0 |
| `posture` | 0.3 |
| `joint_pos_limits`, `action_rate`, `joint_vel` | −1.0, −0.01, −0.005 |

| Task | Actions, runner | Ball | Log dir |
|---|---|---|---|
| `Mjlab-Piplus-Ball-MountMin` | unbounded, entropy 0.01, 24 steps, gamma 0.99 (the setting that balances) | free | `piplus_ball_mountmin` |
| `Mjlab-Piplus-Ball-MountMin-Clip` | joint targets clipped, entropy 0.003, 64 steps, gamma 0.995 | free | `piplus_ball_mountmin_clip` |
| `Mjlab-Piplus-Ball-MountMin-Locked` | as the first | held by rolling resistance all training (probe for cause 5; eval on a free ball) | `piplus_ball_mountmin_locked` |

## Result so far

Three stages (3 cm from scratch, then 6 cm, then 10 cm; `mm1` to `mm3` in the [log](ball_mount_rewards_log.md)): the robot mounts the ball in 0.5 s with the free sole off the floor, survives 97 to 99% of 20 s episodes and is in the 10 cm stance 74% of the last 3 s (CoM 0.030 m from the ball centre, base 0.27 m above the top). The free foot still taps the floor 12% of the time, so the strict success (stance held 90% of the last 3 s) is 5% at 10 cm against a target of 50%, and the existing drop-balance policy does not survive a hand-off. `com_over_ball` turned out not to be needed either (12 terms). Best checkpoint: `piplus_ball_mountmin_clip/*_mm3-c10-s16/model_11997.pt`.

## Success criteria

- Stage 1: at least 50% of mounts hold the stance with the free sole 3 cm clear over the last 3 s (`scripts/eval_mountbalance.py --task <task> --free-foot-clearance 0.03 --probe`).
- Final: the same at 10 cm (the script's default), then the hand-off test (`scripts/eval_handoff.py` (removed; [git history](../README.md#conventions-shared-by-all-experiments))) with `mb3-mix0` as the balance policy.
- If `-Locked` does not load the ball leg either, test cause 5 directly (torque on the ball leg, nominal mass, a lower step) before more reward work.
