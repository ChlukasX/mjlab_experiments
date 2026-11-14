# mount_lineage: the experiments that led to the ball-mount policy `cc2-fall50`

Status: finished (Nov 14 2026).
Code: none. This document links the experiments; each experiment owns its code, its runs and its numbers.
Tasks: `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit-ComCenter` (the task of the selected policy).
Log: none.

## Why

The ball-mount policy took the 18 experiments below, from early Oct 2026 to Nov 14 2026. Each experiment has its own design doc and run log, so the path from the first attempt to the selected policy is spread over many files. This document gives that path in one place: what each experiment tried, what it found, and why the next one followed. It also lists the runs that rebuild the policy from scratch.

Each number below is quoted from the linked document, which owns it. Read the linked document for the conditions of the measurement.

## Result

**Selected policy:** `cc2-fall50` `model_36649`, task `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit-ComCenter`, log group `piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_actfit_comcenter`. The checkpoint is protected (`docs/baseline/task_fingerprints.json`). Selected by the user on Nov 14 2026 after play.

The robot stands beside a free ball (size 3 to 5), steps onto it, and holds a single-leg stance with the other foot in the air. One policy does the mount and the stance. The actor reads only signals that the robot has: joint encoders (with the encoder bias), the IMU (inertial measurement unit), its own last action, and the ball position until it is on the ball (after that, a ball estimate from the leg kinematics). The training model has the leg actuator dynamics and the loop latency (10 to 25 ms) fit to the RoboCup 2026 game rosbags.

| Policy | Survived | Held-stance success | Floor taps per 10 s | Source |
|---|---|---|---|---|
| `af1-actfit` (before the last experiment) | 98% | 0% | 12.4 | [mount_actuator_fit log](../mount_actuator_fit/mount_actuator_fit_log.md) |
| `cc2-fall50` `model_36649` (selected) | 82% | 75% | 0.4 | [mount_com_center log](../mount_com_center/mount_com_center_log.md) |

The policy is not exported yet. The export and the tests in the Bit-Bots stack continue in [deploy_mount](../deploy_mount/deploy_mount.md).

## Design: the path to the policy

### Phase 1: no real mount (Oct 2026)

| Experiment | What it tried | What it found |
|---|---|---|
| [mount_feasibility](../mount_feasibility/mount_feasibility.md) | Can the robot step from the floor onto a size 5 ball? | It looked feasible, but it was wrong: the stance test measured the ankle height, and the robots stood on tiptoe beside the ball |
| [ball_size1](../ball_size1/ball_size1.md) | The same on a size 1 ball | The same tiptoe error |
| [ball_mount](../ball_mount/ball_mount.md) | One policy for mount and balance on a randomized ball (`mb`) | Balance after a drop onto the ball works (`mb3-mix0`); no mount |
| [mount_potential](../mount_potential/mount_potential.md) | A ball lock, a potential-based mount reward, a CEM (cross-entropy method) search (`mc`) | No mount |

### Phase 2: the first real mount

| Experiment | What it tried | What it found |
|---|---|---|
| [ball_mount_rewards](../ball_mount_rewards/ball_mount_rewards.md) | Fewer rewards; the free-foot lift measured at the sole, not the ankle; the clearance staged 3 → 6 → 10 cm (`mm`) | **First real mount:** `mm1-min-clip`, a cold start. Every later mount policy up to `fl1` descends from it by warm starts |

### Phase 3: two policies, then one

| Experiment | What it tried | What it found |
|---|---|---|
| [balance_from_mount](../balance_from_mount/balance_from_mount.md) | A balance policy that takes over from the mount policy (`bfm`) | Hand-off works (`bfm1-warm`) |
| [balance_smooth](../balance_smooth/balance_smooth.md) | Smooth actions that a real actuator can follow (`bs`) | Better hand-off. The two-policy state is the tag `two-policy-chain` |
| [collision_check](../collision_check/collision_check.md) | Check of the collision model of the robot | Model corrections |
| [mount_single](../mount_single/mount_single.md) | One policy for mount and stance, no switch on the robot (`ms`, `bob`) | `ms3-shell`; `bob1` with the ball input only until on the ball. Decision (Oct 14 2026): one policy |
| [deploy_mount](../deploy_mount/deploy_mount.md) | Export of `bob2-up05` to the Bit-Bots stack, sim2sim check, robustness sweep | Export and sim2sim pass, but the policy fails above 15 ms of loop latency |

### Phase 4: robustness by warm starts

| Experiment | What it tried | What it found |
|---|---|---|
| [mount_latency](../mount_latency/mount_latency.md) | One feature per warm start: latency 10-40 ms, big balls, one foot on the ball, level sole, noise ramp, latch at 45°, kinematic ball estimate, free leg (`lat` → `of` → `lv` → `nr` → `kb` → `fl`) | Latency robust (`lat2-big`), but every policy since `mm1` stands with the support thigh turned by about 120° |
| [mount_straight](../mount_straight/mount_straight.md) | Cold starts with the thigh limited to ±0.6 rad and all features at once (`st`) | **Never mounted.** The mount was only learned on an easy task and carried forward by warm starts |

### Phase 5: one cold start through a curriculum

| Experiment | What it tried | What it found |
|---|---|---|
| [mount_curriculum](../mount_curriculum/mount_curriculum.md) | One run from a cold start through 9 stages: mount first, then clearance, smoothing, latency and big balls, posture, noise, blind on the ball, free leg; a stage advances on the running stance share (`cur`) | **Works:** `cur4` → `cur4b` → `cur4c` reach the last stage with a straight support leg. But the held-stance success is 0%: the policy **taps** the floor with the free foot to catch itself. Penalties on the tap (`-NoTap`), a middle thigh range (`-MidThigh`) and a reward for a turned foot (`-Turn`) did not change the gait |

### Phase 6: the tap

| Experiment | What it tried | What it found |
|---|---|---|
| [mount_selfcontact](../mount_selfcontact/mount_selfcontact.md) | A penalty for a right leg close to the left leg (`sg`) | Removed a foot-on-foot prop; the taps stayed |
| [mount_imu_noise](../mount_imu_noise/mount_imu_noise.md) | Half the actor noise on the IMU inputs (`im`) | Not the cause of the taps |
| [mount_tap_limit](../mount_tap_limit/mount_tap_limit.md) | End the episode on taps after the mount, with a budget, then a rate limit (`tl`) | Taps became falls: `tl1` learned to fall before its budget ran out; tighter limits took the survival to near 0 |
| [mount_actuator_fit](../mount_actuator_fit/mount_actuator_fit.md) | Leg actuator dynamics, PD (proportional-derivative) gain scales and latency fit to the game rosbags (`af`) | The fitted model costs `cur4c` much survival; `af1` recovers it on the fitted model. Still tapping |
| [mount_com_center](../mount_com_center/mount_com_center.md) | Dense, linear rewards for a CoM (centre of mass) over the ball centre and at rest relative to the ball, a stance gate that tightens from 0.12 to 0.04 m, then a cost per fall (`cc`) | **The tap gait is gone.** `cc2-fall50` holds the stance or falls. Selected |

### Lessons

1. **Measure what the reward pays for.** The ankle-height stance test passed robots on tiptoe for three experiments. The fix was a test at the sole and a probe that looks at the posture, not only at the reward.
2. **Learn the hard skill on an easy task first.** Every cold start with all features at once failed to mount (`mount_straight`). The mount came from an easy task (`mm1`), and later from the first stage of one curriculum run (`mount_curriculum`).
3. **Give a slope to the cause, not a penalty on the symptom.** The tap was a symptom: the CoM sat about 5 cm behind the ball centre, which is not an equilibrium on a free ball, and the binary stance test passed it. Tap penalties and tap terminations changed taps into falls. Dense terms for the CoM over the ball centre removed the tap.
4. **A termination needs a survival signal.** A termination that no policy can avoid gives no gradient and invites reward hacking (`tl1`).
5. **The model matters.** The fitted actuator dynamics changed the survival of the same policy by more than 20 points.
6. **Compare with one command.** `scripts/eval_policy.py` with a fixed seed and the same play config makes the numbers of different runs comparable.

## Next steps

- Export `cc2-fall50` `model_36649` to the Bit-Bots stack, run the sim2sim check and the robustness sweep, then test in the stack simulation: [deploy_mount](../deploy_mount/deploy_mount.md).
- Open questions of the policy: [mount_com_center](../mount_com_center/mount_com_center.md) (Next steps).

## Re-run

The selected policy comes from one cold start and six resumes or warm starts. No data file is needed. Run in this order; each linked section has the full launch command and the code state:

| Run | Start | Command |
|---|---|---|
| `cur4` | cold start | [mount_curriculum](../mount_curriculum/mount_curriculum.md) (Re-run) |
| `cur4b` | `cur4` `model_12800` | [mount_curriculum](../mount_curriculum/mount_curriculum.md) (Re-run) |
| `cur4c` | `cur4b` `model_21600` | [mount_curriculum](../mount_curriculum/mount_curriculum.md) (Re-run) |
| `af1-actfit` | `cur4c` `model_24450`, copied as `init-from-cur4c-24450` | [mount_actuator_fit log](../mount_actuator_fit/mount_actuator_fit_log.md) |
| `cc1-comcenter` | `af1-actfit` `model_27449`, copied as `init-from-af1-27449` | [mount_com_center log](../mount_com_center/mount_com_center_log.md) |
| `cc1b` | `cc1-comcenter` `model_31448` | [mount_com_center log](../mount_com_center/mount_com_center_log.md) |
| `cc2-fall50` | `cc1b` `model_33650` | [mount_com_center log](../mount_com_center/mount_com_center_log.md) |

The last checkpoint of `cc2-fall50` is the selected policy, `model_36649`.
