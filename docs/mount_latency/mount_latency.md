# mount_latency: the mount policy under robot latency

Branch: `feat/mount-latency` (deleted; created from `feat/deploy-mount`). Commits: `git log --oneline --grep '^mount-latency:'`. Task `Mjlab-Piplus-Ball-MountMin-BlindOnBall-Latency`. Run log: [mount_latency_log.md](mount_latency_log.md).

## Why

The robustness sweep of the deployment candidate `bob2-up05` ([deploy_mount log](../deploy_mount/deploy_mount_log.md)) found that the policy fails above the trained 0-15 ms sensor-to-motor delay: 62% at 20 ms, 0% from 30 ms. A 20 ms delay of the joint or gyro signals alone takes the success to 3-12%. A ROS (Robot Operating System) loop is likely above 15 ms. The loop path is IMU (inertial measurement unit) → node → HCM → motor bus. The real value is not measured. <!-- STE-deviation: the source does not define HCM; the expansion is not guessed --> Zero latency is not realistic. Therefore the target is a policy that works across a high-latency range. A policy tuned to the low end is not the target.

## Changes against `MountMin-BlindOnBall` (`latency=True`)

- **Loop latency as actuator lag, 10-40 ms** (2-8 physics steps of 5 ms). The lag is resampled with hold probability 0.9 per physics step. This gives a correlated delay that jumps now and then (jitter). The task uses actuator lag alone, not observation delay. For the loop, a sensor delay and a command delay add up the same way. The actuator lag has 5 ms resolution. The observation delay has 20 ms resolution (one control step) and is sampled independently for each term. Then the joint readings and the gyro readings would get inconsistent ages.
- **Curriculum:** the range widens linearly from the old 0-15 ms to 10-40 ms over the first 1500 iterations (`latency_curriculum`, logged as `Curriculum/latency/lag_min|max`). A resume restores the step counter of mjlab from the checkpoint. Therefore the counter is zeroed in the init copy of the checkpoint.
- **Encoder bias** ±0.015 → ±0.03 rad. The actor reads the biased angles (`joint_pos_rel(biased=True)`), as the robot does.
- Play config: the final 10-40 ms range.

## Evaluation

**Result `lat1` ([log](mount_latency_log.md)):** with noise and ±0.03 rad bias, the success is 77 / 75 / 65 / 27% at 10 / 20 / 30 / 40 ms (`bob2-up05`: 81 / 6 / 0 / 0%). Size 3-5 balls reach 88-95% at 10-20 ms. Size 1-2 balls reach at most ~50%. Observation noise is the largest remaining cost.

**Caveat (see the [log](mount_latency_log.md)):** the `lat1` and `lat2-big` success rates count a two-foot stance. The free foot rests tilted on the ball (~40% of the load). It passes the single-leg test, because only a flat sole counts as on the ball. Strict success (the free foot never touches the ball): 0% for both.

**Result `of2-clip` ([log](mount_latency_log.md)):** `-OneFoot`: the free foot must not touch the ball. The raw output is clamped to ±7. Strict success, size 4-5, noise and bias: 88 / 93 / 85 / 41% at 10 / 20 / 30 / 40 ms. The robot stands on the flank of the ball with the ankle roll at its stop (level success 0%).

**Result `lv1` ([log](mount_latency_log.md)):** `-Level`: the support sole is near horizontal, with an ankle roll penalty. The posture is fixed (foot near the top, ankle off its stop). But the level success with noise and bias is only 63 / 58 / 35 / 6% at 10 / 20 / 30 / 40 ms (size 4-5 at 20 ms: 70-75%; 85% without noise).

**Result `nr1-up6` ([log](mount_latency_log.md)):** the run starts from `lv1` with a noise curriculum (0.25× → 1×) and `upright` ×3. Level success 65 / 66 / 40 / 7% at 10 / 20 / 30 / 40 ms (size 4-5 at 20 ms: 75-83%; 88% without noise); lean 32° → 28.5°. The gains are small. The noise remains the limit.

**Result `lat2-big` ([log](mount_latency_log.md)):** the run uses size 3-5 balls only and starts from `lat1`. Size 4-5 with noise and bias: 94 / 94 / 89 / 56% at 10 / 20 / 30 / 40 ms (`lat1` on the same balls: 92 / 92 / 85 / 45%). The target is met at 10-20 ms and just missed at 30 ms. The policy is an export candidate if the real latency is ≤30 ms.

Run the sweep of the deploy_mount log with `--obs-noise` on. The training-like condition is the new baseline. Target: ≥90% at every fixed latency from 10 to 40 ms, compared with `bob2-up05` at the same settings. Also run `--obs-delay 1` (20 ms sensor delay on top). This checks that the actuator-lag model covers a sensor-side delay.

## Rewards

This section lists the reward terms of `lat1` and `lat2-big`. Both tasks share them. The weights are the weights used in training. The task defaults differ, and the run passes the `bob2-up05` weights as overrides (see the [log](mount_latency_log.md)). In each step, the reward manager adds value × weight × dt (dt = 0.02 s). Most weights date from the `ball_mount_rewards` and `mount_single` experiments. The history line of each term links to them. Abbreviations in the table and in the term entries: CoM (centre of mass), PD (proportional-derivative), std (standard deviation).

| Term | Weight | Type | Purpose |
|---|---|---|---|
| `single_leg_stance` | 16.0 | reward | 1 per step in the target state: a flat foot on the ball, the free sole more than 10 cm clear, CoM within 0.12 m of the ball centre |
| `free_foot_lift` | 3.0 | reward | free sole's clearance graded 0 to 10 cm while one foot is on the ball (measured at the sole, so tiptoe does not count) |
| `support_on_ball` | 2.0 | reward | 1 per step with a flat foot on the ball and the CoM within 0.12 m (weight on the ball) |
| `stand_tall` | 2.0 | reward | base height above the ball top near 0.30 m (Gaussian, std 0.10) with a foot on the ball; against the crouch |
| `com_over_ball` | 0.0 | (disabled) | linear CoM-to-ball shaping; off since `mm3`, it did not help |
| `foot_to_ball_top` | 2.0 | reward | pull of the nearer ankle toward the ball's top point (Gaussian, std 0.15 m) |
| `foot_lift` | 1.0 | reward | higher ankle raised from standing height (0.05 m) to ball-top height, linear |
| `sole_flat` | 2.0 | reward | sole aligned with the ball surface normal, (1 + cos tilt) / 2, for a foot on the ball |
| `upright` | 2.0 | reward | torso tilt from vertical, exp(−sin² tilt / 0.5²) |
| `posture` | 0.3 | reward | weak pull of the 20 joints toward the home pose (per-joint std) |
| `joint_pos_limits` | −1.0 | penalty | joint angles outside 90% of the range |
| `action_rate` | −0.02 | penalty | squared change of the raw action between control steps |
| `joint_vel` | −0.005 | penalty | squared joint velocity |
| `action_excess` | −4.0 | penalty | commanded targets past the joint ranges before the clip (stops bang-bang commands) |
| `base_yaw_rate` | −0.1 | penalty | squared yaw rate of the base (removed the learned spin) |
| `torque_saturation` | −2.0 | penalty | applied actuator force above 80% of the limit |
| `torque_demand` | −0.5 | penalty | demanded PD torque past the limit, where `torque_saturation` has no gradient |

### Stance on the ball

The stance terms share one gate, `_support_foot(flat=True)`. A foot (ankle roll link) counts as on the ball when all three conditions are true:
- its ankle height is within ±0.03 m (`ON_BALL_Z_HALF`) of ball centre z + r + 0.045 m (`ON_BALL_Z_CENTER`),
- its horizontal distance to the ball centre is below r + 0.05 m (`FOOT_ON_BALL_XY_MARGIN`), and
- its sole is flat on the ball: the angle between the sole normal and the ball-centre-to-foot direction is below 0.3 rad (`MAX_SOLE_TILT`).

r is the randomized radius of each environment (`ball_radius=None`). Two helpers recur. `_com_to_ball` is the horizontal vector from the whole-robot CoM to the ball centre. `_sole_clearance` is the height of the lowest of the 4 sole corners above the floor (0 m for a foot that stands flat).

#### `single_leg_stance`, weight 16.0
- **Function:** `single_leg_stance_lifted` (`src/mjlab_piplus/common/rewards/stance.py`)
- **Measures:** 0 or 1 per step. It is 1 when one foot is on the ball with a flat sole, the other foot is not and its sole is more than 0.10 m above the floor, and the horizontal CoM-to-ball distance is below 0.12 m.
- **Active when:** the term is always evaluated. It is non-zero only in the full target stance.
- **Params:** `sole_clearance` 0.10 m (free sole's lowest corner above the floor); `com_max` 0.12 m; `flat` True (support sole within 0.3 rad of the ball normal). `lifted_z` 0.15 is passed but not read, because `sole_clearance` takes precedence.
- **Why / history:**
  - The original check of Ball-Small passed at floor height. Therefore the mount tasks got their own lift test ([mount_feasibility](../mount_feasibility/mount_feasibility.md)).
  - The clearance was raised from 3 to 10 cm, and the CoM gate was added. Reason: policies crouched or propped a foot on the ball without weight on it ([ball_mount](../ball_mount/ball_mount.md)).
  - Policies exploited the ankle-height test on tiptoe. Therefore `MountMin` measures at the sole with a staged clearance of 3, 6, 10 cm. The weight changed 8 → 16 in `mm3-c10-s16` (74% held at 10 cm against 71%, but jerkier). It stays at 16 ([ball_mount_rewards](../ball_mount_rewards/ball_mount_rewards.md), [log](../ball_mount_rewards/ball_mount_rewards_log.md)).

#### `support_on_ball`, weight 2.0
- **Function:** `support_on_ball` (`src/mjlab_piplus/tasks/ball_mount/piplus_ball_mount_env_cfg.py`)
- **Measures:** 0 or 1 per step. It is 1 when any foot is on the ball with a flat sole and the horizontal CoM-to-ball distance is below 0.12 m. Nothing is required of the other foot.
- **Active when:** the term is always evaluated. It is non-zero only with the weight over the ball.
- **Params:** `com_max` 0.12 m; `flat` True.
- **Why / history:**
  - The term was added at +4. Mount policies had stood on the floor foot with the other foot resting on the ball top (CoM 0.25 m away), while `foot_on_ball` still paid ([ball_mount log](../ball_mount/ball_mount_log.md)). The weight was rebalanced to 2 in `mb4`.
  - The term replaced `foot_on_ball` in `MountMin` (it contains it). Known weakness: 0.12 m is larger than every ball radius. Therefore a two-foot straddle already satisfies the term ([ball_mount_rewards](../ball_mount_rewards/ball_mount_rewards.md)).

#### `com_over_ball`, weight 0.0 (disabled)
- **Function:** `com_over_ball` (`src/mjlab_piplus/tasks/ball_mount/piplus_ball_mount_env_cfg.py`)
- **Measures:** with a foot on the ball, the value is `clamp(1 − d / 0.30, 0, 1)`, where d is the horizontal CoM-to-ball distance in m. The value is 0 to 1 per step. The term contributes nothing at weight 0.
- **Active when:** a foot is on the ball with a flat sole.
- **Params:** `linear_range` 0.30 m, the mount spawn distance (`COM_LINEAR_RANGE`); `flat` True.
- **Why / history:**
  - The term was added at +2 as a dense signal that shifts the weight onto the ball ([ball_mount log](../ball_mount/ball_mount_log.md)).
  - The earlier exp kernel already paid 0.89 in the straddle. Therefore `MountMin` made the kernel linear ([ball_mount_rewards](../ball_mount_rewards/ball_mount_rewards.md)).
  - At weight 0, `mm1-nocom` and `mm2-c06-nocom` matched or beat the runs with the term. Therefore the weight is 0 from `mm3` on (passed as an override) ([ball_mount_rewards log](../ball_mount_rewards/ball_mount_rewards_log.md)).

#### `stand_tall`, weight 2.0
- **Function:** `stand_tall_on_ball` (`src/mjlab_piplus/tasks/ball_mount/piplus_ball_mount_env_cfg.py`)
- **Measures:** with a foot on the ball, the value is `exp(−((h − 0.30) / 0.10)²)`, where h is the base height above the ball top (centre z + r) in m. The value is 1 at h = 0.30 m (`BASE_ABOVE_TOP_TARGET`). The value is 0 to 1 per step.
- **Active when:** any foot is on the ball with a flat sole. The term does not check the other foot or the CoM.
- **Params:** `flat` True.
- **Why / history:**
  - The term was added at +2. Reason: the first mount/balance run held the ball in a deep crouch, with the base 0.19 m above the ball top against ~0.30 m standing tall ([ball_mount](../ball_mount/ball_mount.md)).
  - Weight 4 (`mb2-tall4`) and removal (`mb2-nostand`) changed only how deep the crouch was ([ball_mount log](../ball_mount/ball_mount_log.md)). The weight stays at 2 in `MountMin`, where trained policies collect ~0.9 per step.

#### `free_foot_lift`, weight 3.0
- **Function:** `free_foot_lift` (`src/mjlab_piplus/tasks/ball_mount/piplus_ball_mount_env_cfg.py`)
- **Measures:** the value is `clamp(clearance / 0.10, 0, 1)` of the sole of the free foot (`FREE_FOOT_LIFT` 0.10 m). The value is 0 with the sole on the floor and 1 at 10 cm or more. The value is 0 to 1 per step.
- **Active when:** exactly one foot is on the ball with a flat sole. The term is 0 when both feet or neither foot are on the ball. The term does not check the CoM.
- **Params:** `sole` True (measured at the sole's lowest corner, not the ankle); `flat` True.
- **Why / history:**
  - The term was added at +1 (with `stand_tall`), because the free foot hovered only ~4.5 cm above the floor. The weight is 3 since the `mb4` rebalance ([ball_mount](../ball_mount/ball_mount.md), [log](../ball_mount/ball_mount_log.md)).
  - When the term was measured at the ankle, tiptoe collected up to ~45% of it without unloading the foot. `MountMin` measures at the sole ([ball_mount_rewards](../ball_mount_rewards/ball_mount_rewards.md)).
  - With weight 6 in `mb7` (with `stand_tall` 4, `single_leg_stance` 16), the clearance plateaued below 10 cm ([ball_mount log](../ball_mount/ball_mount_log.md)).

### Approach and orientation

None of these terms depends on the phase. `foot_to_ball_top` and `foot_lift` keep paying during the stance. An ankle on the ball sits above the top. Therefore `foot_lift` is 1 and `foot_to_ball_top` is ~0.91.

#### `foot_to_ball_top`, weight 2.0
- **Function:** `foot_to_ball_top` (`src/mjlab_piplus/common/rewards/mounting.py`)
- **Measures:** the value is `exp(−d² / 0.15²)`, where d is the distance in m from the origin of the nearer ankle roll link to the top point of the ball (centre + radius in world z). The value is 0 to 1 per step. The term is measured at the ankle, not at the sole. A foot that stands flat on the ball has its ankle ~0.045 m above the top. Therefore the term pays ~0.91 in the stance. At spawn (~0.3 m away) the term pays ~0.02.
- **Active when:** always.
- **Params:** `std` 0.15 m (near 0 beyond ~0.3 m); `ball_radius` None (radius of each environment).
- **Why / history:** the term is the first dense pull toward the ball. Its first run (`mount-spike-v1`) learned to stand but never stepped on. The kernel is flat for the first part of the lift. This is the reason for `foot_lift` ([mount_feasibility](../mount_feasibility/mount_feasibility.md), [log](../mount_feasibility/mount_feasibility_log.md)). The weight stays at 2 in `MountMin` ([ball_mount_rewards](../ball_mount_rewards/ball_mount_rewards.md)).

#### `foot_lift`, weight 1.0
- **Function:** `foot_lift` (`src/mjlab_piplus/tasks/ball_mount/piplus_ball_mount_env_cfg.py`)
- **Measures:** the value is `clamp((z_foot − 0.05) / (z_top − 0.05), 0, 1)`, where z_foot is the height of the higher ankle link, 0.05 m is the ankle height when standing on the floor (`STAND_FOOT_Z`), and z_top is the ball top height. The value is 0 when standing and 1 when an ankle reaches ball-top height. The term checks neither the horizontal position nor which foot lifts. The ankle height also rises on tiptoe.
- **Active when:** always.
- **Params:** `ball_radius` None (radius of each environment).
- **Why / history:** the term gives a gradient over the first part of the lift, where `foot_to_ball_top` is flat. The first version measured from 0.10 m and paid 0 throughout (the standing ankle is at 0.05 m). After the fix, mounting started at ~iteration 2000 ([mount_feasibility log](../mount_feasibility/mount_feasibility_log.md)).

#### `sole_flat`, weight 2.0
- **Function:** `sole_flat` (`src/mjlab_piplus/tasks/ball_mount/piplus_ball_mount_env_cfg.py`)
- **Measures:** for each foot on the ball, the value is `(1 + cos(tilt)) / 2`. The term takes the maximum over the feet (0 for a foot not on the ball). tilt is the angle between the sole normal (ankle roll link z axis) and the direction from the ball centre to the ankle (the local surface normal). The value is 1 for a flat sole, 0.5 at 90°, and 0 upside down. At the 0.3 rad "flat" threshold the term pays ~0.98.
- **Active when:** a foot is on the ball by position (ankle height within ±0.03 m of ball top + 0.045 m, horizontal distance below r + 0.05 m). The tilt is not part of this gate.
- **Params:** `ball_radius` None (radius of each environment).
- **Why / history:** the term was added with the flat-sole requirement. The first form, `exp(−tilt² / 0.2²)`, gave no gradient (median sole tilt 112°, none of 44672 samples under 17°). The linear-in-cos form learned the flat sole from scratch ([mount_feasibility log](../mount_feasibility/mount_feasibility_log.md)).

#### `upright`, weight 2.0
- **Function:** `upright` (`mjlab/tasks/velocity/mdp/rewards.py`)
- **Measures:** the value is `exp(−(g_x² + g_y²) / 0.5²)`, where g is the gravity unit vector in the torso link frame. Therefore g_x² + g_y² = sin²(torso tilt from vertical). The value is 0 to 1 per step. At the ~28° lean of the policy the term pays ~0.41 (with the old std 0.2236: ~0.01).
- **Active when:** always.
- **Params:** `std` 0.5 (in units of sin(tilt); overridden from √0.05 ≈ 0.2236); body `torso_link`.
- **Why / history:** the term is inherited from Ball-Small ([ball_small](../ball_small/ball_small.md)). In mount_single round 5, it paid 0.01 on the ball, because the narrow kernel is flat at the 28° lean. Widening the kernel to 0.5 (`bob2-up05`) made it pay 0.47 of 2. The lean stayed at 27-28° ([mount_single log](../mount_single/mount_single_log.md)).

#### `posture`, weight 0.3
- **Function:** `posture` (`mjlab/envs/mdp/rewards.py`)
- **Measures:** the value is `exp(−mean_j((q_j − q0_j)² / σ_j²))` over the 20 joints. It uses the true joint angles (the encoder bias does not enter). q0 is the home pose (`HOME_KEYFRAME_WITH_ARMS`, `src/mjlab_piplus/robot/piplus_constants.py`: hip pitch ±0.5, calf ±1.0, ankle pitch ±0.5, right +, left −, all others 0 rad). The value is 0 to 1 per step.
- **Active when:** always.
- **Params:** σ in rad: hip roll 0.10, hip pitch 0.40, thigh 0.40, calf 0.40, ankle pitch 0.20, ankle roll 0.15, shoulder pitch 0.40, shoulder roll 0.20, upper arm 0.50, elbow 0.30 (set in `ball_small/piplus_ball_small_env_cfg.py`).
- **Why / history:** the term comes from Ball-Small at weight 1 ("stay near home pose", [ball_small](../ball_small/ball_small.md)). The weight was lowered to 0.3 in the mount tasks so that the leg can lift ([mount_feasibility](../mount_feasibility/mount_feasibility.md)). It is unchanged since.

### Regularization penalties

The action covers all 20 joints. target = home + 0.5 × action, clipped to the joint ranges (`bounded_actions`). The actuators apply the target 10-40 ms late.

#### `joint_pos_limits`, weight −1.0
- **Function:** `joint_pos_limits` (`mjlab/envs/mdp/rewards.py`)
- **Measures:** the value is the sum over the 20 joints of the distance, in rad, by which the true joint angle lies outside the soft limits. The soft limits are 90% of the range in the MuJoCo model file (MJCF) about its centre (`soft_joint_pos_limit_factor` 0.9).
- **Active when:** the term is always active. It is 0 while every joint stays inside 90% of its range.
- **Params:** none.
- **Why / history:** the term is the standard term from the Ball-Small base config. It stays in the pruned `MountMin` set ([ball_mount_rewards](../ball_mount_rewards/ball_mount_rewards.md)). It was never tuned.

#### `action_rate`, weight −0.02
- **Function:** `action_rate_l2` (`mjlab/envs/mdp/rewards.py`)
- **Measures:** the value is Σ_j (a_t − a_t−1)² over the 20 raw policy outputs (before scale and clip; one unit = 0.5 rad of target).
- **Active when:** always.
- **Params:** none.
- **Why / history:**
  - The weight doubled from −0.01 to −0.02 from `mm5-exc-ar2` on, together with `action_excess`. Each smoothing step made the mount smoother and more successful ([ball_mount_rewards log](../ball_mount_rewards/ball_mount_rewards_log.md)).
  - A 5× rise at once (−0.05) destroyed the balance policy. Episodes lasted 16-20 steps, because ending the episode was cheaper ([balance_smooth log](../balance_smooth/balance_smooth_log.md)).
  - −0.03 did not lower the action change; `torque_demand` did ([mount_single log](../mount_single/mount_single_log.md)).

#### `joint_vel`, weight −0.005
- **Function:** `joint_vel_l2` (`mjlab/envs/mdp/rewards.py`)
- **Measures:** the value is Σ_j q̇_j² over the 20 joints, in (rad/s)².
- **Active when:** always.
- **Params:** none.
- **Why / history:** the term is a standard term from the base config. It stays in `MountMin` ([ball_mount_rewards](../ball_mount_rewards/ball_mount_rewards.md)). It was never tuned.

#### `action_excess`, weight −4.0
- **Function:** `action_excess` (`src/mjlab_piplus/tasks/ball_mount/piplus_ball_mountbalance_env_cfg.py`)
- **Measures:** the value is the sum over the 20 joints of the distance, in rad, by which the commanded target (home + 0.5 × action, before the clip) lies outside the hard joint range. The robot never receives such a target (it is clipped). The penalty sees how far the command goes past the limit.
- **Active when:** the term is always active. It is 0 for any target inside the range. Therefore it does not pull toward the home pose.
- **Params:** none.
- **Why / history:**
  - The term was added because the ball policies commanded targets ~11 rad from home, bang-bang through the torque limit (balance `bfm1-warm`: mean |action| 22.6, targets outside 58%, legs saturated 64%) ([balance_smooth](../balance_smooth/balance_smooth.md)).
  - The weight rose −0.2 → −1 → −4 on the mount (`mm4` to `mm7-exc200-ar2`): success 62% → 87%, mean |action| 1.24, targets outside 3.8%, action change 0.26 ([ball_mount_rewards log](../ball_mount_rewards/ball_mount_rewards_log.md)).
  - The term aims at the audit limits: mean |action| ≤ 1.0, targets outside the limits ≤ 2%, action change ≤ 0.30.

#### `base_yaw_rate`, weight −0.1
- **Function:** `base_yaw_rate_l2` (`src/mjlab_piplus/tasks/ball_mount/piplus_ball_mountbalance_env_cfg.py`)
- **Measures:** the value is ω_z², the square of the angular velocity of the base about the world vertical, in (rad/s)².
- **Active when:** always.
- **Params:** none.
- **Why / history:**
  - `mm7-exc200-ar2` spun at ~6 rad/s in every environment and depended on the spin (40% fell with the spin zeroed) ([mount_single](../mount_single/mount_single.md)).
  - At −0.05: spin 6.3 → 1.6-1.9 rad/s, success with the changed friction model 48% → 73-80%. At −0.01: too weak. At −0.1: no systematic spin (0.3-0.9 rad/s, no fixed direction), 94-97% ([mount_single log](../mount_single/mount_single_log.md)).

#### `torque_saturation`, weight −2.0
- **Function:** `torque_saturation` (`src/mjlab_piplus/tasks/ball_mount/piplus_ball_mountbalance_env_cfg.py`)
- **Measures:** the value is Σ max(0, |applied force| / force limit − 0.8) over the 20 actuators, in units of the force limit (0 to 0.2 each). The term uses the force that MuJoCo actually applied. This force cannot exceed the limit.
- **Active when:** the term is active when an actuator is above 80% of its force limit (`TORQUE_SATURATION_START`).
- **Params:** none.
- **Why / history:** the term was added because the legs sat at their limit 17% of the time in every mount policy. At −2 (`ms1-sat`), success rose 87% → 94%, but the saturation share barely moved (17% → 16%). The applied force stops at the limit, so there is no gradient beyond it. This is the reason for `torque_demand` ([mount_single](../mount_single/mount_single.md), [log](../mount_single/mount_single_log.md)). The 80% is a placeholder. The acceptable load of the real actuators is not known.

#### `torque_demand`, weight −0.5
- **Function:** `torque_demand` (`src/mjlab_piplus/tasks/ball_mount/piplus_ball_mountbalance_env_cfg.py`)
- **Measures:** the value is Σ max(0, |τ| / force limit − 1) over the 20 joints, in units of the force limit. τ = kp (target − q) − kv q̇ is the PD torque that the policy asks for. It uses the randomized gains of each environment (×0.8-1.2), the clipped target, and the true q and q̇.
- **Active when:** the term is active when the demanded torque exceeds 100% of the limit.
- **Params:** none.
- **Why / history:** the term was added because `torque_saturation` has no gradient once an actuator is at its limit. In `ms2-yaw10-demand`, leg saturation fell 16% → 9%, the actions were smaller and calmer, and success was 95%. The run continued on the hollow-shell ball (`ms3-shell`): 9% → 5% ([mount_single log](../mount_single/mount_single_log.md)).
- **Note (latency tasks):** τ uses the current target. The actuator applies a target 10-40 ms old. Therefore the term penalizes the demand of the command, not the torque that the actuator applies at that moment.

**Result `kb1` ([log](mount_latency_log.md)):** on the ball, the actor gets the ball centre from the pose of the support foot (kinematics, task `-KinBall`) instead of zeros (estimate error median 1.5 cm). The robot computes the centre without the camera. First blind balance: 93% survive 20 s latched at 10-20 ms with noise (zeros: 0%). Single-leg stance 77% of the time. Lean 20.7° (from 28.5°). Level success is only 8% (31% without noise). The held-stance test (≥90% of the last 3 s) drops out on the level sole and the 10 cm clearance.

**Result `fl1` ([log](mount_latency_log.md)):** `-FreeLeg`: the free leg is out of `posture` and is paid to clear the ball. The support hip and knee carry a velocity penalty. The policy is steadier: survival 97% at 20 ms (`kb1` 93%), lean 14.5° (20.7°), level foot 96% of the steps. But level success is 2% (8%): the free foot dips below the 10 cm test line more often. The free leg moved out only a little (ankle share 32 → 35%). <!-- STE-deviation: "a little" is a qualitative result; the number follows in the source -->

## Open

- The support leg is turned ~120° at the thigh (yaw) joint in every policy since the `mm1` cold start. The next step is a cold start with the thigh limited, in [mount_straight](../mount_straight/mount_straight.md).

- The real loop latency: measure it on the robot (joint step command → encoder response, or IMU stamp → motor command + bus time). Then narrow or widen the range.
- Each episode starts with lag 0 until the first resample (the delay buffer of mjlab resets the lag to 0; with hold 0.9 that lasts ~50 ms). This is harmless at the start pose.
- `torque_demand` uses the current target while the actuator applies a delayed one (see the term). This is harmless as a penalty on the command. It is not the torque at that moment.
