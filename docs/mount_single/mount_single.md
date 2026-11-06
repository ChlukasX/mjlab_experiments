# One policy for mount and stance

Branch: `feat/mount-single` (deleted; created from `feat/balance-smooth`). Commits: `git log --oneline --grep '^mount-single:'`.

**Status (Nov 6 2026): superseded.** The single policy was reached here (`ms3-shell` 98%; `bob1` blind on the ball 95%). The policy was exported as `bob2-up05` ([deploy_mount](../deploy_mount/deploy_mount.md)). The work continued in [mount_latency](../mount_latency/mount_latency.md), [mount_straight](../mount_straight/mount_straight.md) and [mount_curriculum](../mount_curriculum/mount_curriculum.md). The open questions at the end of this file are still open.

Decision (Oct 14 2026): continue with a single policy that mounts the ball and holds the single-leg stance. The other option was a mount policy plus a balance policy with a switch. The mount policy `mm7-exc200-ar2` alone reaches 87% (the pair reaches 91 to 96%). One policy needs no switcher on the robot. The tag `two-policy-chain` (commit `746982e`) marks the two-policy state, so the repository can return to it. Runs: [mount_single_log.md](mount_single_log.md). History of the policy: [ball_mount_rewards_log](../ball_mount_rewards/ball_mount_rewards_log.md).

Task: `Mjlab-Piplus-Ball-MountMin-Clip`. Evaluate with `scripts/eval_mountbalance.py --task Mjlab-Piplus-Ball-MountMin-Clip --checkpoint <model> --probe` and `scripts/audit_policy.py`. The script `scripts/audit_policy.py` was removed. Use `eval_mountbalance.py --audit` instead ([git history](../README.md#conventions-shared-by-all-experiments)).

## The spin (Oct 14 2026)

In play, the robot turns on the ball. It always turns in the same direction. The results are measured on `mm7-exc200-ar2`. `--probe` now prints the base and ball yaw rate.

- **Every environment turns counter-clockwise at about 6 rad/s** (one turn per second; 63 rad of heading over 10 s). The ball spins with it at about 5 rad/s. The spin starts during the mount: 8.6 rad/s at 0.5 s, 9.9 rad/s at 1 s.
- **The simulator stops a spinning ball as it should:** a ball that spins at 6 rad/s on the floor under a 130 N load stops within 0.2 s. This holds in CPU (central processing unit) MuJoCo and in mujoco_warp. The live ball-floor contacts have `condim` 6 with torsional friction 0.01.
- **The policy drives the spin and depends on it:** in the test, the spin of the robot and the ball is set to zero at 10 s. Then the policy spins up again to 3 rad/s within 1 s and to the full rate within 2 s. 40% of those environments fall. None of the untouched environments falls.
- **Not a contact-model artifact:** with an elliptic friction cone the spin stays (5.6 rad/s). The spin also stays with torsional friction 0.02 at the feet (5.8 rad/s). Success drops with these changes (88% -> 66% elliptic, 48% elliptic + torsional). Therefore the policy is tuned to the contact details.
- The two-policy balance policy `bs4-exc100-ar2` also spins, but slower (2.1 rad/s).

So the policy learned the spin. Spinning is part of how the policy balances. No term penalized it. The same was seen on Ball-Small (−1.4 rad/s, [ball_small](../ball_small/ball_small.md)). New term: `base_yaw_rate` (squared base yaw rate, weight 0 by default).

## Torque saturation

New term: `torque_saturation` (weight 0 by default). The term is the sum, over the actuators, of the amount by which the applied force goes past 80% of its force limit. The legs sat at their limit 17% of the time in every mount policy. No term acted on it.

New term: `torque_demand` (weight 0 by default). The term is the sum, over the joints, of the amount by which the PD (proportional-derivative) torque that the policy asks for exceeds the force limit. The PD torque uses the own gains of each environment. `torque_saturation` cannot see this demand once the actuator is at its limit.

## Ball randomization (as of Oct 14 2026)

The values are drawn for each environment, once at startup, unless noted (`piplus_ball_mountbalance_env_cfg` and the Ball-Small events that it keeps):

| What | Range |
|---|---|
| Radius | 0.07 to 0.11 m (FIFA size 1 to 5) |
| Mass | linear in the radius from 0.14 to 0.43 kg, times U(0.8, 1.2) |
| Inertia | hollow shell, 2/3 m r^2 (solid sphere, 0.4 m r^2, before Oct 14 2026) |
| Ball sliding friction | 0.3 to 1.2 (acts against the floor; foot-ball contact uses the foot's) |
| Floor sliding friction | 0.3 to 1.2 |
| Contact time constant, ball and feet | 0.01 to 0.03 s |
| Spawn position | ±3 cm along x, ±2 cm along y, per reset |
| Ball position seen by the actor | ±2 cm noise, per step |

<!-- STE-deviation: FIFA is a proper name inside a table cell; the cell stays unchanged -->

Not randomized: the torsional and rolling friction of the ball (0.01, 0.001), its pressure (only through the contact time constant), and disturbances on the ball (no initial velocity, no pushes). The size 1 mass and the give of the ball under the weight of the robot are assumptions. Check them on the real ball.

## Deployment check against the Bit-Bots code (Oct 14 2026)

The robot runs the Hamburg Bit-Bots stack (`bit-bots/bitbots_main`, read at `1bb81db`, Oct 12 2026).

**What is there**
- `bitbots_rl_motion`: ROS (Robot Operating System) nodes that run ONNX (Open Neural Network Exchange) policies at 50 Hz. The nodes use one YAML (YAML Ain't Markup Language) config for each policy (joint order, home pose, kp/kd, action scale, joint signs). The package also has a Pi Plus hardware SDK (software development kit) (`livelybot_hardware_sdk`). An mjlab-trained Pi Plus walk policy is deployed this way (`mjlab_walk_model.yaml`: action scale 0.5, legs kp 55 / kd 1.1, arms 6 / 0.6; these are close to the values of this repository: 50 / 1.1 and 6 / 0.6). Therefore a mount policy needs a new node and a new config. It does not need a new framework.
- Ball perception: the YOEO (You Only Encode Once) network detects the ball in the head camera image (`bitbots_vision`). Inverse perspective mapping projects the detection to the ground with a **fixed ball diameter** (`soccer_ipm`, 0.153 m). Then a Kalman filter in the `odom` frame filters the result (`bitbots_ball_filter`, published at 62 Hz as `ball_position_relative_filtered`). The kick policy (`kick_ball_model.yaml`) reads it in `base_footprint`. The kick policy was trained with a 10 Hz ball history. The Bit-Bots repository has no tag-based (AprilTag or similar) ball detection.

**The policy depends on a live ball position after the mount.** In the test, the ball position of the actor is held at its value from 1 s on (the robot is on the ball by 0.5 s). Then `ms3-shell` falls in every episode (98% -> 0%). When the position is zeroed instead of held, the result is also 0% (`eval_mountbalance.py --ball-obs-freeze 1.0 [--ball-obs-zero]`). The policy uses the position of the ball relative to its base as a balance signal. On the robot, this signal would come from a camera that looks at a ball under its own foot. The signal would pass through a ground-plane projection that assumes the ball lies on the floor. It would also pass through a filter in an odometry frame, which is not meant for a robot that stands on a rolling ball.

Mismatches to settle before deployment:
- **Ball observation:** in training, the policy gets a perfect 3D position in the torso frame every 20 ms with ±2 cm noise. On the robot, it gets a filtered estimate with latency, a lower detection rate, occlusion by the leg of the robot, and a fixed diameter.
- **Head joints:** the real robot has two head joints. A separate head behaviour drives them. In this model they do not exist (commented out in the MJCF: the head is rigid).
- **Gains:** the deployment config sets kp/kd for each policy. The config must carry the values of this repository (legs 50 / 1.1, arms 6 / 0.6). The joint order and the joint signs must match this MJCF.

## Blind on the ball (Oct 14 2026)

Requirements from the robot side: The head camera cannot see a ball in front of the feet or under them. The ball is stationary (unless the robot pushes it). When the robot is on the ball, it balances on proprioception only. The robot decides "on the ball" from the base height while the robot is upright. The base height comes from the IMU (inertial measurement unit).

Task: `Mjlab-Piplus-Ball-MountMin-BlindOnBall` (`blind_on_ball=True`, log dir `piplus_ball_mountmin_blindonball`). The ball-position input of the actor is `ball_pos_until_on_ball`:
- **Before the latch:** the input is the ball position in the base frame plus a perception error. The perception error is constant over the episode (uniform ±3 cm in x and y, ±1 cm in z: the filtered estimate of a stationary ball seen earlier). The per-step noise of ±2 cm is added.
- **The latch:** the latch sets when the base height is above 0.42 m and the torso tilts less than 20° (0.35 rad). Once set, it stays for the episode. Measured on `ms3-shell` (512 environments): with both soles on the floor, the base never exceeds 0.379 m. When the robot stands on the ball, the 5th percentile is 0.439 m on size 1-2, 0.477 m on size 3, and 0.503 m on size 4-5. Every environment crosses 0.42 m at about 0.43 s (at most 0.48 s). The upright condition delays none of them.
- **After the latch:** the input is zeros (plus the per-step noise). A real value always has z about −0.3 m. Therefore zero cannot be mistaken for a ball position. The observation size is unchanged. Therefore earlier checkpoints can resume.
- The critic still sees the true ball state.

`ms3-shell` on this task without retraining reaches 0%. It used the ball position to balance.

Result: `bob1` (resume of `ms3-shell` on this task) reaches 95% (87 / 99 / 100% by ball size). It reaches 90% with a ±5 cm ball-estimate error. See the log.

On the robot, the same input is the estimate of the ball filter, transformed into the base (IMU) frame, until the latch. After the latch, the input is zeros. Set the ball diameter in `soccer_ipm` to the ball in use.

## Open questions

- The acceptable torque level and the acceptable saturation share of the real actuators are not known. 80% is a placeholder.
- The success of the policy depends strongly on the friction model (88% -> 48%). It is not possible to randomize the cone type for each environment. A possible fix: randomize the torsional friction at the feet.
