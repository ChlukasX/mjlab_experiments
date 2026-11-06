# One policy for mount and stance

Branch: `feat/mount-single` (deleted; was off `feat/balance-smooth`). Commits: `git log --oneline --grep '^mount-single:'`.

**Status (Nov 6 2026): superseded.** The single policy was reached here (`ms3-shell` 98%; `bob1` blind on the ball 95%) and exported as `bob2-up05` ([deploy_mount](../deploy_mount/deploy_mount.md)). The work went on in [mount_latency](../mount_latency/mount_latency.md), [mount_straight](../mount_straight/mount_straight.md) and [mount_curriculum](../mount_curriculum/mount_curriculum.md). The open questions at the end of this file are still open.

Decision (Oct 14 2026): continue with a single policy that mounts the ball and holds the single-leg stance, instead of a mount policy plus a balance policy with a switch. The mount policy `mm7-exc200-ar2` alone reaches 87% (the pair 91 to 96%), and one policy needs no switcher on the robot. The two-policy state is tagged `two-policy-chain` (commit `746982e`) to come back to. Runs: [mount_single_log.md](mount_single_log.md). History of the policy: [ball_mount_rewards_log](../ball_mount_rewards/ball_mount_rewards_log.md).

Task `Mjlab-Piplus-Ball-MountMin-Clip`; evaluate with `scripts/eval_mountbalance.py --task Mjlab-Piplus-Ball-MountMin-Clip --checkpoint <model> --probe` and `scripts/audit_policy.py` (removed; now `eval_mountbalance.py --audit`; [git history](../README.md#conventions-shared-by-all-experiments)).

## The spin (Oct 14 2026)

In play the robot turns on the ball, always in the same direction. Measured on `mm7-exc200-ar2` (`--probe` now prints the base and ball yaw rate):

- **Every env turns counter-clockwise at about 6 rad/s** (one turn per second; 63 rad of heading over 10 s), and the ball spins with it at about 5 rad/s. It starts during the mount: 8.6 rad/s at 0.5 s, 9.9 rad/s at 1 s.
- **The simulator stops a spinning ball as it should:** a ball spinning at 6 rad/s on the floor, under a 130 N load, stops within 0.2 s, in CPU MuJoCo and in mujoco_warp alike. The live ball-floor contacts have `condim` 6 with torsional friction 0.01.
- **The policy drives the spin and depends on it:** with the spin set to zero at 10 s (robot and ball), the policy spins up again to 3 rad/s within 1 s and to the full rate within 2 s, and 40% of those envs fall, against none of the untouched ones.
- **Not a contact-model artefact:** with an elliptic friction cone the spin stays (5.6 rad/s), also with torsional friction 0.02 at the feet (5.8 rad/s). Success drops with these changes (88% -> 66% elliptic, 48% elliptic + torsional), so the policy is tuned to the contact details.
- The two-policy balance policy `bs4-exc100-ar2` spins too, slower (2.1 rad/s).

So it is learned: spinning is part of how the policy balances, and nothing penalised it. The same was seen on Ball-Small (−1.4 rad/s, [ball_small](../ball_small/ball_small.md)). New term `base_yaw_rate` (squared base yaw rate, weight 0 by default).

## Torque saturation

New term `torque_saturation` (weight 0 by default): per actuator, how far the applied force goes past 80% of its force limit, summed. The legs sat at their limit 17% of the time in every mount policy, and no term acted on it.

New term `torque_demand` (weight 0 by default): per joint, how far the PD torque the policy asks for (each env's own gains) exceeds the force limit, summed. `torque_saturation` cannot see that once the actuator is at its limit.

## Ball randomization (as of Oct 14 2026)

Per env, drawn once at startup unless noted (`piplus_ball_mountbalance_env_cfg` and the Ball-Small events it keeps):

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

Not randomized: the ball's torsional and rolling friction (0.01, 0.001), its pressure (only through the contact time constant), disturbances on the ball (no initial velocity, no pushes). The size 1 mass and the ball's give under the robot's weight are assumptions to check on the real ball.

## Deployment check against the Bit-Bots code (Oct 14 2026)

The robot will run the Hamburg Bit-Bots stack (`bit-bots/bitbots_main`, read at `1bb81db`, Oct 12 2026).

**What is there**
- `bitbots_rl_motion`: ROS nodes that run ONNX policies at 50 Hz, with per-policy YAML configs (joint order, home pose, kp/kd, action scale, joint signs) and a Pi Plus hardware SDK (`livelybot_hardware_sdk`). An mjlab-trained Pi Plus walk policy is deployed this way (`mjlab_walk_model.yaml`: action scale 0.5, legs kp 55 / kd 1.1, arms 6 / 0.6, close to this repo's 50 / 1.1 and 6 / 0.6), so a mount policy needs a new node and config, not a new framework.
- Ball perception: YOEO network detection in the head camera image (`bitbots_vision`), projected to the ground by inverse perspective mapping with a **fixed ball diameter** (`soccer_ipm`, 0.153 m), then a Kalman filter in the `odom` frame (`bitbots_ball_filter`, published at 62 Hz as `ball_position_relative_filtered`). The kick policy (`kick_ball_model.yaml`) reads it in `base_footprint` and was trained with a 10 Hz ball history. There is no tag-based (AprilTag or similar) ball detection in the repo.

**The policy depends on a live ball position after the mount.** With the actor's ball position held at its value from 1 s on (the robot is on the ball by 0.5 s), `ms3-shell` falls in every episode (98% -> 0%); zeroed instead of held: also 0% (`eval_mountbalance.py --ball-obs-freeze 1.0 [--ball-obs-zero]`). It uses the ball's position relative to its base as a balance signal. On the robot that signal would come from a camera looking at a ball under its own foot, through a ground-plane projection that assumes the ball lies on the floor, and a filter in an odometry frame that is not meant for a robot standing on a rolling ball.

Mismatches to settle before deployment:
- **Ball observation:** perfect 3D position in the torso frame every 20 ms with ±2 cm noise in training, against a filtered estimate with latency, a lower detection rate, occlusion by the robot's own leg, and a fixed diameter on the robot.
- **Head joints:** the real robot has two head joints driven by a separate head behaviour; in this model they do not exist (commented out in the MJCF: the head is rigid).
- **Gains:** the deployment config sets kp/kd per policy; it must carry this repo's values (legs 50 / 1.1, arms 6 / 0.6) and joint order and signs must match this MJCF.

## Blind on the ball (Oct 14 2026)

Requirements from the robot side: the head camera cannot see a ball in front of the feet or under them; the ball is stationary (unless the robot pushes it); once on the ball the robot balances on proprioception only; "on the ball" is decided from the base (IMU) height while upright.

Task `Mjlab-Piplus-Ball-MountMin-BlindOnBall` (`blind_on_ball=True`, log dir `piplus_ball_mountmin_blindonball`): the actor's ball-position input is `ball_pos_until_on_ball`:
- **Before the latch:** ball position in the base frame plus a perception error that is constant over the episode (uniform ±3 cm in x and y, ±1 cm in z: the filtered estimate of a stationary ball seen earlier), plus the per-step noise of ±2 cm.
- **The latch:** base height above 0.42 m with the torso tilted less than 20° (0.35 rad); once set it stays for the episode. Measured on `ms3-shell` (512 envs): with both soles on the floor the base never exceeds 0.379 m; standing on the ball, 5th percentile 0.439 m on size 1-2, 0.477 m on size 3, 0.503 m on size 4-5; every env crosses 0.42 m at about 0.43 s (at most 0.48 s), and the upright condition delays none of them.
- **After the latch:** zeros (plus the per-step noise). A real value always has z about −0.3 m, so zero cannot be mistaken for a ball position; the observation size is unchanged, so earlier checkpoints can be resumed.
- The critic still sees the true ball state.

`ms3-shell` on this task without retraining: 0% (it used the ball position to balance).

Result: `bob1` (resume of `ms3-shell` on this task) reaches 95% (87 / 99 / 100% by ball size), 90% with a ±5 cm ball-estimate error; see the log.

On the robot the same input is: the ball filter's estimate transformed into the base (IMU) frame until the latch, then zeros. The ball diameter in `soccer_ipm` must be set to the ball used.

## Open questions

- The acceptable torque level and saturation share of the real actuators are not known; 80% is a placeholder.
- The policy's success depends strongly on the friction model (88% -> 48%). Randomizing the cone type is not possible per env; torsional friction at the feet could be randomized.
