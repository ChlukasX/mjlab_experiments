# Deploying the ball-mount policy on the Bit-Bots stack

Branch: `feat/deploy-mount` (deleted; was off `feat/mount-single`). Commits: `git log --oneline --grep '^deploy-mount:'`.

**Status (Nov 6 2026): parked.** The export and the simulation-to-simulation (sim2sim) check of `bob2-up05` are done (Oct 29 2026). That policy is not robust to latency above 15 ms (see the [log](deploy_mount_log.md)). The retraining continued in [mount_latency](../mount_latency/mount_latency.md), [mount_straight](../mount_straight/mount_straight.md) and [mount_curriculum](../mount_curriculum/mount_curriculum.md). Those policies are **not exported**. Newest results: `lat2-big` reaches 94 / 94 / 89 / 56% success on size 4-5 balls at 10 / 20 / 30 / 40 ms latency (with noise and bias). `cur4b` `model_21600` reaches survival 96 / 98 / 93 / 68% at the same latencies. It does not hold the stance. The robot-side node is not in this repository. The deploy strategy is still to be decided. Until then, `deploy/` stays out of git and the two scripts below are parked.

Policy: `piplus_ball_mountmin_blindonball/*_bob2-up05/model_39988.pt` (task `Mjlab-Piplus-Ball-MountMin-BlindOnBall`, 97% in simulation, best on size 3-5 balls). It was exported on Oct 29 2026 and replaces `bob1` `model_36989` (see [mount_single log](../mount_single/mount_single_log.md) round 5). The `bob1` export is kept as `deploy/bitbots/mount_ball_bob1.{onnx,yaml}`. The parameter files are identical. The robot-side node goes into `bitbots_main` (`bitbots_rl_motion`; a skeleton on `feature/balance_ball_rl`, not working yet). This repository provides the model, the parameters and a reference of the observation code.

## Files

`deploy/` is not in git (decision of Nov 6 2026). The files below exist only on the machine that exported them.

| File | What |
|---|---|
| `deploy/bitbots/mount_ball.onnx` | policy with the observation normalizer; input `obs` [1, 345] float32, output [1, 20]. Checked against PyTorch: max difference 6.9e-6. |
| `deploy/bitbots/mount_ball.yaml` | `bitbots_rl_motion`-style parameters: joint order, home pose, kp/kd, action scale, joint limits, on-ball latch, 50 Hz |
| `deploy/bitbots/mount_obs.py` | reference observation and action code without Robot Operating System (ROS) (what the node has to reproduce) |
| `scripts/export_mount.py` | regenerates the two files above from a checkpoint (`uv run --with onnxruntime python scripts/export_mount.py --checkpoint <model.pt>`) |
| `scripts/sim2sim_onnx.py` | runs the Open Neural Network Exchange (ONNX) model in the simulation with `mount_obs.py` built from raw signals (encoder angles, quaternion and gyro of the inertial measurement unit (IMU), ball in the IMU frame, base height) |

`scripts/export_mount.py` makes `mount_ball.onnx` and `mount_ball.yaml` again from the checkpoint. `mount_obs.py` is in git history: `git show 993d320^:deploy/bitbots/mount_obs.py > deploy/bitbots/mount_obs.py`. `scripts/sim2sim_onnx.py` imports it. Therefore it runs only where this file exists.

Simulation check (`sim2sim_onnx.py`, 20 s): `bob2-up05` **96% success, 97% survive** with 512 environments (envs) on Oct 29 2026. `bob1`: 93% / 97% with 128 envs. Both runs used the ONNX model, batch-1 inference and the robot-side observation. The target that `mount_obs.py` computes equals the target of the env exactly. Both runs fed encoder angles with the ±0.015 rad encoder bias of the randomization. The training observation never contained this bias (the `joint_pos_rel` of mjlab reads unbiased angles). The policy tolerates that offset (`bob1` 93% against 95%, `bob2-up05` 96% against 97%).

## Interface

Control runs at 50 Hz. Joint order (actions and observations): `r_shoulder_pitch, r_shoulder_roll, r_upper_arm, r_elbow, l_shoulder_pitch, l_shoulder_roll, l_upper_arm, l_elbow, r_hip_pitch, r_hip_roll, r_thigh, r_calf, r_ankle_pitch, r_ankle_roll, l_hip_pitch, l_hip_roll, l_thigh, l_calf, l_ankle_pitch, l_ankle_roll` (all `_joint`). The head is not part of the policy.

The observation has 345 values. It has six terms. Each term is a 5-step history, **oldest first**. The terms are concatenated in the order of the next table. At activation, a history is filled with its first value. The observation has no scaling or clipping.

| Term | Size | Content |
|---|---|---|
| `joint_pos` | 20 | joint angle − home pose (rad) |
| `joint_vel` | 20 | joint velocity (rad/s) |
| `projected_gravity` | 3 | unit gravity vector in the IMU frame (as `GravityHandler`) |
| `base_ang_vel` | 3 | gyro in the IMU frame (rad/s) |
| `actions` | 20 | previous raw network output (before scaling and clipping); zeros at activation |
| `ball_pos` | 3 | ball centre in the IMU frame (m), until the latch; zeros after |

Action: target = home + 0.5 × output, **clipped to the joint limits in the YAML** (the existing `JointHandler` does not clip). Gains (kp: proportional, kd: derivative): legs kp 50 / kd 1.1, arms 6 / 0.6. Home pose: hip pitch ±0.5, calf ±1.0, ankle pitch ±0.5 (right +, left −), all other joints 0. This pose differs from the walk-ready pose of the mjlab walk policy (±0.25 / ±0.65 / ±0.4).

On-ball latch: the latch fires when the base (IMU) height above the floor is more than 0.42 m while the tilt is below 20° (−gravity_z > cos 0.35). From then on, `ball_pos` is zeros until the policy is deactivated. In simulation the base is at most 0.379 m with both feet on the floor. It is at least about 0.44 m on the smallest balls (5th percentile). Therefore the height estimate needs an error well below 2 cm. The robustness sweep ([log](deploy_mount_log.md)) shows that the error matters in one direction only. A low reading (latching late) costs nothing up to −4 cm. A reading 4 cm high fails everything. On the robot, use a threshold of about 0.44 m for margin.

## Plan for the robot side

1. **Node** in `bitbots_rl_motion`, like `mjlab_walk_node`. The node needs these parts: gyro, gravity, joint and robot-state handlers; a ball handler that transforms `ball_position_relative_filtered` into the IMU frame every step; the latch; per-term history buffers (`HistoryBuffer` matches the training convention); an activation trigger (like `rl_kick_active`). `initialize_observation` resets histories, previous action and latch. `deploy/bitbots/mount_obs.py` is the reference to compare against.
2. **Clip** the targets to the joint limits before publishing. Publish the 20 joints with the YAML gains.
3. **Ball estimate:** set `soccer_ipm` `ball_diameter` to the ball used. Check the estimate against a tape measure at the start pose. The policy keeps 90% success up to ±5 cm of error. It drops to about 66% at ±8 cm. The ball is not visible at the start pose. Therefore the robot must see the ball during the approach, or beforehand with the head turned down. The filter holds the ball position.
4. **Base height:** confirm that the source reads about 0.39 m when the robot stands on the floor and rises above 0.42 m when the robot is on the ball. A height measured from the support foot (for example `base_footprint`) does not rise when the support foot is on the ball.
5. **Conventions:** the joint signs are expected to match (this MuJoCo XML Format (MJCF) model follows the robot convention, as the Bit-Bots mjlab walk policy does, `joint_signs` all +1). Verify joint by joint with small offsets. Check the IMU frame orientation against the orientation in the model (IMU at the base link origin, axes of the base).
6. **Start pose:** the robot stands still in the home pose above. The ball centre is 0.19 m + radius ahead of the base, on the right-foot line (0.08 m to the right of the base), within ±3 cm / ±2 cm (the training range). The best ball is size 4-5 (radius 0.10-0.11 m).
7. **Tests in order:** First, run offline: feed recorded robot signals through the node and through `mount_obs.py`. Compare the vectors. Second, hang the robot in a harness with the policy active. Check that it reaches for the place of the ball. Third, put the robot on the floor with the harness, and hold the ball in place. Fourth, use the free ball with the harness on.

## Open on this side

- **Latency (blocker for `bob2-up05`; trained away in later policies, which are not exported):** the training models a 0-15 ms actuator delay. Success is 62% at 20 ms and 0% from 30 ms. A 20 ms delay of the joint or gyro signals alone fails (see the [log](deploy_mount_log.md)). The real sensor-to-motor latency is not measured. It sets the range for the retraining.
- With the training observation noise and encoder bias on, success is 87% (71% on size 1-2 balls). Evaluate with `--obs-noise` from now on.
- The real motor torque limits, and the mass and give of the real ball, are not measured (the leg actuators of the policy sit at the model limit about 4% of the time).
