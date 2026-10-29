# Deploying the ball-mount policy on the Bit-Bots stack

Branch: `feat/deploy-mount` (off `feat/mount-single`) — **export and sim2sim done; the policy is not robust to latency above 15 ms (see the [log](deploy_mount_log.md)), retraining before the robot side continues**

Policy: `piplus_ball_mountmin_blindonball/*_bob2-up05/model_39988.pt` (task `Mjlab-Piplus-Ball-MountMin-BlindOnBall`, 97% in simulation, best on size 3-5 balls; exported Oct 29 2026, replacing `bob1` `model_36989`, see [mount_single log](../mount_single/mount_single_log.md) round 5). The `bob1` export is kept as `deploy/bitbots/mount_ball_bob1.{onnx,yaml}`; the parameter files are identical. The robot-side node goes into `bitbots_main` (`bitbots_rl_motion`; a skeleton on `feature/balance_ball_rl`, not working yet); this repo provides the model, the parameters and a reference of the observation code.

## Files

| File | What |
|---|---|
| `deploy/bitbots/mount_ball.onnx` | policy with the observation normalizer; input `obs` [1, 345] float32, output [1, 20]. Checked against PyTorch: max difference 6.9e-6. |
| `deploy/bitbots/mount_ball.yaml` | `bitbots_rl_motion`-style parameters: joint order, home pose, kp/kd, action scale, joint limits, on-ball latch, 50 Hz |
| `deploy/bitbots/mount_obs.py` | reference observation and action code without ROS (what the node has to reproduce) |
| `scripts/export_mount.py` | regenerates the two files above from a checkpoint (`uv run --with onnxruntime python scripts/export_mount.py --checkpoint <model.pt>`) |
| `scripts/sim2sim_onnx.py` | runs the ONNX model in the simulation with `mount_obs.py` built from raw signals (encoder angles, IMU quaternion and gyro, ball in the IMU frame, base height) |

Sim check (`sim2sim_onnx.py`, 20 s): `bob2-up05` **96% success, 97% survive** (512 envs; Oct 29 2026); `bob1` 93% / 97% (128 envs). Both with the ONNX model, batch-1 inference and the robot-side observation; the target computed by `mount_obs.py` equals the env's exactly. Both runs fed encoder angles with the ±0.015 rad encoder bias of the randomization, which the training observation never contained (mjlab's `joint_pos_rel` reads unbiased angles): the policy tolerates that offset (`bob1` 93% against 95%, `bob2-up05` 96% against 97%).

## Interface

Control at 50 Hz. Joint order (actions and observations): `r_shoulder_pitch, r_shoulder_roll, r_upper_arm, r_elbow, l_shoulder_pitch, l_shoulder_roll, l_upper_arm, l_elbow, r_hip_pitch, r_hip_roll, r_thigh, r_calf, r_ankle_pitch, r_ankle_roll, l_hip_pitch, l_hip_roll, l_thigh, l_calf, l_ankle_pitch, l_ankle_roll` (all `_joint`). The head is not part of the policy.

Observation, 345 values: six terms, each a 5-step history **oldest first**, terms concatenated in this order; a history that starts (activation) is filled with its first value; no scaling or clipping:

| Term | Size | Content |
|---|---|---|
| `joint_pos` | 20 | joint angle − home pose (rad) |
| `joint_vel` | 20 | joint velocity (rad/s) |
| `projected_gravity` | 3 | unit gravity vector in the IMU frame (as `GravityHandler`) |
| `base_ang_vel` | 3 | gyro in the IMU frame (rad/s) |
| `actions` | 20 | previous raw network output (before scaling and clipping); zeros at activation |
| `ball_pos` | 3 | ball centre in the IMU frame (m), until the latch; zeros after |

Action: target = home + 0.5 × output, **clipped to the joint limits in the YAML** (the existing `JointHandler` does not clip). Gains: legs kp 50 / kd 1.1, arms 6 / 0.6. Home pose: hip pitch ±0.5, calf ±1.0, ankle pitch ±0.5 (right +, left −), all else 0; this differs from the mjlab walk policy's walk-ready pose (±0.25 / ±0.65 / ±0.4).

On-ball latch: base (IMU) height above the floor > 0.42 m while the tilt is below 20° (−gravity_z > cos 0.35); from then on `ball_pos` is zeros until the policy is deactivated. In simulation the base is at most 0.379 m with both feet on the floor and at least ~0.44 m on the smallest balls (5th percentile), so the height estimate needs an error well below 2 cm. The robustness sweep ([log](deploy_mount_log.md)) shows the error only matters in one direction: reading low (latching late) costs nothing up to −4 cm, reading 4 cm high fails everything. On the robot use a threshold of ~0.44 m for margin.

## Plan for the robot side

1. **Node** in `bitbots_rl_motion`, like `mjlab_walk_node`: gyro, gravity, joint and robot-state handlers; a ball handler that transforms `ball_position_relative_filtered` into the IMU frame every step; the latch; per-term history buffers (`HistoryBuffer` matches the training convention); an activation trigger (like `rl_kick_active`); `initialize_observation` resets histories, previous action and latch. `deploy/bitbots/mount_obs.py` is the reference to compare against.
2. **Clip** the targets to the joint limits before publishing; publish the 20 joints with the YAML gains.
3. **Ball estimate:** set `soccer_ipm` `ball_diameter` to the ball used. Check the estimate against a tape measure at the start pose; the policy keeps 90% success up to ±5 cm of error and drops to ~66% at ±8 cm. The ball is not visible at the start pose, so it must be seen while approaching or with the head turned down beforehand; the filter holds it.
4. **Base height:** confirm the source reads ~0.39 m standing on the floor and rises above 0.42 m when on the ball. A height measured from the support foot (e.g. `base_footprint`) does not rise when the support foot is on the ball.
5. **Conventions:** joint signs are expected to match (this MJCF follows the robot convention, as the Bit-Bots mjlab walk policy, `joint_signs` all +1); verify joint by joint with small offsets. Check the IMU frame orientation against the model's (IMU at the base link origin, axes of the base).
6. **Start pose:** robot standing still in the home pose above; ball centre 0.19 m + radius ahead of the base, on the right-foot line (0.08 m to the right of the base), within ±3 cm / ±2 cm (the training range). Best ball: size 4-5 (radius 0.10-0.11 m).
7. **Tests in order:** offline, feed recorded robot signals through the node and through `mount_obs.py` and compare the vectors; robot hanging in a harness, policy active, check that it reaches for where the ball is; then on the floor with the harness, ball held in place; then the free ball, harness on.

## Open on this side

- **Latency (blocker):** training models a 0-15 ms actuator delay; success is 62% at 20 ms, 0% from 30 ms, and a 20 ms delay of the joint or gyro signals alone fails (see the [log](deploy_mount_log.md)). The real sensor-to-motor latency is not measured; it sets the range for the retraining.
- With the training observation noise and encoder bias on, success is 87% (71% on size 1-2 balls); evaluate with `--obs-noise` from now on.
- Real motor torque limits and the real ball's mass and give are not measured (the policy's leg actuators sit at the model limit ~4% of the time).
