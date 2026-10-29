# deploy_mount log

Design and interface: [deploy_mount.md](deploy_mount.md).

## Export of `bob2-up05` (Oct 29 2026)

`scripts/export_mount.py --checkpoint logs/rsl_rl/piplus_ball_mountmin_blindonball/2026-10-15_11-49-58_bob2-up05/model_39988.pt` → `deploy/bitbots/mount_ball.{onnx,yaml}` (the `bob1` export is kept as `mount_ball_bob1.*`; the parameter files are identical). ONNX against PyTorch: max difference 6.9e-6.

`scripts/sim2sim_onnx.py` (512 envs, 20 s, batch-1 ONNX, robot-side observation from raw signals with the ±0.015 rad encoder bias): **96% success, 97% survive** (`bob1`: 93% / 97%). Largest observation difference 1.5e-2 (the encoder bias); first-step targets identical.

## Robustness sweep, `bob2-up05` (Oct 29 2026)

One factor at a time on `MountMin-BlindOnBall` (play config: pushes on, ball-estimate error ±3/±3/±1 cm per episode as in training, observation noise off unless stated). `scripts/eval_mountbalance.py`, 2048 envs, 20 s, `--seed 0` (new flag: the seed used to be random, which explains the ±2 points between earlier evals of the same checkpoint). New flags: `--obs-noise`, `--obs-delay` (control steps, 20 ms; `--obs-delay-terms` to pick terms; the previous action is never delayed), `--act-lag` (physics steps, 5 ms), `--encoder-bias` (range and the actor reads biased angles), `--latch-height-bias`, `--kp-range`, `--ball-radius`, `--ball-mass`, `--ball-solid`.

Success by ball size 1-2 / 3 / 4-5 (radius <0.083 / 0.083-0.097 / >0.097 m):

| Condition | Size 1-2 | Size 3 | Size 4-5 | Success | Survive |
|---|---|---|---|---|---|
| baseline, seeds 0 / 1 / 2 | 90-92% | 99% | 99-100% | **96 / 97 / 96%** | 98-99% |
| observation noise on (training levels) | 78% | 96% | 98% | 90% | 95% |
| encoder bias ±0.015 rad, read by the actor (trained bias range, not in the training obs) | 87% | 96% | 98% | 94% | 96% |
| encoder bias ±0.03 | 81% | 90% | 94% | 89% | 92% |
| encoder bias ±0.05 | 65% | 82% | 87% | 78% | 84% |
| **training-like:** noise on + encoder bias ±0.015 | 71% | 94% | 97% | **87%** | 92% |
| actuator lag 15 ms fixed (top of the trained 0-15 ms) | 87% | 97% | 99% | 94% | 96% |
| actuator lag 20 ms fixed | 42% | 68% | 77% | 62% | 66% |
| actuator lag 25 ms fixed | 28% | 55% | 58% | 47% | 49% |
| actuator lag 20-30 ms | 12% | 36% | 43% | 30% | 33% |
| actuator lag 30-40 ms | 0% | 0% | 0% | 0% | 0% |
| observation delay 20 ms, all terms but the previous action | 0% | 0% | 0% | 0% | 0% |
| observation delay 20 ms, `ball_pos` only (256 envs) | | | | 95% | 95% |
| observation delay 20 ms, `projected_gravity` only (256 envs) | | | | 97% | 99% |
| observation delay 20 ms, `joint_pos` only (256 envs) | | | | 5% | 8% |
| observation delay 20 ms, `joint_vel` only (256 envs) | | | | 12% | 19% |
| observation delay 20 ms, `base_ang_vel` only (256 envs) | | | | 3% | 5% |
| latch height estimate +2 cm (latches early) | 91% | 99% | 100% | 96% | 99% |
| latch height estimate −2 cm (latches late) | 92% | 99% | 99% | 97% | 99% |
| latch height estimate +4 cm | 0% | 0% | 0% | 0% | 0% |
| latch height estimate −4 cm | 91% | 98% | 99% | 96% | 98% |
| kp, kd ×0.6-1.4 (trained 0.8-1.2) | 89% | 98% | 99% | 95% | 98% |
| kp, kd ×0.5-0.8 (weak motors) | 78% | 98% | 100% | 92% | 98% |
| ball r 0.077 m (RoboCup ball, `soccer_ipm` default) | 96% | | | 96% | 98% |
| ball r 0.10 m | | | 99% | 99% | 99% |
| ball r 0.11 m | | | 99% | 99% | 99% |
| `bitbots_mujoco_sim` ball: r 0.07 m, 0.205 kg, solid | 89% | | | 89% | 98% |

**Findings**
- **Latency is the blocker.** The policy holds up to the trained 15 ms and falls apart past it: 62% at 20 ms, 47% at 25 ms, 0% from 30 ms. A 20 ms delay of any fast proprioceptive signal (joint angles, joint velocities, gyro) alone takes it to 3-12%; delaying the gravity vector or the ball input does nothing. A ROS loop (IMU → node → HCM → motor bus) is very likely above 15 ms in total, so this policy will most likely not work on the robot as is.
- **The headline 96-97% is optimistic.** All evaluations so far ran without the observation noise the policy was trained with (play config). With the noise and the encoder bias the actor will see on the robot: 87%, and 71% on size 1-2 balls.
- **Encoder bias** beyond the trained ±0.015 rad costs 5 points at ±0.03 and 16 at ±0.05; the calibration of the real joints matters.
- **The latch is safe in one direction only:** a height estimate reading low (late latch) costs nothing even at −4 cm; reading high by 4 cm latches while standing on the floor (base at most 0.379 m) and fails everything. On the robot, raising the threshold to ~0.44 m gives margin for free (equivalent to the −2 cm row: 97%).
- **Gains** ±40% and the RoboCup and size 4-5 balls are fine (96-99%). Weak motors and the solid size-1 ball of the Bit-Bots simulator hurt the small balls only (78% / 89%). For sim2sim in `bitbots_mujoco_sim`, put a size 4-5 ball in the scene.

**Next:** resume `bob2-up05` with observation delay and a wider actuator lag (range to be set from a measured or estimated robot loop latency), encoder bias in the actor observation (`joint_pos_rel(biased=True)`) and a wider bias range; evaluate with `--obs-noise` from now on.
