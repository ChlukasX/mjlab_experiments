# mount_actuator_fit: leg actuator dynamics and latency fit to the game rosbags

Status: finished (Nov 14 2026).
Code: branch `feat/mount-actuator-fit` (off `main`). Files: `robot/piplus_constants.py` (`LEG_DYNAMICS`, `get_spec_with_leg_dynamics`), `tasks/ball_mount_min/piplus_ball_mount_actfit_cfg.py`, `tests/test_actfit.py`.
Tasks: `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit`.
Log: [mount_actuator_fit_log.md](mount_actuator_fit_log.md). Run ID prefix: `af`.

## Why

A team member fit the leg actuators and the IMU (inertial measurement unit) of the Mini Pi Plus to the rosbags of the RoboCup 2026 games (68 bags, 5 robots, swing legs). The fit differs from the simulation of the `-LegGap` task:

| Quantity | Simulation (`-LegGap`) | Rosbag fit |
|---|---|---|
| Leg joint damping | 0.75 Nm s/rad | 0.0-0.1 Nm s/rad |
| Leg joint Coulomb friction (`frictionloss`) | 0.2 Nm | 0.2-0.5 Nm |
| Leg joint armature | 0.044277 kg m² | hips 0.023-0.044, other leg joints about 0.02 kg m² |
| Effective kp / commanded kp | 0.8-1.2 | 0.82-0.97 |
| Effective kd / commanded kd | 0.8-1.2 | 0.69-0.92 |
| Command-to-torque delay | 10-40 ms | 12-13 ms (knee, ankles), 17-18 ms (hips, thigh) |
| Torque-speed curve | none (effort limit 20 Nm) | full torque up to about 3 rad/s, about 0 at 8 rad/s |

The damping in the simulation is about 10 times the fit. A policy trained on a heavily damped leg gets less damping on the robot. The 40 ms latency never occurs in the rosbags.

## Design

Task `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit` (flag `ActFit`, experiment `piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_actfit`). The factory `piplus_ball_mount_actfit_env_cfg` calls the `-LegGap` factory. The observations, the rewards, the actions, and the network are the same, so the `cur4c` actor and critic load.

**One source of truth for the model.** `piplus.xml` stays as exported. `robot/piplus_constants.py::LEG_DYNAMICS` holds one named set per source: `vendor` (the XML values, used by every task before `-ActFit`) and `fit_2026_11` (damping 0.05, armature 0.02, frictionloss 0.3). `get_spec_with_leg_dynamics(base, dynamics)` loads the spec of `base` and sets the leg joint values. The task wraps the `spec_fn` of the `-LegGap` robot. `XmlActuatorCfg` does not apply its `armature` and `frictionloss` fields, so the spec is the only place to set them.

**Randomization** (startup, per environment, absolute values, `mjlab.envs.mdp.dr`):

| Event | Joints | Range |
|---|---|---|
| `leg_damping` | 12 leg joints | 0.0-0.15 Nm s/rad |
| `leg_friction` | 12 leg joints | 0.2-0.5 Nm |
| `hip_armature` | hip pitch, hip roll | 0.02-0.044 kg m² |
| `distal_armature` | thigh, calf, ankle pitch, ankle roll | 0.017-0.025 kg m² |
| `pd_gains` (changed) | all actuators | kp scale 0.75-1.1, kd scale 0.65-1.1 |

**Latency.** The stage curriculum sets the lag range of each stage (`LAG_START` 0-3, `LAG_FINAL` 2-8 physics steps of 5 ms). The curriculum term `lag_cap` runs after `stage` and caps both ends at `LAG_FIT[1]` = 5 steps (25 ms). The delay buffer keeps its size of 8 steps. Play uses `LAG_FIT` = 2-5 steps (10-25 ms). The cap keeps the base curriculum source unchanged, so the fingerprints of the other tasks stay the same.

**Not in this task.** The torque-speed curve (`DcMotorActuatorCfg`) replaces the XML position actuators. `scripts/export_mount.py` reads the gains of the XML actuators, so the curve is a separate step. The IMU noise of the fit (gyro std 0.0015-0.0024 rad/s, accelerometer std 0.013-0.015 m/s², gyro bias up to 0.01 rad/s) is the topic of [mount_imu_noise](../mount_imu_noise/mount_imu_noise.md).

Suggested first runs: evaluate `cur4c` on the play config of this task (`scripts/eval_policy.py`) to measure the gap, then a warm start from `cur4c` `model_24450` at stage 8 with the recipe of `sg1-leggap`.

## Result

The fitted dynamics matter. `cur4c`, trained on the vendor dynamics, loses much of its survival on the fitted model, although the lag is lower. `af1-actfit` (warm start from `cur4c` on this task) recovers the survival and holds the stance longer than `cur4c` on its own task. It still taps the floor with the free foot, and its held-stance success is 0%. It does not cover the old model. The gap check and its numbers are in the [log](mount_actuator_fit_log.md).

`af1-actfit` `model_27449` is the start of [mount_com_center](../mount_com_center/mount_com_center.md), which led to the selected policy.

## Next steps

- The torque-speed curve of the leg actuators is not modelled (see Design).
- The real leg gains are open (Nov 14 2026): the robot model of the Bit-Bots stack simulation (`bitbots_mujoco_sim`) has leg kp 100 and kv 1.0; this model has kp 50 and kv 1.1. The team decides which one the robot uses.

## Re-run

Copy `cur4c` `model_24450.pt` as `logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_actfit/2026-11-12_00-00-00_init-from-cur4c-24450/`. The full launch command of `af1-actfit` is in the [log](mount_actuator_fit_log.md).

