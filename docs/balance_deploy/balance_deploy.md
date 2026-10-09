# Balance policy for the real robot

Branch: `feat/balance-deploy` — **in progress**

Goal: put a balance policy on the real Pi Plus, mounting the ball by hand. Runs and audit results: [balance_deploy_log.md](balance_deploy_log.md).

## Candidates

Only proprioception-only actors can run on the robot (joint encoders and IMU): `Mjlab-Piplus-Ball-Small`, `-Ball-Small-Turf` (size 5 football) and `Mjlab-Piplus-Ball-Balance-Size1` (size 1). The MountBalance actors also take a ball position and need a perception system.

## Audit (`scripts/audit_policy.py`)

Runs a checkpoint in its play config and reports how hard it drives the actuators: size of the actions (1 unit = 0.5 rad of joint-target offset from the home pose), share of commanded targets outside the joint limits, share of joint-steps where the PD torque saturates (legs: ±20 Nm at 0.4 rad of error, arms: ±10 Nm at 1.7 rad), and the mean change of the action per step. The thresholds in the script (mean |action| ≤ 1, ≤ 2% of targets outside the limits, action change ≤ 0.3 per step) are proposals to agree on, not measured limits.

## Finding: the existing balance policies are bang-bang

All three candidates command joint targets tens of radians from the home pose and run their actuators saturated more than half the time (table in the log). In simulation this works because the PD torque is clipped at the actuator limit. On the real robot it depends on the motor driver: targets far beyond the joint range could mean constant maximum torque and chattering, or be clipped, or worse. They should not be deployed as they are.

## Plan

1. **Safe balance policy.** Retrain the drop balance (proprioception-only) with bounded joint targets. The only bounded attempt so far failed (`mb7-mix0-clip`, drops-only with clip + entropy 0.001: 0% survival, against 97% unbounded), and it changed two things at once. Drops-only ablation next: clip only (entropy 0.01), clip + entropy 0.003, unclipped + entropy 0.001; plus action-latency randomization (the action term has `delay_min_lag`/`delay_max_lag`) and the real torque limits. Acceptance: audit within the thresholds and at least 90% success on drops.
2. **Export and checklist.** ONNX export via `mjlab/rl/exporter_utils.py`; document the observation order (joint position and velocity relative to the home pose, projected gravity, base angular velocity, last action, 5-step history) and the manual-mount procedure.
