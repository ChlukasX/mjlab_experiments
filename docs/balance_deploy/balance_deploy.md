# Balance policy for the real robot

Branch: `feat/balance-deploy` (deleted). Commits: `git log --oneline --grep '^balance-deploy:'`. — **audit done; superseded**: [balance_smooth](../balance_smooth/balance_smooth.md) and [mount_single](../mount_single/mount_single.md) did the bounded-action retraining of plan item 1. The audit tool is in that line of branches. Plan item 2 (export and checklist) is open. It now belongs to the deployment of the single policy (see the deployment check in mount_single).

Goal: put a balance policy on the real Pi Plus. A person mounts the ball by hand. Runs and audit results: [balance_deploy_log.md](balance_deploy_log.md).

## Candidates

Only actors with proprioception-only inputs (joint encoders and the inertial measurement unit (IMU)) can run on the robot: `Mjlab-Piplus-Ball-Small`, `-Ball-Small-Turf` (size 5 football) and `Mjlab-Piplus-Ball-Balance-Size1` (size 1). The MountBalance actors also take a ball position. They need a perception system.

## Audit (`scripts/audit_policy.py` (removed; now `eval_mountbalance.py --audit`; [git history](../README.md#conventions-shared-by-all-experiments)))

The audit tool runs a checkpoint in its play config. It reports how hard the policy drives the actuators. It gives four values:

- the size of the actions (1 unit = 0.5 rad of joint-target offset from the home pose)
- the share of commanded targets outside the joint limits
- the share of joint-steps where the proportional-derivative (PD) torque saturates (legs: ±20 Nm at 0.4 rad of error, arms: ±10 Nm at 1.7 rad)
- the mean change of the action per step

The thresholds in the script are mean |action| ≤ 1, ≤ 2% of targets outside the limits, and action change ≤ 0.3 per step. They are proposals to agree on. They are not measured limits.

## Finding: the existing balance policies are bang-bang

All three candidates command joint targets tens of radians from the home pose. They run their actuators saturated more than half the time (table in the log). This works in simulation, because the actuator limit clips the PD torque. On the real robot the result depends on the motor driver. A target far beyond the joint range can mean constant maximum torque and chattering. The driver can also clip it. The result can also be worse. Do not deploy the policies as they are. <!-- STE-deviation: no number in source ("tens of radians"); "worse" names an unknown outcome -->

## Plan

1. **Safe balance policy.** Retrain the drop balance (proprioception-only) with bounded joint targets. The only bounded attempt so far failed (`mb7-mix0-clip`, drops-only with clip + entropy 0.001: 0% survival, against 97% unbounded). It changed two things at once. Run a drops-only ablation next. It has three variants: clip only (entropy 0.01), clip + entropy 0.003, and unclipped + entropy 0.001. Add action-latency randomization (the action term has `delay_min_lag`/`delay_max_lag`) and the real torque limits. Acceptance: the audit is within the thresholds, and the success on drops is at least 90%.
2. **Export and checklist.** Export the policy in Open Neural Network Exchange (ONNX) format with `mjlab/rl/exporter_utils.py`. Document the observation order (joint position and velocity relative to the home pose, projected gravity, base angular velocity, last action, 5-step history) and the manual-mount procedure.
