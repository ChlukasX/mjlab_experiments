# One policy for mount and stance

Branch: `feat/mount-single` (off `feat/balance-smooth`) — **in progress**

Decision (Oct 14 2026): continue with a single policy that mounts the ball and holds the single-leg stance, instead of a mount policy plus a balance policy with a switch. The mount policy `mm7-exc200-ar2` alone reaches 87% (the pair 91 to 96%), and one policy needs no switcher on the robot. The two-policy state is tagged `two-policy-chain` (commit `746982e`) to come back to. Runs: [mount_single_log.md](mount_single_log.md). History of the policy: [ball_mount_rewards_log](../ball_mount_rewards/ball_mount_rewards_log.md).

Task `Mjlab-Piplus-Ball-MountMin-Clip`; evaluate with `scripts/eval_mountbalance.py --task Mjlab-Piplus-Ball-MountMin-Clip --checkpoint <model> --probe` and `scripts/audit_policy.py`.

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

## Open questions

- The acceptable torque level and saturation share of the real actuators are not known; 80% is a placeholder.
- The policy's success depends strongly on the friction model (88% -> 48%). Randomizing the cone type is not possible per env; torsional friction at the feet could be randomized.
