# Collision check of the robot model

Branch: `feat/collision-check` (deleted; was off `feat/balance-smooth`). Commits: `git log --oneline --grep '^collision-check:'`. — **done**: the head box is fitted and the contact stiffness is randomized (applied from Oct 13 2026). The elbow range is still to check on the robot.

Question: in the viewer, the feet are inside the ball and the arms are inside the head. The ball tasks use all 20 leg and arm joints. Is the collision model complete? Does the simulation enforce it?

## Method

- The command `scripts/play_chain.py --steps 1000 --num-envs 256 --dump FILE` (removed; [git history](../README.md#conventions-shared-by-all-experiments)) records simulator states of the chain (mount `mm3-c10-s16`, balance `bs1-clip-exc`; 20 s, every 0.1 s). It also records the largest number of constraint rows and contacts seen.
- The script `scripts/check_collisions.py FILE` (removed; [git history](../README.md#conventions-shared-by-all-experiments)) compares two distances over 2000 states. The first distance is between the **visual meshes** (convex hulls: what the viewer shows). The second is between the **collision primitives** (what the simulation enforces). The script measures both distances for every pair of robot bodies that are not parent and child, and for each foot against ball and floor. Meshes that intersect while the primitives are apart or missing are a gap in the collision model. Limit of the method: convex hulls overstate the overlap of concave parts and of links that nest at a joint.

## Findings (Oct 13 2026)

**1. The solver is not dropping contacts.** The data show at most 93 constraint rows in one environment (env) (limit `njmax` 200) and at most 2037 contacts over 256 envs (limit 25 600).

**2. Foot in the ball: real penetration, from the default soft contact.** The foot's collision primitives (the two inner sole capsules) are inside the ball by more than 2 mm in 54% of the states. The foot mesh agrees with its primitives (57%). Therefore what the viewer shows is what the simulation does.

| | Foot into ball | Ball into floor |
|---|---|---|
| median | 2.8 mm | 2.9 mm |
| 90th percentile | 7.2 mm | 7.5 mm |
| maximum | 23 mm | 18.5 mm |
| small balls (r < 0.083 m), median / p90 | 3.4 / 8.3 mm | 4.3 / 10.2 mm |

All geoms use MuJoCo's default contact (`solref` 0.02 s, `solimp` 0.9 to 0.95). Its resting penetration under gravity is about 3 mm, whatever the mass. The policy pushes into the ball, and this causes the rest. Together, the robot sits about 6 mm (median) and up to 4 cm lower than a rigid ball would hold it. A real football also gives a few mm under 13 kg. <!-- STE-deviation: no number in source ("a few mm") --> Therefore this is not obviously wrong. But the stiffness is an untuned default, and it is not randomized. Foot-ball contact is `condim` 3 with the foot's friction (the foot geoms have priority 1). Therefore the ball's rolling and torsional friction (its `condim` 6) only act against the floor.

**3. Hand in the head: a gap in the collision model, fixed by a larger head box.** The head's collision box covers x −0.024..0.076, y ±0.055, z 0.033..0.077. The head mesh spans x −0.048..0.081, y ±0.086, z −0.013..0.091. The camera bar has no primitive. The left hand and the head meshes intersect in 84% of the states, up to 27 mm. The hand and the camera meshes intersect in 11%. The primitives touch in 0.2%. The next table shows the result with the box resized to the mesh (`get_spec_with_arms(fit_head=True)`, flag `--fit-head` in `play_chain.py` and `eval_handoff.py`):

| | Hand-head meshes intersect | Hand-off success |
|---|---|---|
| as trained | 84% (27 mm) | 83% |
| head box fitted, same policies | 0% | 81% (60 / 92 / 92% by ball size) |

The policies do not depend on the gap. They keep working with it closed.

**4. Forearm against upper arm: unclear.** Their meshes intersect in 14 to 67% of the states (up to 17 mm) with the elbow folded. The primitives intersect in under 1%. The reason: the forearm capsule was deliberately shortened to stay off the upper arm (`_fit_arm_collisions`). Check on the robot, or against the mechanical stop of the joint, whether the real elbow reaches these angles. The hull test alone cannot tell.

**5. Probably harmless hull overlaps.** The hull overlaps are: hip roll link against base (−9 mm) and calf (−20 mm), and shoulder pitch link against upper arm (−8 mm). These links nest around a joint. The hip pitch, hip roll and shoulder links have no primitive of their own (the thigh cylinder sits inside the volume of the hip roll link). No pair of legs, of arm and leg, or of arm and torso meshes intersects.

These bodies have no collision primitive: both shoulder pitch and shoulder roll links, both hip pitch and hip roll links, the camera.

## Applied (Oct 13 2026, on `feat/balance-smooth`)

- **Head box fitted by default** (`get_spec_with_arms(fit_head=True)`; `False` reproduces the old model). The code no longer has the `--fit-head` flags.
- **Contact stiffness randomized** in all randomized-ball tasks (`contact_stiffness` startup event in `piplus_ball_mountbalance_env_cfg`). Each env gets one `solref` time constant from 0.01 to 0.03 s, on the ball and both feet.
- The evaluation of the best pair so far (`mm3-c10-s16` + `bs1-clip-exc`) on the new model, without retraining, gives a hand-off of 77% (58 / 84 / 87% by ball size), against 83% on the model it was trained on.
- The recorded hand-off states come from the old model (the hand is inside the new head box in most of them). Therefore the order is: resume the mount policy on the new model, record its hand-off states again, then resume the balance policy on those. Runs: `mm4-*` in [ball_mount_rewards_log](../ball_mount_rewards/ball_mount_rewards_log.md), then [balance_smooth_log](../balance_smooth/balance_smooth_log.md).

## Proposed changes (as of the check)

1. **Make the fitted head box the default** for the arm tasks. The measured cost for the chain is none. The cost for Ball-Small and Locomotion-Arms is not measured.
2. **Ball contact stiffness:** randomize `solref` on the ball and foot contacts instead of one default (for example time constant 0.01 to 0.03 s; 0.01 is the lower limit at the 5 ms step). Measure the give of the real ball under the weight of the robot. This changes the dynamics. Therefore the mount and balance policies need a resume on it.
3. **Elbow:** measure the range of the real elbow. Then tighten the joint range or lengthen the forearm capsule.
4. Shoulder and hip links: add primitives only if a check on new policies shows real intersections there.
