"""Football constants and builder shared by the ball tasks.

Moved without change from `tasks/ball_small` (radius, mass, `get_ball_spec`, `ROBOT_SPAWN_X`),
`tasks/ball_mount` (`BALL_SIZES`), and `tasks/ball_mount_curriculum` (`MU_ROLL_LOCKED`).
The old modules import these names from here.
`common/balls.py` holds the big ball (radius 0.45 m): `BALL_RADIUS` and `BALL_MASS` there are
other values under the same names, and its `get_ball_spec` builds another ball. Import from
the module you mean. See docs/ball_small/ball_small.md.
"""

import mujoco

BALL_RADIUS = 0.11
BALL_MASS = 0.43

# FIFA ball sizes: (radius m, mass kg). Size 5 is the Ball-Small football.
# ponytail: size 1 is a mini ball, ~44 cm circumference, ~0.14 kg; check against the real ball.
BALL_SIZES = {5: (0.11, 0.43), 1: (0.07, 0.14)}

# Offset right foot over ball top; left foot lands on floor beside ball.
# Right ankle_roll is ~0.05 m lateral from base centre — shift base by same.
ROBOT_SPAWN_X = -0.05  # base shifted left so right foot is over ball

# m: rolling-friction coefficient; cuts ball travel under a dropped robot by ~60%
MU_ROLL_LOCKED = 0.2


def get_ball_spec(radius: float = BALL_RADIUS, mass: float = BALL_MASS) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball_body", pos=[0, 0, radius])
    body.add_freejoint(name="ball_free")
    body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius],
        mass=mass,
        rgba=[0.2, 0.6, 0.2, 1.0],
        contype=1,
        conaffinity=1,
        condim=6,
        friction=[0.8, 0.01, 0.001],
    )
    return spec
