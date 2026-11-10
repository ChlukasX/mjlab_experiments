"""`-ImuNoise` task: lower actor noise on the two IMU terms (docs/mount_imu_noise).

Base: `piplus_ball_mount_selfcontact_env_cfg` (the `-LegGap` task). The model, the rewards, the
curriculum, the action space, and the network are the same, so the `cur4c` actor and critic load.
Only the range of the uniform per-step noise on the actor inputs `projected_gravity` and
`base_ang_vel` changes. Both inputs come from the IMU (inertial measurement unit) of the robot.
The joint encoder terms (`joint_pos`, `joint_vel`), the ball input, the encoder bias, and the loop
latency keep their ranges. The critic never has noise.

The amounts are a first guess: half of the base values (+-0.05 and +-0.2 rad/s, which follow the
mjlab velocity task and are not measured on the Pi Plus). The right amounts come from the datasheet
of the IMU or from a recording of the real IMU at rest (see the section "Sim2real" of the design
doc). The noise curriculum (`set_noise_scale`) scales from the range stored in the term when it
first runs, so the lower range is the full range of every stage.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.utils.noise import UniformNoiseCfg

from mjlab_piplus.tasks.ball_mount_min.piplus_ball_mount_selfcontact_cfg import (
    piplus_ball_mount_selfcontact_env_cfg,
)

# Half range (+-) of the uniform noise per control step. Base values: 0.05 and 0.2.
IMU_NOISE_GRAVITY = 0.025  # projected gravity, unit vector (dimensionless)
IMU_NOISE_ANG_VEL = 0.1  # base angular velocity, rad/s
IMU_NOISE = {"projected_gravity": IMU_NOISE_GRAVITY, "base_ang_vel": IMU_NOISE_ANG_VEL}


def piplus_ball_mount_imunoise_env_cfg(play: bool = False, **kwargs) -> ManagerBasedRlEnvCfg:
    """`-LegGap` task with the IMU noise ranges of `IMU_NOISE`. `kwargs` go to
    `piplus_ball_mount_selfcontact_env_cfg`. In play (and in eval) the actor noise is off unless
    the eval script turns it on; it then uses these ranges."""
    cfg = piplus_ball_mount_selfcontact_env_cfg(play=play, **kwargs)
    terms = cfg.observations["actor"].terms
    for name, half_range in IMU_NOISE.items():
        # A copy of the term: the critic group holds the same term objects as the actor group.
        noise = UniformNoiseCfg(n_min=-half_range, n_max=half_range)
        terms[name] = replace(terms[name], noise=noise)
    return cfg
