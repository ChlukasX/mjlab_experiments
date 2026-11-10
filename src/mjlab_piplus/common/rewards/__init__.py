"""Reward functions that two or more task packages share, by category.

Rule of two: a reward lives in its task package until a second package or a script needs it.
Then it moves here. Each category is one module. Import from the category module, or from
this package (all names are re-exported). The old task modules import these names from here,
so the old import paths still work. See AGENTS.md (rules S7 and S15).

| Module | Reward functions |
|---|---|
| `posture` | `base_height_gauss`, `foot_flatness`, `xy_centering`, `xy_drift_gauss` |
| `stance` | `single_leg_stance_lifted`, `free_foot_contact`; helpers `single_support`, `heading` |
| `mounting` | `foot_to_ball_top`, `mount_potential` |

`ball_control` (the big ball) is in `common/balls.py`. The platform task keeps its own
`foot_flatness` in `tasks/platform`, and `ball_small` keeps its own `foot_on_ball_reward`
and `single_leg_stance`: their bodies differ from the shared ones.
"""

from .mounting import foot_to_ball_top, mount_potential
from .posture import base_height_gauss, foot_flatness, xy_centering, xy_drift_gauss
from .stance import free_foot_contact, heading, single_leg_stance_lifted, single_support

__all__ = [
    "base_height_gauss",
    "foot_flatness",
    "foot_to_ball_top",
    "free_foot_contact",
    "heading",
    "mount_potential",
    "single_leg_stance_lifted",
    "single_support",
    "xy_centering",
    "xy_drift_gauss",
]
