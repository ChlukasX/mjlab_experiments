from pathlib import Path
import os
import re
import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
##
# MJCF and assets.
##

PIPLUS_XML: Path = Path(os.path.dirname(__file__)) / "piplus/piplus.xml"

VISUAL_RGBA = (0.2, 0.2, 0.2, 1.0)
"""Dark gray (80% gray) for the robot's visual meshes. The MJCF paints them
light gray (0.75); they are recolored here so the vendored file stays as
exported. Collision geoms are named ``*_collision[0-9]`` and are left alone."""


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PIPLUS_XML))
  for geom in spec.geoms:
    if geom.type == mujoco.mjtGeom.mjGEOM_MESH and not geom.name:
      geom.rgba = VISUAL_RGBA
  return spec


def get_spec_with_arms() -> mujoco.MjSpec:
  """Like get_spec() but with arm joints and actuators uncommented."""
  xml = PIPLUS_XML.read_text()
  # Uncomment single-line arm joint entries: <!--<joint ... pi_joint_arm ... />-->
  xml = re.sub(r'<!--(<joint[^>]+pi_joint_arm[^>]+/>)-->', r'\1', xml)
  # Uncomment the multi-line arm actuator block: <!--<position .../>...-->
  xml = re.sub(
      r'<!--(\s*(?:<position class="pi_actuator_arm"[^>]*/>\s*)+)-->',
      r'\1', xml, flags=re.DOTALL,
  )
  tmp = PIPLUS_XML.parent / "_arms_tmp.xml"
  tmp.write_text(xml)
  try:
    spec = mujoco.MjSpec.from_file(str(tmp))
  finally:
    tmp.unlink()
  for geom in spec.geoms:
    if geom.type == mujoco.mjtGeom.mjGEOM_MESH and not geom.name:
      geom.rgba = VISUAL_RGBA
  return spec


ACTUATOR_LAG_MIN = 0
ACTUATOR_LAG_MAX = 3

# ACTUATOR_3536 = XmlActuatorCfg(
#   target_names_expr=(
#     "head_yaw_joint",
#     "head_pitch_joint"
#   ),
#   delay_min_lag=ACTUATOR_LAG_MIN,
#   delay_max_lag=ACTUATOR_LAG_MAX,
# )

ACTUATOR_4438 = XmlActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_upper_arm_joint",
    ".*_elbow_joint",
  ),
  delay_min_lag=ACTUATOR_LAG_MIN,
  delay_max_lag=ACTUATOR_LAG_MAX,
)

ACTUATOR_5036 = XmlActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_thigh_joint",
    ".*_calf_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
  ),
  delay_min_lag=ACTUATOR_LAG_MIN,
  delay_max_lag=ACTUATOR_LAG_MAX,
)


HOME_KEYFRAME_WITH_ARMS = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.39),
  joint_pos={
    "r_hip_pitch_joint":       0.5,
    "l_hip_pitch_joint":      -0.5,
    ".*_hip_roll_joint":       0.0,
    ".*_thigh_joint":          0.0,
    "r_calf_joint":            1.0,
    "l_calf_joint":           -1.0,
    "r_ankle_pitch_joint":     0.5,
    "l_ankle_pitch_joint":    -0.5,
    ".*_ankle_roll_joint":     0.0,
    ".*_shoulder_pitch_joint": 0.0,
    ".*_shoulder_roll_joint":  0.0,
    ".*_upper_arm_joint":      0.0,
    ".*_elbow_joint":          0.0,
  },
  joint_vel={".*": 0.0},
)

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.39),
  joint_pos={
    "r_hip_pitch_joint": 0.5,
    "l_hip_pitch_joint": -0.5,
    ".*_hip_roll_joint": 0.0,
    ".*_thigh_joint": 0.0,
    "r_calf_joint": 1.0,
    "l_calf_joint": -1.0,
    "r_ankle_pitch_joint": 0.5,
    "l_ankle_pitch_joint": -0.5,
    ".*_ankle_roll_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_collision[0-9]$",),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

LEG_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"(r|l)_(thigh|calf|ankle_pitch|ankle_roll)_link_collision[0-9]$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"(r|l)_ankle_roll_link_collision[0-9]$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

PIPLUS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    # ACTUATOR_3536,
    # ACTUATOR_4438,
    ACTUATOR_5036,
  ),
  soft_joint_pos_limit_factor=0.9,
)

PIPLUS_ARTICULATION_WITH_ARMS = EntityArticulationInfoCfg(
  actuators=(
    ACTUATOR_4438,
    ACTUATOR_5036,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_piplus_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=PIPLUS_ARTICULATION,
  )


def get_piplus_robot_with_arms_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME_WITH_ARMS,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec_with_arms,
    articulation=PIPLUS_ARTICULATION_WITH_ARMS,
  )
