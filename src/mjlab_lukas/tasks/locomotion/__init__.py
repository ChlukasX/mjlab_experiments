from mjlab.tasks.registry import register_mjlab_task

from .piplus_locomotion_env_cfg import piplus_locomotion_env_cfg, piplus_locomotion_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Piplus-Locomotion",
  env_cfg=piplus_locomotion_env_cfg(),
  play_env_cfg=piplus_locomotion_env_cfg(play=True),
  rl_cfg=piplus_locomotion_ppo_runner_cfg(),
)
