from mjlab.tasks.registry import register_mjlab_task

from .piplus_upright_env_cfg import piplus_upright_env_cfg, piplus_upright_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Piplus-Upright",
  env_cfg=piplus_upright_env_cfg(),
  play_env_cfg=piplus_upright_env_cfg(play=True),
  rl_cfg=piplus_upright_ppo_runner_cfg(),
)
