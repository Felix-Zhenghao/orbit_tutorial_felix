# dt? num of steps? episode? iteration? what do all these mean?

"""
Sample curriculum defined by Orbit team:

def modify_reward_weight(env: RLTaskEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    # Curriculum that modifies a reward weight a given number of steps.

    # Args:
    #     env: The learning environment.
    #     env_ids: Not used since all environments are affected.
    #     term_name: The name of the reward term.
    #     weight: The weight of the reward term.
    #     num_steps: The number of steps after which the change should be applied.
    # 
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name) 
        # <--PAY ATTENTION-->: *_manager.get_term_cfg lets you get reward and modifies that quickly. This is a method under *Manager of orbit.managers
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)

"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.terrains import TerrainImporter

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv

def terrain_levels_vel(
    env: RLTaskEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.orbit.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


# TO LEARN: How to use update_env_origins to adjust the curriculum of terrain.






