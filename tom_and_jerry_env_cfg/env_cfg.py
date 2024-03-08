from __future__ import annotations

import math
from dataclasses import MISSING

from omni.isaac.orbit.assets import ArticulationCfg # Robot configuration model
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.orbit.terrains import TerrainImporterCfg
import omni.isaac.orbit_tasks.locomotion.velocity.mdp as mdp # import the task-specific mdp
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg # TO LEARN: How to use it? What is the information structure of it?
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.orbit.scene import InteractiveSceneCfg

##
# Pre-defined configs
##

from unitree_go1_cfg import UNITREE_GO1_CFG

class MySceneCfg(InteractiveSceneCfg):
    """Terrain scene configuration for a legged robot."""

    # Terrian

    # Robot
    robot: ArticulationCfg = MISSING # set default value as MISSING. You can have no robot, just terrain.

    # Light # TO LEARN: see doc of orbit.sim to know how to configure light.
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

    # Sensor
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    height_scanner = RayCasterCfg(
        prim_path = "{ENV_REGEX_NS}/Robot/base",
        offset = RayCasterCfg.OffsetCfg(pos=(0.0,0.0,20.0)),
        attach_yaw_only = True, # useful for ray-casting height map
        pattern_cfg = patterns.GridPatternCfg(resolution=0.1, size=[1.6,1.0]), # pattern is a sub-module for ray-casting patterns used by the ray-caster
        debug_vis = False,
        mesh_prim_paths=["/World/ground"],
    )


class TomAndJerryEnvCfg:

    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    randomization: RandomizationCfg = RandomizationCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    """
    Possible reason for recommending __post_init__:

    Default Configuration with Customization: This approach allows the environment to be instantiated with a set of 
    default parameters that work out of the box for most cases. However, it also provides an easy and structured way to 
    customize these parameters when necessary. Users can subclass and modify the __post_init__ method to adjust the 
    configuration to their specific needs without changing the class's core initialization logic.
    """

    def __post_init__(self):
        """Post initialization."""
        # robot articulation cfg
        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False






