"""
This is the basic configuration for the velocity-tracking task.
The articulation of the env is not defined.
This env + articulation definition = velocity-tracking task for a specific robot.
"""


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
from .rough_terrain_cfg import ROUGH_TERRAINS_CFG



"""
Scene Definition
"""
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Terrain scene configuration for a legged robot."""

    # Terrian
    terrain = TerrainImporterCfg(
        # If terrain type is generator, then you can define a terrian cfg using TerrainGeneratorCfg
        prim_path="/World/ground",
        terrain_type = "generator",
        # TO LEARN: How to define a TerrainGeneratorCfg
        terrain_generator = ROUGH_TERRAINS_CFG, # The defined TerrainGeneratorCfg
        max_init_terrain_level = 5, # TO LEARN
        collision_group = -1 # TO LEARN
        physics_material=sim_utils.RigidBodyMaterialCfg( 
            # TO LEARN 
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            # TO LEARN
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

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

"""
MDP Settings
Here we go else where and define a task-specific mdp package and import it into this file.
"""

@configclass
class CommandCfg:
    """Here we use velocity command to make the robot track a certain velocity."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        """
        heading_command: If True, the angular velocity command is computed from the heading error, where the target heading is sampled uniformly from provided range. Otherwise, the angular velocity command is sampled uniformly from provided range.
        Then, rel_heading_envs define the probability threshold for the robot to follow heading-based angular velocity command.
        rel_standing_envs is the p threshold for robots to stand still.
        """
        asset_name = "robot",
        resampling_time_range = (10.0,10.0), # TO LEARN: What is the unit of the time_range of resetting command? second? step?
        rel_standing_envs = 0.02,
        rel_heading_envs = 1.0,
        heading_command = True, # Thus, angular velocity is never random, given linear velocity.
        debug_vis = True,
        ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionCfg:
    # TO LEARN: Why this part don't consider the command?

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

@configclass
class ObservationCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """The observations needed by the policy. A group of obs. Comprise of many obs terms."""

        # Obs Terms, order preserved
        """
        Attributes for ObsTerm:
        func: callable
        noise: NoiseCfg
        clip: tuple(float,float) - clipping range for obs after adding noise
        scale: float
        params: dict[str(name of para), Any | SceneEntityCfg (value of para)] - params passed to the func as keyword arguments
        """
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        # TO LEARN: Why write post_init rather than write directly in the config part?
        def __post_init__(self):
            self.enable_corruption = True # Noise defined in terms will take effect
            self.concatenate_terms = True # Terms will be concatenated

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class RandomizationCfg:
    """
    Randomization terms are grouped according to their mode.

    For a typical training process, you may want to randomize in the following modes:
    - “startup”: Randomization term is applied once at the beginning of the training.
    - “reset”: Randomization is applied at every reset.
    - “interval”: Randomization is applied at pre-specified intervals of time.
    """
    
    physics_material = RandTerm(
        # This randomize the material properties of the robot.
        # TO LEARN: What is "the num of shapes per body" in the doc - https://isaac-orbit.github.io/orbit/source/api/orbit/omni.isaac.orbit.envs.mdp.html#omni.isaac.orbit.envs.mdp.randomizations.randomize_rigid_body_material
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
        
    add_base_mass = RandTerm(
        func = mdp.add_body_mass,
        mode = "startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "mass_range": (-5.0, 5.0)},
    )

    base_external_force_torque = RandTerm(
        func = mdp.apply_external_force_torque,
        mode = "reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },             
    )

    reset_base = RandTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}, # It'll appear in the different position with different heading.
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = RandTerm(
        # This is a scaler not an adder. joint_pos *= scaler.
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # TO LEARN: What is "interval_range_s: The range of time in seconds at which the term is applied"?
    # TO LEARN: The method "randomize" of class RandomizationManager: dt – The time step of the environment. What is the relationship between time step and interval_range_s??
    push_robot = RandTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

  
@configclass
class TerminationCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
   

@configclass
class RewardCfg:
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )

    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)



@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

##
# Environment configuration
##

# It seems that the interaction between env components, like curriculum and terrain; command and action, is abstracted away by
#    the implementation of RLTaskEnvCfg. 
    
@configclass
class LocomotionVelocityRoughEnvCfg(RLTaskEnvCfg):
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
                












