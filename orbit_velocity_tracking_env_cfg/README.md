# Orbit Tutorial (now only walk through the velocity_env_cfg.py)

***TO DO:***
- [ ] **Understand curriculum configuration by understanding the source code of manager, especially when is the manager called during simulation and its connection between the ```reset``` method of env**
- [ ] **Figure out what is the action configuration doing and why it is so simple**
- [ ] **Read the API of SceneEntityCfg and write more detailed about its usage**
- [ ] **Understand how to wrap the environment and connect the NN-training part and the env part**
- [ ] **Walk through other files if necessary**


> **All link in this tutorial is linked to the specific part of the source code or official document. Utilize these links to understand the materials deeper.**

> This tutorial is written 25/03/2024 after the launch of v0.4. The latest commit is a30d764. The key feature of v0.4 is the introduction of <a href="https://isaac-orbit.github.io/orbit/source/api/orbit/omni.isaac.orbit.managers.html#event-manager">```EventManager```</a> and deprecation of ```Randomization Manager```.

This tutorial will walk you through the official implementation of the RL environment of the velocity-tracking locomotion task of Unitree GO1. It is exactly the <a href="https://github.com/NVIDIA-Omniverse/Orbit">isaac orbit</a> version of the repository <a href = "https://github.com/leggedrobotics/legged_gym">"legged_gym"</a> by ETH Zurich.

To run the code of this tutorial, create and activate the orbit environment.
```
$ conda activate <name_of_python_orbit_env>
```

## 1. File Structure
The file structure of this tutorial's repository is as follows:<br>
```
orbit_velocity_tracking_env_cfg
├── config
│   └── unitree_go1
│       ├── agent
│       │   └── __init__.py
│       │   └── rsl_rl_cfg.py
│       ├── __init__.py
│       ├── flat_env_cfg.py
│       ├── rough_env_cfg.py
│       ├── unitree_go1_cfg.py
│       └── __init__.py
│
├── mdp
│   ├── __init__.py
│   ├── curriculums.py
│   └── rewards.py
│
├── terrains
│   ├── __init__.py
│   └── rough_terrain_cfg.py
│
├── README.md
├── run_env.py
├── train.py
└── velocity_env_cfg.py
```
**Main idea of the file structure:**
- The *velocity_env_cfg.py* defines the base RL environment fo all robot configuration. Some of its attributes can be modified according to the robot size or task requirement in other files.
- The *config* file contains all configurations of the Unitree GO1 robot (in *config/unitree_go1/unitree_go1_cfg.py*) and tailors the *velocity_env_cfg.py* to the size of Unitree GO1 (in *config/unitree_go1/rough_env_cfg.py* and *flat_env_cfg.py*). The *config/unitree_go1/agent/rsl_rl_cfg* defines the network configuration of PPO for training.
- The *mdp* file contains all user-defined mdp functions. For example, the reward functions, the curriculums, etc.
- The *terrains* file define the training terrains for locomotion tasks.
- Debug: run *run_env.py* to visualize the environment.
- Train: run *train.py* to start training. 

## 2. <a href="https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/locomotion/velocity/velocity_env_cfg.py">velocity_env_cfg.py </a>

### 2.1 scene configuration

The scene includes all 'assets' (real objects) in the environment. For instance, the ground, the light, robots, sensors, etc. We first configure our scene to carry our objects onto the stage!

```python
@configclass # Always use the class wrapper *@configclass* before every configuration
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            # By using {NVIDIA_NUCLEUS_DIR}, it means to import pre-defined asset config in NVIDIA asset store.
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots. The robot configuration is set as missing value. We will fill it in another file.
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg( # To track the terrain's ups-and-downs.
        # {ENV_REGEX_NS} is the regex expression of "/World/envs/env_.*"
        # It enables us to add assets to all parellel environments once and for all.
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # To detect collision by tracking forces applied on each body of the robot.
    # By adding contact_force sensors on Robot/.*, each link of the robot has a sensor to track forces on this link specifically.
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
```

**Key points of the code:**

- Inherit *InteractiveSceneCfg* to define a scene:
```python
from omni.isaac.orbit.scene import InteractiveSceneCfg
```

- <a href="https://isaac-orbit.github.io/orbit/source/api/orbit/omni.isaac.orbit.utils.html#module-omni.isaac.orbit.utils.configclass">Class wrapper</a>. Always use the class wrapper *@configclass* before every configuration. Do 
```python
from omni.isaac.orbit.utils import configclass
```

- <a href="https://isaac-orbit.github.io/orbit/source/api/orbit/omni.isaac.orbit.terrains.html#omni.isaac.orbit.terrains.TerrainImporter">Terrain</a>. The terrain configuration is pre-defined by ETH. By setting terrain_type as 'generator', we can import the pre-defined terrain by setting terrain_generator as our terrain config class.
```python
from .terrains.rough_terrain_cfg import ROUGH_TERRAINS_CFG
```

- NVIDIA_NUCLEUS_DIR. The visual_material of the terrain is imported from the NVIDIA NUCLEUS. It is a pre-defined asset configuration. Huge amount of treasures there!

- Robot. The articulation (robot) configuration is set as missing value. We will fill it in another file.

- Sensors. In this env, each robot has two types of sensors: <a href="https://isaac-orbit.github.io/orbit/source/api/orbit/omni.isaac.orbit.sensors.html#omni.isaac.orbit.sensors.RayCasterCfg">height_scanner</a> to track the terrain's ups-and-downs and <a href="https://isaac-orbit.github.io/orbit/source/api/orbit/omni.isaac.orbit.sensors.html#omni.isaac.orbit.sensors.ContactSensorCfg">contact_forces</a> to detect collision by tracking forces applied on each body of the robot. You can visualize the effect of the sensor to debug (the visualization interface of each sensor is different so check the doc). For instance,, to visualize height_scanner, set debug_vis as True.

- <a href="https://isaac-orbit.github.io/orbit/source/api/orbit/omni.isaac.orbit.scene.html#omni.isaac.orbit.scene.InteractiveScene.env_regex_ns">{ENV_REGEX_NS}</a>. It is the regex expression of "/World/envs/env_.*". If we train the agent in massively parellel environments, we will creat each environment under "env_{index}". Therefore, {ENV_REGEX_NS} enables us to add assets to all parellel environments once and for all. 


### 2.2 Action Configuration

```python
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
```

### 2.3 Observations Configuration

Robots get observations to decide how to act. Here we configure the observation terms of the robot.

```python
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
```

**Key points of the code:**

- Observation. Observation terms are grouped. Each group has terms with the same dimension. A group of terms can be managed together. For instance, the code above has only one obs group cfg called "PolicyCFG" inheritting ObsGroup. We can manage some aspects of all terms in this group through PolicyCFG. Import term cfg class and group cfg class by:
```python
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
```

- <a href="https://isaac-orbit.github.io/orbit/source/api/orbit/omni.isaac.orbit.envs.mdp.html#module-omni.isaac.orbit.envs.mdp.observations">Pre-defined obs term functions</a>. You may notice that in the mdp file of this tutorial, we haven't defined any obs functions. The reason is isaac orbit has many pre-defined functions for creating an obs term in one line of code. For example, the following code will return the linear velocity of the robot base:
```python
base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
```
If we dig deeper, we can see the source code of ***mdp.base_lin_vel***:
```python
def base_lin_vel(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b
```
What it does is the following things: 1. reach the asset through <a href="https://isaac-orbit.github.io/orbit/source/api/orbit/omni.isaac.orbit.managers.html#omni.isaac.orbit.managers.SceneEntityCfg">```SceneEntityCfg```</a>; 2. Get the desired data from the <a href="https://isaac-orbit.github.io/orbit/source/api/orbit/omni.isaac.orbit.assets.html#omni.isaac.orbit.assets.ArticulationData">data container of that asset</a>. **We will cover both of these later**. 

Similarly, orbit has pre-defined reward functions, command functions, etc. They locate at omni.isaac.orbit.envs.mdp. You can use them by:
```python
# In this tutorial, this import is done in the mdp/__init__.py
# Therefore, it will be imported into this file if you import mdp of this tutorial.
from omni.isaac.orbit.envs.mdp import * 
```

- Get access to the asset during simulation using <a href="https://isaac-orbit.github.io/orbit/source/api/orbit/omni.isaac.orbit.managers.html#omni.isaac.orbit.managers.SceneEntityCfg">```SceneEntityCfg```</a>. In many cases, we want to reach the asset during simulation to get or change the states. In the code above, the information height_scanner got is an obs term for the robot. To reach the sensor data, we first need to reach the sensor. Pay attention that we only get the asset configuration through SceneEntityCfg. But most functions use asset cfg as the arguments so that's fine. In our case, the sensor is called "height_scanner", so we get it through:
```python
# sensor's name is "height_scanner" because we do height_scanner = RayCasterCfg(...) in the scene cfg.
SceneEntityCfg("height_scanner")
``` 

- Manage the whole group with ```__post_init__```. As an example of whole-group management, you can change the group's attribute <a href="https://isaac-orbit.github.io/orbit/source/api/orbit/omni.isaac.orbit.managers.html#omni.isaac.orbit.managers.ObservationGroupCfg.enable_corruption">```enable_corruption```</a>.If enable_corruption==True, the observation terms in the group are corrupted by adding noise (if specified), and  otherwise, no corruption is applied. The reason of using ```__post_init__``` is to fine-tune the configuration without changing the original group configuration. In our code case:
```python
def __post_init__(self):
    self.enable_corruption = True
    self.concatenate_terms = True
```


- Instantiate ObsGroupCfg at last. Remember to instantiate all group configuration(s):
```python
policy: PolicyCfg = PolicyCfg()
```

### 2.4 Event Configuration

Observation terms are all defined once and for all when the simulation starts, similar to the reward functions, curriculum rules, etc. But sometimes we want to have some 'special events' happen during simulation. For instance, we can do randomization toward some asset properties to narrow the sim-to-real gap. We use <a href="https://isaac-orbit.github.io/orbit/source/api/orbit/omni.isaac.orbit.managers.html#omni.isaac.orbit.managers.EventTermCfg">```EventTermCfg```</a> to enable special things happen during simulation.

```python
@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
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

    add_base_mass = EventTerm(
        func=mdp.add_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "mass_range": (-5.0, 5.0)},
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

```

**Key points of the code:**

- Choose the mode of an event. Three modes: ***startup***,***reset***,***interval***. The startup event happens only once at the beginning of training; the reset event happens at every time when an env is reset; the interval event happens within a specific time period of the simulation.

- ```physic_dt``` and ```sim_dt```. These attributes of an env is defined as follows:
```python
class Env(RLTaskEnvCfg):
    def __post_init__(self):
        self.decimation = 4 # Unit: physical step(sim.dt or physic_dt)
        self.episode_length_s = 20.0 # Unit: second
        self.sim.dt = 0.005 # Unit: second
```
The ```sim.dt``` (physic_dt) determines how often the physics engine updates the state of the simulated world. A smaller time-step increases the accuracy of the simulation by calculating the physics more frequently, but at the cost of increased computational load. The ```decimation``` determines how many physical steps will the agent take action once. In this case, the agent takes action every ```4``` physical steps (0.005*4 = 0.02 second). Then, according to the ```episode_length_s```, the agent will take 1000 times of action in one episode (if episode not terminated ealier).

- Understand ***reset***.
The following code comes from the source code of <a href="https://isaac-orbit.github.io/orbit/_modules/omni/isaac/orbit/envs/base_env.html#BaseEnv">BaseEnv</a>. The ```_reset_idx``` is called when ```env.reset()``` is called. The ```_reset_idx``` will apply all ```EventTerm``` with a ```mode = "reset"```.
```python
def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(env_ids=env_ids, mode="reset")

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
```

- Understand ***interval***. The interval is sampled uniformly between the specified range for each environment instance. The term is applied on the environment instances where the current time hits the interval time. Use ```interval_range_s``` attribute to control the sampling range of the event's time.


### 2.5 Reward Configuration

We use reward terms to define the reward funtions of the agents.

```Python
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

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

```


**Key points of the code:**

- Same as the situation of observation configuration, there are pre-defined reward functions in the <a href="https://isaac-orbit.github.io/orbit/source/api/orbit/omni.isaac.orbit.envs.mdp.html#module-omni.isaac.orbit.envs.mdp.rewards">```omni.isaac.orbit.envs.mdp```</a>. For example, the term of:
```python
track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
```

- Self-defined reward function. In the ***mdp/rewards.py*** file of this tutorial, two reward functions are defiend. One is defined as followed:
```python
def feet_air_time(env: RLTaskEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward
```
The main idea of defining a reward function (and other self-defined funcs) is: first, reach the asset cfg with ```SceneEntityCfg[<asset_name>]```; second, use the methods of that asset class or data stored in the data container of that asset. For instance, the data container document of a ContactSensor is <a href="https://isaac-orbit.github.io/orbit/source/api/orbit/omni.isaac.orbit.sensors.html#omni.isaac.orbit.sensors.ContactSensorData">link</a>.



### 2.6 Termination Configuration
```python
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
```
- It is a good way to understand termination by looking at the source code of <a href="https://isaac-orbit.github.io/orbit/source/api/orbit/omni.isaac.orbit.envs.mdp.html#module-omni.isaac.orbit.envs.mdp.terminations">pre-defined termination functions</a>:
```python
def illegal_contact(env: RLTaskEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )
```
It returns a boolean value. If True, then the episode is terminated. **The hardest part of defining a function like this, I believe, is to know where the needed data is in the huge data container tensor**. No doc is provided and we need to ```print``` everything.


### Curriculum Configuration

The curriculum part is a good example to learn how mdp functions are called during simulation. Link: <a href="https://github.com/NVIDIA-Omniverse/orbit/blob/a30d764da2367415f1ed357bdcb581e99bb1a9b0/source/extensions/omni.isaac.orbit/omni/isaac/orbit/managers/manager_base.py">```manger_base.py```</a>;  <a href="https://github.com/NVIDIA-Omniverse/orbit/blob/main/source/extensions/omni.isaac.orbit/omni/isaac/orbit/managers/curriculum_manager.py">```curriculum_manager.py```</a>.

```python
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
```

### 2.7 ```__post_init__```

```python
@configclass
class LocomotionVelocityRoughEnvCfg(RLTaskEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # ... Here code instantiates the above mdp cfg is omitted. 

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
```

- Why ```__post_init__```? We use ```__post_init__``` to fine-tune the environment configuration without changing the original definition of the env. This makes the code more readable and debuggable because it explicitly shows what is the task-specific configuration of an environment.
