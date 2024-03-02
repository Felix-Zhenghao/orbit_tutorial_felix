# Understand Environment and Its Components

While a simulation scene or world comprises of different components such as the robots, objects, and sensors (cameras, lidars, etc.), the environment is a higher level abstraction that provides an interface for interacting with the simulation. The environment is comprised of the following components:

- Scene: The scene manager that creates and manages the virtual world in which the robot operates. This includes defining the robot,
static and dynamic objects, sensors, etc.

- Observation Manager: The observation manager that generates observations from the current simulation state and the data gathered 
from the sensors. These observations may include privileged information that is not available to the robot in the real world.
Additionally, user-defined terms can be added to process the observations and generate custom observations. 
For example, using a network to embed high-dimensional observations into a lower-dimensional space.

- Action Manager: The action manager that processes the raw actions sent to the environment and converts them to low-level commands 
that are sent to the simulation. It can be configured to accept raw actions at different levels of abstraction. 
For example, in case of a robotic arm, the raw actions can be joint torques, joint positions, or end-effector poses. 
Similarly for a mobile base, it can be the joint torques, or the desired velocity of the floating base.

- Randomization Manager: The randomization manager that randomizes different elements in the scene. 
This includes resetting the scene to a default state or randomize the scene at different intervals of time. 
The randomization manager can be configured to randomize different elements of the scene such as the masses of objects, 
friction coefficients, or apply random pushes to the robot.

The environment provides a unified interface for interacting with the simulation. However, it **does not include task-specific** quantities such as the reward function, or the termination conditions. These quantities are often specific to defining Markov Decision Processes (MDPs) while the base environment is agnostic to the MDP definition.

**The environment steps forward in time at a fixed time-step. The physics simulation is decimated at a lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured independently using the BaseEnvCfg.decimation (number of simulation steps per environment step) and the BaseEnvCfg.sim.dt (physics time-step) parameters. Based on these parameters, the environment time-step is computed as the product of the two. The two time-steps can be obtained by querying the physics_dt and the step_dt properties respectively.**

- **Physics Time-step (sim.dt)**: The physics time-step is a fundamental parameter that determines how often the physics engine updates the state of the simulated world. A smaller time-step increases the accuracy of the simulation by calculating the physics more frequently, but at the cost of increased computational load. In the context of reinforcement learning, this is crucial for accurately modeling the dynamics of the environment, such as the movement of objects or agents, ensuring that the simulated environment behaves as close to the real world as possible.

- **Simulation Steps (decimation)**: Simulation steps, as described by the term 'decimation' in the provided context, refer to the number of physics simulation steps that are executed for each step of the environment that the RL agent perceives. This is a method to decouple the high-frequency updates required for accurate physics simulation from the lower-frequency decisions made by the RL agent. This discrepancy allows the simulation to maintain a high degree of physical accuracy through frequent updates while managing the computational and cognitive load by reducing the frequency of decision points required of the agent.

Why called 'decimation'? "decimation" refers to the process of reducing the sampling rate or resolution of a dataset or signal. This is achieved by selecting a subset of data points or simulation steps and discarding the rest. The term does not imply destruction in a physical sense.

In scenarios where the environment advances at a finer temporal resolution than the agent's decision-making frequency, defining the observation for the agent involves summarizing or aggregating the information available during the interval between the agent's actions. Why? Because while the physics simulation might update the state of the environment many times between the agent's decisions (due to the decimation factor), the agent typically receives only one observation per decision step.




