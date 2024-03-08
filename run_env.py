"""
This script demonstrates how to run the RL environment for the cartpole balancing task.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Visualize the environment to debug RL environment configuration.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import traceback

import carb

from omni.isaac.orbit.envs import RLTaskEnv

# import the configuration of environment wants to run
from env_cfg import TomAndJerryEnvCfg # This contains the RL env configuration for Unitree GO 1

def main():
    """Main function."""
    # parse the arguments
    env_cfg = TomAndJerryEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = RLTaskEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            
            env.step()
            # sample random actions
            # joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            # obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # output_dict = env.step(joint_efforts)
            # print current orientation of pole
            # print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            #print("-----DICTIONARY OF OUTPUT-----\n",output_dict,"\n-----END OF OUTPUT-----")
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()