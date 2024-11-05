'''To run this script, input the following command in the terminal:
python multi_random_agents --task Iot-Quadcopter-Direct-v0 --num_envs 1 --disable_fabric
'''

from __future__ import annotations

import argparse
from typing import Any

from omni.isaac.lab.app import AppLauncher

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import  parse_env_cfg
def main():
    """Random actions agent with Isaac Lab MARL environment."""
    # Parse arguments


    # Create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Print environment info
    print(f"\n{'='*50}")
    print(f"Environment: {args_cli.task}")
    print(f"Number of agents: {len(env.unwrapped.possible_agents)}")
    print(f"Number of environments: {env.unwrapped.num_envs}")
    print(f"Device: {env.unwrapped.device}")
    print(f"{'='*50}\n")

    # Print spaces for each agent
    for agent in env.unwrapped.possible_agents:
        print(f"Agent: {agent}")
        print(f"  Observation space: {env.unwrapped.observation_space(agent)}")
        print(f"  Action space: {env.unwrapped.action_space(agent)}")
    print(f"{'='*50}\n")

    # Reset environment
    obs_dict, info_dict = env.reset()
    episode_rewards = {agent: 0.0 for agent in env.unwrapped.possible_agents}
    episode_count = 0

    # Simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # Sample random actions for each agent
            actions = {
                agent: (2 * torch.rand(
                    (env.unwrapped.num_envs, env.unwrapped.action_space(agent).shape[0]),
                    device=env.unwrapped.device
                ) - 1)
                for agent in env.unwrapped.possible_agents
            }

            # Step environment
            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(actions)

            # Update episode rewards
            for agent in env.unwrapped.possible_agents:
                episode_rewards[agent] += reward_dict[agent].mean().item()

            # Check if episode ended - modified to handle tensor boolean values
            if any(t.any().item() for t in terminated_dict.values()) or any(t.any().item() for t in truncated_dict.values()):
                episode_count += 1
                print(f"\nEpisode {episode_count} completed:")
                for agent in env.unwrapped.possible_agents:
                    print(f"  {agent} reward: {episode_rewards[agent]:.2f}")
                
                # Reset episode rewards
                episode_rewards = {agent: 0.0 for agent in env.unwrapped.possible_agents}

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()