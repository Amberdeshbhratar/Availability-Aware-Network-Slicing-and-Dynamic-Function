from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import itertools
import gymnasium as gym
import json
from gymnasium import error, spaces, utils
import numpy as np
import random
import math
from custom_env import *
import os

# Create the custom environment
# env = Sliavalilran()

# # Set up the log directory
# log_dir = "./logs/"
# os.makedirs(log_dir, exist_ok=True)

# # Wrap the environment with Monitor and specify the log file location
# env = Monitor(env, log_dir)

# # Define the PPO model
# model = PPO('MlpPolicy', env, verbose=1)

# # Train the model for a certain number of timesteps
# model.learn(total_timesteps=10000)

# # Save the trained model
# model.save("ppo_vnf_placement")

# log_file = os.path.join(log_dir, 'monitor.csv')
# log_data = pd.read_csv(log_file, skiprows=1)  # Skip the first row which is a comment

# # Group by episode and aggregate rewards
# # Assuming 'l' is the episode number and 'r' is the reward
# aggregated_data = log_data.groupby('l')['r'].mean().reset_index()

# # Plot the average reward per episode
# plt.plot(aggregated_data['l'], aggregated_data['r'])
# plt.xlabel('Episode Number')
# plt.ylabel('Average Reward')
# plt.title('PPO Training: Average Reward Per Episode')
# plt.grid()
# plt.show()


# Create the custom environment and log directory
# env = VNFPlacementEnv()
# log_dir = "./logs/"
# os.makedirs(log_dir, exist_ok=True)

# # Wrap the environment with Monitor and specify the log file location
# env = Monitor(env, log_dir)

# # Define the PPO model
# model = PPO('MlpPolicy', env, verbose=1)

# # Train for 100 episodes
# total_episodes = 100
# for episode in range(total_episodes):
#     obs, _ = env.reset()  # Reset environment at the start of each episode
#     done = False
#     while not done:
#         # Predict the action and take a step in the environment
#         action, _ = model.predict(obs)
#         obs, reward, done,truncated, info = env.step(action)
#         model.learn(total_timesteps=1)  # Learn from this timestep

#     print(f"Episode {episode+1}/{total_episodes} completed.")

# # Save the trained model
# model.save("ppo_vnf_placement")

# # Load the Monitor's log file
# log_file = os.path.join(log_dir, 'monitor.csv')
# log_data = pd.read_csv(log_file, skiprows=1)  # Skip the first row which is a comment

# # Group by episode and calculate the average reward per episode
# aggregated_data = log_data.groupby('l')['r'].mean().reset_index()

# # Plot the average reward per episode to show convergence
# plt.plot(aggregated_data['l'], aggregated_data['r'])
# plt.xlabel('Episode Number')
# plt.ylabel('Average Reward')
# plt.title('PPO Training: Average Reward Per Episode (100 Episodes)')
# plt.grid()
# plt.show()


log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

# Create the custom environment
env = Sliavalilran()

# Wrap the environment with Monitor and specify the log file location
env = Monitor(env, log_dir)

# Define the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train for 100 episodes
total_episodes = 150
rewards_per_episode = []  # Store rewards for each episode

for episode in range(total_episodes):
    obs, _ = env.reset()  # Reset environment at the start of each episode
    done = False
    total_reward = 0  # To accumulate the reward for this episode

    # while not done:
    #     # Predict the action and take a step in the environment
    #     action, _ = model.predict(obs)
    #     obs, reward, done,truncated, info = env.step(action)
    #     total_reward += reward  # Accumulate reward

    #     model.learn(total_timesteps=1)  # Learn from this timestep

    # Predict the action and take a step in the environment
    action, _ = model.predict(obs)
    obs, reward, done,truncated, info = env.step(action)
    total_reward += reward  # Accumulate reward

    model.learn(total_timesteps=1)  # Learn from this timestep



    rewards_per_episode.append(total_reward)  # Store total reward for the episode
    print(f"Episode {episode+1}/{total_episodes} completed. Reward: {total_reward}")

# # Calculate moving average (for smoothing)
# window_size = 10  # Define the size of the moving window
# moving_avg_rewards = np.convolve(rewards_per_episode, np.ones(window_size)/window_size, mode='valid')

# Plot the moving average of rewards to show convergence
plt.plot(rewards_per_episode)
plt.xlabel('Episode Number')
plt.ylabel('Reward ')
plt.title('PPO Training: Convergence of Reward Over Episodes')
plt.grid()
plt.show()

