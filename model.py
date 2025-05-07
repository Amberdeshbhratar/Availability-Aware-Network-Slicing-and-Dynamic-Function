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


env = Sliavalilran()
# state = env.reset()
done = False

# Verify that the environment follows the correct API
check_env(env)

# Define the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model for a certain number of timesteps
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_vnf_placement")

# Load the model (optional)
model = PPO.load("ppo_vnf_placement")

# Test the trained model
obs,info = env.reset()


total_reward = 0

requests = {0:"URLLC",
            1:"EMBB",
            2:"MMTC"}

for episode in range(10):
    while not done:
        action, _states = model.predict(obs)
        # vnc,ru,wavelength,request_type = action
        # request_type = random.randint(0,2)
        # action = (vnc,ru,wavelength,request_type)
        # request = ""
        # path = all_paths[rus[ru]]
        # print(action)
        obs, reward, done,truncated, info = env.step(action)
        total_reward += reward
        if done == True:
            break
        # print("request type {} , VNC {} ,Path {}, RU {} , wavelength {}".format(requests[request_type],vnc,path,rus[ru],wavelength))
        
    obs,info = env.reset()
    done = False
    print("Total reward in the Episode {}".format(total_reward))
    total_reward = 0


