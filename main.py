# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:27:57 2019

@author: yxy
"""

## multi-robot formation simulation using APF

import os
from gym import wrappers
import gym
import gym_foa
from APF import APF

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--site-mode', type=str, help="Choose the mode of attachment site: diamond, line, column, square")
parser.add_argument('--save-dir', type=str, help="Set the save path of record, suggesting date.")
args = parser.parse_args()

env = gym.make("rvo_pos_v-v0")
env.seed(1)
save_dir = "./logs/"+args.save_dir
os.makedirs(save_dir) # avoid covering existing files
env = wrappers.Monitor(env, save_dir, lambda episode_id: True)

# env state information (which is static)
robot_num = env.unwrapped.hp_uav_n
sensor_range = env.unwrapped.hp_lidar_range
site_mode = args.site_mode

agent = APF(robot_num, sensor_range, site_mode)

test_num = 10
for it in range(test_num):
    env.reset()
    steps = 0
    while True:
        steps += 1
        # env state information (which is dynamic)
        obs_pos = env.unwrapped.s_obs_pos
        robot_pos = env.unwrapped.s_uav_pos
        goal_pos = env.unwrapped.s_goal
        agent.set_state(robot_pos, obs_pos, goal_pos)
        action = agent.calculate_potential_field()
        # print("action {}".format(action))
        s, r, done, info = env.step(action)
        print("Iteration {}. Step {}. Action {}".format(it, steps, action))
        
        if done:
            break
        