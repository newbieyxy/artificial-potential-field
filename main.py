# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:27:57 2019

@author: yxy
"""

## multi-robot formation simulation using APF

import os
import glob

from gym import wrappers
import gym
import gym_foa
from APF import APF

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--env-name', type=str, default="apf_pos_v-v0", help="Make env (default: apf_pos_v-v0)")
parser.add_argument('--site-mode', type=str, default="square", help="Choose the mode of attachment site: diamond, line, column, square")
parser.add_argument('--save-dir', type=str, default="apf", help="Set the save path of record, suggesting date.")
parser.add_argument('--scene-dir', type=str, default=None, help="Load the environment from this path (default: None).")
args = parser.parse_args()

env = gym.make(args.env_name) # env is set based on global coordinate
env.seed(1)
save_dir = "./logs/"+args.save_dir
os.makedirs(save_dir) # avoid covering existing files
env = wrappers.Monitor(env, save_dir, lambda episode_id: True, force=True)

# env state information (which is static)
try:
    robot_num = env.unwrapped.hp_uav_n
except:
    robot_num = env.unwrapped.s_uav_n
    
sensor_range = env.unwrapped.hp_lidar_range
site_mode = args.site_mode

agent = APF(robot_num, sensor_range, site_mode)

if args.scene_dir is not None:
    iterator = glob.glob(os.path.join(args.scene_dir, "scene*.pkl"))
else:
    iterator = range(1)
    
for it in iterator: 
    if args.scene_dir is not None:
        env.unwrapped.load_scene(it)
    env.reset()
    
    goal_pos = env.unwrapped.s_goal
    obs_r = env.unwrapped.s_obs_r # vector like [r1,r2,...]
    robot_r = env.unwrapped.hp_uav_r # scalar
    agent.reset_state(goal_pos, obs_r, robot_r)
    steps = 0
    while True:
        steps += 1
        # env state information (which is dynamic)
        obs_pos = env.unwrapped.s_obs_pos
        robot_pos = env.unwrapped.s_uav_pos
        # feed the agent with current state
        agent.set_state(robot_pos, obs_pos)
        action = agent.calculate_potential_field() # action is based on global coordinate
        # print("action {}".format(action))
        s, r, done, info = env.step(action)
        print("Iteration {}. Step {}. Action {}".format(it, steps, action))
        
        if done:
            env.video_recorder.capture_frame()  # Save the last frame
            break
        