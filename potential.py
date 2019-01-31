# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:24:29 2019

@author: yxy
"""

## definition of potential

import numpy as np

### 1)avoid_static_obstacles  2)avoid_robots
def repulsive(r, schema): 
    # r for distance; S for sphere; M for margin; schema for parameter mode
    if schema == "avoid_static_obstacles" or schema == "avoid_robots":
        S = 2.0
        M = 0.1
    else:
        raise Exception("No repulsive field for schema {}".format(schema))
        
    if r > S:
        v_magnitude = 0
    elif r > M and r <= S:
        v_magnitude = (S-r)/(S-M)
    else:
        v_magnitude = float('inf')
    return v_magnitude

### 1)move_to_goal 2)maintain_formation 3)move_to_unit_center
def attractive(r, schema): 
    # r for distance; C for controlled zone; D for dead zone; schema for parameter mode
    if schema == "move_to_goal":
        C = 0.
        D = 0.
    elif schema == "maintain_formation":
        C = 1.
        D = 0.
    elif schema == "move_to_unit_center":
        C = 3.
        D = 2.
    else:
        raise Exception("No attractive field for schema {}".format(schema))
    if r > C:
        v_magnitude = 1
    elif r > D and r <= C:
        v_magnitude = (r-D)/(C-D)
    else:
        v_magnitude = 0
    return v_magnitude

def avoid_static_obstacles(robot_pos, obs_pos, obs_pos_r): 
    r = np.linalg.norm(robot_pos - obs_pos) - obs_pos_r # should consider the radius of obstacles
    v_magnitude = repulsive(r, schema="avoid_static_obstacles")
    v_direction = robot_pos - obs_pos
    return v_magnitude, v_direction

def avoid_robots(robot_pos, partner_pos, partner_r):
    r = np.linalg.norm(robot_pos - partner_pos) - partner_r # should consider the radius of partner
    v_magnitude = repulsive(r, schema="avoid_robots")
    v_direction = robot_pos - partner_pos
    return v_magnitude, v_direction

def move_to_goal(robot_pos, goal_pos):
    r = np.linalg.norm(robot_pos - goal_pos)
    v_magnitude = attractive(r, schema="move_to_goal")
    v_direction = goal_pos - robot_pos
    return v_magnitude, v_direction

def maintain_formation(robot_pos, formation_pos):
    r = np.linalg.norm(robot_pos - formation_pos)
    v_magnitude = attractive(r, schema="maintain_formation")
    v_direction = formation_pos - robot_pos
    return v_magnitude, v_direction

def move_to_uint_center(robot_pos, unit_center):
    r = np.linalg.norm(robot_pos - unit_center)
    v_magnitude = attractive(r, schema="move_to_unit_center")
    v_direction = unit_center - robot_pos
    return v_magnitude, v_direction