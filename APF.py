# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:58:37 2019

@author: yxy
"""

## artifial potential field
import potential
import numpy as np

class APF(object):
    def __init__(self, robot_num, sensor_range, site_mode): # env static info
        self.robot_num = robot_num
        self.sensor_range = sensor_range
        self.site_mode = site_mode # attachment-site mode

    def reset_state(self, goal_pos, obs_r, robot_r): # env info which will be changed only after reset
        self.goal_pos = goal_pos
        self.obs_r = obs_r
        self.robot_r = robot_r
        
    def set_state(self, robot_pos, obs_pos): # env dynamic info
        self.robot_pos = robot_pos
        self.obs_pos = obs_pos
        
        
    def calculate_potential_field(self):
        robots_action = []
        for robot_idx in range(self.robot_num):
            ### obs potential
            v_direction_obs = np.zeros(self.robot_pos[0].shape[0])
            v_magnitude_obs = 0.
            weight_obs = 1.1
            for i in range(self.obs_pos.shape[0]):
                v_magnitude_obs_i, v_direction_obs_i = potential.avoid_static_obstacles(self.robot_pos[robot_idx], self.obs_pos[i], self.obs_r[i])
                # print("robot_pos[idx] {} v_direction_obs_i {}".format(self.robot_pos[robot_idx], v_direction_obs_i))
                v_direction_obs += v_direction_obs_i
                v_magnitude_obs += v_magnitude_obs_i
            weight_v_direction_obs = v_direction_obs/np.linalg.norm(v_direction_obs) * weight_obs # normalized and weighted
            
            ### partner potential
            v_direction_partner = np.zeros(self.robot_pos[0].shape[0])
            v_magnitude_partner = 0.
            weight_partner = 1.1
            for i, partner_pos in enumerate(self.robot_pos):
                if i == robot_idx: # np.all(self.robot_pos[robot_idx] == partner_pos)
                    continue
                else:
                    v_magnitude_partner_i, v_direction_partner_i = potential.avoid_robots(self.robot_pos[robot_idx], partner_pos, self.robot_r)
                v_direction_partner += v_direction_partner_i
                v_magnitude_partner += v_magnitude_partner_i
            weight_v_direction_partner = v_direction_partner/np.linalg.norm(v_direction_partner) * weight_partner # normalized and weighted
            
            ### goal potential
            v_magnitude_goal, v_direction_goal = potential.move_to_goal(self.robot_pos[robot_idx], self.goal_pos)
            weight_goal = 0.7
            weight_v_direction_goal = v_direction_goal/np.linalg.norm(v_direction_goal) * weight_goal # normalized and weighted
            
            ### formation potential
            v_direction_formation = np.zeros(self.robot_pos[0].shape[0])
            v_magnitude_formation = 0.
            weight_formation = 1.3
            for i, partner_pos in enumerate(self.robot_pos):
                if i == robot_idx: 
                    continue
                else:
                    dist = np.linalg.norm(self.robot_pos[robot_idx] - self.robot_pos[i])
                    if dist < self.sensor_range: # only consider the partners within sensor range
                        partner_site_pos = self.attachment_site(self.robot_pos[i], self.site_mode)
                        site_dist = np.linalg.norm(self.robot_pos[robot_idx] - partner_site_pos, axis=1)
                        chosen_site_pos = partner_site_pos[np.argmin(site_dist)] # choose the nearest site as the center of attractive potential field
                        v_magnitude_site, v_direction_site = potential.maintain_formation(self.robot_pos[robot_idx], chosen_site_pos)
                        
                        v_direction_formation += v_direction_site
                        v_magnitude_formation += v_magnitude_site
            weight_v_direction_formation = v_direction_formation/np.linalg.norm(v_direction_formation) * weight_formation # normalized and weighted
                            
            
            ### unit center potential
            unit_center = np.average(self.robot_pos)
            v_magnitude_unitcenter, v_direction_unitcenter = potential.move_to_uint_center(self.robot_pos[robot_idx], unit_center)
            weight_unitcenter = 0.6
            weight_v_direction_unitcenter = v_direction_unitcenter/np.linalg.norm(v_direction_unitcenter) * weight_unitcenter # normalized and weighted
            
            # action of one robot
            # v_direction = v_direction_obs + v_direction_partner + v_direction_goal + v_direction_formation + v_direction_unitcenter # [dx, dy] before normalized
            v_direction = weight_v_direction_obs + weight_v_direction_partner + weight_v_direction_goal + weight_v_direction_formation + weight_v_direction_unitcenter # use weighted direction
            normalized_v_direction = v_direction / np.linalg.norm(v_direction)
            v_magnitude = v_magnitude_obs + v_magnitude_partner + v_magnitude_goal + v_magnitude_formation + v_magnitude_unitcenter
            robots_action.append([v_magnitude*normalized_v_direction[0], v_magnitude*normalized_v_direction[1]])

        
        return np.array(robots_action) # shape of robots_action: [[vx1,vy1],[vx2,vy2],[vx3,vy3]]

    # return each attachment-site of one robot based on specific mode
    def attachment_site(self, robot_pos, mode):
        # r for distance from robot to attachment-site; N for number of site; theta for offset of degree
        r = 1.5 # same for all attchment-site mode
        site_pos = []
        if mode == "diamond":
            N = 4
            # direction of site: right-up, left-up, left-bottom, right-bottom
            for i in range(N):
                site_pos_i = [robot_pos[0]+r*np.cos(np.pi/4+np.pi/2*i), robot_pos[1]+r*np.sin(np.pi/4+np.pi/2*i)]
                site_pos.append(site_pos_i)
        elif mode == "line":
            N = 2
            site_pos_1 = [robot_pos[0]-r, robot_pos[1]] # left side site
            site_pos_2 = [robot_pos[0]+r, robot_pos[1]] # right side site
            site_pos = [site_pos_1, site_pos_2]
        elif mode == "column":
            N = 2
            site_pos_1 = [robot_pos[0], robot_pos[1]-r] # lower side site
            site_pos_2 = [robot_pos[0], robot_pos[1]+r] # upper side site
            site_pos = [site_pos_1, site_pos_2]
        elif mode == "square":
            N = 4
            site_pos_1 = [robot_pos[0]+r, robot_pos[1]] # right side site
            site_pos_2 = [robot_pos[0], robot_pos[1]+r] # upper side site
            site_pos_3 = [robot_pos[0]-r, robot_pos[1]] # left side site
            site_pos_4 = [robot_pos[0], robot_pos[1]-r] # lower side site
            site_pos = [site_pos_1, site_pos_2, site_pos_3, site_pos_4]
        else:
            raise Exception("No attachment-site for mode {}".format(mode))
        
        return site_pos