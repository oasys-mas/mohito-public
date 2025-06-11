#----------------------------------------------------------------------------------------------------------------------#
# Title: Env Simulator
# Description: This file contains the Rideshare environment simulator that follows the TaO-MDP Framework
# Author: Gayathri Anil 
# Version: 23.05.01
# Last updated on: 05-03-2023 
#----------------------------------------------------------------------------------------------------------------------#

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
from random import randrange
import math
import numpy as np
import pandas as pd
from copy import deepcopy
import heapq
import torch
from torch_geometric.data import Data
from utils import * 
from torch_geometric.utils import add_self_loops

#num_agents = 3
ridefare_x = 3
relative = True
unaccepted_wait_cost_variable = 0 #-2

# Function to set edge in adjacency matrix
def set_edge(adj_matrix, n1, n2):
    adj_matrix.loc[n1, n2] = 1
    adj_matrix.loc[n2, n1] = 1

class simulatedRideshare():

    def __init__(self, num_agents = 3, grid_len = 5, grid_wid = 5, accept_cost = -0.5, \
        pick_cost = -0.1, move_cost = -0.1, noop_cost = -1, drop_cost = None, \
            no_pass_cost=2, variable_move_cost=False, variable_pick_cost=False, \
                no_pass_reward=False, wait_limit = [5, 10, 10], \
                    open_level = 0.3, task_dist = (0.7, 0.2, 0.1), \
                        pool_limit = 2, pool_limit_cost = -2):
        '''
        Function to initiliase variables for Rideshare domain
        Initialise all agents at a set of locations 
        '''
        
        # grid settings
        self.num_agents = num_agents
        self.grid_length = grid_len
        self.grid_width = grid_wid

        # reward settings
        self.variable_move_cost = variable_move_cost
        self.variable_pick_cost = variable_pick_cost
        self.no_pass_reward = no_pass_reward
        self.accept_cost = accept_cost
        self.pick_cost = pick_cost
        self.move_cost = move_cost
        self.miss_cost = noop_cost
        self.drop_cost = drop_cost
        self.no_pass_cost = no_pass_cost
        self.wait_limit = wait_limit

        # exo settings
        # probability of new passengers entering the simulation
        # self.exo_lambda = open_level
        # self.task_dist = task_dist
        self.pool_limit = pool_limit
        self.pool_limit_cost = pool_limit_cost

        '''
        State details
        layer 0 - all agents' locations (agent index - 0:num_agents-1)
        layer 1 - all accepted passengers (driver index, start, end, fare)
        layer 2 - all riding passenger (driver index, start, end, fare) 
        layer 3 - all unaccepted passengers (end, start, fare)
        '''

        # environment state
        self.state = np.empty((4, self.grid_length, self.grid_width), dtype=object)
        # to fill the matrix with lists: matrix.fill([])
        for i in range(4):
            for j in range(self.grid_length):
                for k in range(self.grid_width):
                    self.state[i, j, k] = []
        
        # intialise list to hold agents' actions
        self.action_list = [None] * self.num_agents

        # intialise list to hold agents' observations
        self.observation_list = []
        
        # assign done and rewards for all agents
        self.done_list = [0] * self.num_agents
        self.reward_list = [0] * self.num_agents

        # action types
        self.action_type_list = ["accept", "pick", "drop", "noop"]

        # placeholder spaces - just to fulfill the requirements of a Gym class
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)

        # node feature settings
        self.agent_node_no = 1
        self.task_node_no = 2
        self.action_node_no = 3
        self.edge_node_no = 0


    def getObsFromState(self, state):
        '''
        Function to get each agent's observation from state
        '''

        observation_list = []
        for agent_no in range(self.num_agents):
            '''
            Observation details
            layer 0 - self locations (agent index)
            layer 1 - all other agents' locations (agent index - 0:num_agents-1)
            layer 2 - self accepted passengers (driver index, start, end, fare)
            layer 3 - self riding passenger (driver index, start, end, fare) 
            layer 4 - all unaccepted passengers (start, end, fare)
            '''
            
            obs = np.empty((5, self.grid_length, self.grid_width), dtype=object)
            for i in range(5):
                for j in range(self.grid_length):
                    for k in range(self.grid_width):
                        obs[i, j, k] = []
            
            # update layer 0 - self position
            loc = np.argwhere(np.vectorize(lambda x: agent_no in x)(state[0]))[0]
            obs[0, loc[0], loc[1]].append(agent_no)
            
            # update layer 1 - others position
            obs[1] = deepcopy(state[0])
            obs[1, loc[0], loc[1]].remove(agent_no)

            # print("BEFORE --- ", state)
            
            # update layer 2 - self accepted passengers
            acc_pass_locs = [(i, j, lst) for i, row in enumerate(state[1]) for j, cell in enumerate(row) for lst in cell if lst and lst[0] == agent_no]
            for loc in acc_pass_locs:
                obs[2, loc[0], loc[1]].append(loc[2])

            # print("AFTER ---", state)
            
            # update layer 3 - self riding passengers
            #print(type(state[2]))
            #print([type(cell) for row in state[2] for cell in row])

            #rid_pass_locs = [(i, j, lst) for i, row in enumerate(state[2]) for j, cell in enumerate(row) for lst in cell if isinstance(lst, list) and lst[0] == agent_no]

            rid_pass_locs = [(i, j, lst) for i, row in enumerate(state[2]) for j, cell in enumerate(row) for lst in cell if lst and lst[0] == agent_no]
            
            for loc in rid_pass_locs:
                obs[3, loc[0], loc[1]].append(loc[2])
            
            # update layer 4 - all unaccepted passengers
            obs[4] = deepcopy(state[3])

            observation_list.append(obs)

        return observation_list


    def step(self, simulated_eps, step, action_list, openness=True):
        '''
        Function to execute one step in the environment
        '''

        '''
        State details
        layer 0 - all agents' locations (agent   - 0:num_agents-1)
        layer 1 - all accepted passengers (driver index, start, end, fare, #entry_timestep)
        layer 2 - all riding passenger (driver index, start, end, fare, #entry_timestep) 
        layer 3 - all unaccepted passengers (start, end, fare, #entry_timestep)
        '''

        # if step == 44 or step == 45:
        #     print("let's take it from here")
        
        #print(action_list)
        reward_list = [0] * self.num_agents
        next_state = deepcopy(self.state)
        current_state = deepcopy(self.state)

        # stats
        completed_wait_time = []
        accepted_wait_time = []

        # # common reward for no unserviced passengers (+2)
        # if self.no_pass_reward and np.sum([1 for sublist in current_state[3].ravel() if sublist]) == 0:
        #     reward_list = [r+self.no_pass_cost for r in reward_list]

        
        # penalty for passengers waiting
        wait_limit = self.wait_limit
        unaccepted_long_wait_cost = -0.1
        unaccepted_wait_cost = unaccepted_wait_cost_variable
        unaccepted_long_count = 0
        unaccepted_count = 0
        general_wait_cost = -0.2
        penalty_list = []
        

        # Iterate through the first dimension of 'var'
        for i, mat in enumerate(current_state):
            if i == 2: # if i in [1, 2]:
                for sub_list in mat.flatten():
                    penalty_list.extend([inner_list[0] for inner_list in sub_list if inner_list and (step-inner_list[3]) > wait_limit[i-1]])
            if i == 3:  # For var[3]
                for sub_list in mat.flatten():
                    unaccepted_long_count += sum(1 for inner_list in sub_list if inner_list and (step-inner_list[2]) > wait_limit[2])
                    unaccepted_count += sum(1 for inner_list in sub_list if inner_list)

        # all agents get a penalty if there are agents who havent' been unaccepted for longer than a certain timesteps
        if unaccepted_long_count > 0:
            reward_list = [r+(unaccepted_long_wait_cost*unaccepted_long_count) for r in reward_list]

        # penalty for having too many unserviced passengers in the environment
        if unaccepted_count > self.num_agents * (self.pool_limit - 0.5):
            reward_list = [r+unaccepted_wait_cost for r in reward_list]

        # # penalty for keeping passengers waiting in the car without being dropped off
        # for p in penalty_list:
        #     reward_list[p] += general_wait_cost



        # transition rules to generate next state and reward
        for idx, agent_action in enumerate(action_list):
        
            # current location of the agent performing agent_action
            agent_loc_x, agent_loc_y = np.argwhere(np.vectorize(lambda x: idx in x)(self.state[0]))[0]

            # action carried out by agent
            agent_action = [int(f) for f in agent_action] #[:-1] + [agent_action[-1]]
            _, start, end, action_type, entry_step = agent_action 
            action_type = self.action_type_list[int(action_type)]

            # overwriting action_type if "miss" event happened
            if agent_action[0] == -1: 
                action_type = "miss"

            # start and end coordinates of the task that the agent has executed
            start_x, start_y = self.gridToCoords(start)
            end_x, end_y = self.gridToCoords(end)


            # -----------------------
            # Accept Action -> action_type == 0
            if (action_type == "accept"):

                # limit on number of passengers that can be accepted by an agent 
                count = sum(1 for level in [current_state[1], current_state[2]] 
                                for row in level 
                                for cell in row 
                                for c in cell
                                if c and c[0] == idx)
                if count > self.pool_limit:
                    reward_list[idx] += self.pool_limit_cost
                else:
                    # move accepted passenger at (start_x, start_y) to accepted passengers list at (end_x, end_y)
                    task_info = next((lst for lst in current_state[3, start_x, start_y] if (lst[1] == (end_x, end_y) and lst[3] == entry_step)), None)
                    next_state[1, start_x, start_y].append([idx] + task_info)
                    next_state[3, start_x, start_y].remove(task_info)
                    reward_list[idx] += 0 #self.accept_cost
                    accepted_wait_time.append(int(step)-int(task_info[-1]))


            # -----------------------
            # Pickup Action -> action_type == 1 - #todo: extend or append?
            elif (action_type == "pick"):
                # if agent is at the pick up location of the passenger, move passenger to riding list
                if (agent_loc_x, agent_loc_y) == (start_x, start_y):
                    task_info = next((lst for lst in current_state[1, start_x, start_y] if lst[2] == (end_x, end_y) and lst[0] == idx), None)
                    next_state[2, agent_loc_x, agent_loc_y].append(task_info)
                    next_state[1, agent_loc_x, agent_loc_y].remove(task_info)
                    
                    # reward
                    reward_list[idx] += self.pick_cost
                

                # else return next location using cityMap() and move the agent to that location
                else: 
                    _, (agent_next_loc_x, agent_next_loc_y) = self.cityMap(self.grid_length, self.grid_width, (agent_loc_x, agent_loc_y), (start_x, start_y)) 
                    # move agent
                    next_state[0, agent_next_loc_x, agent_next_loc_y].append(idx)
                    next_state[0, agent_loc_x, agent_loc_y].remove(idx)
                    
                    # move all riding passengers of the agent
                    lsts = [sublist for sublist in current_state[2, agent_loc_x, agent_loc_y] if isinstance(sublist, list) and sublist and sublist[0] == idx]
                    if len(lsts)>0:
                        next_state[2, agent_next_loc_x, agent_next_loc_y].extend(lsts)
                        next_state[2, agent_loc_x, agent_loc_y] = [sublist for sublist in next_state[2, agent_loc_x, agent_loc_y] if sublist not in lsts]
                    
                    # reward 
                    num_riders = sum(1 for row in current_state[2] for cell in row for sublist in cell if sublist and sublist[0] == idx)
                    num_accepted = sum(1 for row in current_state[1] for cell in row for sublist in cell if sublist and sublist[0] == idx)

                    if num_riders or num_accepted:
                        reward_list[idx] += (self.move_cost / (num_riders + num_accepted))
                    else:
                        reward_list[idx] += self.move_cost


            # -----------------------
            # Drop Action -> action_type == 2    
            elif (action_type == "drop"):
                # if agent is at the drop location of the passenger, move passenger out of the state
                if (agent_loc_x, agent_loc_y) == (end_x, end_y):
                    #print("Type:", type(current_state[2, end_x, end_y]))
                    #print("Value:", current_state[2, end_x, end_y])
                    #task_info = next((lst for lst in current_state[2, end_x, end_y] if isinstance(lst, list) and lst[1] == (start_x, start_y) and lst[0] == idx), None)

                    task_info = next((lst for lst in current_state[2, end_x, end_y] if lst[1] == (start_x, start_y) and lst[0] == idx), None)
                    
                    #if task_info in next_state[2, end_x, end_y]:
                    next_state[2, end_x, end_y].remove(task_info)
                    # reward - ridefare
                    if np.sum(np.vectorize(lambda x: sum(sublist[0] == idx for sublist in x if sublist), otypes=[int])(next_state[2])) > 0:
                        bonus = 4
                    else:
                        bonus = 0
                    # if there is no uniform drop cost
                    if self.drop_cost == None:
                        reward_list[idx] += ((task_info[3]) + bonus)
                    else: 
                        reward_list[idx] += (self.drop_cost)

                    completed_wait_time.append(int(step)-int(task_info[-1]))

                # else return next location using cityMap() and move both agent and passenger to that location
                else: 
                    _, (agent_next_loc_x, agent_next_loc_y) = self.cityMap(self.grid_length, self.grid_width, (agent_loc_x, agent_loc_y), (end_x, end_y))
                    #print(type(current_state[2, agent_loc_x, agent_loc_y]))
                    #print(current_state[2, agent_loc_x, agent_loc_y])
                    #task_info = next((lst for lst in current_state[2, agent_loc_x, agent_loc_y] if isinstance(lst, list) and lst[1] == (start_x, start_y) and lst[2] == (end_x, end_y) and lst[0] == idx), None)

                    task_info = next((lst for lst in current_state[2, agent_loc_x, agent_loc_y] if lst[1] == (start_x, start_y) and lst[2] == (end_x, end_y) and lst[0] == idx), None)
                    
                    if task_info == None:
                        print("stop here")
                        print("start_x, start_y = ", start_x, start_y)

                    # move agent to the next location and remove from previous location
                    next_state[0, agent_next_loc_x, agent_next_loc_y].append(idx)
                    next_state[0, agent_loc_x, agent_loc_y].remove(idx)
                    # move all riding passengers to new location and remove from previous location
                    lsts = [sublist for sublist in current_state[2, agent_loc_x, agent_loc_y] if isinstance(sublist, list) and sublist and sublist[0] == idx]
                    if len(lsts)>0:
                        next_state[2, agent_next_loc_x, agent_next_loc_y].extend(lsts)
                        next_state[2, agent_loc_x, agent_loc_y] = [sublist for sublist in next_state[2, agent_loc_x, agent_loc_y] if sublist not in lsts]
                    #next_state[2, agent_next_loc_x, agent_next_loc_y].append(task_info)
                    #next_state[2, agent_loc_x, agent_loc_y].remove(task_info)
                    # reward
                    # if there is variable reward for pooling
                    # if self.variable_move_cost and np.sum([1 for sublist in current_state[2].ravel() if sublist]) > 1:
                    #     reward_list[idx] += self.move_cost/2
                    # else:

                    num_riders = sum(1 for row in current_state[2] for cell in row for sublist in cell if sublist and sublist[0] == idx)
                    num_accepted = sum(1 for row in current_state[1] for cell in row for sublist in cell if sublist and sublist[0] == idx)
                    if num_riders or num_accepted:
                        reward_list[idx] += (self.move_cost / (num_riders + num_accepted))
                    else:
                        reward_list[idx] += self.move_cost

                    

            # -----------------------
            # Noop Action -> action_type == 3
            elif (action_type == "noop"):
                # reward - 0
                reward_list[idx] += 0 

            # -----------------------
            # Miss Action -> action_type == -1
            elif (action_type == "miss"):
                # reward - penalty
                reward_list[idx] += self.miss_cost 

        if openness:

            if step in simulated_eps:
                task_list = simulated_eps[step]
                num_tasks_added = len(task_list)

                # update next state
                for passenger in task_list:
                    next_state[3, passenger[0][0], passenger[0][1]].append([passenger[0], passenger[1], passenger[2], passenger[3]])

            else:
                num_tasks_added = 0
        else:
            num_tasks_added = 0

            
        # return state
        self.state = next_state
        self.reward_list = reward_list

        # stats
        num_new = sum(len(cell) for row in next_state[3] for cell in row)
        num_accepted = sum(len(cell) for row in next_state[1] for cell in row)
        num_riding = sum(len(cell) for row in next_state[1] for cell in row)

        if step == 0:
            num_tasks_added += num_new

        return next_state, reward_list, [num_accepted, num_riding, num_new, accepted_wait_time, completed_wait_time, num_tasks_added]

    def generateTask(self, step, num_tasks = None, task_dist = (70, 20, 10)):
        '''
        Function to generate new tasks and add to the environment
        '''

        # sample number of tasks to generate
        if num_tasks == None:
            num_tasks = random.choices([1, 2, 3], weights=task_dist)[0]
        task_list = []

        for task_no in range(num_tasks):

            # generating start and end points for passengers randomly
            start_x, start_y = randrange(self.grid_length), randrange(self.grid_width) 
            end_x, end_y = randrange(self.grid_length), randrange(self.grid_width)
            while (end_x, end_y) == (start_x, start_y):
                end_x, end_y = randrange(self.grid_length), randrange(self.grid_width)
            
            # estimating ridefare - setting a minimum fare
            ridefare = (ridefare_x * max(3,  manhattan_distance((start_x, start_y), (end_x, end_y)))) + randrange(-1, 2, 1)
            task_list.append([(start_x, start_y), (end_x, end_y), ridefare, step])

        return task_list
            
    #passengers_list = [[(2, 3), (4, 1), 7], [(0, 3), (3, 0), 7]]

    def reset(self, step, simulated_eps, driver_locations = None, passengers_list = None, accept_list = None, riding_list = None, num_passengers = 2):
        '''
        Function to reset the state of the environment to an initial state
        '''
        # state variable
        # environment state
        self.state = np.empty((4, self.grid_length, self.grid_width), dtype=object)
        # to fill the matrix with lists: matrix.fill([])
        for i in range(4):
            for j in range(self.grid_length):
                for k in range(self.grid_width):
                    self.state[i, j, k] = []

        # update layer 0 with driver locations
        for i in range(self.num_agents):
            if driver_locations:
                self.state[0, driver_locations[i][0], driver_locations[i][1]].append(i) 
            else:
                self.state[0, randrange(self.grid_length), randrange(self.grid_width)].append(i)

        # update layer 1 with accepted passenger details - (driver, start, end, fare)
        # accept_list = [[1, (1, 4), (2, 2), 5], [3, (0, 1), (2, 0), 6]]
        if accept_list:
            for accept in accept_list:
                self.state[1, accept[1][0], accept[1][1]].append([accept[0], accept[1], accept[2], accept[3], accept[4]])

        # update layer 2 with riding passenger details - (driver, start, end, fare)
        # riding_list = [[2, (1, 0), (3, 3), 6]]
        if riding_list:
            for ride in riding_list:
                agent_no = ride[0]
                agent_loc_x, agent_loc_y = np.argwhere(np.vectorize(lambda x: agent_no in x)(self.state[0]))[0]
                self.state[2, agent_loc_x, agent_loc_y].append([ride[0], ride[1], ride[2], ride[3], ride[4]])
        
        # update layer 3 with passenger details - (start, end, fare)
        # passengers_list = [[(2, 3), (4, 1), 7], [(0, 3), (3, 0), 7]]
        if passengers_list == None:
            passengers_list = simulated_eps[step]
        
        for passenger in passengers_list:
            # if len(passenger) != 4:
            #     passenger = passenger[0]
            self.state[3, passenger[0][0], passenger[0][1]].append([passenger[0], passenger[1], passenger[2], passenger[3]])

        # get observations for all agents
        self.observation_list = self.getObsFromState(self.state)

        # get action_list for all agents
        self.action_list = [None] * self.num_agents

        return self.state


    def gridToCoords(self, grid_cell):
        '''
        Function to convert a grid cell number to coordinates
        '''
        x = grid_cell // self.grid_width
        y = grid_cell - (x * self.grid_width)
        return [x, y]

    def coordsToGrid(self, coords):
        '''
        Function to convert cell coordinates to a grid cell number
        '''
        return (coords[0] * self.grid_width) + coords[1]


    def generateGraph(self, obs_list):
        '''
        Function to convert an observation into an interaction graph
        '''

        '''
        Node Features:
        Agent node: [node_type (1), agent_index, location, number of accepted passengers, number of riding passengers]
        Task node: [node_type (2), relative distance to agent, relative direction to agent, index of agent accepted by, index of agent riding with]
        Action node: [node_type (3), start location, end location, action_type, time step of passenger entry into environment]
        Edge node: [node_type (0), 0, 0, 0, 0]

        action_type = ["accept", "pick", "drop", "noop"]
        directions = [same point, N, NE, E, SE, S, SW, W, NW]

        if not use relative mode:
        Task node: [node_type (2), start location, end location, index of agent accepted by, index of agent riding with]
        '''
        graph_list = []
        action_space_list = []
        edge_nodes_list = []
        node_set_list = []

        for idx, obs in enumerate(obs_list):
    
            
            # -----------------------
            # agent nodes
            self_loc = np.argwhere(np.vectorize(lambda x: idx in x)(self.state[0]))[0]
            accepted_ct = np.sum(np.vectorize(lambda x: sum(isinstance(i, list) for i in x), otypes=[int])(obs[2]))
            riding_ct = np.sum(np.vectorize(lambda x: sum(isinstance(i, list) for i in x), otypes=[int])(obs[3]))
            
            agent_nodes = []
            for o in range(self.num_agents):
                if o == idx:
                    agent_nodes.append([self.agent_node_no, idx, self.coordsToGrid(self_loc), accepted_ct, riding_ct])
                else:
                    loc = np.argwhere(np.vectorize(lambda x: o in x)(self.state[0]))[0]
                    agent_nodes.append([self.agent_node_no, o, self.coordsToGrid(loc), -1, -1])


            # -----------------------
            # task nodes
            accepted_task_nodes = [[self.task_node_no, self.coordsToGrid(i[1]), self.coordsToGrid(i[2]), i[0], -1, i[4]] for sublist in obs[2] for item in sublist if isinstance(item, list) and item for i in item]
            riding_task_nodes = [[self.task_node_no, self.coordsToGrid(i[1]), self.coordsToGrid(i[2]), i[0], i[0], i[4]] for sublist in obs[3] for item in sublist if isinstance(item, list) and item for i in item]
            common_task_nodes = [[self.task_node_no, self.coordsToGrid(i[0]), self.coordsToGrid(i[1]), -1, -1, i[3]] for sublist in obs[4] for item in sublist if isinstance(item, list) and item for i in item]
            noop_task_node = [[2, -1, -1, -1, -1]]
            

            # -----------------------
            # action nodes
            noop_action_node = [[3, -1, -1, -1, -1]]
            pick_action_nodes = [[self.action_node_no, tsk[1], tsk[2], 1, tsk[5]] for tsk in accepted_task_nodes]
            drop_action_nodes = [[self.action_node_no, tsk[1], tsk[2], 2, tsk[5]] for tsk in riding_task_nodes]
            common_action_nodes = [[self.action_node_no, tsk[1], tsk[2], 0, tsk[5]] for tsk in common_task_nodes]

            #     # # relative action nodes
            #     # pick_action_nodes = [[self.action_node_no, manhattan_distance(self_loc, self.gridToCoords(tsk[1])), direction(self_loc, self.gridToCoords(tsk[1])), 1, tsk[5]] for tsk in accepted_task_nodes]
            #     # drop_action_nodes = [[self.action_node_no, manhattan_distance(self_loc, self.gridToCoords(tsk[2])), direction(self_loc, self.gridToCoords(tsk[2])), 2, tsk[5]] for tsk in riding_task_nodes]
            #     # common_action_nodes = [[self.action_node_no, manhattan_distance(self_loc, self.gridToCoords(tsk[1])), direction(self_loc, self.gridToCoords(tsk[1])), 0, tsk[5]] for tsk in common_task_nodes]

            # redoing task nodes
            if relative:
                accepted_task_nodes = [[self.task_node_no, manhattan_distance(self_loc, i[1]), direction(self_loc, i[1]), i[0], -1] for sublist in obs[2] for item in sublist if isinstance(item, list) and item for i in item]
                riding_task_nodes = [[self.task_node_no, manhattan_distance(self_loc, i[2]), direction(self_loc, i[2]), i[0], i[0]] for sublist in obs[3] for item in sublist if isinstance(item, list) and item for i in item]
                common_task_nodes = [[self.task_node_no, manhattan_distance(self_loc, i[0]), direction(self_loc, i[0]), -1, -1] for sublist in obs[4] for item in sublist if isinstance(item, list) and item for i in item]
            else: 
                accepted_task_nodes = [[self.task_node_no, self.coordsToGrid(i[1]), self.coordsToGrid(i[2]), i[0], -1] for sublist in obs[2] for item in sublist if isinstance(item, list) and item for i in item]
                riding_task_nodes = [[self.task_node_no, self.coordsToGrid(i[1]), self.coordsToGrid(i[2]), i[0], i[0]] for sublist in obs[3] for item in sublist if isinstance(item, list) and item for i in item]
                common_task_nodes = [[self.task_node_no, self.coordsToGrid(i[0]), self.coordsToGrid(i[1]), -1, -1] for sublist in obs[4] for item in sublist if isinstance(item, list) and item for i in item]
            
            common_action_nodes = common_action_nodes + noop_action_node
            common_task_nodes = common_task_nodes + noop_task_node

            # -----------------------
            # edge nodes
            num_edge_nodes = len(accepted_task_nodes) + len(riding_task_nodes) + (self.num_agents * len(common_task_nodes)) #+ self.num_agents
            edge_nodes = []
            for _ in range(num_edge_nodes):
                node = [0] + [random.uniform(1e-5, 1e-4) for _ in range(4)]
                edge_nodes.append(node)
            #np.random.uniform(1e-5, 1e-4, size=(num_edge_nodes, 5)).tolist() 
            # edge_nodes = [[0, 0, 0, 0, 0]] * num_edge_nodes #np.zeros((num_edge_nodes, 5))

            # -----------------------
            # feature matrix
            feature_matrix = np.vstack(agent_nodes + accepted_task_nodes + riding_task_nodes + common_task_nodes + pick_action_nodes + drop_action_nodes + common_action_nodes + edge_nodes)

            # -----------------------
            # adjacency matrix
            node_names = list(map(lambda ls: 'ag'+str(ls), list(range(self.num_agents)))) +\
                list(map(lambda ls: 'st'+str(ls), list(range(len(accepted_task_nodes)+len(riding_task_nodes))))) + \
                    list(map(lambda ls: 'ct'+str(ls), list(range(len(common_task_nodes))))) + \
                    list(map(lambda ls: 'sa'+str(ls), list(range(len(accepted_task_nodes)+len(riding_task_nodes))))) + \
                        list(map(lambda ls: 'ca'+str(ls), list(range(len(common_action_nodes))))) + \
                            list(map(lambda ls: 'e'+str(ls), list(range(num_edge_nodes))))
                    
            adj_matrix = pd.DataFrame(np.zeros((len(node_names), len(node_names))), index=node_names, columns=node_names)

            e_index = 0
            self_edges = {}
            for tsk_idx in range(len(accepted_task_nodes + riding_task_nodes)):
                adj_matrix.loc['e'+str(e_index), 'ag'+str(idx)] = 1
                adj_matrix.loc['e'+str(e_index), 'st'+str(tsk_idx)] = 1
                adj_matrix.loc['e'+str(e_index), 'sa'+str(tsk_idx)] = 1
                self_edges[node_names.index('e'+str(e_index))] = [feature_matrix[node_names.index('ag'+str(idx))], \
                    feature_matrix[node_names.index('st'+str(tsk_idx))], \
                        feature_matrix[node_names.index('sa'+str(tsk_idx))]]
                e_index = e_index + 1

            for agent_i in range(self.num_agents):
                for tsk_idx in range(len(common_task_nodes)):
                    adj_matrix.loc['e'+str(e_index), 'ag'+str(agent_i)] = 1
                    adj_matrix.loc['e'+str(e_index), 'ct'+str(tsk_idx)] = 1
                    adj_matrix.loc['e'+str(e_index), 'ca'+str(tsk_idx)] = 1
                    if agent_i == idx:
                        self_edges[node_names.index('e'+str(e_index))] = [feature_matrix[node_names.index('ag'+str(idx))], \
                            feature_matrix[node_names.index('ct'+str(tsk_idx))], \
                                feature_matrix[node_names.index('ca'+str(tsk_idx))]]
                    e_index = e_index + 1
            
            
            # -----------------------
            # graph matrix
            # adj_matrix = adj_matrix + adj_matrix.T
            edge_index = np.where(adj_matrix.to_numpy() > 0)
            edge_index = torch.tensor(edge_index)
            edge_index = edge_index.flip(0)
            edge_index, _ = add_self_loops(edge_index)
            
            x = torch.from_numpy(feature_matrix).float()
            #x.requires_grad = True
            data = Data(x=x, edge_index=edge_index)

            graph_list.append(data)

            # -----------------------
            # action_space
            adj_matrix_reset = adj_matrix.reset_index()
            connected_e_nodes = adj_matrix_reset[adj_matrix_reset['index'].str.startswith('e') & (adj_matrix_reset['ag'+str(idx)] == 1)]['index'].tolist()
            connected_a_nodes = []

            for row_name in connected_e_nodes:
                # Select 'ca_' or 'sa_' columns where the value is 1 for the row of interest
                connected_to_row = adj_matrix.loc[row_name, adj_matrix.columns.str.startswith(('ca', 'sa'))]
                connected_to_row = connected_to_row[connected_to_row == 1].index.tolist()
                connected_a_nodes.append(connected_to_row[0])

            connected_e_nodes_indices = adj_matrix_reset[adj_matrix_reset['index'].str.startswith('e') & (adj_matrix_reset['ag'+str(idx)] == 1)].index.tolist()

            nodes = adj_matrix.columns.tolist()
            action_space = feature_matrix[[nodes.index(n) for n in nodes if n in connected_a_nodes]].tolist()

            edge_nodes_list.append(connected_e_nodes_indices)
            action_space_list.append(action_space)
            node_set_list.append(self_edges)

        return graph_list, edge_nodes_list, action_space_list, node_set_list


    def generateCriticGraph(self, state):
        '''
        Function to convert the entire state into an interaction graph
        '''
        out_graph_list = []

        '''
        State details
        layer 0 - all agents' locations (agent index - 0:num_agents-1)
        layer 1 - all accepted passengers (driver index, start, end, fare)
        layer 2 - all riding passenger (driver index, start, end, fare) 
        layer 3 - all unaccepted passengers (end, start, fare)
        '''

        # -----------------------
        # agent nodes
        agent_nodes = []
        agent_locs = []

        for o in range(self.num_agents):
            agent_locs.append(np.argwhere(np.vectorize(lambda x: o in x)(state[0]))[0])
            accepted_ct = np.sum(np.vectorize(lambda x: sum(sublist[0] == o for sublist in x if sublist), otypes=[int])(state[1]))
            riding_ct = np.sum(np.vectorize(lambda x: sum(sublist[0] == o for sublist in x if sublist), otypes=[int])(state[2]))
            agent_nodes.append([self.agent_node_no, o, self.coordsToGrid(agent_locs[o]), accepted_ct, riding_ct])
            
        # -----------------------
        # task nodes
        accepted_task_nodes = [[self.task_node_no, self.coordsToGrid(i[1]), self.coordsToGrid(i[2]), i[0], -1, i[4]] for sublist in state[1] for item in sublist if isinstance(item, list) and item for i in item]
        riding_task_nodes = [[self.task_node_no, self.coordsToGrid(i[1]), self.coordsToGrid(i[2]), i[0], i[0], i[4]] for sublist in state[2] for item in sublist if isinstance(item, list) and item for i in item]
        common_task_nodes = [[self.task_node_no, self.coordsToGrid(i[0]), self.coordsToGrid(i[1]), -1, -1, i[3]] for sublist in state[3] for item in sublist if isinstance(item, list) and item for i in item]
        noop_task_node = [[2, -1, -1, -1, -1]]
        

        # -----------------------
        # action nodes
        pick_action_nodes = [[self.action_node_no, tsk[1], tsk[2], 1, tsk[5]] for tsk in accepted_task_nodes]
        drop_action_nodes = [[self.action_node_no, tsk[1], tsk[2], 2, tsk[5]] for tsk in riding_task_nodes]
        common_action_nodes = [[self.action_node_no, tsk[1], tsk[2], 0, tsk[5]] for tsk in common_task_nodes]
        noop_action_node = [[3, -1, -1, -1, -1]]

        # redoing task nodes

        # if relative:
        #     accepted_task_nodes = [[self.task_node_no, manhattan_distance(self_loc, i[1]), direction(self_loc, i[1]), i[0], -1] for sublist in state[1] for item in sublist if isinstance(item, list) and item for i in item]
        #     riding_task_nodes = [[self.task_node_no, manhattan_distance(self_loc, i[2]), direction(self_loc, i[2]), i[0], i[0]] for sublist in state[2] for item in sublist if isinstance(item, list) and item for i in item]
        #     common_task_nodes = [[self.task_node_no, manhattan_distance(self_loc, i[0]), direction(self_loc, i[0]), -1, -1] for sublist in state[3] for item in sublist if isinstance(item, list) and item for i in item]
        # else: 
        #     accepted_task_nodes = [[self.task_node_no, self.coordsToGrid(i[1]), self.coordsToGrid(i[2]), i[0], -1] for sublist in state[1] for item in sublist if isinstance(item, list) and item for i in item]
        #     riding_task_nodes = [[self.task_node_no, self.coordsToGrid(i[1]), self.coordsToGrid(i[2]), i[0], i[0]] for sublist in state[2] for item in sublist if isinstance(item, list) and item for i in item]
        #     common_task_nodes = [[self.task_node_no, self.coordsToGrid(i[0]), self.coordsToGrid(i[1]), -1, -1] for sublist in state[3] for item in sublist if isinstance(item, list) and item for i in item]

        accepted_task_nodes = [[self.task_node_no, self.coordsToGrid(i[1]), self.coordsToGrid(i[2]), i[0], -1] for sublist in state[1] for item in sublist if isinstance(item, list) and item for i in item]
        riding_task_nodes = [[self.task_node_no, self.coordsToGrid(i[1]), self.coordsToGrid(i[2]), i[0], i[0]] for sublist in state[2] for item in sublist if isinstance(item, list) and item for i in item]
        common_task_nodes = [[self.task_node_no, self.coordsToGrid(i[0]), self.coordsToGrid(i[1]), -1, -1] for sublist in state[3] for item in sublist if isinstance(item, list) and item for i in item]
        
        common_action_nodes = common_action_nodes + noop_action_node
        common_task_nodes = common_task_nodes + noop_task_node

        # -----------------------
        # edge nodes
        num_edge_nodes = len(accepted_task_nodes) + len(riding_task_nodes) + (self.num_agents * len(common_task_nodes))
        edge_nodes = []
        for _ in range(num_edge_nodes):
            node = [0] + [random.uniform(1e-5, 1e-4) for _ in range(4)]
            edge_nodes.append(node)

        # -----------------------
        # adjacency matrix
        node_names = list(map(lambda ls: 'ag'+str(ls), list(range(self.num_agents))))
        feature_matrix = agent_nodes
        for idx in range(self.num_agents):
            num_tasks = len([i for i in accepted_task_nodes if i[3] == idx]) + len([i for i in riding_task_nodes if i[3] == idx])
            temp = list(map(lambda ls: 'st'+str(idx)+"_"+str(ls), list(range(num_tasks)))) + \
                        list(map(lambda ls: 'sa'+str(idx)+"_"+str(ls), list(range(num_tasks))))
            node_names.extend(temp)
            feature_matrix += [i for i in accepted_task_nodes if i[3] == idx] + [i for i in riding_task_nodes if i[3] == idx]
            feature_matrix += [pick_action_nodes[i] for i in range(len(accepted_task_nodes)) if accepted_task_nodes[i][3] == idx] + [drop_action_nodes[i] for i in range(len(riding_task_nodes)) if riding_task_nodes[i][3] == idx]
            

        temp = list(map(lambda ls: 'ct'+str(ls), list(range(len(common_task_nodes))))) + \
                    list(map(lambda ls: 'ca'+str(ls), list(range(len(common_action_nodes))))) + \
                        list(map(lambda ls: 'e'+str(ls), list(range(num_edge_nodes))))
        node_names.extend(temp)
        feature_matrix += common_task_nodes + common_action_nodes + edge_nodes
                            
        adj_matrix = pd.DataFrame(np.zeros((len(node_names), len(node_names))), index=node_names, columns=node_names)

        # adjacency matrix
        e_counter = 0
        for ag_idx in range(self.num_agents):
            t_nodes = [node for node in node_names if (node.startswith(f'st{ag_idx}_') or node.startswith(f'ct'))] 
            a_nodes = [node for node in node_names if (node.startswith(f'sa{ag_idx}_') or node.startswith(f'ca'))] 
            for ix in range(len(t_nodes)):
                set_edge(adj_matrix, 'e'+str(e_counter), 'ag'+str(ag_idx))
                set_edge(adj_matrix, 'e'+str(e_counter), t_nodes[ix])
                set_edge(adj_matrix, 'e'+str(e_counter), a_nodes[ix])
                e_counter += 1

        # -----------------------
        # feature matrix
        feature_matrix = np.vstack(feature_matrix)

        # -----------------------
        # graph matrix
        # adj_matrix = adj_matrix + adj_matrix.T
        edge_index = np.where(adj_matrix.to_numpy() > 0)
        edge_index = torch.tensor(edge_index)
        edge_index = edge_index.flip(0)
        edge_index, _ = add_self_loops(edge_index)
        
        x = torch.from_numpy(feature_matrix).float()
        data = Data(x=x, edge_index=edge_index)

        return data


    def  conflictManager(self, action_list, agents_with_overlap):
        '''
        Function to check if two or more agents have accepted the same passenger
        '''
        overlapping_action = action_list[agents_with_overlap[0]]
        pick_x, pick_y = self.gridToCoords(overlapping_action[1])
        dist = []
        for agent_no in agents_with_overlap:
            ag_x, ag_y = np.argwhere(np.vectorize(lambda x: agent_no in x)(self.state[0]))[0]
            dist.append(math.hypot(ag_x - pick_x, ag_y - pick_y))
        # assigned_agent = agents_with_overlap[dist.index(min(dist))]
        pairs = sorted(zip(dist, agents_with_overlap))

        # Extract the sorted agent indices
        sorted_agents = [agent for _, agent in pairs]
        return sorted_agents
            

    def heuristic(self, a, b):
        '''
        Function to calculate distance 
        '''
        return abs(b[0] - a[0]) + abs(b[1] - a[1])

    def cityMap(self, grid_len, grid_wid, start, goal):
        '''
        Function to compute the best route from a start point to the goal using A* search
        '''

        array = np.zeros((grid_len, grid_wid))

        neighbors = [(0,1),(0,-1),(1,0),(-1,0)]

        close_set = set()
        came_from = {}
        gscore = {start:0}
        fscore = {start:self.heuristic(start, goal)}
        oheap = []

        heapq.heappush(oheap, (fscore[start], start))
        
        while oheap:

            current = heapq.heappop(oheap)[1]

            if current == goal:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                return data, data[-1]

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                tentative_g_score = gscore[current] + 1
                if 0 <= neighbor[0] < array.shape[0]:
                    if 0 <= neighbor[1] < array.shape[1]:                
                        if array[neighbor[0]][neighbor[1]] == 1:
                            continue
                    else:
                        # array bound y walls
                        continue
                else:
                    # array bound x walls
                    continue
                    
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue
                    
                if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
                    
        return False, False


