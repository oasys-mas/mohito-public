#----------------------------------------------------------------------------------------------------------------------#
# Title: Replay Buffer
# Description: This file contains the script to create a replay buffer to train the network
# Author: Gayathri Anil 
# Version: 23.05.01
# Last updated on: 05-03-2023 
#----------------------------------------------------------------------------------------------------------------------#
 
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
    
    def add(self, state, obs, graph, edge_space, act_space, action, edge_value, reward, next_state, next_obs, next_graph, next_edge_space, next_act_space, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, obs, graph, edge_space, act_space, action, edge_value, reward, next_state, next_obs, next_graph, next_edge_space, next_act_space, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        states, obs, graph, edge_space, act_space, actions, edge_values, rewards, next_states, next_obs, next_graph, next_edge_space, next_act_space, dones = zip(*[self.buffer[i] for i in indices])
        
        return states, obs, graph, edge_space, act_space, torch.tensor(actions), edge_values, torch.tensor(rewards), next_states, next_obs, next_graph, next_edge_space, next_act_space, torch.tensor(dones)

    def sequentialSample(self, batch_size, sequence_len):
        num_indices = int(batch_size/sequence_len)
        start_indices = np.random.choice(len(self.buffer) - sequence_len, num_indices).tolist()
        indices = []
        for start_index in start_indices:
            indices.extend(list(range(start_index, start_index+sequence_len)))

        states, obs, graph, edge_space, act_space, actions, edge_values, rewards, next_states, next_obs, next_graph, next_edge_space, next_act_space, dones = zip(*[self.buffer[i] for i in indices])
        return states, obs, graph, edge_space, act_space, torch.tensor(actions), edge_values, torch.tensor(rewards), next_states, next_obs, next_graph, next_edge_space, next_act_space, torch.tensor


class Trajectories:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.trajectory = []
    
    def add(self, eps, step, state, obs, graph, edge_space, act_space, action, edge_value, reward, next_state, next_obs, next_graph, next_edge_space, next_act_space, done):
        if len(self.trajectory) >= self.capacity:
            self.trajectory.pop(0)
        self.trajectory.append((eps, step, state, obs, graph, edge_space, act_space, action, edge_value, reward, next_state, next_obs, next_graph, next_edge_space, next_act_space, done))