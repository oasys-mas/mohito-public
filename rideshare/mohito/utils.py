#----------------------------------------------------------------------------------------------------------------------#
# Title: Utilities File
# Description: This file contains supporting utilities for successful run of the code files in this repository
# Author: Gayathri Anil 
# Version: 23.05.01
# Last updated on: 05-03-2023 
#----------------------------------------------------------------------------------------------------------------------#


import numpy as np
import torch
import csv
import pickle
import os
from replay_buffer import ReplayBuffer
from copy import deepcopy
import random
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv

def configDump(config, results_file):
    '''
    This function dumps the configuration file to local
    '''
    results_loc = './results/' + str(results_file)
    # save config file as a csv
    with open(results_loc + '/config.csv', 'w') as f:
        writer = csv.writer(f)
        for key, value in config.items():
            writer.writerow([key, value])

    # save config file as a txt
    with open(results_loc + '/config.txt', 'w') as f:
        for key, value in config.items():
            f.write(f'{key}: {value}\n')
    


def modelDump(actor_list, critic_list, config, losses, results_file, loss_file, eps):
    '''
    This function dumps the model files to local
    '''
    results_loc = './results/' + str(results_file)
    
    # save losses as a pkl
    with open(loss_file, 'wb') as f:
        pickle.dump(losses, f)

    # save model weights
    if not os.path.isdir(results_loc+ '/model_files/' + str(eps)):
        os.makedirs(results_loc+ '/model_files/' + str(eps))
        
    for ag_idx in range(config["num_agents"]):
        torch.save(actor_list[ag_idx].main.state_dict(), results_loc+'/model_files/'+str(eps)+'/policy_agent'+str(ag_idx)+'.pth')
        torch.save(critic_list[ag_idx].main.state_dict(), results_loc+'/model_files/'+str(eps)+'/critic_agent'+str(ag_idx)+'.pth')
        torch.save(actor_list[ag_idx].target.state_dict(), results_loc+'/model_files/'+str(eps)+'/target_policy_agent'+str(ag_idx)+'.pth')
        torch.save(critic_list[ag_idx].target.state_dict(), results_loc+'/model_files/'+str(eps)+'/target_critic_agent'+str(ag_idx)+'.pth')


def gridToCoords(grid_cell, grid_width):
    '''
    Function to convert a grid cell number to coordinates
    '''
    x = grid_cell // grid_width
    y = grid_cell - (x * grid_width)
    return [x, y]
    

def bufferGenerator(env, num_episodes_replay, steps_per_episode, actor_list, exp_file, save = False):

    exp_buffer = ReplayBuffer()

    for eps in range(num_episodes_replay):

        print("Episode #", eps, "/", num_episodes_replay)

        # 1 - initial state
        current_state = env.reset()
        
        # 1.1 - observations for individual agents
        obs_list = env.getObsFromState(current_state)
        #1.2 - graphs for individual agents
        graph_list, edge_space_list, action_space_list, _ = env.generateGraph(obs_list)
        
        done = False
        eps_reward = [0] * env.num_agents
        action_list = env.action_list

        #reward_across_eps = []
        
        for step in range(steps_per_episode):

            print("Episode #", eps, "- Step #", step, "/", steps_per_episode)

            # 2 - get actions for all agents from actor networks
            action_list = []
            edge_value_list = []

            for ag_idx, graph in enumerate(graph_list):
                edge_value, selected_action = actor_list[ag_idx].getAction(deepcopy(graph), deepcopy(edge_space_list[ag_idx]), deepcopy(action_space_list[ag_idx]), network='main')
                action_list.append(selected_action)
                edge_value_list.append(edge_value)

            # confilct management
            temp_action_list = deepcopy(action_list)
            agents_with_overlap = [i for i in range(len(temp_action_list)) if temp_action_list[i] in temp_action_list[:i]+temp_action_list[i+1:] and temp_action_list[i][3] == 0]
            if len(agents_with_overlap) > 1:
                assigned_agent = env.conflictManager(temp_action_list, agents_with_overlap)
                # Assign Noop action for agent with overlapping accept action
                for agent_i in [x for x in agents_with_overlap if x != assigned_agent]:
                    temp_action_list[agent_i] = [-1, -1, -1, -1, -1] #Noop action

            # 3 - step function to  get next state and rewards
            next_state, reward_list = env.step(temp_action_list)

            # 4 - next set of observations
            next_obs_list = env.getObsFromState(next_state)

            # 5 - next set of graphs
            next_graph_list, next_edge_space_list, next_action_space_list, _ = env.generateGraph(next_obs_list)

            # 4 - [s, o, g, a, r, s_dash, o_dash, g_dash]
            exp_buffer.add(current_state, obs_list, graph_list, edge_space_list, action_space_list, action_list, edge_value_list, reward_list, next_state, next_obs_list, next_graph_list, next_edge_space_list, next_action_space_list, done)
            current_state, obs_list, graph_list, edge_space_list, action_space_list = next_state, next_obs_list, next_graph_list, next_edge_space_list, next_action_space_list

        if save and eps % save_buffer_every_eps == 0:
            with open(exp_file, 'wb') as f:
                pickle.dump(exp_buffer, f)
        
    return exp_buffer

def evalSimulator(eps_num, env, actor_list, steps_per_episode, simulated_eps, eval_file, eval_stats_file, device, num_passengers = None, openness = True, driver_locations = None):

    if openness:
        current_state = env.reset(step = 0, driver_locations = driver_locations, simulated_eps = simulated_eps)
    else:
        current_state = env.reset(step = 0, driver_locations = driver_locations, simulated_eps = simulated_eps)
    
    # 1.1 - observations for individual agents
    obs_list = env.getObsFromState(current_state)
    #1.2 - graphs for individual agents
    graph_list, edge_space_list, action_space_list, _ = env.generateGraph(obs_list)
    
    action_list = env.action_list

    
    for step in range(steps_per_episode):

        # print("Episode #", eps+1, "'- Step #'", step+1, "/", steps_per_episode)

        # 2 - get actions for all agents from actor networks
        action_list = []
        edge_value_list = []

        for ag_idx, graph in enumerate(graph_list):
            graph_device = deepcopy(graph).to(device)
            edge_space_device = torch.tensor(deepcopy(edge_space_list[ag_idx])).to(device)
            action_space_device = torch.tensor(deepcopy(action_space_list[ag_idx])).to(device)
            edge_value, selected_action = actor_list[ag_idx].getAction(graph_device, edge_space_device, action_space_device, network='main', training=False)
            action_list.append(selected_action)
            edge_value_list.append(edge_value)

        # confilct management
        temp_action_list = deepcopy(action_list)
        #agents_with_overlap = [i for i in range(len(temp_action_list)) if temp_action_list[i] in temp_action_list[:i]+temp_action_list[i+1:] and temp_action_list[i][3] == 0]
        agents_with_overlap = [i for i in range(len(temp_action_list)) if temp_action_list[i].tolist() in [x.tolist() for x in temp_action_list[:i]+temp_action_list[i+1:]] and temp_action_list[i][3] == 0]
        if len(agents_with_overlap) > 1:
            assigned_agent = env.conflictManager(temp_action_list, agents_with_overlap)
            # Assign Noop action for agent with overlapping accept action
            for agent_i in [x for x in agents_with_overlap if x != assigned_agent]:
                temp_action_list[agent_i] = [-1, -1, -1, -1, -1] #Noop action

        # todo: remove step from step function
        # 3 - step function to  get next state and rewards
        next_state, reward_list, stats = env.step(step = step, simulated_eps = simulated_eps, action_list = temp_action_list, openness=openness)
        append_stats_to_csv(file_path = eval_stats_file, eps = eps_num, step = step, stats = stats, total_steps = steps_per_episode)

        # 4 - next set of observations
        next_obs_list = env.getObsFromState(next_state)

        # 5 - next set of graphs
        next_graph_list, next_edge_space_list, next_action_space_list, _ = env.generateGraph(next_obs_list)

        # 4 - [s, o, g, a, r, s_dash, o_dash, g_dash]
        # trajectories.add(eps_num, step, current_state, obs_list, graph_list, edge_space_list, action_space_list, action_list, edge_value_list, reward_list, next_state, next_obs_list, next_graph_list, next_edge_space_list, next_action_space_list, done)
        current_state, obs_list, graph_list, edge_space_list, action_space_list = next_state, next_obs_list, next_graph_list, next_edge_space_list, next_action_space_list
        
        for ag_idx in range(env.num_agents):
            with open(eval_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([eps_num, step, ag_idx, reward_list[ag_idx], str(action_list[ag_idx])])

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")


def find_matching_edge_nodes(node_set1, node_set2):
    # Convert the lists of arrays into tuples to make them hashable
    # using frozenset because the order of features in the lists doesn't matter
    # and we want to be able to compare them as sets
    node_set1_tuples = {key: frozenset(tuple(map(tuple, value))) for key, value in node_set1.items()}
    node_set2_tuples = {key: frozenset(tuple(map(tuple, value))) for key, value in node_set2.items()}

    # Find the intersection of values between the two sets
    # creating a reverse mapping of the frozenset of features to the keys
    reverse_mapping1 = {}
    for key, value in node_set1_tuples.items():
        reverse_mapping1.setdefault(value, []).append(key)

    reverse_mapping2 = {}
    for key, value in node_set2_tuples.items():
        reverse_mapping2.setdefault(value, []).append(key)

    # Find the matching sets of features and get the corresponding keys from both dictionaries
    matching_keys = []
    for feature_set in reverse_mapping1.keys() & reverse_mapping2.keys():
        # using product to get all combinations of keys from both dictionaries
        # that share the same feature set
        for key1 in reverse_mapping1[feature_set]:
            for key2 in reverse_mapping2[feature_set]:
                matching_keys.append([key1, key2])

    return matching_keys

def convert_to_float(matched_str):
    if '.' not in matched_str:
        matched_str = matched_str.replace('e', '.0e')
    return float(matched_str)



def generate_task_schedule(steps_per_episode_train, num_new_passengers):
    cutoff_step = int(steps_per_episode_train * 0.9)
    task_distribution = {}
    remaining_tasks = num_new_passengers
    # next_task_step = 11 + random.randint(-3, 3)
    next_task_step = 5 + random.randint(-3, 3)
    increment = int(next_task_step * 2/3)  # Starting interval between task additions.
    interval_growth_rate = 1.01 #1.1  # Growth rate of the interval.

    while remaining_tasks > 0 and next_task_step < cutoff_step:
        # Decide number of tasks to add based on the remaining proportion of the episode.
        max_tasks = min(3, remaining_tasks)  # Ensure we do not schedule more tasks than remaining.
        probs = [0.7 - next_task_step / cutoff_step, 0.2, 0.1 + next_task_step / cutoff_step]

        # Ensure all probabilities are non-negative
        probs = [max(p, 0) for p in probs[:max_tasks]]

        # Normalize probabilities if the sum is greater than zero
        probs_sum = sum(probs)
        if probs_sum > 0:
            probs = [p / probs_sum for p in probs]
        else:
            # If all probabilities are zero, distribute them evenly across the possible tasks
            probs = [1/max_tasks] * max_tasks

        num_tasks_to_add = np.random.choice(range(1, max_tasks + 1), p=probs)

        # Add tasks to the schedule.
        task_distribution[int(next_task_step)] = num_tasks_to_add
        remaining_tasks -= num_tasks_to_add

        # Calculate next addition step.
        next_task_step += (increment + random.randint(-1, 5))
        increment = min(increment * interval_growth_rate, cutoff_step - next_task_step) 

    return task_distribution

# Function to plot time series for a specific episode
def plot_time_series_for_episode(data, episode):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for agent in data['agent_index'].unique():
        episode_data = data[(data['eps'] == episode) & (data['agent_index'] == agent)]
        ax.plot(episode_data['step'], episode_data['reward'], label=f'Agent {agent}')
    
    ax.set_title(f'Time Series of Rewards for Episode {episode}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.legend()
    plt.show()

# Function to plot the average reward per step for each agent across all episodes
def plot_average_rewards_per_step(data):
    fig, ax = plt.subplots(figsize=(10, 5))

    for agent in data['agent_index'].unique():
        avg_rewards = data[data['agent_index'] == agent].groupby('step')['reward'].mean()
        ax.plot(avg_rewards.index, avg_rewards, label=f'Agent {agent}')
    
    ax.set_title('Average Reward per Step for Each Agent Across All Episodes')
    ax.set_xlabel('Step')
    ax.set_ylabel('Average Reward')
    ax.legend()
    plt.show()

def reinitialize_last_layer(model, input_dim, hidden_dim, heads=1):
    # Reinitialize the last GATConv layer
    model.convs[-1] = GATConv(hidden_dim * heads, input_dim, heads=heads)
    # You can also reinitialize the dropout layer if needed
    model.dropouts[-1] = torch.nn.Dropout(0.5)

# This function will calculate the steps per episode for a given episode number
def get_steps_per_episode(episode, initial_steps, min_steps, decay_rate):
    steps = initial_steps - episode * decay_rate
    return max(steps, min_steps)

def append_stats_to_csv(file_path, eps, step, stats, total_steps):
    # Define the column names
    fieldnames = ['eps', 'step', 'num_accepted', 'num_riding', 'num_new', 'accepted_wait_time', 'completed_wait_time', 'num_completed', 'tasks_added', 'total_steps']
    
    completed_mean = np.mean(stats[3]) if len(stats[3])>0 else 0
    # Open the file in append mode, create a new file if it doesn't exist
    with open(file_path, 'a', newline='') as csvfile:
        # Create a DictWriter object
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # If the file is empty, write the header
        csvfile.seek(0, 2)  # Move to the end of the file
        if csvfile.tell() == 0:  # Check if the file is empty
            writer.writeheader()  # Write the header
        
        # Write the stats as a new row
        writer.writerow({
            'eps': eps,
            'step': step,
            'num_accepted': stats[0],
            'num_riding': stats[1],
            'num_new': stats[2],
            'accepted_wait_time': str(stats[3]),
            'completed_wait_time': str(stats[4]),
            'num_completed': len(stats[4]),
            'tasks_added': stats[5],  
            'total_steps': total_steps
        })

def calculate_tasks_for_epoch(epoch, max_epoch, task_range):
    """
    Calculate the number of tasks for the given epoch.

    :param epoch: The current epoch.
    :param max_epoch: The maximum number of epochs.
    :param task_range: A tuple or list with two elements, the min and max number of tasks.
    :return: The number of tasks for the current epoch.
    """
    min_tasks, max_tasks = task_range
    # Calculate the proportion of the way through the epochs
    progress = epoch / max_epoch
    # Linearly interpolate the number of tasks based on the progress
    num_tasks = min_tasks + progress * (max_tasks - min_tasks)
    # Return the number of tasks, rounded to the nearest integer
    return int(round(num_tasks))


def manhattan_distance(loc1, loc2):
    # loc1 = self
    # loc2 = passenger

    # Calculate Manhattan distance
    distance = abs(loc2[0] - loc1[0]) + abs(loc2[1] - loc1[1])

    return(distance)

def direction(loc1, loc2):
    # loc1 = self
    # loc2 = passenger

    # Determine direction: passenger - self
    dx = loc2[0] - loc1[0]
    dy = loc2[1] - loc1[1]

    # directions = [same point, N, NE, E, SE, S, SW, W, NW]
    if dx == 0 and dy == 0:
        direction = 0
    if dx == 0 and dy > 0:
        direction = 1
    elif dx > 0 and dy > 0:
        direction = 2
    elif dx > 0 and dy == 0:
        direction = 3
    elif dx > 0 and dy < 0:
        direction = 4
    elif dx == 0 and dy < 0:
        direction = 5
    elif dx < 0 and dy < 0:
        direction = 6
    elif dx < 0 and dy == 0:
        direction = 7
    elif dx < 0 and dy > 0:
        direction = 8

    return direction

def calculate_vector(x1, y1, x2, y2):
    """Calculate vector from (x1, y1) to (x2, y2) in a grid with (0,0) at top left."""
    return np.array([x2 - x1, y1 - y2])  # Notice the y1 - y2 for the grid orientation

def average_vector(vectors):
    """Calculate the average of a list of vectors."""
    return np.mean(vectors, axis=0)

def angular_distance(v1, v2):
    """Calculate the angular distance between two vectors."""
    if np.linalg.norm(v1) == 0:
        unit_v1 = v1
    else:
        unit_v1 = v1 / np.linalg.norm(v1)

    if np.linalg.norm(v2) == 0:
        unit_v2 = v2
    else:
        unit_v2 = v2 / np.linalg.norm(v2)

    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return angle  # Return the angle in radians

def add_row_to_csv(file_path, row, headers=None):
    # Check if the file exists and if it is empty (which implies a new file or a header is needed)
    file_exists = os.path.isfile(file_path)
    
    # Open the file in append mode, create it if it doesn't exist
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # If the file did not exist or was empty, and headers are provided, write the headers first
        if not file_exists and headers:
            writer.writerow(headers)
        
        # Write the row of data
        writer.writerow(row)

def retrieveEdgeWeights(adj_matrix):
    # Ensure adj_matrix is a NumPy array
    adj_matrix = np.array(adj_matrix)
    
    # Get the indices of non-zero entries in the adjacency matrix
    src, dst = adj_matrix.nonzero()
    
    # Create the edge_index tensor from these indices
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # Extract the weights of the edges from the adjacency matrix
    edge_weights = adj_matrix[src, dst]
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    
    return edge_index, edge_attr


# conflict manager
def executableActions(env, action_list):
    temp_action_list = deepcopy(action_list)
    agents_with_overlap = [i for i in range(len(temp_action_list)) if temp_action_list[i].tolist() in [x.tolist() for x in temp_action_list[:i]+temp_action_list[i+1:]] and temp_action_list[i][3] == 0]

    if len(agents_with_overlap) > 1:
        assigned_agent = env.conflictManager(temp_action_list, agents_with_overlap)
        # Assign Noop action for agent with overlapping accept action
        for agent_i in [x for x in agents_with_overlap if x != assigned_agent]:
            temp_action_list[agent_i] = [-1, -1, -1, -1, -1] #Noop action

    return temp_action_list


