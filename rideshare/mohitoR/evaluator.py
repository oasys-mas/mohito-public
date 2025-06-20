#----------------------------------------------------------------------------------------------------------------------#
# Title: Model Evaluator
# Description: This file contains the script to evaluate the model on a scripted fixed environment
# Author: Gayathri Anil
# Version: 23.05.01
# Last updated on: 05-03-2023
#----------------------------------------------------------------------------------------------------------------------#

import numpy as np
import pickle
import sys
from datetime import datetime
from rideshare.ride import *
from rideshare.simulated_ride import *
from rideshare.utils import *
from rideshare.replay_buffer import Trajectories
import csv
import subprocess
import re
import os
import matplotlib.pyplot as plt
import re

device = torch.device('cpu')
set_seed(50)
seed_value = 27
# seed_value_list = [27, 16, 40, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#---------------------------------------
# executor params
# simulation parameters
steps_per_episode = 100 #int(sys.argv[1])
result_file = 'cuda_06-18_100eps_100pereps_open_agents4_pass16_A20_C20_lrA0001_lrC001_train_per_eps5_grid10x10_GAT_small_normal_accept_limit_fixed_init_passengers'
# 2 - 'cuda_03-18_20000eps_100pereps_open_agents2_pass8_A20_C20_lrA0001_lrC001_train_per_eps5_grid10x10_move_cost-12_bad-2_pool2_bonus4_GAT_small_normal_Areg_Creg_trainer_init_lrs_heads'
# 3 - 'cuda_03-19_20000eps_100pereps_open_agents3_pass12_A20_C20_lrA0001_lrC001_train_per_eps5_grid10x10_move_cost-12_bad-2_pool2_bonus4_GAT_small_normal_Areg_Creg_trainer_init_lrs_heads'
# 4 - 'cuda_03-18_20000eps_100pereps_open_agents4_pass16_A20_C20_lrA0001_lrC001_train_per_eps5_grid10x10_move_cost-12_bad-2_pool2_bonus4_GAT_small_normal_Areg_Creg_trainer_init_lrs_heads'


# num_models = 1000 #int(sys.argv[4])
model_files = [0, 100, 'random', 'simple_time', 'simple_distance']
# ["simple_time", "simple_distance"] #"all_expert"] #, "simple_time", "simple_distance"]#"random", 100, 200] #["all_expert", "simple_time", "simple_distance"] #["all_expert", "simple_distance", "simple_time"] #list(range(0, 4101, 100)) #4 - 2000, 2 - 1600
openness_level_list = [1]
spread = True
attention = True #str(sys.argv[5]) == 'True'
num_agents = int((re.search(r'_agents(\d+)_', result_file)).group(1))
grid_len = int((re.search(r'_grid(\d+)x', result_file)).group(1))
grid_wid = int((re.search(r'_grid(\d+)x', result_file)).group(1))
move_cost = -1.2
action_labels = ["accept", "pick", "drop", "noop"]
save_csv = True #True
save_traj = False
# 4 best agents

#once you've found the bext agents you can avoid running all checkpoints by setting expert agents here.
# all_expert_agents = {j: [result_file, ]}
# all_expert_agents[0] = ['cuda_03-18_20000eps_100pereps_open_agents2_pass8_A20_C20_lrA0001_lrC001_train_per_eps5_grid10x10_move_cost-12_bad-2_pool2_bonus4_GAT_small_normal_Areg_Creg_trainer_init_lrs_heads', 1600, 1, 2]
# all_expert_agents[1] = ['cuda_03-18_20000eps_100pereps_open_agents4_pass16_A20_C20_lrA0001_lrC001_train_per_eps5_grid10x10_move_cost-12_bad-2_pool2_bonus4_GAT_small_normal_Areg_Creg_trainer_init_lrs_heads', 1500, 2, 2]
# all_expert_agents[1] = ['cuda_03-18_20000eps_100pereps_open_agents2_pass8_A20_C20_lrA0001_lrC001_train_per_eps5_grid10x10_move_cost-12_bad-2_pool2_bonus4_GAT_small_normal_Areg_Creg_trainer_init_lrs_heads', 1600, 1, 2]
# all_expert_agents[2] = ['cuda_03-18_20000eps_100pereps_open_agents2_pass8_A20_C20_lrA0001_lrC001_train_per_eps5_grid10x10_move_cost-12_bad-2_pool2_bonus4_GAT_small_normal_Areg_Creg_trainer_init_lrs_heads', 1600, 1, 2]
# all_expert_agents[3] = ['cuda_03-18_20000eps_100pereps_open_agents2_pass8_A20_C20_lrA0001_lrC001_train_per_eps5_grid10x10_move_cost-12_bad-2_pool2_bonus4_GAT_small_normal_Areg_Creg_trainer_init_lrs_heads', 1600, 1, 2]
# all_expert_agents[3] = ['cuda_03-18_20000eps_100pereps_open_agents4_pass16_A20_C20_lrA0001_lrC001_train_per_eps5_grid10x10_move_cost-12_bad-2_pool2_bonus4_GAT_small_normal_Areg_Creg_trainer_init_lrs_heads', 1500, 2, 2]
# all_expert_agents[1] = ['cuda_03-18_20000eps_100pereps_open_agents4_pass16_A20_C20_lrA0001_lrC001_train_per_eps5_grid10x10_move_cost-12_bad-2_pool2_bonus4_GAT_small_normal_Areg_Creg_trainer_init_lrs_heads', 1700, 1, 2]
# all_expert_agents[2] = ['cuda_03-18_20000eps_100pereps_open_agents4_pass16_A20_C20_lrA0001_lrC001_train_per_eps5_grid10x10_move_cost-12_bad-2_pool2_bonus4_GAT_small_normal_Areg_Creg_trainer_init_lrs_heads', 1500, 2, 2]
# all_expert_agents[3] = ['cuda_03-14_20000eps_25pereps_open_agents3_pass36_A20_C20_lrA1e-05_lrC00001_train_per_eps5_grid10x10_move_cost-12_bad-2_pool2_bonus4_GAT_small_normal_Areg_Creg', 4000, 0, 1]

print(attention)

if attention:
    from gat import *
else:
    from gcn import *

eval_file = './testing_episodes/' + "convergence-check-ag" + str(num_agents) + '_w_new_model0.csv'

# eval_file = './baseline-comparisons/' + "ag"+ str(num_agents)+"-ol"+str(openness_level) + '.csv'
eval_plot = './results/' + str(result_file) + '/final_' + str(result_file)

# if os.path.exists(eval_file):
#     os.remove(eval_file)


##['random'] + list(range(500, num_models+1, 500))
#model_file = str(sys.argv[4]) #'9500'

openness = True
passenger_limit = 1000 #int(sys.argv[3]) #15

for openness_level in openness_level_list:
    print("--- Openness Level:", openness_level, "---")

    if openness_level == 1:
        eps_file = './simulated_eps/' + 'simulated_episode_' + str(
            grid_len) + 'x' + str(
                grid_wid) + '_25steps_6pass.pkl'  #str(sys.argv[3]

    elif openness_level == 2:
        if spread:
            eps_file = './simulated_eps/' + 'simulated_episode_' + str(
                grid_len) + 'x' + str(
                    grid_wid) + '_25steps_9pass_spread.pkl'  #str(sys.argv[3])
        else:
            eps_file = './simulated_eps/' + 'simulated_episode_' + str(
                grid_len) + 'x' + str(
                    grid_wid) + '_25steps_9pass.pkl'  #str(sys.argv[3])

    elif openness_level == 3:
        if spread:
            eps_file = './simulated_eps/' + 'simulated_episode_' + str(
                grid_len) + 'x' + str(
                    grid_wid) + '_25steps_12pass_spread.pkl'  #str(sys.argv[3])
        else:
            eps_file = './simulated_eps/' + 'simulated_episode_' + str(
                grid_len) + 'x' + str(
                    grid_wid) + '_25steps_12pass.pkl'  #str(sys.argv[3])

    variable_move_cost = False  #bool(sys.argv[5])
    variable_pick_cost = False  #bool(sys.argv[6])
    no_pass_reward = True  #bool(sys.argv[7])

    accept_cost = 0
    pick_cost = -0.1
    miss_cost = -2
    drop_cost = None
    no_pass_cost = 0
    feature_len = 5
    beta = 0.001  #Exploration

    #---------------------------------------
    # extracting parameters from result file string
    lr_actor = convert_to_float((re.search(r'lrA([\d.e-]+)_',
                                           result_file)).group(1))
    lr_critic = convert_to_float((re.search(r'lrC([\d.e-]+)_',
                                            result_file)).group(1))
    reg_lambda = 0.1  # convert_to_float((re.search(r'reg([\d.e-]+)_', result_file)).group(1))
    num_layers_actor = int((re.search(r'_A(\d+)_', result_file)).group(1))
    num_layers_critic = int((re.search(r'_C(\d+)_', result_file)).group(1))
    grad_clip = 5.0  #convert_to_float((re.search(r'clip([\d.e-]+)_', result_file)).group(1))

    print(num_layers_actor, num_layers_critic)

    # initialise environment
    env = simulatedRideshare(num_agents=num_agents,
                             grid_len=grid_len,
                             grid_wid=grid_wid,
                             accept_cost=accept_cost,
                             pick_cost=pick_cost,
                             move_cost=move_cost,
                             noop_cost=miss_cost,
                             drop_cost=drop_cost,
                             no_pass_cost=no_pass_cost,
                             variable_move_cost=variable_move_cost,
                             variable_pick_cost=variable_pick_cost,
                             no_pass_reward=no_pass_reward)

    print("environment initialised")

    # policy networks
    actor_list = []
    # critic_list = []

    for ag_idx in range(num_agents):
        actor_list.append(
            ActorNetwork(num_state_features=feature_len,
                         LR_A=lr_actor,
                         BETA=beta,
                         num_layers=num_layers_actor))
        # critic_list.append(CriticNetwork(num_state_features=feature_len, num_agents = env.num_agents, LR_C = lr_critic, hidden_dim_critic=50, num_layers = num_layers_critic, grad_clip = grad_clip).to(device))

    print('actors and critics loaded')

    with open(eps_file, 'rb') as f:
        simulated_eps = pickle.load(f)


    if openness:
        all_sublists = [(key, sublist)
                        for key, sublists in simulated_eps.items()
                        for sublist in sublists]
        # sampled_sublists = random.sample(all_sublists, min(passenger_limit_eval, len(all_sublists)))
        simulated_eps_eval = {}
        simulated_eps = {}
        for key, sublist in all_sublists:

            # simulated episode for training
            if key not in simulated_eps:
                simulated_eps[key] = [sublist]
            else:
                simulated_eps[key].append(sublist)

            # simulated episode for evaluation
            if key <= steps_per_episode:
                if key not in simulated_eps_eval:
                    simulated_eps_eval[key] = [sublist]
                else:
                    simulated_eps_eval[key].append(sublist)
    else:
        simulated_eps_eval = {}
        simulated_eps_eval[0] = simulated_eps[0][:passenger_limit]

    print('simulated episodes loaded')

    sum_total = [[]]

    # sum_total = []*len(seed_value_list)
    seed_file = "ag" + str(num_agents) + "-ol" + str(openness_level) + '.pkl'
    with open('./testing_episodes/' + seed_file, 'rb') as f:
        seed_value_list = pickle.load(f)

    if num_agents == 3 and openness_level == 3:
        seed_value_list = seed_value_list + [169, 458, 128]
    # eval_file = './baseline-comparisons/' + "eval-ag"+ str(num_agents)+"-ol"+str(openness_level) + '.csv'

    for s_no, seed_value in enumerate(seed_value_list):

        if len(sum_total) == s_no:
            sum_total.append([])

        sum_total[s_no].append(seed_value)
        for model in model_files:

            set_seed(seed_value)
            print("-------- Model: ", model, "-------- ")
            policy_file = './results/' + str(
                result_file) + '/model_files/' + str(model)
            record_step = []

            if model == 'random':
                for ag_idx in range(num_agents):
                    actor_list[ag_idx].main.load_state_dict(
                        torch.load(policy_file + '/random_actor.pth',
                                   map_location=torch.device('cpu')))
                    actor_list[ag_idx].eval()

            elif model == "simple_time":
                print("First-come First-serve Policy")

            elif model == "simple_distance":
                print("Greedy Distance Policy")

            elif model == 'all_expert':
                actor_list = []
                for ag_idx in range(num_agents):
                    expert_file_name, expert_model_no, expert_agent_no, expert_head_ct = all_expert_agents[
                        ag_idx]
                    pre_trained_model_file = './results/' + expert_file_name + '/model_files/' + str(
                        expert_model_no)
                    actor_list.append(
                        ActorNetwork(num_state_features=feature_len,
                                     LR_A=lr_actor,
                                     BETA=beta,
                                     num_layers=num_layers_actor,
                                     heads=expert_head_ct))
                    actor_list[ag_idx].main.load_state_dict(
                        torch.load(pre_trained_model_file + '/policy_agent' +
                                   str(expert_agent_no) + '.pth',
                                   map_location=torch.device(device)))
                    actor_list[ag_idx].eval()

            else:
                actor_list = []
                # critic_list = []
                for ag_idx in range(num_agents):
                    actor_list.append(
                        ActorNetwork(num_state_features=feature_len,
                                     LR_A=lr_actor,
                                     BETA=beta,
                                     num_layers=num_layers_actor))
                    # critic_list.append(CriticNetwork(num_state_features=feature_len, num_agents = env.num_agents, LR_C = lr_critic, hidden_dim_critic=50, num_layers = num_layers_critic, grad_clip = grad_clip).to(device))
                    actor_list[ag_idx].main.load_state_dict(
                        torch.load(policy_file + '/policy_agent' +
                                   str(ag_idx) + '.pth',
                                   map_location=torch.device('cpu')))
                    # critic_list[ag_idx].main.load_state_dict(torch.load(policy_file + '/critic_agent' + str(ag_idx) + '.pth', map_location=torch.device(device)))
                    actor_list[ag_idx].eval()

            trajectories = Trajectories()

            if openness:
                current_state = env.reset(step=0, simulated_eps=simulated_eps)
            else:
                current_state = env.reset(step=0, simulated_eps=simulated_eps)
            next_done = False

            # 1.1 - observations for individual agents
            obs_list = env.getObsFromState(current_state)
            #1.2 - graphs for individual agents
            graph_list, edge_space_list, action_space_list, _ = env.generateGraph(
                obs_list)

            done = False
            eps_reward = [0] * env.num_agents
            action_list = env.action_list

            reward_across_eps_list = []
            for ag in range(num_agents):
                reward_across_eps_list.append([])

            for step in range(steps_per_episode):

                # print("Episode #", eps+1, "'- Step #'", step+1, "/", steps_per_episode)

                # 2 - get actions for all agents from actor networks
                action_list = []
                edge_value_list = []

                if model == "simple_time":
                    for ag_idx, graph in enumerate(graph_list):
                        wait_time = []

                        if len(action_space_list[ag_idx]) == 1:
                            selected_action = action_space_list[ag_idx][0]
                        else:
                            for task_ in action_space_list[ag_idx][:-1]:
                                wait_time.append(step - task_[-1])
                            selected_action = action_space_list[ag_idx][
                                wait_time.index(max(wait_time))]

                        action_list.append(selected_action)
                        edge_value_list.append(0)
                        # conflict management
                        temp_action_list = deepcopy(action_list)
                        agents_with_overlap = [
                            i for i in range(len(temp_action_list))
                            if temp_action_list[i] in [
                                x for x in temp_action_list[:i] +
                                temp_action_list[i + 1:]
                            ] and temp_action_list[i][3] == 0
                        ]

                elif model == "simple_distance":
                    for ag_idx, graph in enumerate(graph_list):
                        dist_list = []
                        loc_x, loc_y = gridToCoords(graph.x[ag_idx][2].item(),
                                                    grid_wid)

                        if len(action_space_list[ag_idx]) == 1:
                            selected_action = action_space_list[ag_idx][0]
                        else:
                            # ensuring one task at a time
                            filtered_action_space = [
                                sublist
                                for sublist in action_space_list[ag_idx][:-1]
                                if sublist[3] != 0
                            ]
                            if len(filtered_action_space) > 0:
                                temp_action_space = filtered_action_space
                            else:
                                temp_action_space = action_space_list[
                                    ag_idx][:-1]
                            for task_ in temp_action_space:
                                if task_[3] == 0:
                                    dist_list.append(
                                        manhattan_distance(
                                            (loc_x, loc_y),
                                            gridToCoords(task_[1], grid_wid)))
                                elif task_[3] == 1:
                                    dist_list.append(
                                        manhattan_distance(
                                            (loc_x, loc_y),
                                            gridToCoords(task_[1], grid_wid)))
                                elif task_[3] == 2:
                                    dist_list.append(
                                        manhattan_distance(
                                            (loc_x, loc_y),
                                            gridToCoords(task_[2], grid_wid)))
                            selected_action = action_space_list[ag_idx][
                                dist_list.index(min(dist_list))]

                        action_list.append(selected_action)
                        edge_value_list.append(0)
                        # conflict management
                        temp_action_list = deepcopy(action_list)
                        agents_with_overlap = [
                            i for i in range(len(temp_action_list))
                            if temp_action_list[i] in [
                                x for x in temp_action_list[:i] +
                                temp_action_list[i + 1:]
                            ] and temp_action_list[i][3] == 0
                        ]

                else:
                    for ag_idx, graph in enumerate(graph_list):
                        graph_device = deepcopy(graph).to(device)
                        edge_space_device = torch.tensor(
                            deepcopy(edge_space_list[ag_idx])).to(device)
                        action_space_device = torch.tensor(
                            deepcopy(action_space_list[ag_idx])).to(device)
                        edge_value, selected_action = actor_list[
                            ag_idx].getAction(graph_device,
                                              edge_space_device,
                                              action_space_device,
                                              network='main',
                                              training=False)
                        action_list.append(selected_action)
                        edge_value_list.append(edge_value)
                        add_row_to_csv(file_path='./action.csv',
                                       row=[
                                           str(action_space_list[ag_idx]),
                                           selected_action,
                                           edge_value.item()
                                       ],
                                       headers=[
                                           "action_space", "selected_action",
                                           "selected_value"
                                       ])
                        # conflict management
                        temp_action_list = deepcopy(action_list)
                        agents_with_overlap = [
                            i for i in range(len(temp_action_list))
                            if temp_action_list[i].tolist() in [
                                x.tolist() for x in temp_action_list[:i] +
                                temp_action_list[i + 1:]
                            ] and temp_action_list[i][3] == 0
                        ]

                if len(agents_with_overlap) > 1:
                    assigned_agent = "None"
                    sorted_agents = env.conflictManager(
                        temp_action_list, agents_with_overlap)
                    for k in sorted_agents:
                        count = sum(
                            1
                            for level in [current_state[1], current_state[2]]
                            for row in level for cell in row for c in cell
                            if c and c[0] == k)
                        if count == 3:
                            continue
                        else:
                            assigned_agent = k
                            break
                    if assigned_agent != "None":
                        # Assign Noop action for agent with overlapping accept action
                        for agent_i in [
                                x for x in agents_with_overlap
                                if x != assigned_agent
                        ]:
                            temp_action_list[agent_i] = [-1, -1, -1, -1,
                                                         -1]  #Noop action

                # 3 - step function to  get next state and rewards
                next_state, reward_list, _ = env.step(
                    step=step + 1,
                    simulated_eps=simulated_eps,
                    action_list=temp_action_list,
                    openness=openness)


                for ag_idx in range(num_agents):
                    reward_across_eps_list[ag_idx].append(reward_list[ag_idx])

                # 4 - next set of observations
                next_obs_list = env.getObsFromState(next_state)

                # 5 - next set of graphs
                next_graph_list, next_edge_space_list, next_action_space_list, _ = env.generateGraph(
                    next_obs_list)

                # 4 - [s, o, g, a, r, s_dash, o_dash, g_dash]
                trajectories.add(1, step, current_state, obs_list, graph_list,
                                 edge_space_list, action_space_list,
                                 action_list, edge_value_list, reward_list,
                                 next_state, next_obs_list, next_graph_list,
                                 next_edge_space_list, next_action_space_list,
                                 done)
                current_state, obs_list, graph_list, edge_space_list, action_space_list = next_state, next_obs_list, next_graph_list, next_edge_space_list, next_action_space_list

                num_new_passengers = sum(
                    len(sublist) for row in next_state[3] for sublist in row)
                overall_riding_pass = sum(
                    len(sublist) for row in next_state[2] for sublist in row)
                overall_accepted_pass = sum(
                    len(sublist) for row in next_state[1] for sublist in row)

                if next_done:
                    episode_status = "done"
                else:
                    episode_status = "progress"

                if num_new_passengers + overall_riding_pass + overall_accepted_pass == 0:
                    next_done = True

                for ag_idx in range(env.num_agents):

                    riding_pass = sum(
                        len(sublist) for row in next_obs_list[ag_idx][3]
                        for sublist in row)
                    accepted_pass = sum(
                        len(sublist) for row in next_obs_list[ag_idx][2]
                        for sublist in row)

                    # passenger count in cab
                    if riding_pass + accepted_pass > 1:
                        pooling_status = "pooling"
                    elif riding_pass + accepted_pass == 0:
                        pooling_status = "none"
                    else:
                        pooling_status = "single"

                    # ride status
                    if reward_list[ag_idx] > 3:
                        ride_status = "done"
                    else:
                        ride_status = "no"

                    if save_csv:
                        if not os.path.exists(eval_file):
                            with open(eval_file, 'w', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(["ag","ol","seed","type","step","agent","reward","action",
                                    "action-label","entry-step","num_new_passengers","num_accepted_passengers",
                                    "num_riding_passengers","pooling_status","ride_status","eps_status"])

                        with open(eval_file, 'a', newline='') as file:
                            writer = csv.writer(file)
                            if model == "simple_time" or "simple_distance":
                                writer.writerow([
                                    num_agents, openness_level, seed_value,
                                    model, step, ag_idx, reward_list[ag_idx],
                                    str(action_list[ag_idx]),
                                    action_labels[int(
                                        action_list[ag_idx][-2])],
                                    action_list[ag_idx][-1],
                                    num_new_passengers, accepted_pass,
                                    riding_pass, pooling_status, ride_status,
                                    episode_status
                                ])
                            else:
                                writer.writerow([
                                    num_agents, openness_level, seed_value,
                                    model, step, ag_idx, reward_list[ag_idx],
                                    str(action_list[ag_idx]),
                                    action_labels[int(
                                        action_list[ag_idx][-2])],
                                    action_list[ag_idx][-1],
                                    num_new_passengers, accepted_pass,
                                    riding_pass, pooling_status, ride_status,
                                    episode_status
                                ])

            if save_traj:
                with open('./results/' + str(result_file) + '/temp.pkl',
                          'wb') as f:
                    pickle.dump(trajectories, f)

            tot = 0
            print("-------- Model ", model, " - Total Rewards --------")
            for ag in range(num_agents):
                print("Agent #: ", ag, sum(reward_across_eps_list[ag]))
                tot += sum(reward_across_eps_list[ag])
                sum_total[s_no].append(sum(reward_across_eps_list[ag]))

            print("Total = ", tot)  #, " # steps = ", record_step[0]
            sum_total[s_no].append(tot)

    sum_total_df = pd.DataFrame(sum_total)
    # sum_total_df.to_csv("eval_ag"+ str(num_agents)+"-ol"+str(openness_level) +'.csv', index=False)
