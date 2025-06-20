#----------------------------------------------------------------------------------------------------------------------#
# Title: MADDPG Algorithm - Online Trainer Script
# Description: This file contains the script to online train the policy using MADDPG algorithm
# Author: Gayathri Anil 
# Version: 23.05.01
# Last updated on: 05-03-2023 
#----------------------------------------------------------------------------------------------------------------------#

import numpy as np
import gym
import pickle
import os
import subprocess
from datetime import datetime
import sys
from rideshare.ride import *
from rideshare.simulated_ride import *
from rideshare.utils import *
import random

seed_val = 16
set_seed(seed_val)

# ------------------------------------
# command line parameters
num_episodes = 100 #int(sys.argv[1])
num_agents = 4 #int(sys.argv[1])
lr_actor = 0.001 #(slower) learning rate for actor 
lr_critic = 0.01 # learning rate for critic
openness = True #str(sys.argv[3]) == 'True'
attention = True #str(sys.argv[6]) == 'True'
pool_limit = 2 
grid_len = 10
grid_wid = 10
move_cost = -1.2
expert_nudge = False #str(sys.argv[3]) == 'True'
sanity_check = "normal" #str(sys.argv[4]) #"normal", "greedy_nearest", "expert"
trajectory_type = "small" #str(sys.argv[5]) #"fixed", "fixed-limited", "small"
regularise_actor = True
regularise_critic = True
consistency = False
num_new_passengers = (num_agents * 4) #+ (num_agents - 1)
if num_agents == 1:
    num_new_passengers = 4

if trajectory_type == "fixed":
    sim_eps_file = './simulated_eps/simulated_episode_10x10_' + str('500steps') + '.pkl' #'500steps' '500steps_limited
elif trajectory_type == "fixed-limited":
    sim_eps_file = './simulated_eps/simulated_episode_10x10_' + str('500steps_limited') + '.pkl' #'500steps' '500steps_limited
elif trajectory_type == "small":
    sim_eps_file = './simulated_eps/simulated_episode_' + str(int(grid_len)) + 'x' + str(int(grid_wid)) + '_' + str('25steps_6pass') + '.pkl' #'500steps' '500steps_limited
    if sanity_check == "expert":
        expert_trajectory_file = './simulated_eps/expert_trajectory_25_steps.pkl'

print("Exp: ", num_episodes, num_agents, pool_limit, grid_len, grid_wid, move_cost)

relative = True #str(sys.argv[6]) == 'True'
max_wait_time = 20
unaccepted_wait_cost_variable = -2

# ------------------------------------
# Setup parameters
save_model_every_eps = 100
steps_per_episode_train = 100  #int(sys.argv[2]) #30 
min_steps_per_episode_train = 50
steps_per_episode_eval = 100 #int(sys.argv[2]) #30 
train_points_per_eps = 5
batch_size = int(steps_per_episode_train/train_points_per_eps) #18 #int(sys.argv[2]) #16 
variable_move_cost = False #bool(sys.argv[4]) 
variable_pick_cost = False #bool(sys.argv[5])  
no_pass_reward = True #bool(sys.argv[6]) 
variable_reg = False
reg_range = [0.004, 0.002]
wait_limit = [5, 10, 10]
exp_loss = True

# parameters for expert nudging
if expert_nudge:
    if grid_len == 10:
        file = 'cuda_03-14_20000eps_25pereps_open_agents3_pass36_A20_C20_lrA1e-05_lrC00001_train_per_eps5_grid10x10_move_cost-12_bad-2_pool2_bonus4_GAT_small_normal_Areg_Creg' #str(sys.argv[6])
        model_no = 4000
        exp_ag = 0
        head_ct = 1
        epsilon_limit = 0.1
    elif grid_len == 5:
        file = 'cuda_03-21_10000eps_100pereps_open_agents3_pass17_A20_C20_lrA0001_lrC001_train_per_eps5_grid5x5_move_cost-12_bad-2_pool2_bonus4_GAT_small_normal_Areg_Creg_trainer_init_lrs_heads_expnudge'
        model_no = 1400
        exp_ag = 1
        head_ct = 2
        epsilon_limit = 0.1
    pre_trained_model_file = './results/' + file + '/model_files/' + str(model_no)
else:
    pre_trained_model_file = None
    epsilon_limit = 0

epsilon_greedy = True

if expert_nudge:
    epsilon_range = [0.8, 0.05]
    epsilon_decay = 0.00035
else:
    epsilon_range = [0.9, 0.05]
    epsilon_decay = 0.00035

epsilon_fn = lambda epoch: min(epsilon_range) + (max(epsilon_range) - min(epsilon_range)) * math.exp(-epsilon_decay * epoch)
evaluate_model_every_eps = 25

# step reduction 
step_reduction = list(range(steps_per_episode_train, min_steps_per_episode_train-1, -batch_size))
num_decrement = max(int(num_episodes/len(step_reduction)), 1)
episode_ranges = list(range(0, int(num_episodes*3/4), num_decrement))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

if attention:
    from gat import *
else:
    from gcn import *

# ------------------------------------
# Environment parameters
accept_cost = 0
pick_cost = -0.1
move_cost = move_cost
miss_cost = -2
drop_cost = None
no_pass_cost = 0
feature_len = 5

# pool_limit = 2
pool_limit_cost = -2

# ------------------------------------
# Training parameters
beta = 0.001 #Exploration 
reg_lambda_actor = 0.1 #float(sys.argv[3])
consistency_lambda = 0.01 #float(sys.argv[4])
reg_lambda_critic = 0.01#float(sys.argv[5])
grad_clip = 5.0 #float(sys.argv[4])
num_layers_actor = 20
num_layers_critic = 20
gamma = 0.9
soft_update_eps = 20
#soft_update_tau = 0.001
critic_variable_tau = False

soft_update_tau_critic = 0.005 #float(sys.argv[5])#0.005
soft_update_tau_critic_range = [0.1, 0.01]

soft_update_tau_actor = 0.005 #float(sys.argv[6]) #0.005

# ------------------------------------
# Task generator logic
initial_tasks_count = random.randint(num_agents-1, num_agents+3)

# driver locations for eval
driver_locations_eval = [(random.randint(0, grid_len-1), random.randint(0, grid_wid-1)) for _ in range(num_agents)]

# ------------------------------------
# Result file naming

if openness == False:
    result_file = str('cuda_' + datetime.now().strftime("%m-%d")) + '_' + str(num_episodes) + 'eps_' + str(steps_per_episode_train) + 'pereps_closed'
else:
    result_file = str('cuda_' + datetime.now().strftime("%m-%d")) + '_' + str(num_episodes) + 'eps_' + str(steps_per_episode_train) + 'pereps_open'

result_file = result_file + '_agents' + str(num_agents) + '_pass' + str(num_new_passengers)
result_file = result_file + '_A' + str(num_layers_actor) + '_C' + str(num_layers_critic) + '_lrA' + str(lr_actor).replace('.','') + '_lrC' + str(lr_critic).replace('.','') + '_train_per_eps' + str(train_points_per_eps)
result_file = result_file + '_grid' + str(grid_len) +  'x' + str(grid_wid)

if attention:
    result_file = result_file + "_GAT"
else: 
    result_file = result_file + "_GCN"

result_file = result_file + "_" + trajectory_type + "_" + sanity_check 

result_file = result_file + "_accept_limit_fixed_init_passengers"
# ------------------------------------

# ------------------------------------
# Other files

if not os.path.isdir('./results/'+ str(result_file)):
    os.makedirs('./results/'+ str(result_file))

loss_file = './results/' + str(result_file) + '/losses_'+ str(result_file) + '.pkl'
eval_file = './results/' + str(result_file) + '/eval_file_' + str(result_file) + '.csv'
stats_file = './results/' + str(result_file) + '/stats_file_' + str(result_file) + '.csv'
eval_stats_file = './results/' + str(result_file) + '/eval_stats_file_' + str(result_file) + '.csv'



# ------------------------------------
# Config file
config = {
    'num_episodes' : num_episodes,
    'steps_per_episode_train' : steps_per_episode_train, 
    'online': "True",
    'train_points_per_eps': train_points_per_eps, 
    'batch_size' : batch_size, 
    'variable_move_cost': variable_move_cost, 
    'variable_pick_cost': variable_pick_cost,
    'no_pass_reward': no_pass_reward, 
    'beta': beta,
    'lr_actor': lr_actor,
    'lr_critic': lr_critic,
    'regularise_actor': regularise_actor, 
    'regularise_critic': regularise_critic, 
    'reg_lambda_actor': reg_lambda_actor,
    'reg_lambda_critic': reg_lambda_critic,
    'consistency_lambda': consistency_lambda,
    'attention': attention,
    'num_layers_actor': num_layers_actor,
    'num_layers_critic': num_layers_critic,
    'gamma' : gamma, 
    'soft_update_eps' : soft_update_eps,
    'soft_update_tau_actor': soft_update_tau_actor,
    'soft_update_tau_critic': soft_update_tau_critic,
    'soft_update_tau_critic_range': str(soft_update_tau_critic_range),
    'num_agents' : num_agents,
    'grid_len' : grid_len,
    'grid_wid' : grid_wid,
    'accept_cost' : accept_cost,
    'pick_cost' : pick_cost,
    'move_cost' : move_cost,
    'miss_cost' : miss_cost,
    'feature_len' : feature_len,
    'openness': openness,
    'num_new_passengers': num_new_passengers,
    'variable_reg': variable_reg,
    'reg_range': str(reg_range),
    'epsilon_greedy': epsilon_greedy,
    'epsilon_range': str(epsilon_range),
    'pre_trained_model_file': str(pre_trained_model_file)
}
configDump(config, result_file)

print("---------")
print("Parameters")
print(config)


# ------------------------------------
# Environment 
# Register custom environment
#gym.register(id='ride-v0', entry_point='ride:ride')

# Create an instance of the environment
#env = gym.make('ride-v0')

env = rideshare(num_agents=num_agents, grid_len=grid_len, grid_wid=grid_wid, accept_cost=accept_cost, pick_cost=pick_cost, move_cost=move_cost, noop_cost=miss_cost, drop_cost=drop_cost, no_pass_cost=no_pass_cost, variable_move_cost=variable_move_cost, variable_pick_cost=variable_pick_cost, no_pass_reward=no_pass_reward, wait_limit = wait_limit, pool_limit = pool_limit, pool_limit_cost = pool_limit_cost)
simulated_env = simulatedRideshare(num_agents=num_agents, grid_len=grid_len, grid_wid=grid_wid, accept_cost=accept_cost, pick_cost=pick_cost, move_cost=move_cost, noop_cost=miss_cost, drop_cost=drop_cost, no_pass_cost=no_pass_cost, variable_move_cost=variable_move_cost, variable_pick_cost=variable_pick_cost, no_pass_reward=no_pass_reward, wait_limit = wait_limit, pool_limit = pool_limit, pool_limit_cost = pool_limit_cost)
print("Environment Intialised")

# ------------------------------------
# Simulated episode for evaluation

with open(sim_eps_file, 'rb') as f:
    simulated_eps = pickle.load(f)

if sanity_check == "expert":
    with open(expert_trajectory_file, 'rb') as f:
        expert_trajectory = pickle.load(f)

simulated_eps_eval = simulated_eps

# ------------------------------------
# Actor-Critic Networks 

actor_list = []
actor_loss_list = []
total_actor_loss_list = []
reg_list = []
consistency_list = []
previous_actor_param_list = []

critic_list = []
actual_critic_loss_list = []
target_critic_loss_list = []
total_target_critic_loss_list = []
rewards_critic_loss_list = []
critic_loss_list = []     
critic_reg_list = []  
total_critic_list = []  
previous_critic_param_list = []

 
for ag in range(num_agents):
    # set_seed(ag)
    critic_list.append(CriticNetwork(num_state_features=feature_len, num_agents = env.num_agents, LR_C = lr_critic, hidden_dim_critic=50, num_layers = num_layers_critic, grad_clip = grad_clip).to(device))
    actor_list.append(ActorNetwork(num_state_features=feature_len, LR_A=lr_actor, BETA=beta, num_layers = num_layers_actor).to(device))
    previous_actor_param_list.append({name: p.clone().detach() for name, p in actor_list[ag].main.named_parameters()})
    previous_critic_param_list.append({name: p.clone().detach() for name, p in critic_list[ag].main.named_parameters()})

    critic_loss_list.append([])
    critic_reg_list.append([])
    total_critic_list.append([])
    actual_critic_loss_list.append([])
    target_critic_loss_list.append([])
    total_target_critic_loss_list.append([])
    rewards_critic_loss_list.append([])
    actor_loss_list.append([])
    total_actor_loss_list.append([])
    reg_list.append([])
    consistency_list.append([])

print("Actor and Critic Networks Initialised")

# ------------------------------------
# Random policy for random action selection

set_seed(23)
random_actor = ActorNetwork(num_state_features=feature_len, LR_A=lr_actor, BETA=beta, num_layers = num_layers_actor).to(device)
# saving random actor
if not os.path.isdir('./results/' + result_file + '/model_files/random'):
    os.makedirs('./results/' + result_file + '/model_files/random')
torch.save(random_actor.main.state_dict(), './results/' + result_file +'/model_files/random/random_actor.pth')
    
# ------------------------------------
# Expert actor for expert nudge

if expert_nudge:
    # expert policy for expert nudge
    expert_actor = ActorNetwork(num_state_features=feature_len, LR_A=lr_actor, BETA=beta, num_layers = num_layers_actor, heads = head_ct).to(device)
    # expert_critic = CriticNetwork(num_state_features=feature_len, num_agents = num_agents, LR_C = lr_critic, hidden_dim_critic=50, num_layers = num_layers_critic, grad_clip = grad_clip).to(device)
    expert_actor.main.load_state_dict(torch.load(pre_trained_model_file + '/policy_agent' + str(exp_ag) + '.pth', map_location=torch.device(device)))
    expert_actor.target.load_state_dict(torch.load(pre_trained_model_file + '/target_policy_agent' + str(exp_ag) + '.pth', map_location=torch.device(device)))
    # expert_critic.main.load_state_dict(torch.load(pre_trained_model_file + '/critic_agent' + str(exp_ag) + '.pth', map_location=torch.device(device)))
    # expert_critic.target.load_state_dict(torch.load(pre_trained_model_file + '/target_critic_agent' + str(exp_ag) + '.pth', map_location=torch.device(device)))


# ------------------------------------
# ####################################
# ------------------------------------
# Training

epoch, episode_no = 0, 0
epsilon = epsilon_fn(epoch)

while epoch <= num_episodes:

    # episode counter
    episode_no += 1

    # initial state
    if openness: 
        # print("epoch no: ", epoch)
        # if epoch == 0: #<200
        #     seed_val = 16
        #     set_seed(seed_val)
        #     # num_new_passengers = num_agents * 3
        # elif epoch % 100 == 0:
        #     seed_val = random.randint(0,100)
        #     set_seed(seed_val)
        #     # num_new_passengers = num_agents * 4
        # else:
        #     set_seed(seed_val)
        #     # num_new_passengers = num_agents * 4
        
        seed_val = 16
        set_seed(seed_val)
        current_state = env.reset(step = 0, num_passengers=initial_tasks_count, epoch = epoch)
        # set_seed(seed_val)
        # new_seed_val = random.choice([4, 44, 444])
        # set_seed(new_seed_val)
        task_schedule = generate_task_schedule(steps_per_episode_train-10, num_new_passengers-initial_tasks_count)
        # print("hi")
    else:
        current_state = env.reset(step = 0, num_passengers=num_new_passengers, epoch = epoch)


    # observations for individual agents
    obs_list = env.getObsFromState(current_state)
    
    # state graph for critic network
    critic_graph = env.generateCriticGraph(current_state)

    # graphs for individual agents
    graph_list, edge_space_list, action_space_list, node_set_list = env.generateGraph(obs_list)

    done = False
    eps_reward = [0] * env.num_agents
    action_list = env.action_list

    print ("Epoch #", epoch+1, "/", num_episodes, " - ", str(steps_per_episode_train), "steps", " - ", result_file)

    for step in range(steps_per_episode_train):

        if step % batch_size == 0:
            graphs = []
            critic_graphs = []
            edge_space = []
            act_space = []
            actions = []
            action_vals = []
            rewards = []
            next_graphs = []
            next_critic_graphs = []
            next_edge_space = []
            next_act_space = []
            consistency_term = [[] for _ in range(env.num_agents)]

        # 2 - get actions for all agents from actor networks - Epsilon Greedy
        set_seed(step)
        p = random.uniform(0, 1)
        edge_value_list = []
        action_list = []


        # --------------
        # Action selection

        if sanity_check == "normal":

            for ag_idx, graph in enumerate(graph_list):
                epsilon_val = random.uniform(0, 1)
                
                if epsilon_val < epsilon:
                    # random policy based action
                    graph_device = deepcopy(graph).to(device)
                    edge_space_device = torch.tensor(deepcopy(edge_space_list[ag_idx])).to(device)
                    action_space_device = torch.tensor(deepcopy(action_space_list[ag_idx])).to(device)
                    edge_value, selected_action = random_actor.getAction(graph_device, edge_space_device, action_space_device, network='main')
                    action_list.append(selected_action)
                    edge_value_list.append(edge_value)
                    policy_type = "random"

                elif expert_nudge and epsilon_val < epsilon + epsilon_limit:
                    # expert policy 
                    graph_device = deepcopy(graph).to(device)
                    edge_space_device = torch.tensor(deepcopy(edge_space_list[ag_idx])).to(device)
                    action_space_device = torch.tensor(deepcopy(action_space_list[ag_idx])).to(device)
                    edge_value, selected_action = expert_actor.getAction(graph_device, edge_space_device, action_space_device, network='main')
                    action_list.append(selected_action)
                    edge_value_list.append(edge_value)
                    policy_type = "expert"

                else:
                    # learned policy based action
                    graph_device = deepcopy(graph).to(device)
                    edge_space_device = torch.tensor(deepcopy(edge_space_list[ag_idx])).to(device)
                    action_space_device = torch.tensor(deepcopy(action_space_list[ag_idx])).to(device)
                    actor_list[ag_idx].train()
                    edge_value, selected_action = actor_list[ag_idx].getAction(graph_device, edge_space_device, action_space_device, network='main')
                    action_list.append(selected_action)
                    edge_value_list.append(edge_value) 
                    policy_type = "actual"

                # print("Step:", step, "Ag #", ag_idx, "Pol:", policy_type, "Action:", selected_action)
        # policy - expert        
        elif sanity_check == "expert":
            # call expert
            for ag_idx, graph in enumerate(graph_list):
                selected_action = expert_trajectory[ag_idx][step]
                edge_to_be_returned = edge_space_list[ag_idx][action_space_list[ag_idx].index(selected_action)]

                graph_device = deepcopy(graph).to(device)
                edge_space_device = torch.tensor(deepcopy([edge_to_be_returned])).to(device)
                action_space_device = torch.tensor(deepcopy([selected_action])).to(device)
                edge_value, selected_action = random_actor.getAction(graph_device, edge_space_device, action_space_device, network='main')
                action_list.append(selected_action)
                edge_value_list.append(edge_value)

        # conflict management - old
        temp_action_list = deepcopy(action_list)
        agents_with_overlap = [i for i in range(len(temp_action_list)) if temp_action_list[i].tolist() in [x.tolist() for x in temp_action_list[:i]+temp_action_list[i+1:]] and temp_action_list[i][3] == 0]
    
        if len(agents_with_overlap) > 1:
            assigned_agent = env.conflictManager(temp_action_list, agents_with_overlap)
            # Assign Noop action for agent with overlapping accept action
            for agent_i in [x for x in agents_with_overlap if x != assigned_agent]:
                temp_action_list[agent_i] = [-1, -1, -1, -1, -1] #Noop action


        # exo event generation
        # external exo_var generation to regulate the degree of openness
        # if exo_var = None, the domain will generate new tasks randomly 
        if openness and (step in task_schedule):
            exo_var = True
            num_tasks = task_schedule[step] 
            # handling number of unserviced tasks in the environment
            num_unaccepted_passengers = np.sum(np.vectorize(lambda x: sum(isinstance(i, list) for i in x), otypes=[int])(current_state[3]))
            if epoch < 5000 and (num_unaccepted_passengers > (num_agents * pool_limit)):
                num_tasks = 0
            elif epoch >= 5000 and epoch < 8000 and (num_unaccepted_passengers > (num_agents * (pool_limit + 0.5))):
                num_tasks = 0
            elif num_unaccepted_passengers > (num_agents * (pool_limit + 1)) :
                num_tasks = 0
        else:
            exo_var, num_tasks = False, None
        
        # 3 - step function to  get next state and rewards
        next_state, reward_list, stats = env.step(step = step+1, action_list = temp_action_list, openness=openness, exo_var=exo_var, num_tasks=num_tasks)
        
        append_stats_to_csv(file_path = stats_file, eps = episode_no, step = step, stats = stats, total_steps = steps_per_episode_train)
        # 4 - next set of observations
        next_obs_list = env.getObsFromState(next_state)

        # generate a state graph for the criitc network
        next_critic_graph = env.generateCriticGraph(next_state)

        # 5 - next set of graphs
        next_graph_list, next_edge_space_list, next_action_space_list, next_node_set_list = env.generateGraph(next_obs_list)

        #_, _, graphs, edge_space, act_space, actions, action_vals, rewards, _, _, next_graphs, next_edge_space, next_act_space, done
        graphs.append(graph_list)
        critic_graphs.append(critic_graph)
        edge_space.append(edge_space_list)
        act_space.append(action_space_list)
        actions.append(action_list)
        action_vals.append(edge_value_list)
        rewards.append(reward_list)
        next_graphs.append(next_graph_list)
        next_critic_graphs.append(next_critic_graph)
        next_edge_space.append(next_edge_space_list)
        next_act_space.append(next_action_space_list)

        # 4 - [s, o, g, a, r, s_dash, o_dash, g_dash]
        current_state, obs_list, graph_list, critic_graph, edge_space_list, action_space_list, node_set_list = next_state, next_obs_list, next_graph_list, next_critic_graph, next_edge_space_list, next_action_space_list, next_node_set_list

        '''
        Training summary
        1. Sample experiences
        2. Iterate through agents
            1. get next_action and next_action_val using next_graph, for agent i for entire batch from target actor network
            2. actual_q = list of critic(g_i, action_vals of all agents) for entire batch
            3. target_q = list of r_i + gamma * critic'(g_i', action_vals')
            4. update critic

            5. generate actions for entire batch using updated actor 
            6. compute q value using critic(g_i, a_exec)
            7. update actor 
        '''
        
        if (step+1) % batch_size == 0:
            print("step: ", step)

            # step 1 - get next actions and next action values for all agents
            next_action_vals = []
            next_actions = []

            critic_g = []
            next_critic_g = []

            for ag_idx in range(env.num_agents):
                g_device = tuple(x.to(device) for x in list(zip(*next_graphs))[ag_idx]) 
                ed_device = tuple(torch.tensor(x).to(device) for x in list(zip(*next_edge_space))[ag_idx]) 
                act_device = tuple(torch.tensor(x).to(device) for x in list(zip(*next_act_space))[ag_idx])
                actor_list[ag_idx].train()
                policy_val, next_action = actor_list[ag_idx].getBatchAction(g_device, ed_device, act_device, network='target')
                next_action_vals.append(policy_val)
                next_actions.append(next_action)

                critic_g.append([g.detach().clone() for g in critic_graphs])
                next_critic_g.append([g.detach().clone() for g in next_critic_graphs])
            
            action_vals_stack = torch.stack([torch.stack(inner_list) for inner_list in action_vals], dim=0)
            next_action_vals_stack = torch.stack([torch.stack(inner_list) for inner_list in next_action_vals], dim=1)

            # iterate through agents
            for ag_idx in range(env.num_agents):

                # step 2 - actual_q = list of critic(o_i, a) for entire batch 
                action_copy = action_vals_stack.clone().to(device)
                # action_copy = action_vals_stack.detach().clone().to(device)
                critic_g_cuda = tuple(x.to(device) for x in critic_g[ag_idx])

                critic_list[ag_idx].train()
                actual_q = critic_list[ag_idx].forward(critic_g_cuda, action_copy, network='main', training=True) #Q(gi, {ai})
                actual_critic_loss_list[ag_idx].extend([l[0] for l in actual_q.tolist()])

                # step 3 - target_q = list of r_i + gamma * critic'(o_i', a')
                next_action_copy = next_action_vals_stack.clone().to(device)
                # next_action_copy = next_action_vals_stack.detach().clone().to(device)
                next_critic_g_cuda = tuple(x.to(device) for x in next_critic_g[ag_idx])
                critic_list[ag_idx].train()
                target_q = critic_list[ag_idx].forward(next_critic_g_cuda, next_action_copy, network='target', training=True) #Q'(gi_next, pi'(gi_next))
                
                rewards_device = torch.tensor([[s] for s in list(zip(*rewards))[ag_idx]]).to(device)
                target_q = gamma*target_q
                target_critic_loss_list[ag_idx].extend([l[0] for l in target_q.tolist()])
                rewards_critic_loss_list[ag_idx].extend([l[0] for l in rewards_device.tolist()])
                target_q = rewards_device + target_q
                total_target_critic_loss_list[ag_idx].extend([l[0] for l in target_q.tolist()])
                
                # step 4 - critic update
                critic_loss = nn.MSELoss()(actual_q, target_q)
                critic_loss_list[ag_idx].append(critic_loss.item())

                if regularise_critic:
                    critic_dist = torch.sum(torch.stack(list((p - previous_critic_param_list[ag_idx][name]).pow(2).sum() for name, p in critic_list[ag_idx].main.named_parameters())))
                    critic_loss += reg_lambda_critic * critic_dist
                    critic_reg_list[ag_idx].append(critic_dist.item())
                    total_critic_list[ag_idx].append(critic_loss.item())
                    previous_critic_param_list[ag_idx] = {name: p.clone().detach() for name, p in critic_list[ag_idx].main.named_parameters()}
                    critic_list[ag_idx].update(actual_q, target_q, critic_dist*reg_lambda_critic)

                else:
                    critic_list[ag_idx].update(actual_q, target_q, 0)
            
            # step 5 -  generate actions for agent i for entire batch (ex_actions)
            ex_action_vals = []
            ex_actions = []
            for ag_idx in range(env.num_agents):
                graph_c_device = tuple(x.to(device) for x in list(zip(*graphs))[ag_idx]) 
                edge_c_device = tuple(torch.tensor(x).to(device) for x in list(zip(*edge_space))[ag_idx]) 
                action_c_device = tuple(torch.tensor(x).to(device) for x in list(zip(*act_space))[ag_idx])
                actor_list[ag_idx].train()
                policy_val, next_action = actor_list[ag_idx].getBatchAction(graph_c_device, edge_c_device, action_c_device, network='main')
                ex_action_vals.append(policy_val)
                ex_actions.append(next_action)
            
            ex_action_vals_stack = torch.stack([torch.stack(inner_list) for inner_list in ex_action_vals], dim=1)
            
            actor_loss = 0
            for ag_idx in range(env.num_agents):
                # setting actor and critic's gradients to 0
                critic_list[ag_idx].optimizer.zero_grad()
                actor_list[ag_idx].optimizer.zero_grad()

                # step  7 - compute q value using criti(o_i, ex_actions)
                ex_copy = ex_action_vals_stack.to(device) #.detach().clone().to(device)
                ex_graph_cuda = tuple(x.to(device) for x in critic_graphs) 
                critic_list[ag_idx].train()
                q = critic_list[ag_idx].forward(ex_graph_cuda, ex_copy, network='main', training=True)


                # step 8.1 - loss from critic
                loss = -q.mean() 
                actor_loss_list[ag_idx].append(loss.item())
                #critic_eval_list[ag_idx].apppend(-(loss.item()))

                # step 8.2 - reg loss - regularisation with previous network
                if regularise_actor:
                    dist = torch.sum(torch.stack(list((p - previous_actor_param_list[ag_idx][name]).pow(2).sum() for name, p in actor_list[ag_idx].main.named_parameters())))
                    if exp_loss:
                        dist = torch.exp(dist)
                    loss += reg_lambda_actor * dist
                    reg_list[ag_idx].append(dist.item())
                    previous_actor_param_list[ag_idx] = {name: p.clone().detach() for name, p in actor_list[ag_idx].main.named_parameters()}


                actor_loss = actor_loss + loss
                total_actor_loss_list[ag_idx].append(loss.item())

                # zero-ing gradients before back propagation
                actor_list[ag_idx].optimizer.zero_grad()

            # backpropagating loss
            actor_loss.backward()

            # recording normalised gradients of each actor
            for ag_idx in range(env.num_agents):

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(actor_list[ag_idx].main.parameters(), 5)  

                # updating the weights of the policy network
                actor_list[ag_idx].optimizer.step()
                actor_list[ag_idx].scheduler.step()

                # recording actor's gradients
                total_norm = 0.0
                for param in actor_list[ag_idx].parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2

                total_norm = total_norm ** 0.5
                actor_list[ag_idx].gradient_norms.append(total_norm) 

            # step 9 - soft update of main and target networks
            if epoch % soft_update_eps == 0:
                for ag_idx in range(env.num_agents):
                    actor_list[ag_idx].soft_update(tau = soft_update_tau_actor)
                    if critic_variable_tau:
                        tau_critic = max(soft_update_tau_critic_range) - ((max(soft_update_tau_critic_range) - min(soft_update_tau_critic_range)) / num_episodes) * epoch
                    else:
                        tau_critic = soft_update_tau_critic
                    critic_list[ag_idx].soft_update(tau = tau_critic)

            if epoch % save_model_every_eps == 0:
                actor_grad_norms = [actor_list[a_x].gradient_norms for a_x in range(env.num_agents)]
                critic_grad_norms = [critic_list[a_x].gradient_norms for a_x in range(env.num_agents)]
                losses = {"total_actor_loss_list": total_actor_loss_list, "actor_loss_list": actor_loss_list, "reg_list": reg_list, "consistency_list": consistency_list, "critic_loss_list": critic_loss_list, "critic_reg_list": critic_reg_list, "total_critic_list": total_critic_list, "actual_critic_loss_list":actual_critic_loss_list, "target_critic_loss_list":target_critic_loss_list, "rewards_critic_loss_list":rewards_critic_loss_list, "total_target_critic_loss_list":total_target_critic_loss_list, "actor_grad_norms": actor_grad_norms, "critic_grad_norms": critic_grad_norms}
                modelDump(actor_list, critic_list, config, losses, result_file, loss_file, epoch)

            if (epoch+1) % evaluate_model_every_eps == 0:

                evalSimulator(epoch, simulated_env, actor_list, steps_per_episode_eval, simulated_eps, eval_file, eval_stats_file, device, openness = openness, driver_locations=driver_locations_eval)

            # update episode and epsilon
            epoch = epoch+1
            epsilon = epsilon_fn(epoch)
        
    # After each episode, check if we should decrement steps_per_episode_train for the new episode
    # If epoch is beyond 3/4th of the total episodes, set it to the last value in step_reduction
    if epoch > int(num_episodes * 3 / 4):
        steps_per_episode_train = step_reduction[-1]
        # print(f"Episode {epoch}: Steps per episode = {steps_per_episode_train}")
    else:
        # Determine the steps_per_episode_train based on the current episode number
        for i, start_eps in enumerate(episode_ranges):
            if epoch > start_eps:
                steps_per_episode_train = step_reduction[i]
                # print(f"Episode {epoch}: Steps per episode = {steps_per_episode_train}")
    
        

print("Training Completed")

# ------------------------------------
# Model and Configuration Dump

actor_grad_norms = [actor_list[a_x].gradient_norms for a_x in range(env.num_agents)]
critic_grad_norms = [critic_list[a_x].gradient_norms for a_x in range(env.num_agents)]
losses = {"total_actor_loss_list": total_actor_loss_list, "actor_loss_list": actor_loss_list, "reg_list": reg_list, "consistency_list": consistency_list, "critic_loss_list": critic_loss_list, "critic_reg_list": critic_reg_list, "total_critic_list": total_critic_list, "actual_critic_loss_list":actual_critic_loss_list, "target_critic_loss_list":target_critic_loss_list, "rewards_critic_loss_list":rewards_critic_loss_list, "total_target_critic_loss_list":total_target_critic_loss_list, "actor_grad_norms": actor_grad_norms, "critic_grad_norms": critic_grad_norms}


modelDump(actor_list, critic_list, config, losses, result_file, loss_file, "final")
print("Results collated and dumped to local")

# ------------------------------------
# Post Training Analysis

subprocess.run(["python", "loss_analysis_split.py", str(result_file), str(reg_lambda_actor), str(consistency_lambda)])
print("Training Losses Plotted")