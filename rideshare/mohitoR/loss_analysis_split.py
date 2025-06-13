#----------------------------------------------------------------------------------------------------------------------#
# Title: Post-Training Analysis
# Description: This file contains the script to analyse a training run 
# Author: Gayathri Anil 
# Version: 23.05.01
# Last updated on: 05-03-2023 
#----------------------------------------------------------------------------------------------------------------------#

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import sys
import os
from rideshare.utils import *
import re
import ast

result_file = sys.argv[1] #str('cuda_11-12_10000eps_500pereps_online_no_openness_agents2_pass13_tauC0005_tauA0005_A5_C7_lrA00001_lrC0001_train_per_eps20_clip50_regA01_cons001_exploss_GAT') 
reg_lambda = 0.1 #float(sys.argv[2])
consistency_lambda = 0.01 #float(sys.argv[3])

reg = reg_lambda > 0
consistency = consistency_lambda > 0

window_size = 5

loss_file = './results/' + str(result_file) + '/losses_'+ str(result_file) + '.pkl'
plot_file = './results/' + str(result_file) + '/losses_'+ str(result_file)
actor_plot_file = './results/' + str(result_file) + '/actor_losses_'+ str(result_file)
critic_plot_file = './results/' + str(result_file) + '/critic_losses_'+ str(result_file)
split_critic_plot_file = './results/' + str(result_file) + '/split_critic_losses_'+ str(result_file)
grad_plot_file = './results/' + str(result_file) + '/grad_norm_'+ str(result_file)
eval_file = './results/' + str(result_file) + '/eval_file_' + str(result_file) + '.csv'
eval_plot = './results/' + str(result_file) + '/eval_file_' + str(result_file) 
stats_file = './results/' + str(result_file) + '/stats_file_' + str(result_file) + '.csv'
tcr_stats_plot = './results/' + str(result_file) + '/stats_tcr_' + str(result_file) 
wait_stats_plot = './results/' + str(result_file) + '/stats_wait_' + str(result_file) 
eval_stats_file = './results/' + str(result_file) + '/eval_stats_file_' + str(result_file) + '.csv'
eval_tcr_stats_plot = './results/' + str(result_file) + '/eval_stats_tcr_' + str(result_file) 
eval_wait_stats_plot = './results/' + str(result_file) + '/eval_stats_wait_' + str(result_file) 
# extra_stats_plot = './results/' + str(result_file) + '/stats_task_completion_' + str(result_file) 

if '_agents' in result_file:
    num_agents = int((re.search(r'_agents(\d+)_', result_file)).group(1))
else: 
    num_agents = 3

# load losses
with open(loss_file, 'rb') as f:
    losses = pickle.load(f)

actor_loss_list = losses['total_actor_loss_list']
critic_loss_list = losses['critic_loss_list']

# Generate a range of integer values for the episodes
episodes = list(range(1, len(actor_loss_list[0])+1))

if os.path.exists(eval_file):

    data = pd.read_csv(eval_file)
    data = data.set_axis(['model', 'set', 'ag_idx', 'reward', 'action'], axis=1)
    # Convert all 'model' and 'model_files' entries to string
    # data['model'] = data['model'].astype(str)
    # model_files = [str(x) for x in model_files]

    # Convert 'model' to a categorical variable with a defined order
    # data['model'] = pd.Categorical(data['model'], ordered=True)

    # Sort data by 'model'
    # data = data.sort_values('model')

    # Initialize the figure and axes
    fig, axes = plt.subplots(3, 1, figsize=(50, 25))

    # Plot 1: Average reward for each model for each ag_idx
    for ag_idx in data['ag_idx'].unique():
        avg_rewards = data[data['ag_idx'] == ag_idx].groupby('model')['reward'].sum().sort_index()
        axes[0].plot(avg_rewards.index, avg_rewards.values, marker='o', label=f'Agent {ag_idx}')
    axes[0].set_title('Average Reward across epochs by agent')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Average Reward')
    axes[0].legend()

    # Plot 2: Average reward for each model for all ag_idx together
    avg_rewards_all = data.groupby('model')['reward'].sum().sort_index()
    rolling_avg_all = avg_rewards_all.rolling(window=window_size).mean()
    axes[1].plot(avg_rewards_all.index, avg_rewards_all.values, marker='o', label='All Agents')
    axes[1].plot(rolling_avg_all.index, rolling_avg_all.values, linestyle='--', label=f'Moving Avg ({window_size} epochs)')
    axes[1].set_title('Average Reward across epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Average Reward')
    axes[1].legend()
    

    # Plot 3: Number of steps where reward > 3 for each model
    num_steps_greater_than_3 = data[data['reward'] > 3].groupby('model').size().sort_index()
    rolling_avg_steps = num_steps_greater_than_3.rolling(window=window_size).mean()
    axes[2].plot(num_steps_greater_than_3.index, num_steps_greater_than_3.values, marker='o', label='# serviced passengers')
    axes[2].plot(rolling_avg_steps.index, rolling_avg_steps.values, linestyle='--', label=f'Moving Avg ({window_size} epochs)')
    axes[2].set_title('Number of serviced passengers')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('# Steps')
    axes[2].legend()

    plt.tight_layout()
    # save plot to local
    plt.savefig(eval_plot, dpi=300, bbox_inches='tight')
    plt.close()

    print("Eval plots completed")

 

def str_to_list(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return []

def avg_wait_time(x):
    combined_list = sum(x, [])
    if combined_list:
        return sum(combined_list) / len(combined_list)
    else:
        return 0

df = pd.read_csv(stats_file)

df['accepted_wait_time'] = df['accepted_wait_time'].apply(str_to_list)
df['completed_wait_time'] = df['completed_wait_time'].apply(str_to_list)


# Calculate total_steps for each eps
total_steps_per_eps = df.groupby('eps')['total_steps'].mean()



# TCR Plot
plt.figure(figsize=(50, 25))
# tcr_data = df.groupby('eps').apply(lambda x: x['num_completed'].sum() / x['tasks_added'].sum() if x['tasks_added'].sum() != 0 else 0)
num_completed_sum = df.groupby('eps')['num_completed'].sum()
tasks_added_sum = df.groupby('eps')['tasks_added'].sum()
# plt.plot(tcr_data, marker='o')
plt.plot(num_completed_sum, marker='o', label='# completed tasks')
plt.plot(tasks_added_sum, marker='o', label='# tasks')
plt.title('TCR')
plt.xlabel('eps')
plt.ylabel('Tasks')
plt.grid(True)
plt.savefig(tcr_stats_plot, dpi=300, bbox_inches='tight')
plt.close()

# Create a figure with 3 subplots (3 rows, 1 column)
fig, axes = plt.subplots(2, 1, figsize=(50, 25))

# Average Accepted Wait Time Plot
avg_accepted_wait_time = df.groupby('eps')['accepted_wait_time'].apply(avg_wait_time)
axes[0].plot(avg_accepted_wait_time, marker='o')
axes[0].plot(total_steps_per_eps, color='red')
axes[0].set_title('Average Accepted Wait Time vs eps')
axes[0].set_xlabel('eps')
axes[0].set_ylabel('Average Accepted Wait Time')
axes[0].grid(True)

# Average Task Completion Time Plot
avg_completion_wait_time = df.groupby('eps')['completed_wait_time'].apply(avg_wait_time)
axes[1].plot(avg_completion_wait_time, marker='o')
axes[1].plot(total_steps_per_eps, color='red')
axes[1].set_title('Average Task Completion Time vs eps')
axes[1].set_xlabel('eps')
axes[1].set_ylabel('Average Task Completion Time')
axes[1].grid(True)

# Adjust layout
plt.tight_layout()
plt.savefig(wait_stats_plot, dpi=300, bbox_inches='tight')
plt.close()

print("Stats plots for training plotted")

### Eval episode - Stat plots

df = pd.read_csv(eval_stats_file)

df['accepted_wait_time'] = df['accepted_wait_time'].apply(str_to_list)
df['completed_wait_time'] = df['completed_wait_time'].apply(str_to_list)


# Calculate total_steps for each eps
total_steps_per_eps = df.groupby('eps')['total_steps'].mean()

#-------------------------------------------------------------------------

# TCR Plot
plt.figure(figsize=(50, 25))
# tcr_data = df.groupby('eps').apply(lambda x: x['num_completed'].sum() / x['tasks_added'].sum() if x['tasks_added'].sum() != 0 else 0)
num_completed_sum = df.groupby('eps')['num_completed'].sum()
tasks_added_sum = df.groupby('eps')['tasks_added'].sum()
# plt.plot(tcr_data, marker='o')
plt.plot(num_completed_sum, marker='o', label='# completed tasks')
plt.plot(tasks_added_sum, marker='o', label='# tasks')
plt.title('TCR')
plt.xlabel('eps')
plt.ylabel('Tasks')
plt.grid(True)
plt.savefig(eval_tcr_stats_plot, dpi=300, bbox_inches='tight')
plt.close()

# Create a figure with 3 subplots (3 rows, 1 column)
fig, axes = plt.subplots(2, 1, figsize=(50, 25))

# Average Accepted Wait Time Plot
avg_accepted_wait_time = df.groupby('eps')['accepted_wait_time'].apply(avg_wait_time)
axes[0].plot(avg_accepted_wait_time, marker='o')
axes[0].plot(total_steps_per_eps, color='red')
axes[0].set_title('Average Accepted Wait Time vs eps')
axes[0].set_xlabel('eps')
axes[0].set_ylabel('Average Accepted Wait Time')
axes[0].grid(True)

# Average Task Completion Time Plot
avg_completion_wait_time = df.groupby('eps')['completed_wait_time'].apply(avg_wait_time)
axes[1].plot(avg_completion_wait_time, marker='o')
axes[1].plot(total_steps_per_eps, color='red')
axes[1].set_title('Average Task Completion Time vs eps')
axes[1].set_xlabel('eps')
axes[1].set_ylabel('Average Task Completion Time')
axes[1].grid(True)

# Adjust layout
plt.tight_layout()
plt.savefig(eval_wait_stats_plot, dpi=300, bbox_inches='tight')
plt.close()

print("Stats plots for evaluation plotted")

#-------------------------------------------------------------------------

for ag_idx in range(num_agents):

    # Create a new figure with 2 subplots, with shared x-axis
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(50, 25))

    # plot critic loss in the first subplot in blue
    abs_actor_loss = actor_loss_list[ag_idx] #list(map(abs, actor_loss_list[ag_idx]))
    axs[0].plot(episodes, abs_actor_loss, color='brown', label='Actor Loss')
    axs[0].set_title('Actor Loss over epochs')
    axs[0].set_ylabel('Loss')

    # plot critic loss in the first subplot in blue
    axs[1].plot(episodes, critic_loss_list[ag_idx], color='orange', label='Critic Loss')
    axs[1].set_title('Critic Loss over epochs')
    axs[1].set_ylabel('Loss')

    # x-label for the shared x-axis
    fig.text(0.5, 0.04, 'Epochs', ha='center', va='center')

    # Display the plot
    plt.tight_layout()

    # save plot to local
    plt.savefig(plot_file+'_agent'+str(ag_idx), dpi=300, bbox_inches='tight')
    plt.close()

print("Actor-Critic Losses plotted")

#-------------------------------------------------------------------------

if "actor_grad_norms" in losses:

    actor_grad_norms = losses['actor_grad_norms']
    critic_grad_norms = losses['critic_grad_norms']

    for ag_idx in range(num_agents):

        # Create a new figure with 2 subplots, with shared x-axis
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(50, 25))

        # plot critic loss in the first subplot in blue
        axs[0].plot(actor_grad_norms[ag_idx], color='brown', label='Actor')
        axs[0].set_title('Actor Gradient Norm')
        axs[0].set_ylabel('Total Gradient Norm')

        # plot critic loss in the first subplot in blue
        axs[1].plot(critic_grad_norms[ag_idx], color='orange', label='Critic')
        axs[1].set_title('Critic Gradient Norm')
        axs[1].set_ylabel('Total Gradient Norm')

        # x-label for the shared x-axis
        fig.text(0.5, 0.04, 'Epochs', ha='center', va='center')

        # Display the plot
        plt.tight_layout()

        # save plot to local
        plt.savefig(grad_plot_file+'_agent'+str(ag_idx), dpi=300, bbox_inches='tight')
        plt.close()

    print("Actor-Critic Grad Norm plotted")

# ---------------------------------
# Actor Split Plots


actor_loss_list = losses['actor_loss_list']
if 'total_actor_loss_list' in losses:
    consistency_list = losses['consistency_list']
    reg_list = losses['reg_list']
    total_actor_loss_list = losses['total_actor_loss_list']

    for ag_idx in range(num_agents):

        #Compatibility fix for older experiments. This won't be needed for the newer results
        if len(consistency_list[ag_idx]) != len(episodes):
            consistency_list[ag_idx] = consistency_list[ag_idx] + [0] *(len(episodes)  -  len(consistency_list[ag_idx]))

        # Create a new figure with 2 subplots, with shared x-axis
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(50, 25))

        # plot critic loss in the first subplot in blue
        axs[0].plot(episodes, actor_loss_list[ag_idx], color='brown', label='Base Actor Loss')
        if consistency:
            axs[0].plot(episodes, consistency_list[ag_idx], color='orange', label='Cons Loss')
            axs[0].plot(episodes, [consistency_lambda * i for i  in consistency_list[ag_idx]] , color='green', label='Lambda Cons Loss')
        if reg:
            axs[0].plot(episodes, reg_list[ag_idx], color='black', label='Reg Loss')
            axs[0].plot(episodes, [reg_lambda * i for i  in reg_list[ag_idx]] , color='cyan', label='Lambda Reg Loss')
        axs[0].set_title('Parts of Actor Loss over epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        # plot critic loss in the first subplot in blue
        # plot critic loss in the first subplot in blue
        axs[1].plot(episodes, actor_loss_list[ag_idx], color='orange', label='Base Actor Loss')
        axs[1].plot(episodes, total_actor_loss_list[ag_idx], color='brown', label='Total Actor Loss')
        axs[1].set_title('Actor Loss over epochs')
        axs[1].set_ylabel('Loss')
        axs[1].legend()

        # x-label for the shared x-axis
        fig.text(0.5, 0.04, 'Epochs', ha='center', va='center')

        # Display the plot
        plt.tight_layout()

        # save plot to local
        plt.savefig(actor_plot_file+'_agent'+str(ag_idx), dpi=300, bbox_inches='tight')
        plt.close()




print("Detailed Actor Loss plotted")

# New Critic Plots

# if 'total_critic_list' in losses:
#     reg_list = losses['critic_reg_list']
#     actor_loss_list = losses['critic_loss_list']
#     total_actor_loss_list = losses['total_critic_list']
#     critic_reg_lambda = 0.01

#     for ag_idx in range(num_agents):

#         # Create a new figure with 2 subplots, with shared x-axis
#         fig, axs = plt.subplots(2, 1, sharex=True, figsize=(50, 25))

#         # plot critic loss in the first subplot in blue
#         axs[0].plot(episodes, actor_loss_list[ag_idx], color='brown', label='Base Critic Loss')
#         axs[0].plot(episodes, reg_list[ag_idx], color='black', label='Reg Loss')
#         axs[0].plot(episodes, [critic_reg_lambda * i for i  in reg_list[ag_idx]] , color='cyan', label='Lambda Reg Loss')
#         axs[0].set_title('Parts of Critic Loss over epochs')
#         axs[0].set_ylabel('Loss')
#         axs[0].legend()

#         # plot critic loss in the first subplot in blue
#         # plot critic loss in the first subplot in blue
#         axs[1].plot(episodes, actor_loss_list[ag_idx], color='orange', label='Base Critic Loss')
#         axs[1].plot(episodes, total_actor_loss_list[ag_idx], color='brown', label='Total Critic Loss')
#         axs[1].set_title('Critic Loss over epochs')
#         axs[1].set_ylabel('Loss')
#         axs[1].legend()

#         # x-label for the shared x-axis
#         fig.text(0.5, 0.04, 'Epochs', ha='center', va='center')

#         # Display the plot
#         plt.tight_layout()

#         # save plot to local
#         plt.savefig(split_critic_plot_file+'_agent'+str(ag_idx), dpi=300, bbox_inches='tight')

# print("Detailed Critic Loss plotted - New")


# ---------------------------------
# Critic Split Plots

if "rewards_critic_loss_list" in losses:

    actual_q = losses['actual_critic_loss_list']
    target_q = losses['target_critic_loss_list']
    rewards = losses['rewards_critic_loss_list']
    total_target_q = losses['total_target_critic_loss_list']
    critic_loss_list = losses['critic_loss_list']

    if len(rewards[0]) > len(critic_loss_list[0]):#type(rewards[0][0]) is list or 


        for ag_idx in range(num_agents):

            episodes = list(range(1, len(actual_q[ag_idx])+1))

            # Create a new figure with 2 subplots, with shared x-axis
            fig, axs = plt.subplots(5, 1, sharex=True, figsize=(50, 25))

            # plot critic loss in the first subplot in blue
            axs[0].plot(episodes, actual_q[ag_idx], color='brown', label='Actual Q')
            axs[0].set_title('Actual Q-values over epochs')
            axs[0].set_ylabel('Loss')

            # plot critic loss in the first subplot in blue
            axs[1].plot(episodes, target_q[ag_idx], color='orange', label='Target Q')
            axs[1].set_title('Target Q-values over epochs')
            axs[1].set_ylabel('Loss')

            # plot critic loss in the first subplot in blue
            axs[2].plot(episodes, rewards[ag_idx], color='orange', label='Target Q')
            axs[2].set_title('Rewards over epochs')
            axs[2].set_ylabel('Loss')


            # plot critic loss in the first subplot in blue
            axs[3].plot(episodes, total_target_q[ag_idx], color='brown', label='Critic Loss')
            axs[3].set_title('Total Target Loss over epochs')
            axs[3].set_ylabel('Loss')

            # plot critic loss in the first subplot in blue
            sqr = [val**2 for val in np.subtract(total_target_q[ag_idx],actual_q[ag_idx])]
            axs[4].plot(episodes, sqr, color='brown', label='Critic Loss')
            axs[4].set_title('Critic Loss over epochs')
            axs[4].set_ylabel('Loss')

            # x-label for the shared x-axis
            fig.text(0.5, 0.04, 'Epochs', ha='center', va='center')

            # Display the plot
            plt.tight_layout()

            # save plot to local
            plt.savefig(critic_plot_file+'_agent'+str(ag_idx), dpi=300, bbox_inches='tight')
            plt.close()

    else: 


        for ag_idx in range(num_agents):

            # Create a new figure with 2 subplots, with shared x-axis
            fig, axs = plt.subplots(5, 1, sharex=True, figsize=(50, 25))

            # plot critic loss in the first subplot in blue
            axs[0].plot(episodes, actual_q[ag_idx], color='brown', label='Actual Q')
            axs[0].set_title('Actual Q-values over epochs')
            axs[0].set_ylabel('Loss')

            # plot critic loss in the first subplot in blue
            axs[1].plot(episodes, target_q[ag_idx], color='orange', label='Target Q')
            axs[1].set_title('Target Q-values over epochs')
            axs[1].set_ylabel('Loss')

            # plot critic loss in the first subplot in blue
            axs[2].plot(episodes, rewards[ag_idx], color='orange', label='Target Q')
            axs[2].set_title('Rewards over epochs')
            axs[2].set_ylabel('Loss')

            # plot critic loss in the first subplot in blue
            axs[3].plot(episodes, total_target_q[ag_idx], color='brown', label='Critic Loss')
            axs[3].set_title('Total Target Loss over epochs')
            axs[3].set_ylabel('Loss')

            # plot critic loss in the first subplot in blue
            axs[4].plot(episodes, critic_loss_list[ag_idx], color='brown', label='Critic Loss')
            axs[4].set_title('Critic Loss over epochs')
            axs[4].set_ylabel('Loss')

            # x-label for the shared x-axis
            fig.text(0.5, 0.04, 'Epochs', ha='center', va='center')

            # Display the plot
            plt.tight_layout()

            # save plot to local
            plt.savefig(critic_plot_file+'_agent'+str(ag_idx), dpi=300, bbox_inches='tight')
            plt.close()

print("Detailed Critic Loss plotted")