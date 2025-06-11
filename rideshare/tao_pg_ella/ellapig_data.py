#----------------------------------------------------------------------------------------------------------------------#
# Title: Ellapig Data Management
# Description: This file contains functions for recording and plotting data from the Ellapig & actor-critic base-learner networks.
# Author: Matthew Sentell
# Version: 1.00.00
#----------------------------------------------------------------------------------------------------------------------#

from types import SimpleNamespace
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_base(losses, results_group='tmp', result_id='tmp', log_done=False):
    '''Plots the provided actor and critic losses.'''
    Path(f'./results/{results_group}').mkdir(parents=True, exist_ok=True) # validate directory

    target = [ # prep losses for plotting
        SimpleNamespace(title='Actor Loss by Iteration', x='Iteration', y='Actor Loss', color='blue', data=losses.actor),
        SimpleNamespace(title='Critic Loss by Iteration', x='Iteration', y='Critic Loss', color='orange', data=losses.critic),
        SimpleNamespace(title='Avgerage Episodic Reward', x='Iteration', y='Reward', color='black', data=losses.reward)
    ]

    fig, plots = plt.subplots(2, 2, figsize=(16, 9)) # prep plots
    for sub, plot in enumerate(col for row in plots for col in row): # make each plot
        if sub < len(target):
            plot.plot(list(range(1, len(device_safe(target[sub].data))+1)), device_safe(target[sub].data), color=target[sub].color)
            plot.set_title(target[sub].title)
            plot.set_ylabel(target[sub].y)
            plot.set_xlabel(target[sub].x)
        else:
            fig.delaxes(plot)

    # store final plots
    plt.tight_layout()
    plt.savefig(f'./results/{results_group}/{result_id}_losses.png', dpi=300, bbox_inches='tight')
    plt.close()

    if log_done:
        print(f'Stored to: results/{results_group}/{result_id}_losses.png')

def plot_full(stats, results_group='tmp', result_id='', log_done=True, store=False):
    '''Plots the provided actor, critic, & task-specific latent space losses and average immediate rewards.'''

    # validate directory & dump stats
    Path(f'./results/{results_group}').mkdir(parents=True, exist_ok=True)
    if store:
        with open(f'./results/{results_group}/{result_id}_losses.pkl', 'wb') as f:
            pickle.dump(stats, f)

        # subfunctions
    internalize = lambda cat: torch.stack([vars(loss)[cat] for loss in stats.losses]).mean(dim=-2)
    externalize = lambda cat: torch.stack([vars(loss)[cat].mean() for loss in stats.losses])

    # prep stats for plotting
    actor = 'actor' if hasattr(stats.losses[0], 'actor') else 'policy'
    if hasattr(stats.losses[0], 'latent'):
        latent_actor = 'latent'
        latent_critic = None
    else:
        latent_actor = 'latent_policy'
        latent_critic = 'latent_critic'
    target = [
        SimpleNamespace(title='Average Actor Loss across Tasks', x='Learning Iteration', y='Actor Loss', color='blue', data=externalize(actor)),
        SimpleNamespace(title='Average Critic Loss across Tasks', x='Learning Iteration', y='Critic Loss', color='orange', data=externalize('critic')),
        SimpleNamespace(title='Average TLS Loss across Tasks', x='Learning Iteration', y='Task Latent Space Loss', color='pink', data=externalize(latent_actor)),
        SimpleNamespace(title='Average Immediate Reward', x='Learning Iteration', y='Immediate Reward', color='black', data=stats.rewards),
        SimpleNamespace(title='Average Actor Loss per Task', x='Base Learner Iteration', y='Actor Loss', color='blue', data=internalize(actor)),
        SimpleNamespace(title='Average Critic Loss per Task', x='Base Learner Iteration', y='Critic Loss', color='orange', data=internalize('critic')),
        SimpleNamespace(title='Average TLS Loss per Task', x='Internal Iteration', y='Task Latent Space Loss', color='pink', data=internalize(latent_actor)),
    ]
    if latent_critic is not None:
        target.insert(3, SimpleNamespace(title='Average Critic TLS Loss across Tasks', x='Learning Iteration', y='Critic Task Latent Space Loss', color='cyan', data=externalize(latent_critic)))
        target.append(SimpleNamespace(title='Average Critic TLS Loss per Task', x='Internal Iteration', y='Critic Task Latent Space Loss', color='cyan', data=internalize(latent_critic)))

    # processing task data
    if hasattr(stats, 'tasks'):
        if hasattr(stats.tasks.list[0][0], '__iter__') and all(len(iteration) == 1 for iteration in stats.tasks.list):
            stats.tasks.list = [episode for iteration in stats.tasks.list for episode in iteration]
        if hasattr(stats.tasks.list[0][0], '__iter__'):
            step_max = torch.tensor([len(episode) for iteration in stats.tasks.list for episode in iteration], device=device).max()
            stats.tasks.list = [torch.stack([torch.nn.functional.pad(torch.tensor(episode, dtype=float, device=device), (0, step_max - len(episode)), 'constant', 0) for episode in iteration]).mean(dim=-2) for iteration in stats.tasks.list]
            closed = torch.tensor([l[0] for l in stats.tasks.list], dtype=float, device=device).std() < 1
            
            steptasks = torch.stack([torch.nn.functional.pad(torch.tensor(tasklist, dtype=float, device=device), (0, step_max - len(tasklist)), 'constant', 0) for tasklist in stats.tasks.list]).mean(dim=-2)
            steptasks = SimpleNamespace(title='Average Number of Tasks per Step', x='Episode Step', y='Tasks Observed', color='green', data=steptasks)

        else:
            closed = all(l[0] == count for l, count in zip(stats.tasks.list, stats.tasks.count))
            step_max = torch.tensor([len(episode) for episode in stats.tasks.list], device=device).max()
            
            steptasks = torch.stack([torch.nn.functional.pad(torch.tensor(tasklist, dtype=float, device=device) / count, (0, step_max - len(tasklist)), 'constant', 0) for tasklist, count in zip(stats.tasks.list, stats.tasks.count)]).mean(dim=-2)
            steptasks = SimpleNamespace(title='Normalized No. Tasks per Step', x='Episode Step', y='Tasks Observed', color='green', data=steptasks)
            
        deltask = [next(-a for a, _ in [np.polyfit(range(len(tasklist)), tasklist, 1)]) for tasklist in stats.tasks.list]
        deltask = SimpleNamespace(title='Average Task Reduction across Tasks', x='Learning Iteration', y='Best-Fit Normalized Task Reduction', color='green', data=deltask)
        
        if closed:
            zerotimes = [list(tasklist).index(0) if 0 in tasklist else len(tasklist) for tasklist in stats.tasks.list]
            zerotimes = SimpleNamespace(title='Average Time to Zero Tasks', x='Learning Iteration', y='Steps', color='magenta', data=zerotimes)
        else:
            zerotimes = SimpleNamespace(title='Tasks Encountered per Learning Iteration', x='Learning Iteration', y='Tasks', color='magenta', data=stats.tasks.count)

        target.insert(4, deltask)
        # target.append(SimpleNamespace(title='Avg. Normalized Immediate Reward', x='Learning Iteration', y='Normalized Immediate Reward', color='black', data=[r/t*avg_count for r, t in zip(stats.rewards, stats.tasks.count)]))
        target.append(zerotimes)
        target.append(steptasks)

    # plot each target
    fig, plots = plt.subplots(2, -(len(target) // -2), figsize=(21, 9)) # prep plots
    agent = result_id.split('_')[-1]
    fig.suptitle(f"Agent #{agent} Learning Data for {result_id[:-len(agent) - 1]}")
    for sub, plot in enumerate(col for row in plots for col in row): # make each plot
        if sub < len(target):
            if not hasattr(device_safe(target[sub].data)[0], "cpu"):
                plot.plot(list(range(1, len(device_safe(target[sub].data))+1)), device_safe(target[sub].data), color=target[sub].color)
            else:
                plot.plot(list(range(1, len(device_safe(target[sub].data)) + 1)), device_safe(target[sub].data), color=target[sub].color)
            plot.set_title(target[sub].title)
            plot.set_ylabel(target[sub].y)
            plot.set_xlabel(target[sub].x)
        else:
            fig.delaxes(plot)
    # store final plots
    plt.tight_layout()
    plt.savefig(f'./results/{results_group}/{result_id}_losses.png', dpi=300, bbox_inches='tight')
    plt.close()

    if log_done:
        print(f'Stored to: results/{results_group}')

def plot_task(losses, results_group='tmp', result_id='task', log_done=False):
    '''Plots the provided BL actor & critic losses and the local latent space losses.'''
    Path(f'./results/{results_group}').mkdir(parents=True, exist_ok=True) # validate directory
    with open(f'./results/{results_group}/{result_id}_losses.pkl', 'wb') as f:
        pickle.dump(losses, f)

    target = [ # prep losses for plotting
        SimpleNamespace(title='Actor Loss by BL Iteration', x='Base Learner Iteration', y='Actor Loss', color='blue', data=losses.actor),
        SimpleNamespace(title='Critic Loss by BL Iteration', x='Base Learner Iteration', y='Critic Loss', color='orange', data=losses.critic),
        SimpleNamespace(title='TLS Loss by Internal Iteration', x='Internal Iteration', y='Task Latent Space Loss', color='pink', data=losses.latent),
    ]

    fig, plots = plt.subplots(2, 2, figsize=(16, 9)) # prep plots
    fig.suptitle(f'\"{result_id}\" Losses')
    for sub, plot in enumerate(col for row in plots for col in row): # make each plot
        if sub < len(target):
            plot.plot(list(range(1, len(device_safe(target[sub].data))+1)), device_safe(target[sub].data), color=target[sub].color)
            plot.set_title(target[sub].title)
            plot.set_ylabel(target[sub].y)
            plot.set_xlabel(target[sub].x)
        else:
            fig.delaxes(plot)

    # store final plots
    plt.tight_layout()
    plt.savefig(f'./results/{results_group}/{result_id}_losses.png', dpi=300, bbox_inches='tight')
    plt.close()

    if log_done:
        print(f'Stored to: results/{results_group}/{result_id}_losses.png')

def plot_runs(trajectories, results_group='tmp', result_id='run', log_done=True):

    target=[]
    max_step = torch.tensor([len(episode) for episode in trajectories], device=device).max().item()

    # action pie chart
    actions = [step.reaction.item() % 4 for episode in trajectories for step in episode]
    actions = [actions.count(x) for x in range(4)]
    target.append(SimpleNamespace(title='Action Frequency', labels=['noöp', 'accept', 'pickup', 'dropoff'], sizes=actions))

    # action time chart
    actions = [[episode[step].reaction.item() % 4 for episode in trajectories if len(episode) > step] for step in range(max_step)]
    actions = [[step.count(x) / len(step) for x in range(4)] for step in actions]
    target.append(SimpleNamespace(title='Action Frequency per Step', x='Step', y='Action Frequency', data=actions, normal=True, fill=True))

    # task continuity per step
    tasks = [[[*step.observation][int(step.reaction.item() // 4)] for step in episode if len([*step.observation]) > 0] for episode in trajectories]
    tasktime = [torch.tensor([1 if episode[step] == episode[step - 1] else 0 for episode in tasks if len(episode) > step], dtype=float, device=device).mean().item() for step in range(1, max_step)]
    tasktime = [[time, 1 - time] for time in tasktime]
    # target.append(SimpleNamespace(title='Avg. Task Continuity per Step', x='Step', y='Task Continuity', data=tasktime, normal=True, fill=True))

    # pooling dropoffs per step
    drops = [[torch.tensor([episode[step].reward > 1 and (x < 1 or len([True for task in episode[step].observation.values() if task[-1] > 1]) > 1) for episode in trajectories if len(episode) > step], dtype=float, device=device).sum().item() for x in range(2)]for step in range(max_step)]
    target.append(SimpleNamespace(title='Pooling vs Non-Pooling Dropoffs per Step', x='Step', y='Dropoffs', labels=['non-pooling', 'pooling'], data=drops, fill=True))
    # print([[[f'>{key}<' if step.action.item() // 2 == list(step.observation.keys()).index(key) else key for key in step.observation.keys() if int(key.split('.')[-1]) > 1] for step in episode if step.reward > 1 and len([True for task in step.observation.keys() if int(task.split('.')[-1]) > 1]) > 1] for episode in trajectories if len(episode) < max_step and any(step.reward > 1 and len([True for task in step.observation.keys() if int(task.split('.')[-1]) > 1]) > 1 for step in episode)])
    # print('Early Finish w/ Pooling:', len([True for episode in trajectories if len(episode) < max_step and any(step.reward > 1 and len([True for task in step.observation.values() if int(task[-1]) > 1]) > 1 for step in episode)]))
    # print('Any w/ Pooling:', len([True for episode in trajectories if any(step.reward > 1 and len([True for task in step.observation.values() if int(task[-1]) > 1]) > 1 for step in episode)]))
    # print('Any Early Finish:', len([True for episode in trajectories if len(episode) < max_step]))

    # tasks seen per step
    # observations = torch.tensor([torch.tensor([len(episode[step].observation) for episode in trajectories if len(episode) > step], dtype=float, device=device).mean().item() for step in range(max_step)], dtype=float, device=device).max().item()
    observations = [[torch.tensor([len([True for task in episode[step].observation.values() if task[-1] == x]) for episode in trajectories if len(episode) > step], dtype=float, device=device).mean().item() if x > 0 else torch.tensor([len([True for substep in episode[:step] if substep.reward > 1]) for episode in trajectories if len(episode) > step], dtype=float, device=device).mean().item() for x in range(4)] for step in range(max_step)]
    target.append(SimpleNamespace(title='Avg. Task Breakdown per Step', x='Step', y='Tasks Observed', labels=['completed', 'available', 'waiting', 'riding'], data=observations, fill=True))

    # rewards per step
    rewards = [torch.tensor([episode[step].reward for episode in trajectories if len(episode) > step], dtype=float, device=device) for step in range(max_step)]
    rwd_err = [step.std() for step in rewards]
    # rewards = [[step.max(), step.mean(), step.min()] for step in rewards]
    rewards = [step.mean() for step in rewards]
    target.append(SimpleNamespace(title='Immediate Reward per Step', x='Step', y='Average Reward', color='black', data=rewards, err=rwd_err))

    # raw action pie chart
    actions = [step.action.item() % 4 for episode in trajectories for step in episode]
    actions = [actions.count(x) for x in range(4)]
    target.append(SimpleNamespace(title='Raw Action Frequency', labels=['noöp', 'accept', 'pickup', 'dropoff'], sizes=actions))

    # raw action time chart
    actions = [[episode[step].action.item() % 4 for episode in trajectories if len(episode) > step] for step in range(max_step)]
    actions = [[step.count(x) / len(step) for x in range(4)] for step in actions]
    target.append(SimpleNamespace(title='Raw Action Frequency per Step', x='Step', y='Action Frequency', data=actions, normal=True, fill=True))

    # task continuity pie chart
    # tasks = torch.tensor(tasktime, device=device).mean()
    tasks = torch.tensor([1 if episode[step] == episode[step - 1] else 0 for step in range(1, max_step) for episode in tasks if len(episode) > step], dtype=float, device=device).mean().item()
    tasks = [tasks, 1 - tasks]
    # target.append(SimpleNamespace(title='Overall Task Continuity', labels=['Continuous', 'Discontinuous'], sizes=tasks))

    dropcount = [0 if step.reward <= 1 else 2 if len([True for task in step.observation.values() if task[-1] > 1]) > 1 else 1 for episode in trajectories for step in episode]
    dropcount = [dropcount.count(x) for x in range(1,3)]
    # target.append(SimpleNamespace(title='Overall Dropoff Pooling', labels=['Non-Pooling', 'Pooling'], sizes=dropcount))

    # pooling vs non-pooling efficacy per episode
    pooling = [(2 if any(step.reward > 1 and len([True for task in step.observation.values() if int(task[-1]) > 1]) > 1 for step in episode) else 3) if len(episode) < max_step else (1 if any(step.reward > 1 and len([True for task in step.observation.values() if int(task[-1]) > 1]) > 1 for step in episode) else 0) for episode in trajectories]
    pooling = [pooling.count(x) for x in range(4)]
    target.append(SimpleNamespace(title='Pooling Efficacy', labels=['Unfinished w/o Pooling', 'Unfinished w/ Pooling', 'Finished w/ Pooling', 'Finished w/o Pooling'], sizes=pooling))

    # episode time chart
    episodes_step = [len([True for episode in trajectories if len(episode) > step]) for step in range(max_step)]
    target.append(SimpleNamespace(title='Active Episodes per Step', x='Step', y='Incomplete Episodes', color='magenta', data=episodes_step, bottom=0))

    # cumulative rewards per step
    rewards = [torch.tensor([sum([substep.reward for substep in episode[:step]]) for episode in trajectories if len(episode) > step], dtype=float, device=device) for step in range(max_step)]
    rwd_err = [step.std() for step in rewards]
    # rewards = [[step.max().item(), step.mean().item(), step.min().item()] for step in rewards]
    rewards = [step.mean() for step in rewards]
    target.append(SimpleNamespace(title='Cumulative Reward per Step', x='Step', y='Average Reward', color='black', data=rewards, err=rwd_err))

    # plot each target
    fig, plots = plt.subplots(2, -(len(target) // -2), figsize=(21, 9)) # prep plots
    agent = result_id.split('_')[-1]
    fig.suptitle(f"Agent #{agent} Policy Statistics for {result_id[:-len(agent) - 1]}")
    for sub, plot in enumerate(col for row in plots for col in row): # make each plot
        if sub < len(target):
            if hasattr(target[sub], 'data'):
                if hasattr(target[sub], 'color'):
                    if hasattr(target[sub], 'err'):
                        plot.errorbar(list(range(1, len(device_safe(target[sub].data))+1)), device_safe(target[sub].data), yerr=device_safe(target[sub].err), color=target[sub].color, ecolor='gray')
                    else:
                        plot.plot(list(range(1, len(device_safe(target[sub].data))+1)), device_safe(target[sub].data), color=target[sub].color)
                elif hasattr(target[sub], 'fill'):
                    for i in range(len(device_safe(target[sub].data)[0])):
                        subdata = [sum(device_safe(target[sub].data)[element][i:]) for element in range(len(device_safe(target[sub].data)))]
                        if hasattr(target[sub], 'labels'):
                            plot.fill_between(list(range(1, len(device_safe(target[sub].data))+1)), subdata, label=target[sub].labels[i])
                        else:
                            plot.fill_between(list(range(1, len(device_safe(target[sub].data))+1)), subdata)
                else:
                    plot.plot(list(range(1, len(device_safe(target[sub].data))+1)), device_safe(target[sub].data))
                plot.set_title(target[sub].title)
                plot.set_ylabel(target[sub].y)
                plot.set_xlabel(target[sub].x)
                plot.grid()
                if hasattr(target[sub], 'labels'):
                    plot.legend()
                if hasattr(target[sub], 'normal'):
                    plot.set_ylim(0, 1)
                elif hasattr(target[sub], 'bottom'):
                    plot.set_ylim(bottom=target[sub].bottom)
            else:
                plot.pie(target[sub].sizes, labels=target[sub].labels, autopct='%.0f%%')
                plot.set_title(target[sub].title)
        else:
            fig.delaxes(plot)
    # store final plots
    plt.tight_layout()
    plt.savefig(f'./results/{results_group}/{result_id}_losses.png', dpi=300, bbox_inches='tight')
    plt.close()

    if log_done:
        print(f'Stored to: results/{results_group}')

def plot_runs_decomposed(trajectories, results_group='tmp', result_id='run', log_done=True):

    target=[]
    max_step = torch.tensor([len(episode) for episode in trajectories], device=device).max().item()

    # action pie chart
    actions = [0 if int(step.reaction.item()) % 2 == 0 else int(list(step.observation.keys())[int(step.reaction.item()) // 2].split('.')[-1]) for episode in trajectories for step in episode]
    actions = [actions.count(x) for x in range(4)]
    target.append(SimpleNamespace(title='Action Frequency', labels=['noöp', 'accept', 'pickup', 'dropoff'], sizes=actions))

    # action time chart
    actions = [[0 if episode[step].reaction.item() % 2 == 0 else int(list(episode[step].observation.keys())[int(episode[step].reaction.item()) // 2].split('.')[-1]) for episode in trajectories if len(episode) > step] for step in range(max_step)]
    actions = [[step.count(x) / len(step) for x in range(4)] for step in actions]
    target.append(SimpleNamespace(title='Action Frequency per Step', x='Step', y='Action Frequency', data=actions, normal=True, fill=True))

    # task continuity per step
    tasks = [[[*step.observation][int(step.reaction.item() // 2)] for step in episode if len([*step.observation]) > 0] for episode in trajectories]
    tasktime = [torch.tensor([1 if episode[step] == episode[step - 1] else 0 for episode in tasks if len(episode) > step], dtype=float, device=device).mean().item() for step in range(1, max_step)]
    tasktime = [[time, 1 - time] for time in tasktime]
    # target.append(SimpleNamespace(title='Avg. Task Continuity per Step', x='Step', y='Task Continuity', data=tasktime, normal=True, fill=True))

    # pooling dropoffs per step
    drops = [[torch.tensor([episode[step].reward > 1 and (x < 1 or len([True for task in episode[step].observation.keys() if int(task.split('.')[-1]) > 1]) > 1) for episode in trajectories if len(episode) > step], dtype=float, device=device).sum().item() for x in range(2)]for step in range(max_step)]
    target.append(SimpleNamespace(title='Pooling vs Non-Pooling Dropoffs per Step', x='Step', y='Dropoffs', labels=['non-pooling', 'pooling'], data=drops, fill=True))
    # print([[[f'>{key}<' if step.action.item() // 2 == list(step.observation.keys()).index(key) else key for key in step.observation.keys() if int(key.split('.')[-1]) > 1] for step in episode if step.reward > 1 and len([True for task in step.observation.keys() if int(task.split('.')[-1]) > 1]) > 1] for episode in trajectories if len(episode) < max_step and any(step.reward > 1 and len([True for task in step.observation.keys() if int(task.split('.')[-1]) > 1]) > 1 for step in episode)])
    # print('Early Finish w/ Pooling:', len([True for episode in trajectories if len(episode) < max_step and any(step.reward > 1 and len([True for task in step.observation.keys() if int(task.split('.')[-1]) > 1]) > 1 for step in episode)]))
    # print('Any w/ Pooling:', len([True for episode in trajectories if any(step.reward > 1 and len([True for task in step.observation.keys() if int(task.split('.')[-1]) > 1]) > 1 for step in episode)]))
    # print('Any Early Finish:', len([True for episode in trajectories if len(episode) < max_step]))

    # tasks seen per step
    # observations = torch.tensor([torch.tensor([len(episode[step].observation) for episode in trajectories if len(episode) > step], dtype=float, device=device).mean().item() for step in range(max_step)], dtype=float, device=device).max().item()
    observations = [[torch.tensor([len([True for task in episode[step].observation.keys() if int(task.split('.')[-1]) == x]) for episode in trajectories if len(episode) > step], dtype=float, device=device).mean().item() if x > 0 else torch.tensor([len([True for substep in episode[:step] if substep.reward > 1]) for episode in trajectories if len(episode) > step], dtype=float, device=device).mean().item() for x in range(4)] for step in range(max_step)]
    target.append(SimpleNamespace(title='Avg. Task Breakdown per Step', x='Step', y='Tasks Observed', labels=['completed', 'available', 'waiting', 'riding'], data=observations, fill=True))

    # rewards per step
    rewards = [torch.tensor([episode[step].reward for episode in trajectories if len(episode) > step], dtype=float, device=device) for step in range(max_step)]
    rwd_err = [step.std() for step in rewards]
    # rewards = [[step.max(), step.mean(), step.min()] for step in rewards]
    rewards = [step.mean() for step in rewards]
    target.append(SimpleNamespace(title='Immediate Reward per Step', x='Step', y='Average Reward', color='black', data=rewards, err=rwd_err))

    # raw action pie chart
    actions = [0 if step.action.item() % 2 == 0 else int(list(step.observation.keys())[int(step.action.item()) // 2].split('.')[-1]) for episode in trajectories for step in episode]
    actions = [actions.count(x) for x in range(4)]
    target.append(SimpleNamespace(title='Raw Action Frequency', labels=['noöp', 'accept', 'pickup', 'dropoff'], sizes=actions))

    # raw action time chart
    actions = [[0 if episode[step].action.item() % 2 == 0 else int(list(episode[step].observation.keys())[int(episode[step].action.item()) // 2].split('.')[-1]) for episode in trajectories if len(episode) > step] for step in range(max_step)]
    actions = [[step.count(x) / len(step) for x in range(4)] for step in actions]
    target.append(SimpleNamespace(title='Raw Action Frequency per Step', x='Step', y='Action Frequency', data=actions, normal=True, fill=True))

    # task continuity pie chart
    # tasks = torch.tensor(tasktime, device=device).mean()
    tasks = torch.tensor([1 if episode[step] == episode[step - 1] else 0 for step in range(1, max_step) for episode in tasks if len(episode) > step], dtype=float, device=device).mean().item()
    tasks = [tasks, 1 - tasks]
    # target.append(SimpleNamespace(title='Overall Task Continuity', labels=['Continuous', 'Discontinuous'], sizes=tasks))

    # pooling to non pooling dropoffs
    dropcount = [0 if step.reward <= 1 else 2 if len([True for task in step.observation.keys() if int(task.split('.')[-1]) > 1]) > 1 else 1 for episode in trajectories for step in episode]
    dropcount = [dropcount.count(x) for x in range(1,3)]
    # target.append(SimpleNamespace(title='Overall Dropoff Pooling', labels=['Non-Pooling', 'Pooling'], sizes=dropcount))

    # pooling vs non-pooling efficacy per episode
    pooling = [(2 if any(step.reward > 1 and len([True for task in step.observation.keys() if int(task.split('.')[-1]) > 1]) > 1 for step in episode) else 3) if len(episode) < max_step else (1 if any(step.reward > 1 and len([True for task in step.observation.keys() if int(task.split('.')[-1]) > 1]) > 1 for step in episode) else 0) for episode in trajectories]
    pooling = [pooling.count(x) for x in range(4)]
    target.append(SimpleNamespace(title='Pooling Efficacy', labels=['Unfinished w/o Pooling', 'Unfinished w/ Pooling', 'Finished w/ Pooling', 'Finished w/o Pooling'], sizes=pooling))

    # episode time chart
    episodes_step = [len([True for episode in trajectories if len(episode) > step]) for step in range(max_step)]
    target.append(SimpleNamespace(title='Active Episodes per Step', x='Step', y='Incomplete Episodes', color='magenta', data=episodes_step, bottom=0))

    # cumulative rewards per step
    rewards = [torch.tensor([sum([substep.reward for substep in episode[:step]]) for episode in trajectories if len(episode) > step], dtype=float, device=device) for step in range(max_step)]
    rwd_err = [step.std() for step in rewards]
    # rewards = [[step.max().item(), step.mean().item(), step.min().item()] for step in rewards]
    rewards = [step.mean() for step in rewards]
    target.append(SimpleNamespace(title='Cumulative Reward per Step', x='Step', y='Average Reward', color='black', data=rewards, err=rwd_err))

    # plot each target
    fig, plots = plt.subplots(2, -(len(target) // -2), figsize=(21, 9)) # prep plots
    agent = result_id.split('_')[-1]
    fig.suptitle(f"Agent #{agent} Policy Statistics for {result_id[:-len(agent) - 1]}")
    for sub, plot in enumerate(col for row in plots for col in row): # make each plot
        if sub < len(target):
            if hasattr(target[sub], 'data'):
                if hasattr(target[sub], 'color'):
                    if hasattr(target[sub], 'err'):
                        plot.errorbar(list(range(1, len(device_safe(target[sub].data))+1)), device_safe(target[sub].data), yerr=device_safe(target[sub].err), color=target[sub].color, ecolor='gray')
                    else:
                        plot.plot(list(range(1, len(device_safe(target[sub].data))+1)), device_safe(target[sub].data), color=target[sub].color)
                elif hasattr(target[sub], 'fill'):
                    for i in range(len(device_safe(target[sub].data)[0])):
                        subdata = [sum(device_safe(target[sub].data)[element][i:]) for element in range(len(device_safe(target[sub].data)))]
                        if hasattr(target[sub], 'labels'):
                            plot.fill_between(list(range(1, len(device_safe(target[sub].data))+1)), subdata, label=target[sub].labels[i])
                        else:
                            plot.fill_between(list(range(1, len(device_safe(target[sub].data))+1)), subdata)
                else:
                    plot.plot(list(range(1, len(device_safe(target[sub].data))+1)), device_safe(target[sub].data))
                plot.set_title(target[sub].title)
                plot.set_ylabel(target[sub].y)
                plot.set_xlabel(target[sub].x)
                plot.grid()
                if hasattr(target[sub], 'labels'):
                    plot.legend()
                if hasattr(target[sub], 'normal'):
                    plot.set_ylim(0, 1)
                elif hasattr(target[sub], 'bottom'):
                    plot.set_ylim(bottom=target[sub].bottom)
            else:
                plot.pie(target[sub].sizes, labels=target[sub].labels, autopct='%.0f%%')
                plot.set_title(target[sub].title)
        else:
            fig.delaxes(plot)
    # store final plots
    plt.tight_layout()
    plt.savefig(f'./results/{results_group}/{result_id}_losses.png', dpi=300, bbox_inches='tight')
    plt.close()

    if log_done:
        print(f'Stored to: results/{results_group}')

def plot_training(rewards:list[list[float]], *, dest="results/traincomp.png"):
    """Function for plotting the results of training evaluation."""

    # Constants for plotting
    x=list(range(0, 151 ,10))
    colors=["blue", "red", "cyan", "magenta"] #, "cyan", "yellow", "magenta"]
    labels=["2 Agents, 8 Tasks", "3 Agents, 12 Tasks", "4 Agents, 16 Tasks"]

    # Make two separate plots for readability.
    fig, plot = plt.subplots(2, 1, sharex=True, sharey=True)

    # Plot full-service training.
    plot[0].set_title("Full-Service Task Training")
    plot[1].set_title("Per-Stage Task Training")
    for f in range(0, (split := len(rewards) // 2), 2):
    # for f in range(0, len(rewards), 2):

        # Plot full-service training.
        y=rewards[f]
        yerr=rewards[f+1]
        plot[0].plot(x, y, color=colors[c:=(f//2)%len(colors)], label=labels[(f//2)%(split//2)])
        plot[0].fill_between(x, y-yerr, y+yerr, color=colors[c], alpha=0.5, lw=0)

        # Plot per-stage training.
        y=rewards[s:=f+split]
        yerr=rewards[s+1]
        plot[1].plot(x, y, color=colors[c])
        plot[1].fill_between(x, y-yerr, y+yerr, color=colors[c], alpha=0.5, lw=0)

    # Label plots.
    fig.legend(labelcolor=colors)
    plot[1].set_xlabel("Episodes of Training")
    for p in plot:
        p.grid(axis='y')
        p.spines["top"].set_visible(False)
        p.spines["right"].set_visible(False)
        p.set_xticks(x)
        p.set_ylabel("Rewards")

    # Store results.
    plt.tight_layout()
    plt.savefig(dest, dpi=300, bbox_inches='tight')
    plt.close()

def plot_rewards(rewards:list[list[float]], *, dest="results/taskcomp.png"):
    """Function for plotting the results of reward evaluation."""

    # Constants for plotting
    width=.33
    colors=["black", "grey"]
    labels = ["Full-Service", "Per-Stage"]
    xticks=["2 Agents,\n6 Tasks", "2 Agents,\n9 Tasks", "2 Agents,\n12 Tasks", "3 Agents,\n6 Tasks", "3 Agents,\n9 Tasks", "3 Agents,\n12 Tasks"]
    x=np.arange(len(xticks))

    # Make two separate plots for readability.
    fig, plot = plt.subplots(1, 1, figsize=(8, 3))

    # Plot pooling data.
    bar=0
    rewards = [[r for reward in rewards[:len(rewards)//2] for r in reward], [r for reward in rewards[len(rewards)//2:] for r in reward]]
    for r, reward in enumerate(rewards):
        offset=width*bar
        plot.bar(x+offset, reward, width, label=labels[r], color=colors[r])
        bar+=1


    # Label plots.
    fig.legend()
    plot.set_title("Rewards by Task")
    plot.grid(axis='y')
    for s in plot.spines.values():
        s.set_visible(False)
    plot.set_xticks(x+width/2, xticks)
    plot.set_yticks(range(-500, 51, 100))
    plot.set_ylabel("Rewards")

    # Store results.
    plt.tight_layout()
    plt.savefig(dest, dpi=300, bbox_inches='tight')
    plt.close()

def plot_policies(rewards:list[list[float]], *, dest="results/polcomp.png"):
    """Function for plotting the results of policy evaluation."""

    # Constants for plotting
    width = 0.2
    colors = ["red", "blue", "green", "purple"]
    labels = ["MOHITO", "FIFO", "Greedy", "OTPG-ELLA"]
    xticks = ["2 Agents,\n6 Tasks", "2 Agents,\n9 Tasks", "3 Agents,\n6 Tasks", "3 Agents,\n9 Tasks", "3 Agents,\n12 Tasks"]
    x=np.arange(len(xticks))

    # Make two separate plots for readability.
    fig, plot = plt.subplots(1, 1, figsize=(8, 3))

    # Plot pooling data.
    for r, reward in enumerate(rewards):
        offset = width * r
        plot.bar(x+offset, reward, width, label=labels[r], color=colors[r])


    # Label plots.
    fig.legend()
    plot.set_title("Rewards by Policy")
    plot.grid(axis='y')
    for s in plot.spines.values():
        s.set_visible(False)
    plot.set_xticks(x+width*1.5, xticks)
    plot.set_yticks(range(-150, 101, 50))
    plot.set_ylabel("Rewards")

    # Store results.
    plt.tight_layout()
    plt.savefig(dest, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pooling(service:list[list[float]], *, dest="results/poolcomp.png"):
    """Function for plotting the results of pooling evaluation."""

    # Constants for plotting
    width = 0.33
    colors = ["grey", "black"]
    labels = ["Unpooled Serviced","Unpooled Unserviced","Pooled Serviced","Pooled Unserviced"]
    xticks = ["2 Agents,\n6 Tasks", "2 Agents,\n9 Tasks", "2 Agents,\n12 Tasks", "3 Agents,\n6 Tasks", "3 Agents,\n9 Tasks", "3 Agents,\n12 Tasks"]
    x = np.arange(len(xticks))
    served = [sum(s) for s in service[:len(service)//2]] # Sum each statset to for percentage calculations.

    # Make two separate plots for readability.
    fig, plot = plt.subplots(2, 1, sharex=True, sharey=True)

    # Plot pooling data.
    for s, (servestat, poolstat) in enumerate(zip(service.T[::2], service.T[1::2])):
        offset = width * s
        plot[0].bar(x + offset, (servestat[:len(poolstat)//2] + poolstat[:len(poolstat)//2]) / served, width, label=labels[s*2], color=colors[1])
        plot[0].bar(x + offset, servestat[:len(poolstat)//2] / served, width, label=labels[s*2+1], color=colors[0])
        plot[1].bar(x + offset, (servestat[len(poolstat)//2:] + poolstat[len(poolstat)//2:]) / served, width, color=colors[1])
        plot[1].bar(x + offset, servestat[len(poolstat)//2:] / served, width, color=colors[0])


    # Label plots.
    fig.legend(labels=["Unserviced", "Serviced"])
    plot[0].set_title("Full-Service Task Pooling Efficacy")
    plot[1].set_title("Per-Stage Task Pooling Efficacy")
    for p in plot:
        p.grid(axis='y')
        for s in p.spines.values():
            s.set_visible(False)
        p.set_xticks(x + width / 2, [f"!P    P\n{t}" for t in xticks])
        p.set_ylabel("% Passengers")

    # Store results.
    plt.tight_layout()
    plt.savefig(dest, dpi=300, bbox_inches='tight')
    plt.close()

def plot_data(target:str, plotter:str, dest:str=None, episodic:int=None, delta=False):
    '''
    Plots the data from the provided datafile using the specified plotter.
    Optionally, a new destinaiton may be specified, and an episodic-to-immediate reward divisor may be provided.
    '''
    # find sourcefile(s)
    if 'results' not in target:
        target = f'results/{target}'
    if '.pkl' == target[-4:]:
        with open(target, 'rb') as f:
            data = pickle.load(f)
            if episodic != None:
                data.rewards = [reward / episodic for reward in data.rewards]
            print(f'Found \"{target}\"'.replace('//', '/'))
    else:
        for f in os.listdir(target):
            if f[-4:] == '.pkl' and 'loss' in f:
                plot_data(target + f'/{f}', plotter, dest, episodic, delta)
        return True

    # set result groud & id
    dest = target.split('/') if dest is None else dest.split('/')
    if 'results' in dest:
        results_group = '/'.join(dest[dest.index('results')+1:-1])
    else:
        results_group = '/'.join(dest[:-1])
    if '_losses.pkl' in dest[-1]:
        result_id = dest[-1][:-11]
    elif '.pkl' in dest[-1]:
        result_id = dest[-1][:-4]
    else:
        result_id = dest[-1]

    # plot data with the desired plotter
    match plotter:
        case 'base':
            plotter = plot_base
        case 'task':
            plotter = plot_task
        case 'full':
            plotter = plot_full
        case 'runs':
            plotter = plot_runs
    plotter(data, results_group, result_id, True)

def store(results_group='tmp', policies=None, losses=None, config:dict=None):
    Path(f'./results/{results_group}').mkdir(parents=True, exist_ok=True) # validate directory
    result_id = results_group.split('/')[-1]
    if policies != None:
        for agent, policy in enumerate(policies):
            with open(f'./results/{results_group}/{result_id}_{agent}_policy.pkl', 'wb') as f:
                pickle.dump(policy, f)
    if losses != None:
        for agent, loss in enumerate(losses):
            with open(f'./results/{results_group}/{result_id}_{agent}_losses.pkl', 'wb') as f:
                pickle.dump(loss, f)
    if config != None:
            with open(f'./results/{results_group}/{result_id}_config.pkl', 'wb') as f:
                pickle.dump(config, f)

def load(src:str, partial=True):
    '''
    Returns a training session's configuration, agent policy(ies), and resultant data given a source destination in the results directory.
    Paremeters:
        src (str): The source identifier for the relevant training session.
        trainer (str): The subdirectory(ies) for the relevant trainer or training session.
        partial (bool, optional): Whether to allow partial loading or to raise errors for missing files.
    '''

    # variable initialization
    if src[:7] != 'results':
        src = f'results/{src}'

    target = src.split('/')[-1]
    args = None
    agents = []
    stats = []

    if not Path(f'{src}/').exists():
        while not Path('/'.join(src.split('/')[:-1])).exists():
            src = '/'.join(src.split('/')[:-1])
            target = src.split('/')[-1]
        raise FileNotFoundError(f"Directory \"{target}\" not found in \"{'/'.join(src.split('/')[:-1])}/\"")

    # load training configuration
    try:
        with open(f'{src}/{target}_config.pkl', 'rb') as f:
            args = SimpleNamespace(**pickle.load(f))
    except FileNotFoundError as e:
        if partial:
            print(f'Missing configuration file for \"{src}\".')
            args = None
        else:
            raise FileNotFoundError(f'Missing configuration file for \"{src}\".').with_traceback(e.__traceback__) from None

    
    # load training progress
    try:
        if args is not None:
            for agent in range(args.no_agents if hasattr(args, 'no_agents') else 1):
                with open(f'{src}/{target}_{agent}_policy.pkl', 'rb') as f:
                    agents.append(pickle.load(f))
                with open(f'{src}/{target}_{agent}_losses.pkl', 'rb') as f:
                    stats.append(pickle.load(f))
        else: raise FileNotFoundError
    except FileNotFoundError: # agent(s) not in separate files, try monofiles
        try:
            with open(f'{src}/{target}_policy.pkl', 'rb') as f:
                agents = pickle.load(f)
        except FileNotFoundError:
            if partial:
                print(f'Missing policy file(s) from \"{src}\".')
                agents = None
            else:
                raise FileNotFoundError(f'Missing policy file(s) from \"{src}\".').with_traceback(e.__traceback__) from None
        try:
            with open(f'{src}/{target}_losses.pkl', 'rb') as f:
                stats = pickle.load(f)
        except FileNotFoundError:
            if partial:
                print(f'Missing loss file(s) from \"{src}\".')
                stats = None
            else:
                raise FileNotFoundError(f'Missing loss file(s) from \"{src}\".').with_traceback(e.__traceback__) from None
    
    # return config, policies, and results
    return args, agents, stats

def device_safe(data):
    if hasattr(data, "cpu"):
        return data.cpu()
    elif hasattr(data, "__iter__") and len(data) > 0:
        return [device_safe(d) for d in data]
    else:
        return data
