#----------------------------------------------------------------------------------------------------------------------#
# Title: Ellapig Training Functions
# Description: This file contains the training functions for the Ellapig & actor-critic base-learner networks.
# Author: Matthew Sentell
# Version: 0.06.08
# Last updated on: 2024.03.28
#----------------------------------------------------------------------------------------------------------------------#

from enum import IntEnum
from types import SimpleNamespace
from ellapig_net import *
from ellapig_data import *
from ride import *
import numpy as np
import datetime
import torch

LINE = '--------------------------------'
DATE = f'{datetime.datetime.now():%Y.%m.%d}'

def train_base(iterations=500, episodes=10, steps=500, latent_dimension=32, layers=0, learning_rate=1e-2, notifications=10, result_dir:str=None, result_id:str=None):
    env = gym.make('CartPole-v1', render_mode=None)
    agent = ACagent(4, latent_dimension, 2, learning_rate=learning_rate, layers=layers)
    trajectories = []
    losses = SimpleNamespace(actor=[], critic=[], reward=[])
    notify = iterations // notifications
    if result_dir is None:
        result_dir = f'base/{DATE}'
    if result_id is None:
        result_id = f'i{iterations}e{episodes}s{steps}'
    for iteration in range(iterations):
        i = iteration + 1
        if (i) % notify == 0 or i == iterations:
            print(f'Iteration # {i} / {iterations}')
        trajectories = []
        # agent.critic.optimizer = optim.Adam([agent.critic.policy], learning_rate / i)
        for episode in range(episodes):
            # print (f'  Episode # {episode + 1} / {episodes}')
            observation, info = env.reset()
            trajectory = []
            for step in range(steps):
                # print(f'       Step # {step + 1} / {steps}')
                traj = {'observation': torch.tensor(observation, dtype=float, device=device)}
                traj['action'], traj['value'] = agent.forward(traj['observation'])
                observation, traj['reward'], terminated, truncated, info = env.step(traj['action'].item())
                trajectory.append(SimpleNamespace(**traj))
                if terminated or truncated:
                    break
            trajectories.append(trajectory)
        losses.reward.append(torch.tensor([torch.tensor([step.reward for step in episode], dtype=float, device=device).sum() for episode in trajectories], dtype=float, device=device).mean().item())
        loss_actor, loss_critic = agent.optimize(trajectories, trajectories)
        losses.actor.append(loss_actor.item())
        losses.critic.append(loss_critic.item())
        # print(agent.actor.policy)
        # if i == iterations:
        #     print(agent.hessian(trajectories))
    # print(losses)
    env.close()
    plot_base(losses, result_dir, result_id, log_done=True)
    return losses

def train_base_batch(batch_size=8, iterations=128, episodes=32, steps=512, latent_dimension=32, layers=0, learning_rate=1e-2, notifications=10, result_dir:str=None, result_id:str=None):
    if result_dir is None:
        result_dir = f'base/{DATE}'
    if result_id is None:
        result_id = f'b{batch_size}i{iterations}e{episodes}s{steps}'
    losses = []
    for i in range(batch_size):
        print(f'{LINE}\nTEST NO. {i+1}')
        losses.append(train_base(iterations, episodes, steps, latent_dimension, layers, learning_rate, notifications, result_dir=result_dir, result_id=f'{result_id}_{i}'))
    losses = SimpleNamespace(
        actor = torch.stack([torch.tensor(loss.actor, dtype=float, device=device) for loss in losses]).mean(dim=-2),
        critic = torch.stack([torch.tensor(loss.critic, dtype=float, device=device) for loss in losses]).mean(dim=-2),
        reward = torch.stack([torch.tensor(loss.reward, dtype=float, device=device) for loss in losses]).mean(dim=-2)
    )
    # print(losses)
    plot_base(losses, result_dir, result_id, log_done=True)

def train_cart(iterations=128, episodes=32, steps=512, double=False, latent_dimension=8, latent_task_dimension=8, layers=0, tasks=1, learning_rate=1e-3, learning_steps=128, discount=0.9,
    notify_task=True, fixed=False, plot_partial=False, _agent=None, _stats=None):

    config = locals()
    config['_agent'] = config['_stats'] = None

    # initialize environment, agent, and results
    print(F'{LINE}\nINITIALIZING...')
    env = gym.make('CartPole-v1', render_mode=None)
    if _agent is None:
        agent = Duellapig(
            observation_dimension=4,
            action_dimension=2,
            latent_dimension=latent_dimension,
            latent_task_dimension=latent_task_dimension,
            layers=layers,
            learning_rate=learning_rate,
            learning_steps=learning_steps,
            discount_factor=discount
        ) if double else Linellapig(
            observation_dimension=4,
            action_dimension=2,
            latent_dimension=latent_dimension,
            latent_task_dimension=latent_task_dimension,
            layers=layers,
            learning_rate=learning_rate,
            learning_steps=learning_steps,
            discount_factor=discount
        )
    else:
        agent = _agent
    result_dest = f"cart/{DATE}/i{iterations}e{episodes}{f'_t{tasks}' if tasks > 1 else ''}"
    store(result_dest, config=config)
    stats = SimpleNamespace(rewards=[], losses=[]) if _stats is None else _stats
    offset = len(stats.rewards)
    print(f'Experiment Specifications:\n  {iterations} learning iterations over {episodes}-episode, {steps}-step trajectories\n  {learning_steps} internal iterations at {learning_rate} LR, discounted {discount}\n  {latent_dimension} shared x {latent_task_dimension} task dimensions')
    print(f"Environment Specifications: \"CartPole-v1\"{f' simulated multitasking {tasks} tasks' if tasks > 1 else ''}")
    # print(f'Result Destination: {result_dest}')

    # gather tajectories and learn from them
    print(F'{LINE}\nITERATING...')
    for i in range(offset + 1, iterations + 1):
        print(f'Iteration # {i} / {iterations}')
        trajectories = []
        for episode in range(episodes):
            #print (f'  Episode # {ep_no + 1} / {episodes}')
            observation, info = env.reset()
            trajectory = []
            task = f'mvmt_{randrange(tasks)}'
            for step in range(steps):
                #print(f'       Step # {step_no + 1} / {steps}')
                traj = {'observation': {task:torch.tensor(observation, dtype=float, device=device)}}
                traj['action'] = agent.forward(traj['observation'])
                observation, traj['reward'], terminated, truncated, info = env.step(traj['action'].item())
                trajectory.append(SimpleNamespace(**traj))
                if terminated or truncated:
                    break
            trajectories.append(trajectory)
        if fixed:
            queue = agent.queue
            for j in range(1 ,iterations + 1):
                print(f'Fixing Trajectories...\nIteration # {j} / {iterations}')
                agent.queue = queue
                stats.losses.append(agent.optimize(trajectories, log_task=notify_task, plot_tasks=plot_partial))
                stats.rewards.append(torch.tensor([torch.tensor([step.reward for step in episode], dtype=float, device=device).sum() for episode in trajectories], dtype=float, device=device).mean())
            break
        stats.losses.append(agent.optimize(trajectories, log_task=notify_task, plot_tasks=plot_partial))
        stats.rewards.append(torch.tensor([torch.tensor([step.reward for step in episode], dtype=float, device=device).sum() for episode in trajectories], dtype=float, device=device).mean())

    # plot results
    print(F'{LINE}\nPLOTTING...')
    store(result_dest, policies=[agent], losses=[stats])
    plot_full(stats, result_dest, f'{result_dest[15:]}_0')
    return result_dest

def train_ride(iterations=128, episodes=32, steps=60, double=False, latent_dimension=8, latent_task_dimension=8, layers=0, learning_rate=1e-3, learning_steps=128, discount=0.9, notify_task=True, fixed=False, plot_tasks:str=None, plot_partial:float=None,
    no_agents=3, grid=SimpleNamespace(l=5, w=5), costs=SimpleNamespace(accept=0, pick = -0.1, move = -0.1, miss = -1, drop = None, all_accepted=0),
    no_tasks=1, openness=False, learning=True, raw_action=False, clean=False, decomposed=False,
    _agents=None, _stats=None):

    if plot_partial is not None:
        partializer = iterations if learning else episodes
        plot_partial:int = None if int(-plot_partial) | int(plot_partial) >= partializer else int(plot_partial if plot_partial >= 1 else partializer * plot_partial if plot_partial > 0 else partializer * -plot_partial) if plot_partial > -1 else partializer // -plot_partial
        if plot_partial == 0: plot_partial = None
        del partializer
    config = locals()
    config['_agents'] = config['_stats'] = None

    # local functions to handle gym-nonstandard environment
    # observation converter
    tensorfy = lambda elems: torch.cat([tensorfy(elem) if hasattr(elem, '__iter__') else torch.tensor([elem], dtype=float, device=device) for elem in elems]) if len(elems) > 0 else torch.tensor(elems, dtype=float, device=device)
    def interpret_decomposed(observation, test=False):
        '''
        Specific to the Rideshare environment, converts a single agent's observation
        into a task-specific observation vector for individual task processing.
        Returns both the original joint and the new task-vectorized observations.
        '''

        ObsLayer = IntEnum('ObsLayer', zip(['SELF', 'OTHER', 'ACCEPT', 'RIDE', 'OPEN'], range(5))) # observation layers
        TaskState = IntEnum('TaskState', ['OPEN', 'ACCEPT', 'RIDE']) # task states

        base = [] # will contain own location, others' locations, and the numbbers of accepted & riding passengers
        base.append(np.argwhere(np.vectorize(lambda x: len(x) > 0)(observation[ObsLayer.SELF]))[0])
        base.append([loc for loc in np.argwhere(np.vectorize(lambda x: len(x) > 0)(observation[ObsLayer.OTHER])) for other in observation[ObsLayer.OTHER, loc[0], loc[1]]])
        # get info for passengers currently riding with the agent
        riding = [rider for rider in observation[ObsLayer.RIDE, base[0][0], base[0][1]]]
        # get info for accepted passengers awaiting pickup
        accepted = [acc for loc in np.argwhere(np.vectorize(lambda x: len(x) > 0)(observation[ObsLayer.ACCEPT])) for acc in observation[ObsLayer.ACCEPT, loc[0], loc[1]]]
        # get info for passengers that have yet to be accepted by anyone
        available = [avail for loc in np.argwhere(np.vectorize(lambda x: len(x) > 0)(observation[ObsLayer.OPEN])) for avail in observation[ObsLayer.OPEN, loc[0], loc[1]]]
        base.append(len(accepted))
        base.append(len(riding))
        base.append(len(available))
        tasks = {} # combine tasks in to a single hashmap TODO: fix overwriting of same src-dest pairs!
        for target in riding: # tasks en route
            tasks['.'.join('.'.join(str(j) for j in i) for i in target[1:3] + [[int(TaskState.RIDE)]])] = tensorfy([base, target[1:]]).to(device)
        for target in accepted: # tasks awaiting pickup
            tasks['.'.join('.'.join(str(j) for j in i) for i in target[1:3] + [[int(TaskState.ACCEPT)]])] = tensorfy([base, target[1:]]).to(device)
        for target in available: # tasks available to accept
            tasks['.'.join('.'.join(str(j) for j in i) for i in target[:2] + [[int(TaskState.OPEN)]])] = tensorfy([base, target]).to(device)
        if test:
            print(tasks)
        return tasks # return both joint and task-vectorized observations
    def interpret(observation, test=False):
        '''
        Specific to the Rideshare environment, converts a single agent's observation
        into a task-specific observation vector for individual task processing.
        Returns both the original joint and the new task-vectorized observations.
        '''

        ObsLayer = IntEnum('ObsLayer', zip(['SELF', 'OTHER', 'ACCEPT', 'RIDE', 'OPEN'], range(5))) # observation layers
        TaskState = IntEnum('TaskState', ['OPEN', 'ACCEPT', 'RIDE']) # task states

        base = [] # will contain own location, others' locations, and the numbbers of accepted & riding passengers
        base.append(np.argwhere(np.vectorize(lambda x: len(x) > 0)(observation[ObsLayer.SELF]))[0])
        base.append([loc for loc in np.argwhere(np.vectorize(lambda x: len(x) > 0)(observation[ObsLayer.OTHER])) for other in observation[ObsLayer.OTHER, loc[0], loc[1]]])
        # get info for passengers currently riding with the agent
        riding = [rider for rider in observation[ObsLayer.RIDE, base[0][0], base[0][1]]]
        # get info for accepted passengers awaiting pickup
        accepted = [acc for loc in np.argwhere(np.vectorize(lambda x: len(x) > 0)(observation[ObsLayer.ACCEPT])) for acc in observation[ObsLayer.ACCEPT, loc[0], loc[1]]]
        # get info for passengers that have yet to be accepted by anyone
        available = [avail for loc in np.argwhere(np.vectorize(lambda x: len(x) > 0)(observation[ObsLayer.OPEN])) for avail in observation[ObsLayer.OPEN, loc[0], loc[1]]]
        base.append(len(accepted))
        base.append(len(riding))
        base.append(len(available))
        tasks = {} # combine tasks in to a single hashmap TODO: fix overwriting of same src-dest pairs!
        for target in riding: # tasks en route
            tasks['.'.join('.'.join(str(j) for j in i) for i in target[1:3])] = tensorfy([base, target[1:], TaskState.RIDE]).to(device)
        for target in accepted: # tasks awaiting pickup
            tasks['.'.join('.'.join(str(j) for j in i) for i in target[1:3])] = tensorfy([base, target[1:], TaskState.ACCEPT]).to(device)
        for target in available: # tasks available to accept
            tasks['.'.join('.'.join(str(j) for j in i) for i in target[:2])] = tensorfy([base, target, TaskState.OPEN]).to(device)
        if test:
            print(tasks)
        return tasks # return both joint and task-vectorized observations
    # action converter
    exterpret_decomposed = lambda action, observation, actions=2, Get=IntEnum('TaskInfo', ['STATE','STEP', 'FARE', 'END_Y', 'END_X', 'START_Y', 'START_X']): next(
        [ # converts a single, integer action from a multitask policy to a 5x1 integer array useable by the rideshare domain.
            3, # action indicator
            int(task[-Get.START_X] * grid.w + task[-Get.START_Y]), # start coords to grid
            int(task[-Get.END_X] * grid.w + task[-Get.END_Y]), # end coords to grid
            int(task[-Get.STATE] - 1), # action type is automatic in this environment (if not noöp)
            int(task[-Get.STEP]) # step from task initialization
        ] for task in [list(observation.values())[int(action) // actions].tolist()
            + [float(list(observation.keys())[int(action) // actions].split('.')[-1])]] # local task variable
    ) if action % actions != 0 else [3, -1, -1, 3, -1] # noöp by default
    exterpret = lambda action, observation, actions=4, Get=IntEnum('TaskInfo', ['STATE','STEP', 'FARE', 'END_Y', 'END_X', 'START_Y', 'START_X']): next(
        [ # converts a single, integer action from a multitask policy to a 5x1 integer array useable by the rideshare domain.
            3, # action indicator
            int(task[-Get.START_X] * grid.w + task[-Get.START_Y]), # start coords to grid
            int(task[-Get.END_X] * grid.w + task[-Get.END_Y]), # end coords to grid
            int(task[-Get.STATE] - 1), # action type is automatic in this environment (if not noöp)
            int(task[-Get.STEP]) # step from task initialization
        ] for task in [list(observation.values())[int(action) // actions].tolist()] # local task variable
    ) if int(action) % actions != 0 else [3, -1, -1, 3, -1] # noöp by default
    # conlict manager
    deconflict = lambda actions, NOOP=[3, -1, -1, 3, -1], MISS=[3, -1, -1, -1, -1]: [MISS if action != NOOP and any([action == other for other in actions[:agent]]) else action for agent, action in enumerate(actions)]

    # initialize agents, environment, and results
    print(F'{LINE}\nINITIALIZING...')
    if _agents is None:
        agents = [
            Duellapig(
                observation_dimension=2 * no_agents + 10,
                action_dimension=4,
                latent_dimension=latent_dimension,
                latent_task_dimension=latent_task_dimension,
                layers=layers,
                learning_rate=learning_rate,
                learning_steps=learning_steps,
                discount_factor=discount
            ) for agent in range(no_agents)
        ] if double else [
            Linellapig(
                observation_dimension=2 * no_agents + (10 if not decomposed else 9),
                action_dimension=4 if not decomposed else 2,
                latent_dimension=latent_dimension,
                latent_task_dimension=latent_task_dimension,
                layers=layers,
                learning_rate=learning_rate,
                learning_steps=learning_steps,
                discount_factor=discount
            ) for agent in range(no_agents)
        ]
    else:
        agents = _agents
    if not learning and type(agents[0]) is not Simpellapig:
        agents = [Simpellapig(agent) for agent in agents]
    no_agents = len(agents) # if agents are imported, use the number from the import. - might change later

    env = rideshare(
        num_agents=no_agents, 
        grid_len=grid.l,
        grid_wid=grid.w,
        accept_cost = costs.accept,
        pick_cost = costs.pick,
        move_cost = costs.move,
        noop_cost = costs.miss,
        drop_cost = None, # costs.drop,
        no_pass_cost = costs.all_accepted,
        variable_move_cost = False,
        variable_pick_cost = False,
        no_pass_reward = True
    )

    # print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(torch.cuda.current_device()))

    if learning:
        result_dest = f"ride/{DATE}/{'i' if not clean else 'x'}{iterations}{f'e{episodes}' if episodes > 1 else ''}_{'o' if openness else 'f' if fixed else'c'}{no_tasks}a{no_agents}{'r' if raw_action else ''}"
        store(result_dest, config=config) # store configuration if initialization was successful
    else:
        result_dest = f"ride/{DATE}/e{episodes}_{'o' if openness else 'f' if fixed else 'x' if clean and not raw_action else 'c'}{no_tasks}a{no_agents}{'x' if clean and raw_action else 'r' if _agents is None else f'l{iterations}'}"
    stats = [SimpleNamespace(rewards=[], losses=[], tasks=SimpleNamespace(list=[], count=[])) for agent in agents] if _stats is None else _stats
    offset = len(stats[0].rewards)
    
    # print experiment & environment specifications
    if learning: # learning exp specs != non-learning exp specs
        print('Experiment Specifications:')
        print(f"  {iterations} learning iterations over {episodes}-episode, {steps}-step{' decomposed' if decomposed else ''}{' expert' if clean else ' raw-action' if raw_action else ''} trajectories")
        print(f"  {learning_steps} internal iterations at {learning_rate} LR, discounted {discount}")
        print(f"  {latent_dimension} shared x {latent_task_dimension} task dimensions")
        if plot_partial is not None:
            print(f'  partial plotting every {plot_partial} iterations')
    else:
        print(f"Experiment Specifications:\n  {episodes} non-learning episodes of {steps} steps each{', decomposed' if decomposed else ''}")
    print(f"Environment Specifications:\n  \"Rideshare\" {'OPEN' if openness else f'{no_tasks if not clean else (no_tasks + 1) * no_agents} Tasks'} with {no_agents} Agent(s) on a {grid.l}x{grid.w} Grid\n  Costs: Accept={costs.accept}, Pickup={costs.pick}, Move={costs.move}, Dropoff={None}, AllAccepted={costs.all_accepted}")
    # print(f'Result Destination: {result_dest}')

    # gather tajectories and learn from them
    print(f'{LINE}\nITERATING...')
    if not (g_mode := False):
        for i in range(offset + 1, iterations + 1):
            if learning: # non-learning runs only have one iteration
                print(f'Iteration # {i} / {iterations}')
            trajectories = [[] for agent in agents]
            trajectories_critic = []
            if fixed and not learning: # non-learning runs can have a fixed start
                start = [[[randrange(0, grid.l), randrange(0, grid.w)] for agent in range(no_agents)], env.generateTask(step=0, num_tasks=no_tasks)]
            for episode in range(1, episodes + 1):
                if not learning and ((plot_partial is not None and episode % plot_partial == 0) or episode == episodes):
                    print (f'Episode # {episode} / {episodes}') # learning runs *should* only have one episode per iteration (acc. requested design specs)
                if clean and (learning or (raw_action and not learning)):
                    state, targets = make_clean_trajectory(env, extra_passengers=no_tasks)
                    observation = env.getObsFromState(state)
                    trajectory = [[] for agent in agents]
                    if learning: # agents' queues must be manually filled since they are not being used directly & only specific task are relevant
                        for agent, target in zip(agents, targets):
                            for t in target:
                                if not decomposed:
                                    if t not in agent.queue:
                                        agent.queue.append(t)
                                else:
                                    for subt in [str(subtarget) for subtarget in range(1, 4)]:
                                        if '.'.join([t, subt]) not in agent.queue:
                                            agent.queue.append('.'.join([t, subt]))
                    for step in range(steps):
                        traj=[{'observation': interpret(observation[agent]) if not decomposed else interpret_decomposed(observation[agent])} for agent in range(no_agents)]
                        if all([len(traj[agent]['observation']) == 0 for agent in range(no_agents)]):
                            break
                        for a in range(no_agents):
                            traj[a]['tasks'] = len(traj[a]['observation'])
                            if learning:
                                traj[a]['action'] = pool_targets(traj[a]['observation'], targets[a]) if not decomposed else pool_targets_decomposed(traj[a]['observation'], targets[a])
                            else:
                                traj[a]['action'] = traj[a]['reaction'] = pool_targets(traj[a]['observation'], targets[a]) if not decomposed else pool_targets_decomposed(traj[a]['observation'], targets[a])
                        actions = deconflict([exterpret(traj[agent]['action'].item(), traj[agent]['observation']) if not decomposed else exterpret_decomposed(traj[agent]['action'].item(), traj[agent]['observation']) for agent in range(no_agents)]) # exterpret the raw actions into the environment
                        state, rewards, _ = env.step(len(trajectory), actions, openness) # progress the environment
                        observation = env.getObsFromState(state) # collect the new full observation
                        for agent in range(no_agents):
                            traj[agent]['reward'] = rewards[agent]
                            trajectory[agent].append(SimpleNamespace(**traj[agent]))
                else:
                    if clean and not learning:
                        state, _ = make_clean_trajectory(env, extra_passengers=no_tasks)
                    else:
                        state = env.reset(0, num_passengers=no_tasks, epoch=i if learning else episode) if learning or not fixed else env.reset(0, num_passengers=no_tasks, driver_locations=start[0], passengers_list=start[1])
                    observation = env.getObsFromState(state)
                    trajectory = [[] for agent in agents] # initialize new trajectory
                    for step in range(steps):
                        #print(f'       Step # {step_no + 1} / {steps}')
                        traj = [{'observation': interpret(observation[agent]) if not decomposed else interpret_decomposed(observation[agent])} for agent in range(no_agents)] # interpret the full observation into a dict of task-specific observations
                        if not openness and all([len(traj[agent]['observation']) == 0 for agent in range(no_agents)]): # cuts closed environments short when no tasks remain
                            break
                        for a, agent in enumerate(agents):
                            traj[a]['tasks'] = len(traj[a]['observation'])
                            traj[a]['action'] = agent.forward(traj[a]['observation']) # generate the raw action
                        actions = deconflict([exterpret(traj[agent]['action'].item(), traj[agent]['observation']) if not decomposed else exterpret_decomposed(traj[agent]['action'].item(), traj[agent]['observation']) for agent in range(no_agents)]) # exterpret the raw actions into the environment
                        state, rewards, _ = env.step(len(trajectory), actions, openness) # progress the environment
                        observation = env.getObsFromState(state) # collect the new full observation
                        for agent in range(no_agents):
                            if learning and not raw_action and actions[agent][3] != -1: # match result to actual action unless otherwise specified, keeping misses as the originally attempted action (to appropriately reflect punishments)
                                if decomposed:
                                    traj[agent]['action'] = traj[agent]['action'] - traj[agent]['action'] % 2 + (actions[agent][3] + 1) % 2
                                else:
                                    traj[agent]['action'] = traj[agent]['action'] - traj[agent]['action'] % 4 + (actions[agent][3] + 1) % 4
                            elif not learning:
                                if decomposed:
                                    traj[agent]['reaction'] =  traj[agent]['action'] - traj[agent]['action'] % 2 + (1 if (actions[agent][3] + 1) % 4 != 0 else 0)
                                else:
                                    traj[agent]['reaction'] =  traj[agent]['action'] - traj[agent]['action'] % 4 + (actions[agent][3] + 1) % 4
                            traj[agent]['reward'] = rewards[agent] # record the immediate reward
                            trajectory[agent].append(SimpleNamespace(**traj[agent])) # add step to current trajectory
                for agent in range(no_agents): # add completed trajectory to actor & critic trajectory lists
                    trajectories[agent].append(trajectory[agent])
                    trajectories_critic.append(trajectory[agent])
            if fixed and learning: # use the same trajectory set for all learning if fixed
                queue = [agent.queue for agent in agents]
                print('Fixing Trajectories...')
                for j in range(1, iterations + 1):
                    print(f'Fixed Iteration # {j} / {iterations}')
                    for a, agent in enumerate(agents):
                        agent.queue = queue[a]
                        stats[a].tasks.list.append([[step.tasks for step in episode] for episode in trajectories[a]])
                        stats[a].tasks.count.append(len(agent.queue))
                        stats[a].losses.append(agent.optimize(trajectories[a], critique=trajectories_critic, log_task=notify_task, plot_tasks=plot_tasks, prefix=f'Agent # {a + 1} / {no_agents} '))
                        stats[a].rewards.append(torch.stack([torch.tensor([step.reward for step in episode], dtype=float, device=device).mean() for episode in trajectories[a]]).mean())
                break
            elif learning: # store relevant data for future plotting & do a learning iteration for each agent
                for a, agent in enumerate(agents):
                    stats[a].tasks.list.append([[step.tasks for step in episode] for episode in trajectories[a]])
                    stats[a].tasks.count.append(len(agent.queue))
                    stats[a].losses.append(agent.optimize(trajectories[a], critique=trajectories_critic, log_task=notify_task, plot_tasks=plot_tasks, prefix=f'Agent # {a + 1} / {no_agents} '))
                    stats[a].rewards.append(torch.stack([torch.tensor([step.reward for step in episode], dtype=float, device=device).mean() for episode in trajectories[a]]).mean())
            else: # store and plot trajectory data from the non-learning run
                print(F'{LINE}\nPLOTTING...')
                if plot_tasks: # disabled by default for taking up too much space.
                    store(result_dest, losses=trajectories)
                else:
                    store(result_dest)
                for a in range(no_agents):
                    if not decomposed:
                        plot_runs(trajectories[a], result_dest, f'{result_dest[15:]}_{a}')
                    else:
                        plot_runs_decomposed(trajectories[a], result_dest, f'{result_dest[15:]}_{a}')
                return result_dest

            # store partial results
            if plot_partial is not None and i % plot_partial == 0:
                partial_policies = [Simpellapig(agent) for agent in agents]
                config['learning'] = False # reflect inability for learning resumption on these policies
                config['iterations'] = i # record iterations so far rather than unreached total
                store(f'{result_dest}/i{i}', config=config, policies=partial_policies, losses=stats)
    else:
        stats = gayathri_mode(env=env,
            agents=agents,
            interpret=interpret_decomposed if decomposed else interpret,
            deconflict=deconflict,
            exterpret=exterpret_decomposed if decomposed else exterpret,
            plot=plot_runs_decomposed if decomposed else plot_runs,
            stats=stats,
            iterations=iterations if learning else episodes,
            learning=learning,
            result_dest=result_dest,
            notif=plot_partial)
        if not learning:
            return result_dest
    # plot final results
    print(F'{LINE}\nPLOTTING...')
    store(result_dest, policies=agents, losses=stats)
    for a in range(no_agents):
        plot_full(stats[a], result_dest, f'{result_dest[15:]}_{a}')
    return result_dest # used for resume chaining

def make_clean_trajectory(env:rideshare, extra_passengers=1):
    '''generates a pairs of pooling-optimal tasks for each agent in the environment, and returns an intialized environment and second tasks pairs by agent for use with pool_targets.'''
    
    # intialize random tasks for each agent
    tasks_full = env.generateTask(step=0, num_tasks=env.num_agents) # full task info
    tasks = [['.'.join([str(tasks_full[agent][0][0]), str(tasks_full[agent][0][1]), str(tasks_full[agent][1][0]), str(tasks_full[agent][1][1])])] for agent, _ in enumerate(tasks_full)] # per-agent task assignment
    
    # add optimally-poolable passengers for each agent
    for agent in tasks:
        for seat in range(extra_passengers):
            task_prime = [int(no) for no in agent[-1].split('.')] # for optimality, always build from the last assigned task
            task_path = [task_prime[0], task_prime[3]] if randrange(2) < 1 else [task_prime[2], task_prime[1]] # which corner in travel path
            # subtask = in_path_pool(task_prime, task_path) # TODO: add generator argument... maybe
            subtask = break_path_pool(task_prime)
            if all(subtask not in a for a in tasks): # ignore duplicates
                agent.append(subtask)
                task_ints = [int(t) for t in agent[-1].split('.')]
                tasks_full.append([(task_ints[0], task_ints[1]), (task_ints[2], task_ints[3]), 3 * max(3, abs(task_ints[0] - task_ints[2]) + abs(task_ints[1] - task_ints[3]) + randrange(-1, 2, 1)), 0])
    
    return env.reset(step=0, driver_locations=[tasks_full[agent][0] for agent in range(env.num_agents)], passengers_list=tasks_full), tasks

def pool_targets(observation, targets):
    '''
        runs through accepting, picking up, and dropping off a list of targets for efficent pooling in the rideshare domain.
        (returns a marmotellapig action, see train_ride.exterpret to see how this is input into the environment)
    '''

    # accept all the targets
    if target := next((t for t in targets if t in observation and observation[t][-1] == 1), None):
        return torch.tensor(list(observation.keys()).index(target) * 4 + 1, dtype=float, device=device)
    # pickup all the targets
    if target := next((t for t in targets if t in observation and observation[t][-1] == 2), None):
        return torch.tensor(list(observation.keys()).index(target) * 4 + 2, dtype=float, device=device)
    # dropoff all the targets
    if target := next((t for t in targets[::-1] if t in observation and observation[t][-1] == 3), None):
        return torch.tensor(list(observation.keys()).index(target) * 4 + 3, dtype=float, device=device)
    # noop if all targets completed
    return torch.tensor(0, dtype=float, device=device)

def pool_targets_decomposed(observation, targets):
    '''
        runs through accepting, picking up, and dropping off a list of targets for efficent pooling in the rideshare domain.
        (returns a marmotellapig action, see train_ride.exterpret to see how this is input into the environment)
    '''

    observation = [('.'.join(obs.split('.')[:-1]), int(obs.split('.')[-1])) for obs in observation]
    # accept all the targets
    if target := next((t for t in targets if (t, 1) in observation), None):
        return torch.tensor(observation.index((target, 1)) * 2 + 1, dtype=float, device=device)
    # pickup all the targets
    if target := next((t for t in targets if (t, 2) in observation), None):
        return torch.tensor(observation.index((target, 2)) * 2 + 1, dtype=float, device=device)
    # dropoff all the targets
    if target := next((t for t in targets[::-1] if (t, 3) in observation), None):
        return torch.tensor(observation.index((target, 3)) * 2 + 1, dtype=float, device=device)
    # noop if all targets completed
    return torch.tensor(0, dtype=float, device=device)

def in_path_pool(prime, path, cat=None, first=True):
    '''Generates a poolable task along the provided corner-based path.'''
    if cat is None: # if not provided a partial location for task placement, pick at random.
        cat = randrange(4)
    match cat:
        case 0: # first leg in-path pickup
            if first and path[0] == prime[0]:
                return in_path_pool(prime, path, 1, False)
            try: 
                return '.'.join([str(randrange(path[0], prime[0])) if path[0] != prime[0] else str(prime[0]), str(path[1]), str(prime[2]), str(prime[3])])
            except ValueError:
                return '.'.join([str(randrange(prime[0], path[0])), str(path[1]), str(prime[2]), str(prime[3])])
        case 1: # second leg in-path pickup
            if first and path[1] == prime[1]:
                return in_path_pool(prime, path, 0, False)
            try: 
                return '.'.join([str(path[0]), str(randrange(path[1], prime[1])) if path[1] != prime[1] else str(prime[1]), str(prime[2]), str(prime[3])])
            except ValueError:
                return '.'.join([str(path[0]), str(randrange(prime[1], path[1])), str(prime[2]), str(prime[3])])
        case 2: # first leg in-path dropoff
            if first and path[0] == prime[2]:
                return in_path_pool(prime, path, 3, False)
            try:
                return '.'.join([str(prime[0]), str(prime[1]), str(randrange(path[0], prime[2])) if path[0] != prime[2] else str(prime[2]), str(path[1])])
            except ValueError:
                return '.'.join([str(prime[0]), str(prime[1]), str(randrange(prime[2], path[0])), str(path[1])])
        case 3: # second leg in-path dropoff
            if first and path[1] == prime[3]:
                return in_path_pool(prime, path, 2, False)
            try: 
                return '.'.join([str(prime[0]), str(prime[1]), str(path[0]), str(randrange(path[1], prime[3])) if path[1] != prime[3] else str(prime[3])])
            except ValueError:
                return '.'.join([str(prime[0]), str(prime[1]), str(path[0]), str(randrange(prime[3], path[1]))])

def break_path_pool(prime, cat=None):
    '''Generates a poolable task within the prime task's travel area that shares either a pickup or dropoff point.'''
    if cat is None: # if not provided a partial location for task placement, pick at random.
        cat = randrange(2)
    if prime[2] < prime[0]:
        prime[0], prime[2] = prime[2], prime[0]
    if prime[3] < prime[1]:
        prime[1], prime[3] = prime[3], prime[1]
    match cat:
        case 0: # break-path pickup
            x = randrange(prime[0] + 1, prime[2]) if prime[2] - prime[0] > 1 else prime[0]
            y = randrange(prime[1] + 1, prime[3]) if prime[3] - prime[1] > 1 else prime[1]
            if (x, y) == (prime[0], prime[1]):
                if prime[0] != prime[2]:
                    x = prime[2]
                else:
                    y = prime[3]
            return '.'.join([
                str(x),
                str(y),
                str(prime[2]),
                str(prime[3])
            ])
        case 1: # break-path dropoff
            x = randrange(prime[0], prime[2] - 1) if prime[2] - prime[0] > 1 else prime[2]
            y = randrange(prime[1], prime[3] - 1) if prime[3] - prime[1] > 1 else prime[3]
            if (x, y) == (prime[2], prime[3]):
                if prime[0] != prime[2]:
                    x = prime[0]
                else:
                    y = prime[1]
            return '.'.join([
                str(prime[0]),
                str(prime[1]),
                str(x),
                str(y)
            ])

def gayathri_mode(env: rideshare, agents, interpret, exterpret, deconflict, plot, stats, iterations, learning=True, result_dest=None, notif=100):
    if notif is None:
        notif = iterations
    tasks = [[(4, 1), (0, 8), 32, 0], [(0, 1), (5, 4), 26, 0], [(4, 8), (7, 2), 28, 0]]
    task_queue = [[(3, 7), (9, 0), 39, 2], [(3, 3), (2, 9), 21, 4], [(7, 4), (8, 7), 17, 7]]
    driver_locations = [[6, 2], [1, 5], [5, 6]]
    action_queues = [
        ['4.1.0.8.1', '4.1.0.8.2', '4.1.0.8.2', '4.1.0.8.2', '4.1.0.8.2', '3.3.2.9.1', '3.3.2.9.2', '3.3.2.9.2', '3.3.2.9.2', '3.3.2.9.2', '3.3.2.9.3', '3.3.2.9.3', '3.3.2.9.3', '3.3.2.9.3', '3.3.2.9.3', '3.3.2.9.3', '3.3.2.9.3', '3.3.2.9.3', '4.1.0.8.3', '4.1.0.8.3', '4.1.0.8.3', '4.1.0.8.3'],
        ['0.1.5.4.1', '0.1.5.4.2', '0.1.5.4.2', '0.1.5.4.2', '0.1.5.4.2', '0.1.5.4.2', '0.1.5.4.2', '0.1.5.4.3', '0.1.5.4.3', '7.4.8.7.1', '0.1.5.4.3', '0.1.5.4.3', '0.1.5.4.3', '0.1.5.4.3', '0.1.5.4.3', '0.1.5.4.3', '0.1.5.4.3', '7.4.8.7.2', '7.4.8.7.2', '7.4.8.7.2', '7.4.8.7.3', '7.4.8.7.3', '7.4.8.7.3', '7.4.8.7.3', '7.4.8.7.3'],
        ['4.8.7.2.1', '4.8.7.2.2', '4.8.7.2.2', '3.7.9.0.1', '4.8.7.2.2', '4.8.7.2.2', '3.7.9.0.2', '3.7.9.0.2', '3.7.9.0.2', '4.8.7.2.3', '4.8.7.2.3', '4.8.7.2.3', '4.8.7.2.3', '4.8.7.2.3', '4.8.7.2.3', '4.8.7.2.3', '4.8.7.2.3', '4.8.7.2.3', '4.8.7.2.3', '3.7.9.0.3', '3.7.9.0.3', '3.7.9.0.3', '3.7.9.0.3', '3.7.9.0.3']
    ]
    if not learning:
        trajectories = [[] for agent in agents]
    for i in range(1, iterations + 1 if not learning else 2):
        trajectory = [[] for agent in agents] # initialize new trajectory
        state = env.reset(step=0, driver_locations=driver_locations, passengers_list=tasks)
        for step in range(25 if learning else 100):
            if task := next((task for task in task_queue if task[-1] == step), None):
                state[3, task[0][0], task[0][1]].append(task)
            observation = env.getObsFromState(state)
            traj=[{'observation': interpret(observation[agent])} for agent in range(len(agents))]
            if learning:
                for agent in range(len(agents)):
                    traj[agent]['tasks'] = len(traj[agent]['observation'])
                    traj[agent]['action'] = traj[agent]['reaction'] = torch.tensor(list(traj[agent]['observation'].keys()).index(action_queues[agent][step]) * 2 + 1, dtype=float, device=device) if len(action_queues[agent]) > step else torch.tensor(0, dtype=float, device=device)
            else:
                for a, agent in enumerate(agents):
                    traj[a]['tasks'] = len(traj[a]['observation'])
                    # traj[a]['action'] = traj[a]['reaction'] = torch.tensor(list(traj[a]['observation'].keys()).index(action_queues[a][step]) * 2 + 1, dtype=float, device=device) if len(action_queues[a]) > step else torch.tensor(0, dtype=float, device=device)
                    traj[a]['action'] = traj[a]['reaction'] = agent.forward(traj[a]['observation'])
            actions = deconflict([exterpret(traj[agent]['action'].item(), traj[agent]['observation']) for agent in range(len(agents))]) # exterpret the raw actions into the environment
            state, rewards, _ = env.step(len(trajectory), actions, False) # progress the environment
            for agent in range(len(agents)):
                traj[agent]['reward'] = rewards[agent]
                trajectory[agent].append(SimpleNamespace(**traj[agent]))
        if not learning:
            if i % notif == 0:
                print(f'Episode # {i} / {iterations}')
            for agent in range(len(agents)):
                trajectories[agent].append(trajectory[agent])
    if learning:
        for i in range(1, iterations + 1):
            print(f'Iteration # {i} / {iterations}')
            for a, agent in enumerate(agents):
                agent.queue = list(set(action for queue in action_queues for action in queue))
                stats[a].tasks.list.append([[step.tasks for step in episode] for episode in [trajectory[a]]])
                stats[a].tasks.count.append(len(agent.queue))
                stats[a].losses.append(agent.optimize([trajectory[a]], log_task=False, plot_tasks=False, prefix=f'Agent # {a + 1} / {len(agents)} '))
                stats[a].rewards.append(torch.stack([torch.tensor([step.reward for step in episode], dtype=float, device=device).mean() for episode in [trajectory[a]]]).mean())
    else:
        print(F'{LINE}\nPLOTTING...')
        store(result_dest)
        for agent in range(len(agents)):
            plot(trajectories[agent], result_dest, f'{result_dest[15:]}_{agent}')
        return result_dest
    
    return stats

def resume(src:str, extension:int=None, partial:float=None):

    # load training data to resume
    print(f'{LINE}\nRETRIEVING PREVIOUS TRAINING...')
    args, agents, stats = load(src)
    if args is None: raise AttributeError(f'No relevant configuration data found for \"{src}\".')
    print('Found configuration file.')
    if agents is None and stats is not None: raise AttributeError(f'No relevant policy data to continue training on \"{src}\".')
    elif agents is not None: print('Found policy file(s).')
    
    # apply desired changes to training upon resumption
    if extension is not None:
        args.iterations = extension if extension > 0 else ((len(stats[0].rewards) if hasattr(stats, '__iter__') else len(stats.rewards)) if stats is not None else 0) - extension
    if partial is not None:
        args.plot_partial = partial

    # set trainer by results subdirectory
    trainer = src.split('/')
    trainer = trainer[0] if trainer[0] != 'results' else trainer[1]

    # attempt to set trainer by argument list if trainer subdirectory does not match
    if trainer not in ['ride', 'cart']:
        match len(vars(args)):
            case train_ride.__code__.co_argcount:
                trainer = 'ride'
            case train_cart.__code__.co_argcount:
                trainer = 'cart'
            case _:
                print('Configuration type unsupported.')
                return None

    # resume training
    if trainer == 'ride':
        if stats is None or len(stats[0].rewards) < args.iterations:
            get = lambda arg: getattr(args, arg, train_ride.__defaults__[train_ride.__code__.co_varnames.index(arg)])
            return train_ride(iterations=args.iterations, episodes=args.episodes, steps=args.steps,
                double=get('double'), latent_dimension=args.latent_dimension, latent_task_dimension=args.latent_task_dimension, layers=get('layers'),
                learning_rate=args.learning_rate, learning_steps=args.learning_steps, discount=args.discount,
                notify_task=args.notify_task, fixed=args.fixed, plot_tasks=args.plot_tasks, plot_partial=args.plot_partial,
                no_agents=args.no_agents, grid=args.grid, costs=args.costs, no_tasks=args.no_tasks, openness=args.openness,
                learning=True, raw_action=args.raw_action, decomposed=get('decomposed'), clean=get('clean'), _agents=agents, _stats=stats)
        else:
            print(f'Desired training ({args.iterations} iterations) already complete!')
            return src
    elif trainer == 'cart':
        if stats is None or len(stats[0].rewards) < args.iterations:
            get = lambda arg: getattr(args, arg, train_cart.__defaults__[train_cart.__code__.co_varnames.index(arg)])
            return train_cart(
                iterations=args.iterations, episodes=args.episodes, steps=args.steps,
                double=get('double'),
                latent_dimension=args.latent_dimension, latent_task_dimension=args.latent_task_dimension, layers=get('layers'),
                learning_rate=args.learning_rate, learning_steps=args.learning_steps, discount=args.discount,
                tasks=args.tasks, notify_task=args.notify_task, fixed=args.fixed, plot_partial=args.plot_partial,
                _agent=agents[0] if agents is not None else None,
                _stats=stats[0] if stats is not None else None)
        else:
            print(f'Desired training ({args.iterations} iterations) already complete!')
            return src

def run_ride(src=None, episodes=1000, steps=100, grid=SimpleNamespace(l=5, w=5), costs=SimpleNamespace(accept=-0.5, pick = -0.1, move = -0.1, miss = -1, drop = None, all_accepted=2),
    no_agents=3, no_tasks=1, decomp=False, clean=False, expert=False, openness=False, store_trajectories=False, fixed=False, notify=True):

    # load policy data
    print(f'{LINE}\nDEFINING POLICIES...')
    if src is not None:
        config, agents, _ = load(src)
        if agents is None:
            raise AttributeError(f'No relevant policy data found in \"{src}\"')
        else:
            i = getattr(config, 'iterations', 1)
            decomp = getattr(config, 'decomposed', False)
            expert = False
            print(f'Found {len(agents)} policies.')
    else:
        agents = None
        print(f'Using {no_agents} random policies')

    # generate and plot trajectories
    train_ride(iterations=i if src is not None else 1, episodes=episodes, steps=steps, plot_partial=notify, grid=grid, costs=costs, no_tasks=no_tasks, openness=openness,
        learning=False, clean=clean if not expert else expert, raw_action=expert, decomposed=decomp, _agents=agents, no_agents=no_agents, plot_tasks=store_trajectories, fixed=fixed)
