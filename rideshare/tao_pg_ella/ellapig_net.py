#----------------------------------------------------------------------------------------------------------------------#
# Title: Marmotellapig Networks
# Description: 
#   This file contains the marmotellapig & actor-critic base-learner networks implementing Multi-Agent, Open-Task PG-ELLA using actor-critic methodology.
#   Based on the PG-ELLA as described in http://proceedings.mlr.press/v32/ammar14.pdf
#   Based on the implementation found in https://github.com/cdcsai/Online_Multi_Task_Learning (originally, this implementation has turned out to be greatly flawed and a big redo was needed.)
# Author: Matthew Sentell
# Version: 1.00.01
#----------------------------------------------------------------------------------------------------------------------#

from enum import IntEnum
from types import SimpleNamespace
from ellapig_data import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Use GPU if available.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Used later in actor-Critic implementation.
NetType = IntEnum('NetType', ['main', 'target'])

class TaskLatentSpaces(nn.Module):
    """Class for storing task-specific actor & critic latent spaces (TLS) and associated data. Requires TLS dimension for initialization."""
    def __init__(self, latent_task_dimension:int):
        super(TaskLatentSpaces, self).__init__()
        # Initialize policy TLS.
        self.policy_latent_space = torch.rand(latent_task_dimension, dtype=float, device=device)
        self.policy_optimal = None
        self.policy_hessian = None
        # Initialize critic TLS.
        self.critic_latent_space = torch.rand(latent_task_dimension, dtype=float, device=device)
        self.critic_optimal = None
        self.critic_hessian = None

class TaskLatentSpace(nn.Module):
    """Class for storing a task-specific latent space (TLS) and associated data. Requires TLS dimension for initialization. (Deprecated)"""
    def __init__(self, latent_task_dimension:int):
        super(TaskLatentSpace, self).__init__()
        # Initialize TLS.
        self.latent_space = torch.rand(latent_task_dimension, dtype=float, device=device)
        self.critic = None # Task-Specific Critic to be stored in assocation.
        self.optimal = None
        self.hessian = None

class PolicyNetwork(nn.Module):
    """
    Class defining a PyTorch Linear network with the provided dimensions.

    Dimensions:
        Input Dimension: The expected size inputs.
        Latent Dimension: The size of hidden layers in the network.
        Layers: The number of extra (after the first) hidden layers in the network.
        Output Dimension: The size to output after forwarding input through the network.
    """
    def __init__(self, input_dimension:int, latent_dimension:int, layers:int, output_dimension:int, params:torch.Tensor):
        super(PolicyNetwork, self).__init__()
        if True: # layers > 0: # Old check that shouldn't be needed, but see else comment.
            # Initialize policy layers.
            self.subpol = nn.ModuleList([nn.Linear(input_dimension, latent_dimension, dtype=float, bias=False, device=device)]).to(device) # IN to HL 1
            for layer in range(layers): # Every HL after 1
                self.subpol.append(nn.Linear(latent_dimension, latent_dimension, dtype=float, bias=False, device=device))
            self.subpol.append(nn.Linear(latent_dimension, output_dimension, dtype=float, bias=False, device=device)) # HL final to OUT
            # Define policy weights from provided parameters.
            if params is not None:
                self.subpol[0].weight = nn.Parameter(params[:input_dimension * latent_dimension].view(latent_dimension, input_dimension).to(device)) # IN to HL 1
                for l, layer in enumerate(self.subpol[1:-1]): # Every HL after 1
                    layer.weight = nn.Parameter(params[input_dimension * latent_dimension + l * pow(latent_dimension, 2):input_dimension * latent_dimension + (l + 1) * pow(latent_dimension, 2)].view(latent_dimension, latent_dimension).to(device))
                self.subpol[-1].weight = nn.Parameter(params[input_dimension * latent_dimension + layers * pow(latent_dimension, 2):].view(output_dimension, latent_dimension).to(device)) # HL final to OUT
        else: # Last time I deleted this section, the network would not work correctly. It shouldn't be used, so IDK why.
            self.observational = nn.Linear(input_dimension, latent_dimension, dtype=float, bias=False)
            self.actional = nn.Linear(latent_dimension, output_dimension, dtype=float, bias=False)
            if params is not None:
                self.observational.weight = nn.Parameter(params[:input_dimension * latent_dimension].view(latent_dimension, input_dimension))
                self.actional.weight = nn.Parameter(params[input_dimension * latent_dimension:].view(output_dimension, latent_dimension))
            self.subpol = None
    
    def forward(self, input):
        """Function for passing input through the network."""
        if self.subpol is not None: # This should always be True now, but...
            output = self.subpol[0](input) # IN to HL 1
            for layer in self.subpol[1:]: # HL 1 to HL ? to OUT
                output = layer(F.relu(output))
        else:
            output = self.actional(F.relu(self.observational(input)))
        return output

    def deparam(self):
        """Function to return the policy parameters as a single, flat vector of weights."""
        if self.subpol is not None: # This should always be True now, but...
            return torch.cat([layer.weight.flatten() for layer in self.subpol])
        else:
            return torch.cat([self.observational.weight.flatten(), self.actional.weight.flatten()])

class Duellapig():
    """Class for Marmotellapig learning and operation with both actors and critics. (Deprecated)"""
    def __init__(self, observation_dimension, latent_dimension, action_dimension, latent_task_dimension, layers=0, learning_rate=1e-3, learning_steps=256, discount_factor=0.9):
        super(Duellapig, self).__init__()

        # store dimensional variables
        self.observation_dimension = observation_dimension # task features, len(t)
        self.latent_dimension = latent_dimension # hidden dimension
        self.layers = layers
        self.action_dimension = action_dimension # actions per task
        self.policy_dimension = observation_dimension * latent_dimension + action_dimension * latent_dimension + layers * pow(latent_dimension, 2) # total actor policy parameters, len(θ(t)) = d(π)
        self.critic_dimension = observation_dimension * latent_dimension + latent_dimension + layers * pow(latent_dimension, 2) # total critic policy parameters, len(θ(t)) = d(c)
        self.latent_task_dimension = latent_task_dimension # task-spec. vector lengths, len(s(t)) = k

        # store learning variables
        self.learning_rate = learning_rate
        self.learning_steps = learning_steps
        self.discount = discount_factor
        self.queue = [] # pre-preparable queue of tasks to learn in optimization
        self.tasks = {} # task-specific vectors

        # initialize shared latent spaces
        self.policy_latent_space = torch.zeros(self.policy_dimension, self.latent_task_dimension, dtype=float, device=device)
        self.critic_latent_space = torch.zeros(self.critic_dimension, self.latent_task_dimension, dtype=float, device=device)

        # initialize incremental partial shared latent space elements
        self.policy_latent_space_numerator = torch.zeros(self.policy_dimension * latent_task_dimension, dtype=float, device=device)
        self.policy_latent_space_denominator = torch.zeros(self.policy_dimension * latent_task_dimension, self.policy_dimension * latent_task_dimension, dtype=float, device=device)
        self.critic_latent_space_numerator = torch.zeros(self.critic_dimension * latent_task_dimension, dtype=float, device=device)
        self.critic_latent_space_denominator = torch.zeros(self.critic_dimension * latent_task_dimension, self.critic_dimension * latent_task_dimension, dtype=float, device=device)

    def optimize(self, trajectories, critique=None, log_task=False, plot_tasks=None, avg_losses=True, prefix=''):
        '''
        Updates latent spaces provided a set of trajectories.

        Parameters: 
            trajectories (list(Trajectory) of lists(Episode) of Step(observation, action, reward)): Episodic observation-action-reward vectors for actor learning.
            critque (list(Trajectory) of lists(Episode) of Step(observation, action, reward)): Episodic observation-action-reward vectors for critic learning.
            test (bool[optional]): Test marker. If True, logs values as they are generated.
            log_task (bool[optional]): Notification marker. If True, displays a notification for each new task started.
            plot_task (str[optional]): Notification marker. If not None, creates a chart of internal losses for each task, overwriting single file if specified.
            avg_losses (bool[optional]): Record marker. If True, displays a notification for each new task started.
            prefix (str[optional]): Notification Marker. Text used to preface task notifications.
        '''

        # local functions
        denominatize = lambda taskspace, critic=False: torch.kron(torch.outer(taskspace.policy_latent_space, taskspace.policy_latent_space), taskspace.policy_hessian) if not critic else torch.kron(torch.outer(taskspace.critic_latent_space, taskspace.critic_latent_space), taskspace.critic_hessian)
        numeratize = lambda taskspace, critic=False: torch.kron(taskspace.policy_latent_space, torch.matmul(taskspace.policy_optimal, taskspace.policy_hessian)) if not critic else torch.kron(taskspace.critic_latent_space, torch.matmul(taskspace.critic_optimal, taskspace.critic_hessian))

        # prepare local variables
        losses = {}
        if len(self.queue) == 0:
            for episode in trajectories: 
                for step in episode:
                    for task in step.observation.keys():
                        if task not in self.queue:
                            self.queue.append(task)

        # optimize shared and local latent spaces for each observed task
        for task in self.queue:
            if log_task:
                print(f'{prefix}task # {self.queue.index(task) + 1} / {len(self.queue)} \"{task}\"')

            # check that provided trajectories and critique actually contain this task
            if (local_trajectories := self.filter(trajectories, task)) == []:
                continue
            elif critique is None or (local_critique := self.filter(critique, task)) == []: # if no critique was provided, then this check is redundant
                local_critique = local_trajectories

            # get task data
            if local := self.tasks.get(task, None): # if task exists, store its existing data locally and remove its influence on the shared latent space before updating
                self.policy_latent_space_denominator -= denominatize(local)
                self.policy_latent_space_numerator -= numeratize(local)
                self.critic_latent_space_denominator -= denominatize(local, True)
                self.critic_latent_space_numerator -= numeratize(local, True)
            else: # if task is new, initialize local data
                self.tasks[task] = local = TaskLatentSpaces(self.latent_task_dimension)
            
            # gather and evaluate trajectories from the current policy
            policy = PolicyNetwork(
                input_dimension=self.observation_dimension,
                latent_dimension=self.latent_dimension,
                layers=self.layers,
                output_dimension=self.action_dimension,
                params=torch.matmul(self.policy_latent_space.detach(), local.policy_latent_space.detach()).flatten()
            )
            critic = PolicyNetwork(
                input_dimension=self.observation_dimension,
                latent_dimension=self.latent_dimension,
                layers=self.layers,
                output_dimension=1,
                params=torch.matmul(self.critic_latent_space.detach(), local.critic_latent_space.detach()).flatten()
            )
            local.policy_optimal, local.policy_hessian, local.critic_optimal, local.critic_hessian, internal_losses = self.evaluate(local_trajectories, local_critique, policy, critic, task)

            # reinitialize shared latent space zero-columns
            for col in range(self.policy_latent_space.shape[1]):
                if torch.count_nonzero(self.policy_latent_space.T[col]) == 0:
                    self.policy_latent_space.T[col] = torch.rand(self.policy_latent_space.shape[0], dtype=float, device=device)
            for col in range(self.critic_latent_space.shape[1]):
                if torch.count_nonzero(self.critic_latent_space.T[col]) == 0:
                    self.critic_latent_space.T[col] = torch.rand(self.critic_latent_space.shape[0], dtype=float, device=device)

            # update the local task-specific latent spaces
            lose = lambda L, s, a, H : 0.5 * torch.linalg.norm(s, ord=1) + torch.matmul(torch.matmul(a - torch.matmul(L, s), H), a - torch.matmul(L, s))
            new_policy_latent_space, new_critic_latent_space = local.policy_latent_space.detach().clone(), local.critic_latent_space.detach().clone()
            new_policy_latent_space.requires_grad_()
            new_critic_latent_space.requires_grad_()
            policy_optimizer = optim.Adam([new_policy_latent_space], lr=self.learning_rate)
            critic_optimizer = optim.Adam([new_critic_latent_space], lr=self.learning_rate)
            local_losses = SimpleNamespace(policy=internal_losses.policy, critic=internal_losses.critic, latent_policy=[], latent_critic=[])
            for i in range (self.learning_steps):
                policy_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                policy_loss = lose(L=self.policy_latent_space, s=new_policy_latent_space, a=local.policy_optimal, H=local.policy_hessian)
                critic_loss = lose(L=self.critic_latent_space, s=new_critic_latent_space, a=local.critic_optimal, H=local.critic_hessian)
                local_losses.latent_policy.append(policy_loss.item())
                local_losses.latent_critic.append(critic_loss.item())
                policy_loss.backward()
                critic_loss.backward()
                policy_optimizer.step()
                critic_optimizer.step()
            losses[task] = local_losses
            new_policy_latent_space.requires_grad_(False)
            new_critic_latent_space.requires_grad_(False)
            local.policy_latent_space = nn.Parameter(new_policy_latent_space.detach()).requires_grad_(False)
            local.critic_latent_space = nn.Parameter(new_critic_latent_space.detach()).requires_grad_(False)

            # update the partial shared latent spaces
            self.policy_latent_space_denominator += denominatize(local)
            self.policy_latent_space_numerator += numeratize(local)
            self.critic_latent_space_denominator += denominatize(local, True)
            self.critic_latent_space_numerator += numeratize(local, True)

            # store changes
            self.tasks[task] = local

        # clear the queue
        self.queue = []

        # update the shared latent spaces
        self.policy_latent_space.weight = torch.matmul(torch.inverse(torch.div(self.policy_latent_space_denominator, len(self.tasks)) + torch.eye(self.policy_dimension * self.latent_task_dimension, self.policy_dimension * self.latent_task_dimension, device=device) * 0.1), torch.div(self.policy_latent_space_numerator, len(self.tasks))).view(self.policy_dimension, self.latent_task_dimension)
        self.critic_latent_space.weight = torch.matmul(torch.inverse(torch.div(self.critic_latent_space_denominator, len(self.tasks)) + torch.eye(self.critic_dimension * self.latent_task_dimension, self.critic_dimension * self.latent_task_dimension, device=device) * 0.1), torch.div(self.critic_latent_space_numerator, len(self.tasks))).view(self.critic_dimension, self.latent_task_dimension)

        # average losses across taks and return them
        if avg_losses:
            average = lambda cat: torch.stack([torch.tensor(vars(loss)[cat], dtype=float, device=device) for loss in losses.values()]).mean(dim=-2)
            losses = SimpleNamespace(
                policy = average('policy'),
                critic = average('critic'),
                latent_policy = average('latent_policy'),
                latent_critic = average('latent_critic')
            )
        return losses

    def evaluate(self, trajectories, critique, policy=None, critic=None, task=None, plot_evaluations=False):
        '''
        Evaluate the current policy over a set of trajectories.

        Parameters: 
            trajectories (list(Trajectory) of lists(Episode) of Step(observation, action, reward)): Episodic observation-action-reward vectors for actor learning.
            critque (list(Trajectory) of lists(Episode) of Step(observation, action, reward)): Episodic observation-action-reward vectors for critic learning.
            policy (nn.Linear?[optional]): actor policy parameters to evaluate. If present, initializes the actor to match.
            ciritc (nn.Linear?[optional]): critic policy parameters to evaluate. If present, initializes the critic to match.
            task (str[optional]): Task id indicator. (unused)
            test (bool[optional]): Test marker. (unused)
            task (bool[optional]): Task id indicator. (unused)
        '''

        base_learner = ACagent(
            observation_dimension=self.observation_dimension,
            latent_dimension=self.latent_dimension,
            layers=self.layers,
            action_dimension=self.action_dimension,
            learning_rate=self.learning_rate,
            discount=self.discount,
            actor=policy,
            critic=critic
        )
        losses = SimpleNamespace(policy=[], critic=[])

        # use the base learner optimization to get the local optimal
        for i  in range(self.learning_steps):
            loss_policy, loss_critic = base_learner.optimize(trajectories, critique)
            losses.policy.append(loss_policy.item())
            losses.critic.append(loss_critic.item())
        base_learner.requires_grad_(False)
        if plot_evaluations:
            plot_base(losses, result_id=task)
        
        # get hessian from the base learner's local optimal 
        policy_hessian, critic_hessian = base_learner.hessian(trajectories, critique)

        # returns: local optimal, actor hessian gradient, local critical optimal, critic hessian gradient, & internal losses
        return base_learner.actor.params(), policy_hessian, base_learner.critic.params(), critic_hessian, losses

    def filter(self, trajectories, task):
        '''
        Returns a task-specificized trajectories list.

        Prameters:
            trajectories (list of lists of SimpleNameSpaces(observation, action, reward)): Episodic observation-action-reward vectors.
            task (key): Task key to filter for.
        '''
        filtered_trajectories = []
        for episode in trajectories:
            filtered_episode = []
            for step in episode:
                if task in step.observation:
                    filtered_episode.append(SimpleNamespace( # convert full step into task-specific step
                        observation = step.observation[task], # get only the task-relevant observation
                        action = step.action.detach().clone() % self.action_dimension if step.action.item() // self.action_dimension == list(step.observation.keys()).index(task) else torch.tensor(0, dtype=float, device=device), # non-task actions are effectively noöps
                        reward = torch.tensor(step.reward, dtype=float, device=device)
                    ))
            if len(filtered_episode) > 0:
                filtered_trajectories.append(filtered_episode)
        # print(f'{trajectories}\n{filtered_trajectories}')
        return filtered_trajectories

    def forward(self, tasks):
        policy = torch.empty(0, dtype=float)
        for task, obs in tasks.items():
            if not task in self.queue: # mark task for update
                self.queue.append(task)
            if task in self.tasks: # add existing task-specific policy results if available
                local_policy = PolicyNetwork(
                    input_dimension=self.observation_dimension,
                    latent_dimension=self.latent_dimension,
                    layers=self.layers,
                    output_dimension=self.action_dimension,
                    params=torch.matmul(self.policy_latent_space.detach(), self.tasks[task].policy_latent_space.detach()).flatten()
                )
                policy = torch.cat((policy, local_policy.forward(obs)))
            else: # add random action values if no policy found
                policy = torch.cat((policy, torch.rand(self.action_dimension, dtype=float, device=device)))
        return Categorical(torch.softmax(policy, dim=0)).sample() if len(policy) > 0 else torch.tensor(0, dtype=float, device=device)

class Linellapig():
    """Class for Marmotellapig learning and operation using individual task-specific critics."""
    def __init__(self, observation_dimension:int, latent_dimension:int, action_dimension:int, latent_task_dimension:int, layers=0, learning_rate=1e-3, learning_steps=256, discount_factor=0.9):
        super(Linellapig, self).__init__()

        # Store dimensional variables.
        self.observation_dimension = observation_dimension # task features, len(t)
        self.latent_dimension = latent_dimension # hidden dimension
        self.layers = layers # number of extra (after the first) hidden layers
        self.action_dimension = action_dimension # actions per task
        self.policy_dimension = observation_dimension * latent_dimension + action_dimension * latent_dimension + layers * pow(latent_dimension, 2) # total policy parameters, len(θ(t)) = d
        self.latent_task_dimension = latent_task_dimension # task-specific vector lengths, len(s(t)) = k

        # Store learning variables.
        self.learning_rate = learning_rate # Learning rate for actor, critic, and TLS backpropagation.
        self.learning_steps = learning_steps # Number of actor, critic, and TLS backpropagation steps 
        self.discount = discount_factor # Weight of immediate vs forward planning.
        self.queue = [] # Tasklist used for learning, will need to be filled through operation or external assignment.
        self.tasks = {} # Task-specific vectors and their associated values.

        # Initialize shared latent space.
        self.latent_space = torch.zeros(self.policy_dimension, self.latent_task_dimension, dtype=float, device=device)

        # Initialize incremental partial shared latent space elements (later referred to as shared latent space constructors).
        self.latent_space_numerator = torch.zeros(self.policy_dimension * latent_task_dimension, dtype=float, device=device)
        self.latent_space_denominator = torch.zeros(self.policy_dimension * latent_task_dimension, self.policy_dimension * latent_task_dimension, dtype=float, device=device)

    def optimize(self, trajectories:list[list[tuple[dict[int|str, list[float]], int, float]]], critique:list[list[tuple[dict[int|str, list[float]], int, float]]]=None, test=False, log_task=False, plot_tasks=None, avg_losses=True, prefix=''):
        """
        Updates latent spaces provided a set of trajectories.

        Parameters: 
            trajectories (list(Trajectory) of lists(Episode) of Step(observation, action, reward)): Episodic observation-action-reward vectors for actor learning.
            critque (list(Trajectory) of lists(Episode) of Step(observation, action, reward)): Episodic observation-action-reward vectors for critic learning.
            test (bool[optional]): Test marker. If True, logs values as they are generated. (Deprecated)
            log_task (bool[optional]): Notification marker. If True, displays a notification for each new task started.
            plot_task (str[optional]): Notification marker. If not None, creates a chart of internal losses for each task, overwriting single file if specified. (Deprecated)
            avg_losses (bool[optional]): Record marker. If True, displays a notification for each new task started.
            prefix (str[optional]): Notification Marker. Text used to preface task notifications.
        """

        # local functions
        denominatize = lambda taskspace: torch.kron(torch.outer(taskspace.latent_space, taskspace.latent_space), taskspace.hessian) # used for task-specific parts of the shared latent denominator
        numeratize = lambda taskspace: torch.kron(taskspace.latent_space, torch.matmul(taskspace.optimal, taskspace.hessian)) # used for the task-specific parts of the sahred latent numerator
        
        # reinitialize shared latent space zero-columns
        for i in range(self.latent_space.shape[1]):
            if torch.count_nonzero(self.latent_space.T[i]) == 0:
                self.latent_space.T[i] = torch.rand(self.latent_space.shape[0], dtype=float, device=device)

        # Optimize local latent spaces for each task in queue
        losses = {}
        for t, task in enumerate(self.queue, 1):
            if log_task:
                print(f'{prefix}task # {t} / {len(self.queue)} \"{task}\"')

            # Check that provided trajectories and critique actually contain this task.
            if (local_trajectories := self.filter(trajectories, task)) == []:
                continue # Skip tasks without trajectories.
            elif critique is None or (local_critique := self.filter(critique, task)) == []:
                local_critique = local_trajectories # Use actor trajectories as critique if noe was provided or that provided lacks this task.

             # If task has an existing local latent space, make a copy, then remove its influence on the shared latent space constructors.
            if (local := self.tasks.get(task, None)) is not None:
                self.latent_space_denominator -= denominatize(local) # Remove local influence from the shared latent denominator.
                self.latent_space_numerator -= numeratize(local) # Remove local influence from the shared latent numerator.
            # If task is new, initialize a new local latent space for it.
            else:
                self.tasks[task] = local = TaskLatentSpace(self.latent_task_dimension)

            # Evaluate the current policy from the provided trajectories with the base learner.
            policy = PolicyNetwork( # Define actor policy from shared and local latent spaces.
                input_dimension=self.observation_dimension,
                latent_dimension=self.latent_dimension,
                layers=self.layers,
                output_dimension=self.action_dimension,
                params=torch.matmul(self.latent_space.detach(), local.latent_space.detach()).flatten() if torch.count_nonzero(self.latent_space).item() > 0 else None
            )
            critic = PolicyNetwork( # Define the critic policy from the task-specific stored critic.
                input_dimension=self.observation_dimension,
                latent_dimension=self.latent_dimension,
                layers=self.layers,
                output_dimension=1,
                params=local.critic
            )
            local.optimal, local.hessian, local.critic, internal_losses = self.evaluate(local_trajectories, local_critique, policy, critic, task, test)

            # Update the local task-specific latent space.
            lose = lambda L=self.latent_space, s=local.latent_space, a=local.optimal, H=local.hessian : 0.001 * torch.linalg.norm(s, ord=1) + torch.matmul(a - torch.matmul(L, s), (a - torch.matmul(L, s)))
            # Make a temporary, optimizable copy of the local latent space.
            new_latent_space = local.latent_space.detach().clone()
            new_latent_space.requires_grad_()
            # Prepare for backpropagation.
            optimizer = optim.Adam([new_latent_space], lr=self.learning_rate)
            local_losses = SimpleNamespace(actor=internal_losses.actor, critic=internal_losses.critic, latent=[])
            # Backpropagation steps
            for i in range(self.learning_steps):
                optimizer.zero_grad()
                loss = lose(s=new_latent_space)
                local_losses.latent.append(loss.item())
                loss.backward()
                optimizer.step()
            # Store backpropagation results.
            losses[task] = local_losses
            # Update the local latent space to match the now-optimized copy.
            new_latent_space.requires_grad_(False)
            local.latent_space = nn.Parameter(new_latent_space).requires_grad_(False)

            # Update the shared latent space constructors.
            self.latent_space_denominator += denominatize(local)
            self.latent_space_numerator += numeratize(local)

            # Store the local latent space to its task-specific space in the TLS dictionary.
            self.tasks[task] = local

        # Empty fuly-processed queue.
        self.queue = []

        # Update shared latent space from the updated constructors.
        self.latent_space.weight = torch.matmul(torch.inverse(torch.div(self.latent_space_denominator, len(self.tasks)) + torch.eye(self.policy_dimension * self.latent_task_dimension, self.policy_dimension * self.latent_task_dimension, device=device) * 0.1), torch.div(self.latent_space_numerator, len(self.tasks))).view(self.policy_dimension, self.latent_task_dimension)

        # Average losses across taks and return them.
        if avg_losses:
            average = lambda category: torch.stack([torch.tensor(vars(loss)[category], dtype=float, device=device) for loss in losses.values()]).mean(dim=-2)
            losses = SimpleNamespace(
                actor = average('actor'),
                critic = average('critic'),
                latent = average('latent')
            )
        return losses

    def evaluate(self, trajectories:list[list[tuple[dict[int|str, list[float]], int, float]]], critique:list[list[tuple[dict[int|str, list[float]], int, float]]], policy:PolicyNetwork=None, critic:PolicyNetwork=None, task:int|str=None, test=False, plot_evaluations=False) -> tuple[list, list, list, tuple[list, list]]:
        '''
        Evaluate the current policy over a set of trajectories.

        Parameters: 
            trajectories (list(Trajectory) of lists(Episode) of Step(observation, action, reward)): Episodic observation-action-reward vectors for actor learning.
            critque (list(Trajectory) of lists(Episode) of Step(observation, action, reward)): Episodic observation-action-reward vectors for critic learning.
            policy (nn.Linear?[optional]): actor policy parameters to evaluate. If present, initializes the actor to match.
            ciritc (nn.Linear?[optional]): critic policy parameters to evaluate. If present, initializes the critic to match.
            task (str[optional]): Task id indicator. (Deprecated)
            test (bool[optional]): Test marker. (Deprecated)
            task (bool[optional]): Task id indicator. (Deprecated)
        '''

        # Initialize the base learner to the provided actor & critic policies.
        base_learner = ACagent(
            observation_dimension=self.observation_dimension,
            latent_dimension=self.latent_dimension,
            layers=self.layers,
            action_dimension=self.action_dimension,
            learning_rate=self.learning_rate,
            discount=self.discount,
            actor=policy,
            critic=critic
        )
        # Initialize base-learner losses.
        losses = SimpleNamespace(actor=[], critic=[])

        # Use base learner optimization to get the local optimal.
        for i  in range(self.learning_steps):
            if trajectories == []:
                raise ValueError('Missing Trajectories')
            loss_actor, loss_critic = base_learner.optimize(trajectories, critique)
            losses.actor.append(loss_actor.item())
            losses.critic.append(loss_critic.item())
        base_learner.requires_grad_(False)
        # Base-learner plotting (Deprecated)
        if plot_evaluations:
            plot_base(losses, result_id=task)
        
        # Get the actor's hessian gradient from the base learner's local optimal.
        hessian, _ = base_learner.hessian(trajectories)

        # returns: local optimal, est. hessian gradient thereof, local critical optimal, & internal losses
        return base_learner.actor.params(), hessian, base_learner.critic.params(), losses

    def filter(self, trajectories:list[list[tuple[dict[int|str, list[float]], int, float]]], task:int|str, mode='no-op') -> list[list[list]]:
        """
        Returns a task-specificized trajectories list.

        Prameters:
            trajectories (list of lists of SimpleNameSpaces(observation, action, reward)): Episodic observation-action-reward vectors.
            task (key): Task key to filter for.
            mode (str): How to treat filtered foreign actions.
        """

        # Initialize filtered trajectory list.
        filtered_trajectories = []
        for episode in trajectories:
            # Initialize filtered episodic trajectory lsit.
            filtered_episode = []
            for step in episode:
                # Only include steps where the filter target is present.
                if task in step.observation:
                    # Adjust the broader action to be task-specific according to the deisred strategy.
                    if mode == 'no-op': # non-task actions are effectively noöps
                        action = step.action.detach().clone() % self.action_dimension if step.action.item() // self.action_dimension == list(step.observation.keys()).index(task) else torch.tensor(0, dtype=float, device=device)
                    elif mode == 'ignore': # non-task actions are ignored
                        action = step.action.detach().clone() % self.action_dimension if step.action.item() // self.action_dimension == list(step.observation.keys()).index(task) else torch.tensor(-1, dtype=float, device=device)
                    # elif mode == 'friendly': # non-task actions are effectively still progressing the task (rideshare specific)
                    #     action = step.action.detach().clone() % self.action_dimension if step.action.item() // self.action_dimension == list(step.observation.keys()).index(task) else step.observation[task][-1].detach().clone()
                    else: # non-task actions are effectively noöps
                        action = step.action.detach().clone() % self.action_dimension if step.action.item() // self.action_dimension == list(step.observation.keys()).index(task) else torch.tensor(0, dtype=float, device=device)
                    # Do not include steps with negative (invalid) actions.
                    if action >= 0:
                        filtered_episode.append(SimpleNamespace( # convert full step into task-specific step
                            observation = step.observation[task], # get only the task-relevant observation
                            action = action,
                            reward = torch.tensor(step.reward, dtype=float, device=device)
                        ))
            # Only include the episode if the task was present somewhere therein.
            if len(filtered_episode) > 0:
                filtered_trajectories.append(filtered_episode)
        # Return the final list of filtered trajectories.
        return filtered_trajectories

    def forward(self, tasks:dict[int|str, list[float]]):
        """Function for selecting an action provided an observation interpreted by task."""

        # Initialize the policy as a zero-sized empty tensor.
        policy = torch.empty(0, dtype=float, device=device)

        # Append task-specific action spaces for each task in the observation.
        for task, obs in tasks.items():
            
            # Mark task for update (this can be removed if queues are assigned externally).
            if not task in self.queue:
                self.queue.append(task)

            # Append the existing task-specific policy results if available.
            if task in self.tasks:
                local_policy = PolicyNetwork(
                    input_dimension=self.observation_dimension,
                    latent_dimension=self.latent_dimension,
                    layers=self.layers,
                    output_dimension=self.action_dimension,
                    params=torch.matmul(self.latent_space.detach(), self.tasks[task].latent_space.detach()).flatten()
                ).to(device)
                policy = torch.cat((policy, local_policy.forward(obs)))
            # Add randomly-valued action if task-specific policy can be found. (allows for an initial random policy as well as partially random for operation partially-learned environments)
            else:
                policy = torch.cat((policy, torch.rand(self.action_dimension, dtype=float, device=device)))
            
        # Return final action choice. (if no tasks were observed returns 0 as that is a noop in the test environments)
        return Categorical(torch.softmax(policy, dim=0)).sample() if len(policy) > 0 else torch.tensor(0, dtype=float, device=device)

class Simpellapig():
    """Class for compact Marmotellapig operation (incapable of learning)."""
    def __init__(self, learned:Linellapig|Duellapig):
        super(Simpellapig, self).__init__()

        # define policy-shape parameters
        self.observation_dimension = learned.observation_dimension # size of expected task-specific inputs.
        self.latent_dimension = learned.latent_dimension # size of hidden layers
        self.layers = learned.layers # number of extra (after the frist) hidden layers
        self.action_dimension = learned.action_dimension # size to output after forwarding the input through task-specific policy networks.
        
        # define policy weights
        if hasattr(learned, 'latent_space'): # from Linellapig
            self.latent_space = learned.latent_space.to(device)
            self.tasks = dict([(task, SimpleNamespace(latent_space=taskspace.latent_space.to(device))) for task, taskspace in learned.tasks.items()])
        else: # from Duellapig
            self.latent_space = learned.policy_latent_space.to(device)
            self.tasks = dict([(task, SimpleNamespace(latent_space=taskspace.policy_latent_space.to(device))) for task, taskspace in learned.tasks.items()])

    def forward(self, tasks:dict[int|str, list[float]]):
        """Function for selecting an action provided an observation interpreted by task."""

        # Initialize the policy as a zero-sized empty tensor.
        policy = torch.empty(0, dtype=float, device=device)

        # Append task-specific action spaces for each task in the observation.
        for task, obs in tasks.items():

            # Append the existing task-specific policy results if available.
            if task in self.tasks:
                local_policy = PolicyNetwork(
                    input_dimension=self.observation_dimension,
                    latent_dimension=self.latent_dimension,
                    layers=self.layers,
                    output_dimension=self.action_dimension,
                    params=torch.matmul(self.latent_space.detach(), self.tasks[task].latent_space.detach()).flatten()
                )
                policy = torch.cat((policy, local_policy.forward(obs)))
            # Add randomly-valued action if task-specific policy can be found. (allows for operation partially-learned environments via a partially-random policy)
            else:
                policy = torch.cat((policy, torch.rand(self.action_dimension, dtype=float, device=device)))
            
        # Return final action choice. (if no tasks were observed returns 0 as that is a noop in the test environments)
        return Categorical(torch.softmax(policy, dim=0)).sample() if len(policy) > 0 else torch.tensor(0, dtype=float, device=device)

class ACnet(nn.Module):
    """Class for components of the Neural Network Implementation of Actor-Critic"""
    def __init__(self, observation_dimension:int, latent_dimension:int, action_dimension:int, layers=0, net_type=NetType.main, learning_rate=1e-3, policy:PolicyNetwork=None):
        super(ACnet, self).__init__()

        # Record if this is the actor or the critic, and adjust dimensions as needed.
        self.net_type = net_type
        if net_type is not NetType.main:
            action_dimension = 1
        
        # Use existing policy if provided.
        if policy is not None:
            self.subpol = policy
        else:
            self.subpol = PolicyNetwork(
                input_dimension=observation_dimension,
                latent_dimension=latent_dimension,
                layers=layers,
                output_dimension=action_dimension,
                params=torch.rand(observation_dimension * latent_dimension + action_dimension * latent_dimension + layers * pow(latent_dimension, 2), dtype=float, device=device)
            )

        # Prepare for later optimization.
        self.optimizer = optim.Adam(self.subpol.parameters(), learning_rate)

    def forward(self, observation:list):
        """Function for forwarding input through the network."""
        return Categorical(F.softmax(self.subpol(observation), dim=-1)) if self.net_type == NetType.main else self.subpol(observation)
    
    def optimize(self, loss):
        """Function for performing one step of backpropagation."""
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def params(self):
        """Function for getting the weights of this ntwork."""
        return self.subpol.deparam().detach()
    
    def requires_grad_(self, require=True):
        """Function for dis/enabling gradient requirement in this network."""
        self.subpol.requires_grad_(require)

class ACagent():
    """Base Learner Implementation of Actor-Critic"""
    def __init__(self, observation_dimension:int, latent_dimension:int, action_dimension:int, layers=0, learning_rate=1e-3, discount=0.9, actor:PolicyNetwork=None, critic:PolicyNetwork=None):
        super(ACagent, self).__init__()

        # Store network dimensions.
        self.discount = discount # discount factor for forward planning
        self.observation_dimension = observation_dimension # size of expected inputs.
        self.latent_dimension = latent_dimension # size of hidden layers
        self.layers = layers # number of extra (after the first) hidden layers.
        self.action_dimension = action_dimension # size to output from the actor after forwarding inputs through the network.

        # Define actor & critic components.
        self.actor = ACnet(observation_dimension, latent_dimension, action_dimension, layers, learning_rate=learning_rate, policy=actor).to(device)
        self.critic = ACnet(observation_dimension, latent_dimension, action_dimension, layers, NetType.target, learning_rate, critic).to(device)
        if critic is not None: # IDK why this was necessary here, but there's an error if it's not.
            self.critic.requires_grad_()

    def forward(self, observation:list[float]):
        """Returns the action distribution of the agent's actor network and the state value of the agent's critic network for the provided observation."""
        return self.actor.forward(observation).sample(), self.critic.forward(observation)
    
    def optimize(self, trajectories:list[list[tuple[dict[int|str, list[float]], int, float]]], critique:list[list[tuple[dict[int|str, list[float]], int, float]]]) -> tuple[float, float]|tuple[None, None]:
        """Runs both of the agent's networks through one iteration of Adam optimization from the provided trajectories."""

        # There's no loss to calculate over empty trajectories.
        if len(trajectories) > 0:
            # critic loss & optimization step
            loss_critic = torch.cat([(returned - torch.cat([self.critic.forward(step.observation) for step in critique[episode]])).pow(2).mean().unsqueeze(0) for episode, returned in enumerate(self.returns(critique))]).mean()
            self.critic.optimize(loss_critic)
            # actor loss & optimization step
            loss_actor = torch.cat([(torch.cat([self.actor.forward(step.observation).log_prob(step.action).unsqueeze(0) for step in trajectories[episode]])
                * (returned - torch.cat([self.critic.forward(step.observation) for step in trajectories[episode]])).detach()).mean().unsqueeze(0) for episode, returned in enumerate(self.returns(trajectories))]).mean()
            self.actor.optimize(loss_actor)
            # return calculated losses
            return loss_actor, loss_critic
        else:
            return None, None

    def returns(self, trajectories:list[list[tuple[dict[int|str, list[float]], int, float]]]):
        """Function for calculating discounted rewards. From https://github.com/yc930401/Actor-Critic-pytorch/blob/master/Actor-Critic.py (expanded to cover multiple episodes)."""

        returns = []
        for episode in trajectories:
            r = self.critic.forward(episode[-1].observation)
            returned = []
            for step in reversed(range(len(episode))):
                r = episode[step].reward + self.discount * r
                returned.insert(0, r)
            returns.append(returned)
        return [torch.cat(returned).detach() for returned in returns]

    def requires_grad_(self, require=True):
        """Wrapper function to set grad requirement for both the actor and critic networks."""
        self.actor.requires_grad_(require)
        self.critic.requires_grad_(require)
        
    def hessian(self, trajectories:list[list[tuple[dict[int|str, list[float]], int, float]]], critique:list[list[tuple[dict[int|str, list[float]], int, float]]]=None) -> tuple[list[float], None]:
        """
        Function for calculating the hessian gradient of the actor network.
        From https://discuss.pytorch.org/t/runtimeerror-element-0-of-variables-does-not-require-grad-and-does-not-have-a-grad-fn/11074/68 (you might need to scroll a bit).
        The previous, simpler version consistently returned all 0s after implementing new policy forms (hidden layers & multilayer support).
        """

        test_net = PolicyNetwork(self.observation_dimension, self.latent_dimension, self.layers, self.action_dimension, self.actor.params())
        loss = lambda policy: torch.cat([(torch.cat([Categorical(F.softmax(test_net(step.observation), dim=-1)).log_prob(step.action).unsqueeze(0) for step in trajectories[episode]]) * (returned - torch.cat([self.critic.forward(step.observation) for step in trajectories[episode]])).detach()).mean().unsqueeze(0) for episode, returned in enumerate(self.returns(trajectories))]).mean()
        del_attr = lambda obj, names: delattr(obj, names[0]) if len(names) == 1 else del_attr(getattr(obj, names[0]), names[1:])
        set_attr = lambda obj, names, val: setattr(obj, names[0], val) if len(names) == 1 else set_attr(getattr(obj, names[0]), names[1:], val)
        def paramize(net):
            orig = tuple(net.parameters())
            names =[]
            for name, p in list(net.named_parameters()):
                del_attr(net, name.split('.'))
                names.append(name)
            return orig, names
        def deparamize(net, names, params):
            for name, p in zip(names, params):
                set_attr(net, name.split('.'), p)
        def hess(*paramized):
            deparamize(test_net, names, paramized)
            return loss(test_net)
        params, names = paramize(test_net)
        params = tuple(p.detach().requires_grad_() for p in params)
        hessian = torch.cat([col.flatten() for row in torch.autograd.functional.hessian(hess, params) for col in row]).reshape(len(self.actor.params()), len(self.actor.params()))
        
        # None would be critic, but that wasn't needed by the time this was implemented. 
        return hessian, None
