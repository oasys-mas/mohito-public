#----------------------------------------------------------------------------------------------------------------------#
# Title: Marmotellapig's Rideshare Trainer
# Description: This file contains a trainer for the various ellapig-derived networks in the rideshare domain.
# Author: Matthew Sentell
# Version: 1.00.01
#----------------------------------------------------------------------------------------------------------------------#

from enum import IntEnum
from dataclasses import dataclass
from typing import Callable
from typing import Any
from rideshare.tao_pg_ella.ellapig_net import *
from rideshare.tao_pg_ella.ellapig_data import *
from rideshare.ride import *
import pandas as pd
import numpy as np
import pickle
import datetime
import torch

LINE = '--------------------------------'
DATE = f'{datetime.datetime.now():%Y.%m.%d}'

# Helper Functions & Data Classes

def tensorflatten(elems:list):
    """Function for nested lists into one tensor."""
    if len(elems) <= 0:
        return torch.tensor(elems, dtype=float, device=device)
    else:
        return torch.cat([tensorflatten(elem) if hasattr(elem, "__iter__") \
            else torch.tensor([elem], dtype=float, device=device) for elem in elems])
    
@dataclass
class Grid:
    """Class for defining Rideshare grid dimensions."""
    len: int = 10
    wid: int = 10

@dataclass
class Costs:
    """
    Class for defining Rideshare cost values.\n
    Cost Meanings:
        Miss: Awarded to losing agents when multiple agents compete for the same passenger.
        Accept: Awarded for successfully accepting an available passenger.
        Move: Awarded for attempting to pick up an accepted passenger or drop off a riding passenger from the incorrect tile. Optional variability.
        Pick: Awarded for successfully traveling to and collecting a passenger into a vehicle. Optional variability.
        Drop: Awarded for successfully servicing a passenger in their entirety. Task-defined by default.
        No Passengers Remaining: Awarded for clearing all available passengers within an environment.
        Pool Limit Exceeded: Awarded for attempting to accept more passengers than can fit in a vehicle.
    """
    miss: float = -2
    accept: float = 0
    move: float = -1.2
    pick: float = -0.1
    variable_move: bool = False
    variable_pick: bool = False
    drop: float = None
    no_passengers_remaining: float = 0
    pool_limit_exceeded: float = -2

@dataclass
class WaitLimits:
    """Class for defining limits on the number of steps for which passengers will await service."""
    unpicked: int = 5
    undropped: int = 10
    unaccepted: int = 10

@dataclass
class EllapigDimensions:
    """Class for defining the shape of an Ellapig network."""
    observation: int
    action: int
    latent_shared: int = 32
    latent_task: int = 8
    layers: int = 0

@dataclass
class Trajectory:
    """Class for defining trajectory data."""
    observation:dict[int|str, list[float]]
    tasks:int
    action:int
    reaction:int # Used to account for automatic actioning in rideshare.
    reward:float

# Main Class

class RideshareTrainer:
    def __init__(self, env:rideshare=None, agents:list[Linellapig]|list[Duellapig]=None):        
        
        # Constant for converting typekeys to types.
        self.agent_types = { "linear": Linellapig, "double": Duellapig }

        # Assign environment and agents if provided.
        self.env: rideshare = env
        self.agents = agents

        # Match agent flags to agents.
        if agents is not None:
            self.agent_type = type(agents[0])
            self.learnable = isinstance(agents[0], Linellapig|Duellapig)
            self.decomposed = agents[0].action_dimension <= 2
        else:
            self.agent_type = None
            self.learnable = False
            self.decomposed = False

    # Environment Customization:

    def reinit_environment(self, agents:int=3, *, grid=Grid(), costs=Costs(), clear_reward=False, pooling_limit=2, wait_limit=WaitLimits()):
        """
        Function for initializing a new rideshare environment.

        Parameters:
            Agents: Value for the expected number of agents in the environment.
            Grid: Values for the environment's grid dimensions.
            Costs: Values for various cost values for action results within the environment.
            Clear Reward: Flag for granting of the "No Passengers Remaining" reward after clearing all available passengers within the environment.
            Pooling Limit: Value for the maximum passenger capacity of vehicles within the environment.
            Wait Limit: Values for the number of steps passengers will await service at various states of their task.
        """
        self.env = rideshare(
            num_agents = agents,
            grid_len = grid.len, grid_wid = grid.wid,
            noop_cost = costs.miss, accept_cost = costs.accept, pick_cost = costs.pick, move_cost = costs.move, drop_cost = costs.drop,
            variable_move_cost=costs.variable_move, variable_pick_cost=costs.variable_pick,
            no_pass_reward=clear_reward, no_pass_cost=costs.no_passengers_remaining, pool_limit = pooling_limit, pool_limit_cost = costs.pool_limit_exceeded,
            wait_limit = [wait_limit.unpicked, wait_limit.undropped, wait_limit.unaccepted]
        )

    def load_environment(self, source:str, *, safe=True) -> bool:
        """
        Function for initializing the environment from a config file.
        """

        # Attempt file opening and processing.
        try:
            with open(source, "rb") as f:
                config = pickle.load(f)
        except Exception as e:
            if safe:
                return None
            else:
                raise e
        
        params = {
            "num_agents": "agents",
            "grid_length": "grid", "grid_width": "grid", 
            "variable_move_cost": "costs", "variable_pick_cost": "costs", "accept_cost": "costs", "pick_cost": "costs", "move_cost": "costs", "miss_cost": "costs", "drop_cost": "costs", "no_pass_cost": "costs", "pool_limit_cost": "costs",
            "no_pass_reward": "clear_reward",
            "pool_limit": "pooling_limit",
            "wait_limit": "wait_limit"
        }

        if isinstance(config, dict) and any([param in config for param in params.keys()]):

            # Make list-comprehended variables to avoid repeat operation later.
            paranames = [n for n in self.reinit_environment.__code__.co_varnames if n != "self"]
            conparams = [v for k, v in params.items() if k in config]

            # Function to fill parameter value from loaded configuration or default.
            get = lambda attr, defaults=self.reinit_environment.__defaults__, index=paranames.index: \
                config[attr] if attr in config else defaults[index(params[attr])]
            
            # Get simple parameters.
            agents = get("num_agents")
            clear_reward = get("no_pass_reward")
            pooling_limit = get("pool_limit")

            # Grid subconstruction.
            if "grid" in conparams:
                grid = Grid(
                    get("grid_length", Grid.__init__.__defaults__, lambda _: 0),
                    get("grid_length", Grid.__init__.__defaults__, lambda _: 1))
            else:
                grid = Grid()

            # Costs subconstruction.
            if "costs" in conparams:
                # Make list-comprehended variable to avoid repeat operation later.
                coparanames = [n for n in Costs.__init__.__code__.co_varnames if n != "self"]
                # Function to fill parameter value from loaded configuration or default.
                get_cost = lambda attr: get(f"{attr}_cost", Costs.__init__.__defaults__, coparanames.index)
                # Build Costs parameter.
                costs =  Costs(
                    get_cost("miss"),
                    get_cost("accept"),
                    get_cost("move"),
                    get_cost("pick"),
                    get_cost("variable_move"),
                    get_cost("variable_pick"),
                    get_cost("drop"),
                    get("no_pass_cost", Costs.__init__.__defaults__, lambda _: coparanames.index("no_passengers_remaining")),
                    get("pool_limit_cost", Costs.__init__.__defaults__, lambda _: coparanames.index("pool_limit_exceeded")))
            else:
                costs = Costs()
            
            # Wait limits subconstruction.
            if "wait_limit" in conparams:
                # Wait limit configuration is stored as a list instead of a dataclass, so conversion is necessary.
                wait_limit = [limit for limit in get("wait_limit")]
                wait_limit = WaitLimits(wait_limit[0], wait_limit[1], wait_limit[2])
            else:
                wait_limit = WaitLimits()

            # Build environment and return true on success
            self.reinit_environment(agents, grid, costs, clear_reward, pooling_limit, wait_limit)
            return True

        # Return false if loaded configuration cannot build an environment
        return False

    # Environment Helpers

    def interpreter(self, observation:list, task_from:Callable[[list, list, int, int], tuple[str, list[float]]]) -> dict[str, list[float]]:
        """
        Specific to the Rideshare environment: Converts a single agent's observation into a dictionary of task-specific observation vectors for individual task processing.
        
        Parameters:
            Observation: Value from Rideshare getObsFromState formatted as a 5-layer nested list with each layer being a grid-scale map of lists of that layers representative features.
            Task From: Function for converting the base + task-specifc observation elements and task atate into a task-specific key-value pair.
        """

        # Human-readable enum for observation layers.
        ObsLayer = IntEnum("ObsLayer", zip(["SELF", "OTHER", "ACCEPT", "RIDE", "OPEN"], range(5)))

        # Collect a base observation containing own location; others' locations; and the numbers of available, accepted, & riding passengers.
        base = []
        base.append(np.argwhere(np.vectorize(lambda x: len(x) > 0)(observation[ObsLayer.SELF]))[0]) # own location
        # other's locations
        base.append([loc for loc in np.argwhere(np.vectorize(lambda x: len(x) > 0)(observation[ObsLayer.OTHER])) for other in observation[ObsLayer.OTHER, loc[0], loc[1]]])
        # Collect task-specific info for passengers currently riding with the agent.
        riding = [passenger for passenger in observation[ObsLayer.RIDE, base[0][0], base[0][1]]]
        # Collect task-specific info for accepted passengers awaiting pickup.
        accepted = [passenger for loc in np.argwhere(np.vectorize(lambda x: len(x) > 0)(observation[ObsLayer.ACCEPT])) for passenger in observation[ObsLayer.ACCEPT, loc[0], loc[1]]]
        # Collect task-specific info for passengers that have yet to be accepted by anyone.
        available = [passenger for loc in np.argwhere(np.vectorize(lambda x: len(x) > 0)(observation[ObsLayer.OPEN])) for passenger in observation[ObsLayer.OPEN, loc[0], loc[1]]]
        base.append(len(accepted)) # total accepted passengers
        base.append(len(riding)) # total riding passengers
        base.append(len(available)) # total available passengers

        # Create a dict of per-task observations.
        tasks = {}
        for state, targets in enumerate([available, accepted, riding], start=1): # Prioritize in-progress tasks by overwriting new ones that have matching keys.
            for target in targets:
                key, val = task_from(base, target, state, int(state > 1))
                tasks[key] = val
        return tasks
    
    def exterpreter(self, action:float|int, observation:dict[int|str, list[float]], task_from:Callable[[tuple[str, list[float]]], list[float]], *, actions=4):
        """
        Specific to the Rideshare environment: Converts a single, integer action from a multitask policy to a 5x1 integer array useable by the rideshare domain.
        
        Parameters:
            Action: Value for the Ellapig networks' action choice.
            Observation: Dictionary of task-specific observations.
            Task From: Function for converting chosen task-specific observation into a state-inclusive representative vector from which the rideshare action node may be generated.
            Actions: Value for the size of task-specific action spaces.
        """

        # Human-readable enum for parts of the task-specific observation vector
        Get = IntEnum("TaskInfo", ["STATE","STEP", "FARE", "END_Y", "END_X", "START_Y", "START_X"])

        # Conversion is only necessary for non-noop actions.
        if int(action) % actions != 0:
            task = task_from(list(observation.items())[int(action) // actions]) # local task variable

            # Convert Ellapig action result to rideshare action node.
            return [
                3, # action indicator
                int(task[-Get.START_X] * self.env.grid_width + task[-Get.START_Y]), # start coords to grid
                int(task[-Get.END_X] * self.env.grid_width + task[-Get.END_Y]), # end coords to grid
                int(task[-Get.STATE] - 1), # action type is automatic in this environment (if not noöp)
                int(task[-Get.STEP]) # task initialization step
            ]
        else:
            return [3, -1, -1, 3, -1] # noöp by default

    # Agent Customization:

    def new_agent(dimensions:EllapigDimensions, agent_type:Linellapig|Duellapig=Linellapig, learning_rate=1e-3, learning_steps=128, discount=0.9) -> Linellapig | Duellapig:
        """
        Returns one new agent matching the provided specifications.

        Parameters:
            Decomposed: Flag for task-by-action instead of task-by-service.
            Dimensions: Values for the shape of the agents' networks.
            Agent Type: Value for the desired Ellapig implementation type.
            Learning Rate: Value for Adam optimizer to use during backpropigation.
            Learning Steps: Value for the number of backpropigation steps to use for base and per-task learning.
            Discount: Value for discounting forward planning. 
        """
        
        return agent_type(
                observation_dimension=dimensions.observation,
                action_dimension=dimensions.action,
                latent_dimension=dimensions.latent_shared,
                latent_task_dimension=dimensions.latent_task,
                layers=dimensions.layers,
                learning_rate=learning_rate,
                learning_steps=learning_steps,
                discount_factor=discount
            )
    
    def load_agent(source:str, *, safe=True, simplify=False) -> Linellapig | Duellapig | Simpellapig:
        """
        Attempts to load and return a single agent from a pickle file.

        Parameters:
            Source: File path to load from.
            Safe: Flag to ingore errors with a None return.
            simplify: Flag to convert any success into a Simpellapig compact policy.
        """

        # Attempt file opening and processing.
        try:
            with open(source, "rb") as f:
                agent = pickle.load(f)
        except Exception as e:
            if safe:
                return None
            else:
                raise e
            
        # Attempt loaded data interpretation as a usable agent.
        else:
            if isinstance(agent, Simpellapig): # Use already-simplified agent.
                return agent
            elif isinstance(agent, Linellapig | Duellapig): # Use valid agent, but simplify if needed.
                return agent if not simplify else Simpellapig(agent)
            elif safe: # Ignore error?
                return None
            else: # Error!
                raise TypeError(f"Agent is not an Ellapig derivative. ({type(agent)})")

    def reload_agents(self, source:str, *, hardmatch=True, simplify=False, agent_type=None) -> int:
        """
        Function for loading an existing set of agents.

        Parameters:
            Source: The file or directory path of the desired agent(s).
            Hardmatch: Flag to fail if any agents are different types.
            Simplify: Flag to disable learning and use a compact form of each agent.

        Returns the length of the new set of agents, or 0 if there was no replacement.
        """

        # Affirm path validity.
        if (path := Path(source)).exists():
            agents = []

            # Convert directory or file path to iterable.
            if (paths := list(path.iterdir()) if path.is_dir() else [path] if path.is_file() else None) is not None:

                # Append successfully loaded agents.
                for f in paths:
                    if (agent := RideshareTrainer.load_agent(f, simplify=simplify)) is not None:
                        agents.append(agent)

            # Affirm there was at least one successfully loaded agent.
            if len(agents) > 0:

                # Affirm agents are of a unified type.
                if agent_type is None:
                    agent_type = type(agents[0]) # Define a type if necessary.
                elif not any(isinstance(a, agent_type) for a in agents):
                    return 0 # Fail if no agents of the desired type were successfully loaded.
                if hardmatch: # Hardmatch fails on disaffirmation.
                    for a in agents:
                        if not isinstance(a, agent_type):
                            return 0
                else: # Softmatch trims agents of incorrect types
                    agents = [a for a in agents if isinstance(a, agent_type)]

                # Update the trainer's agent list and type.
                self.agents = agents
                self.agent_type = agent_type
                self.decomposed = self.agents[0].action_dimension <= 2
                self.learnable = not simplify
                return len(self.agents)
        
        # Fail on invalid path.
        return 0

    def reinit_agents(self, agents:int, *, dimensions:EllapigDimensions, agent_type="linear", learning_rate=1e-3, learning_steps=128, discount=0.9):
        """
        Simple Agent initialization. May not match environment!!!
        
        Parameters:
            Decomposed: Flag for task-by-action instead of task-by-service.
            Dimensions: Values for the shape of the agents' networks.
            Agent Type: Key for the desired Ellapig implementation.
            Learning Rate: Value for Adam optimizer to use during backpropigation.
            Learning Steps: Value for the number of backpropigation steps to use for base and per-task learning.
            Discount: Value for discounting forward planning. 
        """

        # Affirm agent number and type validity.
        if agents > 0 and agent_type in self.agent_types:
                
                # Generate new agents & update related info.
                self.agents = [RideshareTrainer.new_agent(
                    dimensions=dimensions,
                    agent_type=self.agent_types[agent_type],
                    learning_rate=learning_rate,
                    learning_steps=learning_steps,
                    discount=discount) for _ in range(agents)]
                self.decomposed = dimensions.action <= 2
                self.learnable = True
                return agents
        return 0

    def reinit_agents_to_env(self, decomposed=False, *, dimensions=EllapigDimensions(0, 0), agent_type="linear", learning_rate=1e-3, learning_steps=128, discount=0.9):
        """
        General Agent initialization to match environment.
        
        Parameters:
            Decomposed: Flag for task-by-action instead of task-by-service.
            Dimensions: Values for the shape of the agents' networks. Observation and action dimensions will be overwritten to match the environment.
            Agent Type: Key for the desired Ellapig implemntation.
            Learning Rate: Value for Adam optimizer to use during backpropigation.
            Learning Steps: Value for the number of backpropigation steps to use for base and per-task learning.
            Discount: Value for discounting forward planning. 
        """

        # Affirm agent number and type validity.
        if agent_type in self.agent_types:

            # Overwrite action & observation dimensions to match environment and decomposition status.
            self.decomposed = decomposed
            if self.decomposed:
                dimensions.observation = 2 * self.env.num_agents + 9
                dimensions.action = 2
            else:
                dimensions.observation = 2 * self.env.num_agents + 10
                dimensions.action = 4

            # Generate new agents & update related info.
            self.agents = [RideshareTrainer.new_agent(
                dimensions=dimensions,
                agent_type=self.agent_types[agent_type],
                learning_rate=learning_rate,
                learning_steps=learning_steps,
                discount=discount) for _ in range(self.env.num_agents)]
            self.learnable = True
            return len(self.agents)
        
        return 0

    # Experiment Customization:

    def train(self, iterations=1000, episodes=1, steps=100, passengers=8, plot_partial=50):
        """
        Function for learning policies.

        Parameters:
            Iterations: Value for number of iterations over which to learn the policies.
            Episodes: Value for the number of episodes to gather trajectories from each iteration.
            Steps: Value for the number of steps in each episode of each iteration.
            Passengers: Value to the maximum number of pssengers to add over the course of each episode.
            Plot Partial: Value for policy-checkpointing frequency.
        """

        # Affirm operability.
        if len(self.agents) != self.env.num_agents or not self.learnable:
            return None

        # Collect experiment Configuration.
        config = {"learning": self.learnable, "iterations": 0}
        env = ["num_agents", "grid_length", "grid_width", 
            "variable_move_cost", "variable_pick_cost", "no_pass_reward", "accept_cost", "pick_cost", "move_cost", "miss_cost", "drop_cost", "no_pass_cost", "wait_limit",
            "pool_limit", "pool_limit_cost"]
        for attr in env:
            config[attr] = getattr(self.env, attr)

        # Set result target storage location and initialize training statistics.
        result_dest = f"ride/{DATE}/i{iterations}{f'e{episodes}' if episodes > 1 else ''}_o{passengers}a{len(self.agents)}"
        store(results_group=result_dest, config=config) # store configuration for future access
        stats = [SimpleNamespace(rewards=[], losses=[], tasks=SimpleNamespace(list=[], count=[])) for a in range(len(self.agents))]
        
        # Specification Printing
        print("Experiment Specifications:") # Agent & learning details.
        print(f"  {iterations} learning iterations over {episodes}-episode, {steps}-step{' decomposed' if self.decomposed else ''} trajectories")
        print(f"  {self.agents[0].learning_steps} internal iterations at {self.agents[0].learning_rate} LR, discounted {self.agents[0].discount}")
        print(f"  {self.agents[0].latent_dimension} shared x {self.agents[0].latent_task_dimension} task dimensions")
        if plot_partial is not None: # Checkpointing notice
            print(f"  partial plotting every {plot_partial} iterations")
        print(f"Environment Specifications:") # Environment details.
        print(f"  \"Rideshare\" {passengers}{' Open'} Tasks with {len(self.agents)} Agent(s) on a {self.env.grid_length}x{self.env.grid_width} Grid")
        print(f"  Costs: Accept={self.env.accept_cost}, Pickup={self.env.pick_cost}, Move={self.env.move_cost}, Dropoff={self.env.drop_cost}, AllAccepted={self.env.no_pass_cost}")

        # Set various subfunctions by decomposition status.
        if self.decomposed:
            interpret = lambda observation: self.interpreter(
                observation = observation,
                task_from = lambda base, target, state, start:
                    ('.'.join('.'.join(str(j) for j in i) for i in target[start:start+2] + [[int(state)]]), # task state placed in task key
                        tensorflatten([base, target[start:]]).to(device)))
            task_state = lambda k, _: int(k.split('.')[-1]) # task state from task key (used in simultaneous task limiting)
            exterpret = lambda action, observation: self.exterpreter(
                action = action,
                observation = observation,
                task_from = lambda observation: observation[1].tolist() + [float(observation[0].split('.')[-1])], # task state needs to be appended from key
                actions = 2) # 2 actions per task
        else:
            interpret = lambda observation: self.interpreter(
                observation = observation,
                task_from = lambda base, target, state, start:
                    ('.'.join('.'.join(str(j) for j in i) for i in target[start:start+2]),
                        tensorflatten([base, target[start:], state]).to(device))) # task state placed in task observation
            task_state = lambda _, v: int(v[-1]) # task state from task observation (used in simultaneous task limiting)
            exterpret = lambda action, observation: self.exterpreter(
                action = action,
                observation = observation,
                task_from = lambda observation: observation[1].tolist(),
                actions = 4) # 4 actions per task

        # Gradually relax the maximum simultaneous tasks.
        task_max = [int(len(self.agents) * (self.env.pool_limit + mod / 2.0)) for mod in range(0, 3)]

        # Iterate trhough training.
        iters = iterations + 1 # just here for output readability & later reference as a maximuum value
        for i in range(1, iters):
            print(f'Iteration # {i} / {iterations}')            

            # Gather trajectories.
            trajectories, critique = self.gather_trajectories(
                episodes=episodes,
                steps=steps,
                passengers=passengers,

                interpret = interpret,
                exterpret = exterpret,
                terminate = lambda step, schedule, observations: step > max(schedule.keys()) and all([len(o) == 0 for o in observations]),
                deconflict = lambda actions, NOOP=[3, -1, -1, 3, -1], MISS=[3, -1, -1, -1, -1]: [MISS if action != NOOP and any([action == other for other in actions[:agent]]) else action for agent, action in enumerate(actions)],
                react = lambda attempt, actual, actions=2 if self.decomposed else 4: (attempt - attempt % actions + (actual + 1) % actions, attempt),

                seeder = lambda e: i * episodes + e,
                task_limiter = lambda tasks, observation: tasks if sum([int(task_state(k, v) == 1) for k, v in observation.items()]) <= task_max[int(i / iters * len(task_max))] else 0
            )

            # Update learning statistics.
            for a, agent in enumerate(self.agents):
                agent.queue = list(set(key for episode in trajectories[a] for step in episode for key in step.observation.keys()))
                stats[a].tasks.list.append([[step.tasks for step in episode] for episode in trajectories[a]])
                stats[a].tasks.count.append(len(agent.queue))
                stats[a].losses.append(agent.optimize(trajectories[a], critique=critique, log_task=False, prefix=f"Agent # {a + 1} / {len(self.agents)} "))
                stats[a].rewards.append(torch.stack([torch.tensor([step.reward for step in episode], dtype=float, device=device).mean() for episode in trajectories[a]]).mean())

            # store partial results & checkpoint policies
            if plot_partial is not None and (i % plot_partial == 0 or i == iterations):
                partial_policies = [Simpellapig(agent) for agent in self.agents] # simplify policies for compactness
                config["learning"] = False # reflect inability for learning resumption on these policies
                config["iterations"] = i # record iterations so far rather than unreached total
                store(f"{result_dest}/i{i}", config=config, policies=partial_policies, losses=stats)

        # plot final results & store learned policies
        print(f"{LINE}\nPLOTTING...")
        store(result_dest, policies=self.agents, losses=stats)
        for a in range(len(self.agents)):
            plot_full(stats[a], result_dest, f'{result_dest[15:]}_{a}')
        return result_dest # used for resume chaining

    def train_std(self, decomposed=False, iterations=1000, plot_partial=50, eval:int=None):
        """
        Function to run the standard training suite, covering three setups in a 10x10 environment: 2, 3, and 4 agents with fourfold passengers over 100 time steps.

        Parameters:
            Decomposed: Flag for task-by-action instead of task-by-service.
            Iterations: Value for number of iterations over which to learn the policies. Standard setup has one episode per iteration.
            Plot Partial: Value for policy-checkpointing frequency.
            Eval: Value & flag for immediately evaluating the learned policy over this many episodes upon each completion of learning.
        """

        # Decomposition check
        dimensions = EllapigDimensions(0, 0, 40, 4) if decomposed else EllapigDimensions(0, 0)

        # Learn at 2, 3, & 4 agents with 8, 12, & 16 passengers respectively
        for a in range(2,5):
            self.reinit_environment(agents=a)
            self.reinit_agents_to_env(
                decomposed = decomposed,
                dimensions = dimensions)
            target = self.train(passengers=a*4, iterations=iterations, plot_partial=plot_partial)
            if eval is not None:
                self.run_std(target, learned_iterations=iterations, episodes=eval)

    def eval_training(self, source:str|list[str], dest:str="results/training.csv", *, overwrite=False, episodes=16, passengers=6):
        """
        Function for evaluation change over all the internal checkpoints of a training session. Writes to a csv by default.
        
        Parameters:
            Source: Value for directory or direcories containing training.
            Dest: Value for the target CSV.
            Overwrite: Flag to overwrite the destination file instead of adding to it.
            Episodes: The number of episodes to average for each checkpoint.
        """

        # Single source arugment iterability assignment.
        if isinstance(source, str):
            source = [source]

        # Overwrite argument to file opener mode conversion.
        if overwrite or not Path(dest).exists():
            with open(dest, 'w') as f:
                print("0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150", file=f)

        # Use standardized seeding across evaluation runs.
        seeder = lambda x: x # * 10

        # Results should be processed for mean & standard deviaiton of cumulative episodic rewards summed across agents.
        def process(trajectories:list[list[list[list[float]]]], critique:list[list[list[float]]]) -> tuple[float, float]:
            """Subfunction that returns the mean and standard deviation of rewards from a given trajectory/critique set."""
            rewards = torch.tensor([sum([sum([step.reward for step in trajectory[e]]) for trajectory in trajectories]) for e in range(episodes)], dtype=float)
            return rewards.mean().item(), rewards.std().item()

        # Evaluate training over the full source list.
        for src in source:

            # Affirm source validity
            if (path := Path(src)).exists() and path.is_dir() and len(paths:=[p for p in path.iterdir() if p.is_dir()]) > 0 and (a := self.reload_agents(paths[0])) > 0:
                print(f"Loaded {len(paths)} checkpoints containing {a} agents from {src}.") # Friendly update on current progress.

                # Initialize results lists for this training session.
                rewards, deviations = [], []

                # Evaluate random (unlearned or 0 learning iterations) policy in a standard environment.
                self.reinit_environment(a)
                self.reinit_agents_to_env()
                reward, deviation = self.run(episodes=episodes, passengers=passengers, iterations=0, prelude=False, seeder=seeder, process=process)
                rewards.append(str(reward))
                deviations.append(str(deviation))

                # Evalaute subsequent learning checkpoints.
                for p in paths:

                    # Affirm checkpointed policy validity.
                    if self.reload_agents(p, simplify=True) > 0:
                        # Evaluate random checkpointed policy in a standard environment.
                        self.reinit_environment(agents=len(self.agents))
                        reward, deviation = self.run(episodes=episodes, passengers=passengers, iterations=int(str(p).split('i')[-1]), prelude=False, seeder=seeder, process=process)
                        rewards.append(str(reward))
                        deviations.append(str(deviation))
                    else:
                        print(f"{p} agents not found.")
                
                # Write single-training session's results as the next to lines of the destination csv.
                with open(dest, 'a') as f:
                    print(','.join(rewards), file=f)
                    print(','.join(deviations), file=f)
            else:
                print(f"{src} not found.")

        # Plot Final results from csv. (use this call separately in case of crash)
        plot_training(pd.read_csv(dest, dtype=float).to_numpy(dtype=float))

    def run(self, episodes=1000, steps=100, passengers=6, *,
        iterations=0,
        prelude=True,
        seeder:Callable[[int], int]=lambda episode: int(datetime.datetime.now().microsecond),
        process:Callable[[list[list[list[list[float]]]], list[list[list[float]]]], Any]=None):
        """
        Function for evaluating learned policies.

        Parameters:
            Episodes: Value for the number of episodes to average over.
            Steps: Value for the number of steps in each episode.
            Passengers: Value to the maximum number of pssengers to add over the course of each episode.
            Iterations: Value for number of iterations over which the to-be-evaluated policy was learned. (used in filenaming)
            Prelude: Flag for disabling the prelude printout.
            Seeder: Function for generating a seed for each episode's initialization and scheduling from episode number.
            Process: Function for processing the resulting trajectories differently from standard plotting.
        """

        # Affirm operability.
        if len(self.agents) != self.env.num_agents:
            return None
        
        # Prelude
        if prelude:
            print(f"Experiment Specifications:")
            print(f"  {episodes} non-learning episodes of {steps} steps each{', decomposed' if self.decomposed else ''}")
            print(f"Environment Specifications:")
            print(f"  \"Rideshare\" {passengers}{' Open'} Tasks with {len(self.agents)} Agent(s) on a {self.env.grid_length}x{self.env.grid_width} Grid")
            print(f"  Costs: Accept={self.env.accept_cost}, Pickup={self.env.pick_cost}, Move={self.env.move_cost}, Dropoff={self.env.drop_cost}, AllAccepted={self.env.no_pass_cost}")

        # Decomposition check to set various subfunctions
        if self.decomposed:
            interpret = lambda observation: self.interpreter(
                observation = observation,
                task_from = lambda base, target, state, start:
                    ('.'.join('.'.join(str(j) for j in i) for i in target[start:start+2] + [[int(state)]]), # task state placed in task key
                        tensorflatten([base, target[start:]]).to(device)))
            exterpret = lambda action, observation: self.exterpreter(
                action = action,
                observation = observation,
                task_from = lambda observation: observation[1].tolist() + [float(observation[0].split('.')[-1])], # task state needs to be appended from key
                actions = 2) # 2 actions per task
            react = lambda attempt, actual: (attempt, attempt - attempt % 2 + int((actual + 1) % 4 != 0)) # attempt actions = 2, actual actions = 4
            plot = plot_runs_decomposed
        else:
            interpret = lambda observation: self.interpreter(
                observation = observation,
                task_from = lambda base, target, state, start:
                    ('.'.join('.'.join(str(j) for j in i) for i in target[start:start+2]),
                        tensorflatten([base, target[start:], state]).to(device))) # task state placed in task observation
            exterpret = lambda action, observation: self.exterpreter(
                action = action,
                observation = observation,
                task_from = lambda observation: observation[1].tolist(),
                actions = 4) # 4 actions per task
            react = lambda attempt, actual: (attempt, attempt - attempt % 4 + (actual + 1) % 4) # attempt actions = actual actions = 4
            plot = plot_runs

        # Data collection
        trajectories, critique = self.gather_trajectories(
            episodes=episodes,
            steps=steps,
            passengers=passengers,

            interpret = interpret,
            exterpret = exterpret,
            terminate = lambda step, schedule, observations: step > max(schedule.keys()) and all([len(o) == 0 for o in observations]),
            deconflict = lambda actions, NOOP=[3, -1, -1, 3, -1], MISS=[3, -1, -1, -1, -1]: [MISS if action != NOOP and any([action == other for other in actions[:agent]]) else action for agent, action in enumerate(actions)],
            react = react,

            seeder=seeder
        )

        # Data processing.
        if process is not None: # Use provided data processor if available.
            return process(trajectories, critique)
        else: # Standard plotting if no data processor provided.
            print(f"{LINE}\nPLOTTING...")
            store(result_dest := f"ride/{DATE}/e{episodes}_o{passengers}a{len(self.agents)}{'r' if iterations <= 0 else f'l{iterations}'}")
            for a in range(len(self.agents)):
                plot(trajectories[a], result_dest, f'{result_dest[15:]}_{a}')
            return result_dest
    
    def run_std(self, source:str, learned_iterations=1, episodes=1000):
        """
            Function to run the standard set of policy evaluations. The standard set covers three openness levels with 6, 9, and 12 passengers each.
            All evaluations are performed on a 10x10 grid with default costs.
        """

        # Assign random agents if requested.
        if source == "random":
            self.reinit_environment()
            self.reinit_agents_to_env()
            learned_iterations = 0
            
        # Assign random decomposed agents if requested.
        elif source == "randecomp":
            self.reinit_environment()
            self.reinit_agents_to_env(decomposed=True)
            learned_iterations = 0
            
        # Assign learned agents if requested.
        else:
            self.reload_agents(source, simplify=True)
            self.reinit_environment(len(self.agents))

        # Run policy evaluation in environments with 6, 9, & 12 passengers.
        for p in range(6, 13, 3):
            self.run(iterations=learned_iterations, passengers=p, episodes=episodes)

    def eval_rewards(self, source:str|list[str], dest="results/rewards.csv", *, overwrite=False, episodes=16):
        """
        Function for reward evaluation across learned policies. Writes to a csv by default.
        
        Parameters:
            Source: Value for directory or direcories containing policy.
            Dest: Value for the target CSV.
            Overwrite: Flag to overwrite the destination file instead of adding to it.
            Episodes: The number of episodes to average for each policy.
        """

        # Single source arugment iterability assignment.
        if isinstance(source, str):
            source = [source]

        # Overwrite argument to file opener mode conversion.
        if overwrite or not Path(dest).exists():
            with open(dest, 'w') as f:
                print("6,9,12", file=f)

        # Use standardized seeding across evaluation runs.
        seeder = lambda x: x # * 10

        for src in source:
            # Affirm source validity & Initiate standard environment.
            if (path := Path(src)).exists() and path.is_dir() and (a := self.reload_agents(src, simplify=True)) > 0:
                self.reinit_environment(a)

                # Friendly update on current progess.
                print(f"Loaded {a} agents from {src}.")

                # Initialize this policy's rewards list.
                rewards = []

                # Run policy evaluation in environments with 6, 9, & 12 passengers.
                for p in [6, 9, 12]:
                    rewards.append(str(self.run(
                        episodes=episodes,
                        passengers=p,
                        iterations=150,
                        prelude=False,
                        seeder=seeder,
                        process=lambda trajectories, critique: torch.tensor([sum([sum([step.reward for step in trajectory[e]]) for trajectory in trajectories]) for e in range(episodes)], dtype=float)
                            .mean()
                            .item())))

                # Each evaluation set adds a line to the csv
                with open(dest, 'a') as f:
                    print(','.join(rewards), file=f)
            else:
                print(f"{src} invalid or not found.")

        # Plot Final results from csv. (use this call separately in case of crash)
        plot_rewards(pd.read_csv(dest, dtype=float).to_numpy(dtype=float))

    def eval_pooling(self, source:str|list[str], dest="results/pooling.csv", *, overwrite=False, episodes=16):
        """
        Function for evaluation pooling efficacy of learned policies. Writes to a csv by default.
        
        Parameters:
            Source: Value for directory or direcories containing policies.
            Dest: Value for the target CSV.
            Overwrite: Flag to overwrite the destination file instead of adding to it.
            Episodes: The number of episodes to average for each policy.
        """

        # Single source arugment iterability assignment.
        if isinstance(source, str):
            source = [source]

        # Overwrite argument to file opener mode conversion.
        if overwrite or not Path(dest).exists():
            with open(dest, 'w') as f:
                print("serviced,unserviced,poolserviced,poolunserviced", file=f)

        # Use standardized seeding across evaluation runs.
        seeder = lambda x: x # * 10

        def service_tracker(trajectories:list[list[list[list[float]]]], critique:list[list[list[float]]], *, passengers=6):
            """Function for getting passenger pooling & servicing data from trajectories."""

            # Prep passenger counts.
            unpooled_unserviced = 0
            unpooled_serviced = 0
            pooled_unserviced = 0
            pooled_serviced = 0
            
            # Calculate the above from the trajectories gathered.
            for e in range(episodes):
                for trajectory in trajectories:

                    # Prep driver data.
                    tasks = set() # All tasks attempted to be serviced.
                    dropped = set() # All tasks successfully serviced.
                    pooled = set() # All tasks serviced alongside another task.

                    # Gather set data
                    for step in trajectory[e]:
                        pooling = set() # See later use.

                        # Look at all observed passengers
                        for k, v in step.observation.items():
                            # Ignore unaccepted passengers.
                            if (ts := task_state(k, v)) > 1:
                                tasks.add(passenger_from(k))
                                # Passengers in the car might be pooling with someone else.
                                if ts > 2:
                                    pooling.add(passenger_from(k))

                        # Which passenger was dropped off if one was.
                        if step.reward > 1:
                            dropped.add(passenger_from(list(step.observation.keys())[step.action // actions]))

                        # More than one passenger are in a car together.
                        if len(pooling) > 1:
                            for passenger in pooling:
                                pooled.add(passenger)

                    # Sum stats across agents.
                    # unpooled_unserviced += len(tasks - pooled - dropped)
                    unpooled_serviced += len(tasks & dropped - pooled)
                    pooled_unserviced += len(tasks & pooled - dropped)
                    pooled_serviced += len(tasks & pooled & dropped)

            # Average stats across episodes.
            # unpooled_unserviced /= episodes
            unpooled_serviced /= episodes
            pooled_unserviced /= episodes
            pooled_serviced /= episodes

            # Include unaccepted passengers (overwrites previous exclusion).
            unpooled_unserviced = passengers - unpooled_serviced - pooled_unserviced - pooled_serviced

            # Return the full statset gathered.
            return unpooled_serviced, unpooled_unserviced, pooled_serviced, pooled_unserviced

        for src in source:
            # Affirm source validity & Initiate standard environment.
            if (path := Path(src)).exists() and path.is_dir() and (a := self.reload_agents(src, simplify=True)) > 0:
                self.reinit_environment(a)

                # Friendly update on current progess.
                print(f"Loaded {a} agents from {src}.")

                # Decomposition check on loaded agents.
                if self.decomposed:
                    actions=2
                    task_state = lambda k, _: int(k.split('.')[-1]) # task state from task key
                    passenger_from = lambda task: '.'.join(task.split('.')[:-1])
                else:
                    actions=4
                    task_state = lambda _, v: int(v[-1]) # task state from task observation
                    passenger_from = lambda task: task

                # Run policy evaluation in environments with 6, 9, & 12 passengers.
                for p in [6, 9, 12]:
                    service = self.run(
                        episodes=episodes,
                        passengers=p,
                        iterations=150,
                        prelude=False,
                        seeder=seeder,
                        process=lambda trajectories, critique: service_tracker(trajectories, critique, passengers=p))

                    # Each evaluation adds a line to the csv
                    with open(dest, 'a') as f:
                        print(','.join([str(s) for s in service]), file=f)
            else:
                print(f"{src} invalid or not found.")
        
        # Plot Final results from csv. (use this call separately in case of crash)
        plot_pooling(pd.read_csv(dest, dtype=float).to_numpy(dtype=float))

    def gather_trajectories(self,
        interpret:Callable[[list], dict[str, list[float]]],
        exterpret:Callable[[float|int, dict[str, list[float]]], list[int]],
        terminate:Callable[[int, dict[int, int], list[dict[str, list[float]]]], bool],
        deconflict:Callable[[list[list[int]]], list[list[int]]],
        react:Callable[[int, int], tuple[int, int]],
        episodes=1,
        steps=100,
        *,
        seeder:Callable[[int], int]=lambda episode: int(datetime.datetime.now().microsecond),
        task_limiter:Callable[[int, dict[str, list[float]]], int]=lambda tasks, observation: tasks,
        passengers=6):
        """
        Returns trajectories for each agent and a collective critique from operating over one or more episodes.
        
        Parameters:
            Interpret: Function for converting Rideshare getObsFromState to task-specific observation dictionaries.
            Exterpret: Function for converting Ellapig action choice value and task-specific observation into Rideshare action node.
            Terminate: Function for determining if the environment is in a terminable state.
            Deconflict: Function for removing duplicates from a list of Rideshare action nodes.
            React: Function for reprocessing Ellapig action choice to match corresponding final Rideshare action node.
            Seeder: Function for generating a seed for each episode's initialization and scheduling from episode number.
            Task Limiter: Function for limiting the scheduled passenger additions by maximum number of simultaneous tasks.
            Episodes: Value for the number of episodes to gather trajectories from.
            Steps: Value for the maximum number of steps this episode.
            Passengers: Value for the maximum number of pssengers to add over the course of each episode.
        """

        # Initialize trajectories and critique.
        trajectories = [[] for a in range(len(self.agents))]
        critique = []

        # Run episodes.
        for episode in range(episodes):
            # print(f"Episode # {episode} / {episodes}")

            # Set seed for episode initialization and scheduling.
            seed = seeder(episode)

            # Run and collect trajectory from episode.
            for a, trajectory in enumerate( # Split trajectories by agent.
                self.run_episode( # Run episode.
                    interpret = interpret,
                    exterpret = exterpret,
                    terminate = terminate,
                    deconflict = deconflict,
                    react = react,
                    task_limiter = task_limiter,
                    steps = steps,
                    openness = True,
                    seed = seed,
                    passengers = passengers)):
                
                # Trajectory collection.
                trajectories[a].append(trajectory) # Collect trajectories by agent.
                critique.append(trajectory) # Collect unified critique.

        # Return collected trajectories.
        return trajectories, critique

    def run_episode(self,
        interpret:Callable[[list], dict[str, list[float]]],
        exterpret:Callable[[float|int, dict[str, list[float]]], list[int]],
        terminate:Callable[[int, dict[int, int], list[dict[str, list[float]]]], bool],
        deconflict:Callable[[list[list[int]]], list[list[int]]],
        react:Callable[[int, int], tuple[int, int]],
        task_limiter:Callable[[int, dict[str, list[float]]], int],
        steps=100,
        *,
        openness=True,
        seed=42,
        passengers=6):
        """
        Returns a single-episode trajectory for each agent from operation.
        
        Parameters:
            State: Current (initial) state of the environment.
            Interpret: Function for converting Rideshare getObsFromState to task-specific observation dictionaries.
            Exterpret: Function for converting Ellapig action choice value and task-specific observation into Rideshare action node.
            Terminate: Function for determining if the environment is in a terminable state.
            Deconflict: Function for removing duplicates from a list of Rideshare action nodes.
            React: Function for reprocessing Ellapig action choice to match corresponding final Rideshare action node.
            Task Limiter: Function for limiting the scheduled passenger additions by maximum number of simultaneous tasks.
            Steps: Value for the maximum number of steps this episode.
            Openness: Flag for allowing new passengers after environment initialization.
            Seed: Value for randomization used in episode initialization and scheduling.
            Passengers: Value used for episode scheduling.
        """

        # Initialize trajectory for each agent.
        trajectory = [[] for a in range(len(self.agents))]
        # drivers = [[1,1],[1,8],[8,1],[8,8]] # Used for fixing driver locations (must be set and (un)commented manually).

        # Initialize episode
        set_seed(seed)
        state = self.env.reset(  # Initialize episode.
            step = 0,
            # driver_locations=[drivers[a] for a in range(len(self.agents))], # Used for fixing driver locations (must be set and (un)commented manually).
            num_passengers = (existing := random.randint(len(self.agents) - 1, len(self.agents) + 3))) # from Gayathri's trainer

        # Schedule episode
        set_seed(seed)
        schedule = generate_task_schedule(steps, passengers - existing)
        if schedule == {}:
            schedule[-1] = 0

        # Run episode steps
        for step in range(steps):
            #print(f"       Step # {step_no + 1} / {steps}")

            # Observe the current state
            observation = self.env.getObsFromState(state) # collect the new full observation
            traj = [{"observation": interpret(observation[a])} for a in range(len(self.agents))] # interpret the full observation into a dict of task-specific observations
            
            # Respond as agents to state observation by terminating or selecting actions.
            if terminate(step, schedule, [traj[a]["observation"] for a in range(len(self.agents))]):
                break
            for a, agent in enumerate(self.agents):
                traj[a]["action"] = agent.forward(traj[a]["observation"]) # generate the raw action
            actions = deconflict([exterpret(traj[a]["action"].item(), traj[a]["observation"]) for a in range(len(self.agents))]) # exterpret the raw actions into the environment
            
            # Respond as the environment to the selected actions.
            if openness and (step in schedule):
                exo_var = True
                new_tasks = task_limiter(schedule[step], traj[0]["observation"])
            else:
                exo_var, new_tasks = False, None
            state, rewards, _ = self.env.step(len(trajectory), actions, openness=openness, exo_var=exo_var, num_tasks=new_tasks) # progress the environment
            
            # Update trajectory progress.
            for a in range(len(self.agents)):
                traj[a]["action"], traj[a]["reaction"] = react(traj[a]["action"], actions[a][3])
                trajectory[a].append(Trajectory(
                    observation = traj[a]["observation"],
                    tasks = len(traj[a]["observation"]),
                    action = traj[a]["action"],
                    reaction = traj[a]["reaction"],
                    reward = rewards[a]
                )) # add step to current trajectory

        # Retun completed trajectory.
        return trajectory

# Initializes an instance of the rideshare trainer for later use.
r = RideshareTrainer()

# Example calls for standard training of full-service and per-stage tasks respectively.
# r.train_std(iterations=150, plot_partial=10, eval=100)
# r.train_std(decomposed=True, iterations=150, plot_partial=10, eval=100)

# Example call for evaluation over training checkpoints across various training runs.
# r.eval_training([
#         "results/ride/2024.04.16/i150_o8a2/",
#         "results/ride/2024.04.16/i150_o12a3/",
#         "results/ride/2024.04.16/i150_o16a4/",
#         "results/ride/2024.04.17/i150_o8a2/",
#         "results/ride/2024.04.17/i150_o12a3/",
#         "results/ride/2024.04.18/i150_o16a4/"],
#     dest="results/training.csv", overwrite=True)

# plot_training(pd.read_csv("results/training.csv", dtype=float).to_numpy(dtype=float))

# Example call for evaluation over rewards across training runs by decomposition.
# r.eval_rewards([
#         "results/ride/2024.04.16/i150_o8a2/i150",
#         "results/ride/2024.04.16/i150_o12a3/i150",
#         "results/ride/2024.04.16/i150_o16a4/i150",
#         "results/ride/2024.04.17/i150_o8a2/i150",
#         "results/ride/2024.04.17/i150_o12a3/i150",
#         "results/ride/2024.04.18/i150_o16a4/i150"],
#     overwrite=True)

# plot_rewards(pd.read_csv("results/rewards.csv", dtype=float).to_numpy(dtype=float))

# Example call for evaluation over pooling across training runs by decomposition.
# r.eval_pooling([
#         "results/ride/2024.04.16/i150_o8a2/i150",
#         "results/ride/2024.04.16/i150_o12a3/i150",
#         "results/ride/2024.04.16/i150_o16a4/i150",
#         "results/ride/2024.04.17/i150_o8a2/i150",
#         "results/ride/2024.04.17/i150_o12a3/i150",
#         "results/ride/2024.04.18/i150_o16a4/i150"],
#     overwrite=False)

# plot_pooling(pd.read_csv("results/pooling.csv", dtype=float).to_numpy(dtype=float))
