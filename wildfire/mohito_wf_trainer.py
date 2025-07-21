import argparse, yaml, tqdm, os, copy, random, sys
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any

from free_range_zoo.utils.env import BatchedAECEnv
from free_range_zoo.envs.wildfire.configs.uai_experiment import UAI_2025_ol_config
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from free_range_zoo.envs import wildfire_v0
from mohito.mohito_wrapper import mohito_hypergraph_wrapper_v0
from mohito.mohito import ActorReturn, CriticReturn
from mohito.mohito import Mohito


parser = argparse.ArgumentParser(
    description="MOHITO Wildfire Trainer",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '-ol',
    '--openness_level',
    type=int,
)
parser.add_argument(
    '-m',
    "--mohito_config_file",
    type=str,
)
parser.add_argument(
    '-o',
    '--output_dir',
    type=str,
)
parser.add_argument(
    '-bs',
    '--base_spread',
    type=float,
    default=None,
    help=
    "Base spread rate of the fire. If None, uses the default value for the openness level.",
)
parser.add_argument(
    '-rip',
    '--random_ignition_prob',
    type=float,
    default=None,
    help=
    "Probability of random ignition. If None, uses the default value of 0.1.",
)

args = parser.parse_args()

checkpoint_path = os.path.join(args.output_dir, "mohito_checkpoints")
validation_path = os.path.join(args.output_dir, "mohito_validation")

try:
    os.mkdir(args.output_dir)
except:
    print("Output directory already exists!")
    pass

try:
    os.mkdir(checkpoint_path)
except:
    print("mohito_checkpoint folder already exists!")
    pass

try:
    os.mkdir(validation_path)
except:
    print("mohito_validation folder already exists!")
    pass

# %%
with open(args.mohito_config_file, 'r') as file:
    mohito_hyperparameters = yaml.safe_load(file)

print(yaml.dump(mohito_hyperparameters, default_flow_style=False))

training_parameters = mohito_hyperparameters.pop('training')
convergence_parameters = mohito_hyperparameters.pop('convergence')
validation_parameters = mohito_hyperparameters.pop('validation')
environment_parameters = mohito_hyperparameters.pop('environment')


# %%
conf_params = {
    # 'openness_level': 1,
    # 'starting_state': 1,
    'fire_types': [
    ],
    'fire_rewards': [0, 20, 400],
    'burnout_penalty': [0, -10, -25],
    'base_spread':
    args.base_spread if args.base_spread is not None else
    'ol',  # will be set to the default value for the openness level
    'random_ignition_prob': args.random_ignition_prob if args.random_ignition_prob is not None else 0.1,
    'intensity_increase_prob':
    training_parameters[
        'intensity_increase_prob'],  # 1.0 for always intensity increase
    'new_ignition_temperatures': [2, 3],
    'use_stochastic_ignition_temperature':
    True
}
#%%


#!starting states for experimentation
ss_train = training_parameters['starting_states']

with open(
    os.path.join(args.output_dir, 'command_line_args.yaml'), 'w') as f:
    yaml.dump({
        'openness_level': args.openness_level,
        'mohito_config_file': args.mohito_config_file,
        'output_dir': args.output_dir,
    } | conf_params | environment_parameters | mohito_hyperparameters | training_parameters | validation_parameters | convergence_parameters, f)

conf: dict[int, list[UAI_2025_ol_config]] = {}

for openness_level in range(3):
    #composition over the starting states
    OL_conf = [
        UAI_2025_ol_config(openness_level=openness_level + 1,
                           starting_state=ss,
                           **conf_params) for ss in ss_train
    ]
    conf[openness_level + 1] = OL_conf


if training_parameters['reprod_training']:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(training_parameters['seed'])
    random.seed(training_parameters['seed'])
    np.random.seed(training_parameters['seed'])

# %% [markdown]
# get agents

# %%
env = wildfire_v0.parallel_env(
    max_steps=100,
    parallel_envs=1,
    configuration=conf[args.openness_level][0],
    device=torch.device('cpu'),
    log_directory=None,
    override_initialization_check=True,
    **environment_parameters,
    terminate_on_no_fires=not training_parameters['terminate_on_no_fires'],
    verbose=False)
env.reset()

num_agents = len(env.agents)
agent_names = copy.copy(env.agents)

env.close()
print(agent_names)

# %%

agent_policies = {
    name: Mohito(name=name, **mohito_hyperparameters)
    for name in agent_names
}

print(agent_policies[agent_names[0]])

# %%


class MohitoCriticController:
    """
    Holds the replay buffer, handles exchanging observations + actions between policies at training time, and running validation.
    """

    def __init__(self,
                 agent_names: List[str],
                 buffer_size: int,
                 batch_size: int,
                 experiences_before_update: int,
                 backups_per_update: int,
                 steps_per_checkpoint: int,
                 steps_per_validation: int,
                 validation_envs: List[BatchedAECEnv],
                 validation_seeds: List[int],
                 k_convergence: int,
                 convergence_diff_bound: float,
                 minimum_convergence_bound: float,
                 wait_until_full: bool = True):
        """
        Args:
            agent_names (List[str]): List of agent names to update.
            buffer_size (int): Size of the replay buffer.
            batch_size (int): Size of the batch to sample from the replay buffer.

            experiences_before_update (int): How many new experiences to add to the replay buffer before sampling.
            backups_per_update (int): How many times to backup / update per reaching the # of new experiences.
            #For the original paper implementation new_experiences_per = 1, and num_updates_per = 1.

            steps_per_validation (int): How many steps to take before running validation.
            steps_per_checkpoint (int): How many steps to take before saving a checkpoint.
            validation_envs List[(BatchedAECEnv)]: Environments to use for validation (different starting states).
            validation_seeds (List[int]): List of seeds to use for validation.

            k_convergence (int): Number of validation runs to consider convergence.
            convergence_diff_bound (float): The bound for convergence, if the difference in k rewards are all less than or equal to the current reward, we have converged.
            minimum_convergence_bound (float): The minimum reward to consider convergence.
            wait_until_full (bool): If True, will not sample from the replay buffer until it is full.
        """
        self.agent_names = agent_names
        self.has_converged = {name: False for name in agent_names}

        self.buffer = []
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        assert self.batch_size <= self.buffer_size, "Batch size must be less than or equal to buffer size."

        self.experiences_before_update = experiences_before_update
        self.backups_per_update = backups_per_update
        self.wait_until_full = wait_until_full

        self.k_convergence = k_convergence
        self.convergence_diff_bound = convergence_diff_bound
        self.minimum_convergence_bound = minimum_convergence_bound
        self._last_k_R = []  # last k rewards for convergence checks

        self._experience_counter = 0
        self._validation_counter = 0
        self._total_validation_counter = 0
        self._update_counter = 0
        self.do_first_validation = True
        self._checkpoint_counter = 0
        self._total_checkpoints = 0

        self.validation_envs = validation_envs
        self.steps_per_validation = steps_per_validation
        self.steps_per_checkpoint = steps_per_checkpoint
        self.validation_seeds = validation_seeds

    def __call__(self, policies: Dict[str, Mohito], o, a, r,
                 o_prime) -> Tuple[bool, Dict[str, bool]]:
        """
        Adds the experience to the replay buffer, then if the buffer is full, and it is time to update then sample and update policies.

        Args:
            policies (Dict[str, Mohito]): Dictionary of policies to update.
            o (Dict[str, Any]): Observations from the environment.
            a (Dict[str, Any]): Actions taken by the policies.
            r (Dict[str, float]): Rewards received from the environment.
            o_prime (Dict[str, Any]): Next observations from the environment.
        Returns:
            Tuple[bool, Dict[str, bool]]: Tuple of whether the buffer was full and a dictionary of convergence checks for each policy following validation reqs.

        """
        #add experience to the replay buffer
        self.buffer.append({'o': o, 'a': a, 'r': r, 'o_prime': o_prime})

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            self._experience_counter += 1

        did_update = False

        #update
        if (len(self.buffer) == self.buffer_size or not self.wait_until_full):
            if (self._experience_counter %
                    self.experiences_before_update == 0):
                for j in range(self.backups_per_update):
                    batch = random.sample(self.buffer, self.batch_size)
                    self.update(policies, batch)
                    self._update_counter += 1

                self._experience_counter = 0
                did_update = True

            #validate
            if (self._validation_counter % self.steps_per_validation == 0
                    and self._validation_counter > 0):
                R = self.validate(policies)
                self._validation_counter = 0
                self._total_validation_counter += 1

                #check convergence
                if len(self._last_k_R) == self.k_convergence:

                    if torch.all(
                            torch.abs(torch.stack(self._last_k_R) -
                                      R) <= self.convergence_diff_bound
                    ) and R > self.minimum_convergence_bound:
                        #if the last k rewards are all less than or equal to the current reward, we have converged
                        self.has_converged = {
                            name: True
                            for name in policies.keys()
                        }

                    self._last_k_R.pop(0)
                self._last_k_R.append(R)
            else:
                self._validation_counter += 1

            if (self._checkpoint_counter % self.steps_per_checkpoint == 0
                    and self._checkpoint_counter > 0) or any(
                        self.has_converged.values()):
                for name, p in policies.items():
                    torch.save(
                        p.state_dict(),
                        f'{checkpoint_path}/{name}_{self._total_checkpoints}.pt'
                        if not any(self.has_converged.values()) else
                        f'{checkpoint_path}/{name}_{self._total_checkpoints}_converged.pt'
                    )
                self._checkpoint_counter = 0
                self._total_checkpoints += 1
            else:
                self._checkpoint_counter += 1

        elif self.do_first_validation:
            self.validate(policies)
            self.do_first_validation = False

        return did_update, self.has_converged

    def validate(self, policies: Dict[str, Mohito]) -> float:
        """
        Runs validation and returns the average reward for use in convergence checks.
        """

        #override dropout
        [p.eval() for p in policies.values()]

        R_ss = []

        for ss_ind, validation_env in enumerate(self.validation_envs):

            ss = validation_env.user_logged_ss_ind

            R = []
            for seed in self.validation_seeds:
                obs, info = validation_env.reset(
                    seed=seed,
                    options={
                        'log_description':
                        f'updates:{self._update_counter};validation:{self._total_validation_counter};seed:{seed};ss:{ss};openness_level:{args.openness_level}'
                    })
                r = []
                step = 0

                while not torch.all(validation_env.finished):
                    [p.observe(obs[p.name][0]) for p in policies.values()]

                    agent_actions = {
                        agent_name:
                        agent_policies[agent_name].act(
                            action_space=env.action_space(agent_name),
                            return_logits=True)
                        for agent_name in env.agents
                    }  # Policy action determination here

                    actions = {
                        agent_name: act[0]
                        for agent_name, act in agent_actions.items()
                    }
                    logits = {
                        agent_name + "_logits": act[1]
                        for agent_name, act in agent_actions.items()
                    }

                    obs, rewards, terminations, truncations, info = validation_env.step(
                        actions)
                    r.append(rewards['firefighter_1'])

                    logit_file = os.path.join(validation_path, 'logit.csv')

                    pd.DataFrame(
                        logits | {
                            'r_1': rewards['firefighter_1'],
                            'step': step,
                            'seed': seed,
                            'ss': ss,
                            'checkpoint': self._total_checkpoints
                        }).to_csv(logit_file,
                                  mode='a',
                                  header=not os.path.exists(logit_file))
                    step += 1

                R.append(torch.stack(r).sum())
            R_ss.append(torch.stack(R, dim=0))

        #stack by starting state
        R = torch.stack(R_ss, dim=0)
        #average over starting states
        R = R.mean(dim=0)

        #?log validation means here (if you want some non logfile logging)

        #set dropout
        [p.train() for p in policies.values()]

        return R.mean()

    def update(self, policies: Dict[str, Mohito],
               batch: List[Tuple[Dict[str, Any]]]):
        """
        Get ed and ed_prime for updates, then update the critic and actor networks.

        Args:
            policies (Dict[str, Mohito]): Dictionary of policies to update.
            batch (List[Tuple[Dict[str, Any]]]): Batch of experiences to sample from the replay buffer.
        """

        #get ed and ed_prime for updates
        hyperedges = {
            name: p.update_forward(batch)
            for name, p in policies.items()
        }

        #update the networks
        for name, p in policies.items():
            p.update(batch, hyperedges=hyperedges)


# %%
envs = [
    wildfire_v0.parallel_env(
        max_steps=100,
        parallel_envs=1,
        configuration=conf[args.openness_level][ss_ind],
        device=torch.device('cpu'),  #gpu is slower typically...
        log_directory=None,
        override_initialization_check=True,
        **environment_parameters,
        terminate_on_no_fires=training_parameters['terminate_on_no_fires'],
        verbose=False) for ss_ind, ss in enumerate(ss_train)
]

val_envs = [
    (
    ss,
    wildfire_v0.parallel_env(
        max_steps=100,
        parallel_envs=1,
        configuration=conf[args.openness_level][ss_ind],
        device=torch.device('cpu'),  #gpu is slower typically...
        log_directory=validation_path,
        override_initialization_check=False,
        **{
            **environment_parameters,
            **validation_parameters['env_params']
        },
        verbose=True))
        for ss_ind, ss in enumerate(ss_train)
]

for j, (env, (ss_ind,val_env)) in enumerate(zip(envs, val_envs)):
    env.reset()
    env = action_mapping_wrapper_v0(env)
    env.reset()
    env = mohito_hypergraph_wrapper_v0(env)
    env.reset(seed=training_parameters['seed'])
    envs[j] = env

    val_env.reset()
    val_env = action_mapping_wrapper_v0(val_env)
    val_env.reset()
    val_env = mohito_hypergraph_wrapper_v0(val_env)
    val_env.reset()
    val_env.user_logged_ss_ind = ss_ind
    val_envs[j] = val_env

# %%
update_controller = MohitoCriticController(
    agent_names=agent_names,
    buffer_size=training_parameters['buffer_size'],
    batch_size=training_parameters['batch_size'],
    experiences_before_update=training_parameters['experiences_before_update'],
    backups_per_update=training_parameters['backups_per_update'],
    steps_per_checkpoint=training_parameters['steps_per_checkpoint'],
    wait_until_full=training_parameters['wait_until_full'],
    validation_envs=val_envs,
    steps_per_validation=validation_parameters['frequency'],
    validation_seeds=validation_parameters['seeds'],
    k_convergence=convergence_parameters['k_convergence'],
    convergence_diff_bound=convergence_parameters['convergence_diff_bound'],
    minimum_convergence_bound=convergence_parameters[
        'minimum_convergence_bound'])

round_robin = 0

with tqdm.tqdm(total=training_parameters['num_episodes'], ascii="=/🐦") as pbar:
    for ep_no in range(training_parameters['num_episodes']):
        pbar.update(1)
        # Reset the environment
        if training_parameters['starting_state_strategy'] == 'round_robin':
            env = envs[round_robin]
            round_robin = (round_robin + 1) % len(envs)
        elif training_parameters['starting_state_strategy'] == 'random':
            env = random.choice(envs)
        else:
            raise ValueError(
                f"Unknown starting state strategy: {training_parameters['starting_state_strategy']}"
            )

        observations, infos = env.reset(seed=training_parameters['seed']+ep_no)
        while not torch.all(env.finished):

            # Get actions / Hyperedges from actors
            actor_outputs: Dict[str, ActorReturn] = {
                name:
                p(observations[name],
                  action_space=env.action_space(name),
                  explore=True)
                for name, p in agent_policies.items()
            }

            # Select actions
            agent_actions = {
                agent_name: actor_outputs[agent_name].task_action
                for agent_name in env.agents
            }

            observation_prime, rewards, terminations, truncations, infos = env.step(
                agent_actions)

            # Call utility to update the policies, validate, and check convergence
            did_update_did_converge = update_controller(
                policies=agent_policies,
                o=observations,
                a=agent_actions,
                r=rewards,
                o_prime=observation_prime,
            )

            did_update, did_converge = did_update_did_converge

            if any(list(did_converge.values())):
                print(f"Convergence reached at episode {ep_no}.")
                sys.exit(0)

            observations = observation_prime
