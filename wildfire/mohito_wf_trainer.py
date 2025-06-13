import os, copy, tqdm, argparse, random, yaml
from typing import Dict, List, Tuple, Any
import torch

from free_range_zoo.utils.env import BatchedAECEnv
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from free_range_zoo.envs.wildfire.configs.uai_experiment import UAI_2025_ol_config
from free_range_zoo.envs import wildfire_v0

from mohito.mohito import ActorReturn, Mohito
from mohito.mohito_wrapper import mohito_hypergraph_wrapper_v0


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

args = parser.parse_args()

checkpoint_path = os.path.join(args.output_dir, "mohito_checkpoints")
validation_path = os.path.join(args.output_dir, "mohito_validation")

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
validation_parameters = mohito_hyperparameters.pop('validation')

# %%

conf_params = {
    # 'openness_level': 1,
    # 'starting_state': 1,
    'fire_types': [ #see UAI_2025_ol_config for details
        [["j0", "j2", "j0"], ["j1", "j2", "j1"]],  #ss0
        [["j1", "j2", "j1"], ["j0", "j2", "j0"]],  #ss1
        [["j0", "j2", "j0"], ["j1", "j2", "j1"]],  #ss2
    ],
    'fire_rewards': [0, 20, 400],
    'burnout_penalty': [0, -10, -25],
    'base_spread': 'ol', #determine base spread by OL
    'random_ignition_prob': 0.1
}

conf: dict[int, list[UAI_2025_ol_config]] = {}

for openness_level in range(3):
    #composition over the starting states
    OL_conf = [
        UAI_2025_ol_config(openness_level=openness_level + 1,
                           starting_state=j,
                           **conf_params) for j in range(3)
    ]
    conf[openness_level + 1] = OL_conf




if training_parameters['reprod_training']:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(training_parameters['seed'])

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
    show_bad_actions=False,
    observe_other_suppressant=False,
    observe_other_power=False,
    observe_burn_time=True,
)
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
                self.validate(policies)
                self._validation_counter = 0
                self._total_validation_counter += 1
            else:
                self._validation_counter += 1

            if (self._checkpoint_counter % self.steps_per_checkpoint == 0
                    and self._checkpoint_counter > 0):
                for name, p in policies.items():
                    torch.save(
                        p.state_dict(),
                        f'{checkpoint_path}/{name}_{self._total_checkpoints}.pt'
                    )
                self._checkpoint_counter = 0
                self._total_checkpoints += 1
            else:
                self._checkpoint_counter += 1

        elif self.do_first_validation:
            self.validate(policies)
            self.do_first_validation = False

        return did_update, self.has_converged

    def validate(self, policies: Dict[str, Mohito]):

        #override dropout
        [p.eval() for p in policies.values()]

        R = []

        for ss, validation_env in enumerate(self.validation_envs):
            for seed in self.validation_seeds:
                obs, info = validation_env.reset(
                    seed=seed,
                    options={
                        'log_description':
                        f'updates:{self._update_counter};validation:{self._total_validation_counter};seed:{seed};ss:{ss};openness_level:{args.openness_level}'
                    })
                r = []
                while not torch.all(validation_env.finished):
                    [p.observe(obs[p.name][0]) for p in policies.values()]
                    actions = {
                        name:
                        p.act(action_space=validation_env.action_space(name))
                        for name, p in policies.items()
                    }
                    obs, rewards, terminations, truncations, info = validation_env.step(
                        actions)
                    r.append(rewards['firefighter_1'])

                R.append(torch.stack(r).sum())

        #?used for external validation logging if wanted
        R = torch.stack(R)


        #set dropout
        [p.train() for p in policies.values()]

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
        configuration=conf[args.openness_level][ss],
        device=torch.device('cpu'),  #gpu is slower typically...
        log_directory=None,
        override_initialization_check=True,
        show_bad_actions=False,
        observe_other_suppressant=training_parameters['observe_other_suppressant'],
        observe_other_power=False,
        observe_burn_time=True,
    ) for ss in range(3)
]

val_envs = [
    wildfire_v0.parallel_env(
        max_steps=100,
        parallel_envs=1,
        configuration=conf[args.openness_level][ss],
        device=torch.device('cpu'),  #gpu is slower typically...
        log_directory=validation_path,
        override_initialization_check=False,
        show_bad_actions=False,
        observe_other_suppressant=training_parameters['observe_other_suppressant'],
        observe_other_power=False,
        observe_burn_time=True,
        ) for ss in range(3)
]

for j, (env, val_env) in enumerate(zip(envs, val_envs)):
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
    validation_seeds=validation_parameters['seeds'])

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

        observations, infos = env.reset()
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
            did_update_did_converge: Tuple[bool,
                                           Dict[str,
                                                bool]] = update_controller(
                                                    policies=agent_policies,
                                                    o=observations,
                                                    a=agent_actions,
                                                    r=rewards,
                                                    o_prime=observation_prime,
                                                )

            observations = observation_prime
