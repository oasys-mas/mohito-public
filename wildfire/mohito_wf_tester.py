from free_range_zoo.envs.wildfire.configs.uai_experiment import UAI_2025_ol_config
from free_range_zoo.envs import wildfire_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from free_range_zoo.wrappers.space_validator import space_validator_wrapper_v0
from mohito.mohito_wrapper import mohito_hypergraph_wrapper_v0
import torch
import tqdm, os, yaml
from mohito.mohito import Mohito
import argparse

#%%
torch.use_deterministic_algorithms(True, warn_only=True)

parser = argparse.ArgumentParser(description='MOHITO wildfire experiment')
parser.add_argument('-o',
                    '--output_folder',
                    type=str,
                    help='Output directory for results')
parser.add_argument('-m',
                    '--mohito_config',
                    type=str,
                    help='mohito configuration file')
parser.add_argument('-p',
                    '--policy_path',
                    type=str,
                    help='path to the policy root directory')
parser.add_argument('-ol',
                    '--openness_level',
                    type=int,
                    choices=[1, 2, 3],
                    help='Openness level for the experiment')
parser.add_argument('--low',
                    type=int,
                    default=0,
                    help='Lowest checkpoint to test')
parser.add_argument('--high',
                    type=int,
                    default=-1,
                    help='Highest checkpoint to test')

parser.add_argument('--seed',
                    type=int,
                    default=30,
                    help='Random seed for reproducibility')
parser.add_argument(
    '-n',
    type=int,
    default=60,
    help=
    'Number of episodes to run for each openness level. Note reprod is dependent on both seed and n!'
)

args = parser.parse_args()

#%%

#seed for testing
episodes_per_ss = args.n // 3
assert args.n % 3 == 0, "Number of episodes must be divisible by 3 for equal distribution across starting states."

agent_names = ['firefighter_1', 'firefighter_2']

#%%
conf_params = {
    # 'openness_level': 1,
    # 'starting_state': 1,
    'fire_types': [
        [["j0", "j2", "j0"], ["j1", "j2", "j1"]],  #ss0
        [["j1", "j2", "j1"], ["j0", "j2", "j0"]],  #ss1
        [["j0", "j2", "j0"], ["j1", "j2", "j1"]],  #ss2
    ],
    'fire_rewards': [0, 20, 400],
    'burnout_penalty': [0, -10, -25],
    'base_spread':
    'ol',
    'random_ignition_prob':
    0.1
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

#%%
with open(args.mohito_config, 'r') as file:
    mohito_hyperparameters = yaml.safe_load(file)

train_conf = mohito_hyperparameters.pop('training')
mohito_hyperparameters.pop('validation')

all_checkpoints = list(
    set([
        f.split('_')[-1].split('.pt')[0] for f in os.listdir(args.policy_path)
    ]))

filter_checkpoints = [
    check for check in all_checkpoints
    if (int(check) <= args.high or args.high == -1) and int(check) >= args.low
]

seeds = list(range(args.seed, args.seed + 1 + args.n))

print(f"Testing MOHITO over start seed: {args.seed}-{args.seed + args.n}")
print("Openness level:", args.openness_level)
print("Checkpoints to test:", filter_checkpoints[0], "-",
      filter_checkpoints[-1])
print("Episodes per starting state:", episodes_per_ss)
print("Total episodes:", len(filter_checkpoints) * episodes_per_ss * 3)

with tqdm.tqdm(total=episodes_per_ss * 3 * len(filter_checkpoints),
               ascii="=/🐦") as pbar:
    for checkpoint in filter_checkpoints:

        agent_policies = {
            name: Mohito(name=name, **mohito_hyperparameters)
            for name in agent_names
        }
        for agent, policy in agent_policies.items():
            model_weight_path = os.path.join(args.policy_path,
                                             f'{agent}_{checkpoint}.pt')

            p = policy.parameters()

            policy.load_state_dict(
                torch.load(model_weight_path, weights_only=True))

            assert p != policy.parameters(
            ), f'Policy {agent} has no parameters loaded from {model_weight_path}'

            policy.eval()  # Set policy to evaluation mode

            

        ol_conf = conf[args.openness_level]

        for ss, ol_ss_conf in enumerate(ol_conf):

            j_ss_offset = 0

            env = wildfire_v0.parallel_env(
                max_steps=100,
                parallel_envs=1,
                configuration=ol_ss_conf,
                device=torch.device('cpu'),
                log_directory=os.path.join(
                    args.output_folder,
                    f"policy;MOHITO{checkpoint}_ol;{args.openness_level}_ss;{ss}"
                ),
                override_initialization_check=False,
                show_bad_actions=False,
                observe_other_suppressant=train_conf[
                    'observe_other_suppressant'],
                observe_other_power=False,
                observe_burn_time=True,
            )
            env.reset()
            env = action_mapping_wrapper_v0(env)
            env = space_validator_wrapper_v0(env)
            env = mohito_hypergraph_wrapper_v0(env)

            for j in range(episodes_per_ss):

                torch.manual_seed(seeds[j_ss_offset + j])

                observations, infos = env.reset(
                    seed=seeds[j_ss_offset + j],
                    options={
                        'log_description':
                        f"policy;MOHITO_ol;{args.openness_level}_ss;{ss}_ep;{j}"
                    })

                while not torch.all(env.finished):

                    for agent_name, agent in agent_policies.items():
                        agent.observe(
                            observations[agent_name])  # Policy observation

                    agent_actions = {
                        agent_name:
                        agent_policies[agent_name].act(
                            action_space=env.action_space(agent_name))
                        for agent_name in env.agents
                    }  # Policy action determination here

                    observations, rewards, terminations, truncations, infos = env.step(
                        agent_actions)
                pbar.update(1)
            env.close()

            j_ss_offset += episodes_per_ss
