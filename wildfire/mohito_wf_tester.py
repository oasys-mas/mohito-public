from free_range_zoo.envs.wildfire.configs.uai_experiment import UAI_2025_ol_config
from free_range_zoo.envs import wildfire_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from free_range_zoo.wrappers.space_validator import space_validator_wrapper_v0
from mohito.mohito_wrapper import mohito_hypergraph_wrapper_v0
import torch
import tqdm, os, yaml
from mohito.mohito import Mohito
import argparse
import pandas as pd

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

try:
    os.mkdir(args.output_folder)
except FileExistsError:
    pass


#dump to .yaml file in the same folder
with open(os.path.join(args.output_folder,'command_line_args.yaml'), 'w') as f:
    yaml.dump(vars(args), f, default_flow_style=False)


#seed for testing
episodes_per_ss = args.n

agent_names = ['firefighter_1', 'firefighter_2']

with open(args.mohito_config, 'r') as file:
    mohito_hyperparameters = yaml.safe_load(file)

train_conf = mohito_hyperparameters.pop('training')
mohito_hyperparameters.pop('validation')
mohito_hyperparameters.pop('convergence')
environment_parameters = mohito_hyperparameters.pop('environment')

#!starting states for experimentation (assume ss for training == ss for testing)
ss_train = train_conf['starting_states']

#%%
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
    train_conf['intensity_increase_prob'],  # 1.0 for always intensity increase
    'new_ignition_temperatures': [2, 3], #intensities to be sampled from for new fires
    'use_stochastic_ignition_temperature': #whether fires are sampled from the new_ignition_temperatures
    True
}

conf: dict[int, list[UAI_2025_ol_config]] = {}

for openness_level in range(3):
    #composition over the starting states
    OL_conf = [
        UAI_2025_ol_config(openness_level=openness_level + 1,
                           starting_state=ss,
                           **conf_params) for ss in ss_train
    ]
    conf[openness_level + 1] = OL_conf

#%%


all_checkpoints = list(
    set([
        (f.split('_')[2].replace('.pt',''), '_'.join(f.split('_')[2:])) for f in os.listdir(args.policy_path)
    ]))

filter_checkpoints = [
    check[1] for check in all_checkpoints
    if (int(check[0]) <= args.high or args.high == -1) and int(check[0]) >= args.low
]

seeds = list(range(args.seed, args.seed+args.n))

print(f"Testing MOHITO over start seed: {args.seed}-{args.seed*2}")
print("Openness level:", args.openness_level)
print("Checkpoints to test:", filter_checkpoints[0], "-",filter_checkpoints[-1])
print("Episodes per starting state:", episodes_per_ss)
print("Total episodes:", len(filter_checkpoints) * episodes_per_ss * 4)

with tqdm.tqdm(total=episodes_per_ss * len(ss_train) * len(filter_checkpoints),
               ascii="=/🐦") as pbar:
    for checkpoint in filter_checkpoints:

        agent_policies = {
            name: Mohito(name=name, **mohito_hyperparameters)
            for name in agent_names
        }
        for agent, policy in agent_policies.items():
            model_weight_path = os.path.join(args.policy_path,
                                             f'{agent}_{checkpoint}')

            print(f'Loading policy {agent} from {model_weight_path}')

            p = policy.parameters()

            policy.load_state_dict(
                torch.load(model_weight_path))

            assert p != policy.parameters(
            ), f'Policy {agent} has no parameters loaded from {model_weight_path}'

            policy.eval()  # Set policy to evaluation mode


        assert all([ not p.training  for p in agent_policies.values()]), \
            f'Policies are not in evaluation mode after loading weights from {args.policy_path}'
            

        ol_conf = conf[args.openness_level]

        for ss_ind, ol_ss_conf in enumerate(ol_conf):

            ss = ss_train[ss_ind]
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
                observe_other_suppressant=environment_parameters['observe_other_suppressant'],
                observe_other_power=False,
                observe_relative_task_index=environment_parameters['observe_relative_task_index'],
                observe_burn_time=True,
                terminate_on_no_fires= False,
                verbose=True
            )
            env.reset()
            env = action_mapping_wrapper_v0(env)
            env = space_validator_wrapper_v0(env)
            env = mohito_hypergraph_wrapper_v0(env)

            for ep, seed in enumerate(seeds):

                torch.manual_seed(seed)

                observations, infos = env.reset(
                    seed=seed,
                    options={
                        'log_description':
                        f"policy;MOHITO_ol;{args.openness_level}_ss;{ss}_seed;{seed}_episode;{ep}",
                    })

                step = 1
                while not torch.all(env.finished):

                    for agent_name, agent in agent_policies.items():
                        agent.observe(
                            observations[agent_name][0]
                        )  # Policy observation

                    obs_output = {
                        agent_name+"_nodes": f'{observations[agent_name][0].x.tolist()}'
                    for agent_name, agent in agent_policies.items()}

                    obs_output = obs_output | {
                        agent_name + "_edge_index": f'{observations[agent_name][0].edge_index.tolist()}'
                        for agent_name in env.agents
                    }


                    agent_actions = {
                        agent_name:
                        agent_policies[agent_name].act(
                            action_space=env.action_space(agent_name), return_logits=True)
                        for agent_name in env.agents
                    }  # Policy action determination here

                    actions = {
                        agent_name: act[0] for agent_name, act in agent_actions.items()
                    }
                    logits = {
                        agent_name+"_logits": act[1] for agent_name, act in agent_actions.items()
                    }
                    observations, rewards, terminations, truncations, infos = env.step(
                        actions)

                    logit_file = os.path.join(
                        args.output_folder,
                        os.path.join(
                           f"policy;MOHITO{checkpoint}_ol;{args.openness_level}_ss;{ss}",
                            f"0_logits.csv"
                        )
                    )
                    pd.DataFrame(logits | obs_output | {'r_1': rewards['firefighter_1'], 'step': step, 'seed': seed, 'episode': ep}).to_csv(
                        logit_file,
                        mode='a',
                        header = not os.path.exists(logit_file)
                    )
                    step += 1

                pbar.update(1)
            env.close()

            j_ss_offset += episodes_per_ss
