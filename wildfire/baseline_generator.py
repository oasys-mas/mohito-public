import argparse, yaml, os, tqdm
import torch

from free_range_zoo.envs.wildfire.configs.uai_experiment import UAI_2025_ol_config
from free_range_zoo.envs import wildfire_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from free_range_zoo.wrappers.space_validator import space_validator_wrapper_v0
from free_range_zoo.envs.wildfire.baselines import NoopBaseline, RandomBaseline, WeakestBaseline, FifoBaseline


torch.use_deterministic_algorithms(True, warn_only=True)

parser = argparse.ArgumentParser(
    description="Generate baseline results for UAI 2025 Wildfire Experiment",
)

parser.add_argument(
    '--seed', type=int, default=42,
    help='Seed for random number generation (default: 42)',
)
parser.add_argument(
    "-n", type=int)

parser.add_argument(
    "-c", "--continue_through_no_fires", action='store_true', default=False,
)

parser.add_argument(
    '-int', '--intensity_increase_prob', type=float, default=1.0,
    help='Probability of intensity increase for the fire (default: 1.0)',
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
parser.add_argument(
    '-o', 
    '--output_folder',
    type=str,
)

#!determines which starting states are used for the experiment
starting_state = [0,1,2]

args = parser.parse_args()

print("Arguments:")
for arg in vars(args):
    print(f"  {arg}: {getattr(args, arg)}")

baseline_output_folder = args.output_folder

if not os.path.exists(baseline_output_folder):
    os.makedirs(baseline_output_folder)

#dump to .yaml file in the same folder
with open(os.path.join(baseline_output_folder,'baseline_args.yaml'), 'w') as f:
    yaml.dump(vars(args), f, default_flow_style=False)


# %%
#seed for testing
episodes_per_ss = args.n

# %%
conf_params = {
    'fire_types': [
    ],
    'fire_rewards': [0, 20, 400],
    'burnout_penalty': [0, -10, -25],
    'base_spread':
    args.base_spread if args.base_spread is not None else
    'ol',  # will be set to the default value for the openness level
    'random_ignition_prob': args.random_ignition_prob if args.random_ignition_prob is not None else 0.1,
    'intensity_increase_prob': args.intensity_increase_prob,
    'new_ignition_temperatures': [2, 3],
    'use_stochastic_ignition_temperature':
    True
}

conf: dict[int, list[UAI_2025_ol_config]] = {}

seeds = list(range(args.seed, args.seed+args.n))

for openness_level in range(3):
    #composition over the starting states
    OL_conf = [
        UAI_2025_ol_config(openness_level=openness_level + 1,
                           starting_state=ss,
                           **conf_params) for ss in starting_state
    ]
    conf[openness_level + 1] = OL_conf

# %% [markdown]
# ## Test Load Environment (determine fixed number of agents)

# %%
baselines = [NoopBaseline, RandomBaseline, WeakestBaseline, FifoBaseline]

with tqdm.tqdm(total=episodes_per_ss * len(starting_state) * 3 * len(baselines),
               ascii="=/🐦") as pbar:
    for baseline in baselines:
        for ol, ol_conf in conf.items():

            if args.base_spread is not None:
                if ol != 2:
                    continue

            j_ss_offset = 0

            for ss, ol_ss_conf in enumerate(ol_conf):

                env = wildfire_v0.parallel_env(
                    max_steps=100,
                    parallel_envs=1,
                    configuration=ol_ss_conf,
                    device=torch.device('cpu'),
                    log_directory=os.path.join(
                        baseline_output_folder,
                        f"test_logging_policy;{baseline.__name__}_ol;{ol}_ss;{ss}"),
                    override_initialization_check=True,
                    show_bad_actions=False,
                    observe_other_suppressant=False,
                    observe_other_power=False,
                    observe_burn_time=True,
                    verbose=True,
                    terminate_on_no_fires= not args.continue_through_no_fires
                )
                env.reset()
                env = action_mapping_wrapper_v0(env)
                env = space_validator_wrapper_v0(env)

                for j, seed in enumerate(seeds):

                    torch.manual_seed(seed)

                    observations, infos = env.reset(
                        seed=seed,
                        options={
                            'log_description':
                            f"policy;{baseline.__name__}_ol;{ol}_ss;{ss}_seed;{seed}_episode;{j}"
                        })

                    agents = {
                        env.agents[0]:
                        baseline(agent_name="firefighter_1", parallel_envs=1),
                        env.agents[1]:
                        baseline(agent_name="firefighter_2", parallel_envs=1),
                    }

                    while not torch.all(env.finished):

                        for agent_name, agent in agents.items():
                            agent.observe(
                                observations[agent_name])  # Policy observation

                        agent_actions = {
                            agent_name:
                            agents[agent_name].act(
                                action_space=env.action_space(agent_name))
                            for agent_name in env.agents
                        }  # Policy action determination here

                        observations, rewards, terminations, truncations, infos = env.step(
                            agent_actions)
                    pbar.update(1)
                env.close()

                j_ss_offset += episodes_per_ss
