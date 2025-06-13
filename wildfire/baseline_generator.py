import argparse, tqdm, os
import torch

from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from free_range_zoo.wrappers.space_validator import space_validator_wrapper_v0
from free_range_zoo.envs.wildfire.configs.uai_experiment import UAI_2025_ol_config
from free_range_zoo.envs import wildfire_v0
from free_range_zoo.envs.wildfire.baselines import NoopBaseline, RandomBaseline, WeakestBaseline, FifoBaseline



parser = argparse.ArgumentParser(
    description="Generate baseline results for UAI 2025 Wildfire Experiment",
)

parser.add_argument(
    '--seed', type=int, default=42,
    help='Seed for random number generation (default: 42)',
)
parser.add_argument(
    "-n",
    help="number of episodes / starting state (will perform n * starting states * OLs many episodes)",
    type=int
)

args = parser.parse_args()


# %%
#seed for testing
episodes_per_ss = args.n  // 3
baseline_output_folder = 'baseline_output'

# %%


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
    'base_spread': 'ol',
    'random_ignition_prob': 0.1
}

conf: dict[int, list[UAI_2025_ol_config]] = {}

seeds = list(range(args.seed, args.seed+1+args.n))

for openness_level in range(3):
    #composition over the starting states
    OL_conf = [
        UAI_2025_ol_config(openness_level=openness_level + 1,
                           starting_state=j,
                           **conf_params) for j in range(3)
    ]
    conf[openness_level + 1] = OL_conf

# %% [markdown]
# ## Test Load Environment (determine fixed number of agents)

# %%

torch.use_deterministic_algorithms(True, warn_only=True)

baselines = [NoopBaseline, RandomBaseline, WeakestBaseline, FifoBaseline]

with tqdm.tqdm(total=episodes_per_ss * 3 * 3 * len(baselines),
               ascii="=/🐦") as pbar:
    for baseline in baselines:
        for ol, ol_conf in conf.items():
            
            # seed starting state offset
            j_ss_offset = 0

            for ss, ol_ss_conf in enumerate(ol_conf):

                env = wildfire_v0.parallel_env(
                    max_steps=100,
                    parallel_envs=1,
                    configuration=ol_ss_conf,
                    device=torch.device('cpu'),
                    log_directory=os.path.join(
                        baseline_output_folder,
                        f"test_logging_policy;{baseline.__name__}_ol;{ol}_ss;{ss}"), #key for logging
                    override_initialization_check=True,
                    show_bad_actions=False,
                    observe_other_suppressant=False,
                    observe_other_power=False,
                    observe_burn_time=True,
                )
                env.reset()
                env = action_mapping_wrapper_v0(env)
                env = space_validator_wrapper_v0(env)

                for j in range(episodes_per_ss):

                    torch.manual_seed(seeds[j_ss_offset + j])

                    observations, infos = env.reset(
                        seed=seeds[j_ss_offset + j],
                        options={
                            'log_description':
                            f"policy;{baseline.__name__}_ol;{ol}_ss;{ss}_ep;{j}"
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
