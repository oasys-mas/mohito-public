from typing import Dict, Any, List, Tuple
import pickle
import torch

from free_range_zoo.envs import wildfire_v0
from free_range_zoo.utils.env import BatchedAECEnv
from free_range_zoo.envs.wildfire.env.structures.configuration import RewardConfiguration, \
    FireConfiguration, StochasticConfiguration, AgentConfiguration, WildfireConfiguration

from mohito.wrappers.action_task import action_mapping_wrapper_v0
from mohito.wrappers.space_validator import space_validator_wrapper_v0
from mohito.wrappers.mohito_wrapper import mohito_hypergraph_wrapper_v0

def load_wildfire_environment(config: Dict[str, Any],
                              log_dir: str = None) -> BatchedAECEnv:
    """
    Args:
        config (Dict[str, Any]): 'environment' configuration from yaml file.
        log_dir (str, optional): Directory to save logs. Defaults to None (no logging).
    Returns:
        BatchedAECEnv: The configured wildfire environment wrapped for Mohito.
    """
    if 'config_file' in config:
        with open(config['config_file'], mode='rb') as f:
            env_config = pickle.load(f)
    else:

        ten_attack = {k: torch.tensor(v, dtype=torch.int) if isinstance(v, list) else v for k, v in config['config']['agent'].items()}
        ten_attack['equipment_states'] = ten_attack['equipment_states'].to(torch.float)
        ten_attack['possible_capacities'] = ten_attack['possible_capacities'].to(torch.float)
        fighter = AgentConfiguration(**ten_attack)


        ten_fire = {k: torch.tensor(v, dtype=torch.int) if isinstance(v, list) else v for k, v in config['config']['fire'].items()}
        fire = FireConfiguration(**ten_fire)

        ten_reward = {k: torch.tensor(v) if isinstance(v, list) else v for k, v in config['config']['reward'].items()}
        reward = RewardConfiguration(**ten_reward)
        ten_stochastic = {k: torch.tensor(v) if isinstance(v, list) else v for k, v in config['config']['stochastic'].items()}
        stochastic = StochasticConfiguration(**ten_stochastic)

        env_config = WildfireConfiguration(**config['config']['env'],
                                           agent_config=fighter,
                                           fire_config=fire,
                                           reward_config=reward,
                                           stochastic_config=stochastic)

    env = wildfire_v0.parallel_env(
        configuration=env_config,
        log_directory=log_dir,
        render_mode=None,
        **config['init'])

    env.reset()
    env = space_validator_wrapper_v0(env)
    env.reset()
    env = action_mapping_wrapper_v0(env)
    obs, _ = env.reset()
    env = mohito_hypergraph_wrapper_v0(env)
    env.reset()

    return env




#https://stackoverflow.com/questions/7204805/deep-merge-dictionaries-of-dictionaries-in-python
def merge_dict(a: dict,
               b: dict,
               path: list = []) -> Tuple[Dict[str, Any], List[str]]:
    """
    Args:
        a (dict): Dictionary to merge into.
        b (dict): Dictionary to merge from.
        path (list, optional): Path of keys (for error messages).
    
    Returns:
        Tuple[Dict[str, Any], List[str]]: Merged dictionary and list of decisions made.
    """
    decisions = []

    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                #keep a
                decisions.append(('replace', path + [str(key)], b[key]))
        else:
            a[key] = b[key]
    return a, decisions