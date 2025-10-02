from typing import List, Dict, Tuple, Any
import random, torch, os
import pandas as pd

from free_range_zoo.utils.env import BatchedAECEnv
from mohito.mohito import Mohito


class MohitoCriticController:
    """
    Holds the replay buffer, handles exchanging observations + actions between policies at training time, and running validation.
    """

    def __init__(self,
                 agent_names: List[str],
                    checkpoint_path: str,
                    validation_path: str,
                    envs: Dict[str, BatchedAECEnv],
                    steps_per_checkpoint: int,
                    training: Dict[str, Any],
                    validation: Dict[str, Any],
                    convergence: Dict[str, Any],
                 ):
        """
        Args:
            #manual entry
            agent_names (List[str]): List of agent names to update.
            checkpoint_path (str): Path to save checkpoints.
            validation_path (str): Path to save validation logs.
            envs (Dict[str, env]): List of environments to use for validation (different configs).

            #by yaml
            steps_per_checkpoint (int): How many steps to take before saving a checkpoint.

            training:
                buffer_size (int): Size of the replay buffer.
                batch_size (int): Size of the batch to sample from the replay buffer.
                experiences_before_update (int): How many new experiences to add to the replay buffer before sampling.
                backups_per_update (int): How many times to backup / update per reaching the # of new experiences.
                wait_until_full (bool): If True, will not sample from the replay buffer until it is full.
            validation:
                seeds (List[int]): List of seeds to use for validation.
                frequency (int): How many steps to take before running validation.
            convergence:
                k_convergence (int): Number of validation runs to consider convergence.
                convergence_diff_bound (float): The bound for convergence, if the difference in k rewards are all less than or equal to the current reward, we have converged.
                minimum_convergence_bound (float): The minimum reward to consider convergence.
        """
        self.agent_names = agent_names
        self.has_converged = {name: False for name in agent_names}

        self.buffer = []
        self.batch_size = training['batch_size']
        self.buffer_size = training['buffer_size']
        assert self.batch_size <= self.buffer_size, "Batch size must be less than or equal to buffer size."

        self.experiences_before_update = training['experiences_before_update']
        self.backups_per_update = training['backups_per_update']
        self.wait_until_full = training['wait_until_full']

        self.k_convergence = convergence['k_convergence']
        self.convergence_diff_bound = convergence['convergence_diff_bound']
        self.minimum_convergence_bound = convergence['minimum_convergence_bound']
        self._last_k_R = []  # last k rewards for convergence checks

        self._experience_counter = 0
        self._validation_counter = 0
        self._total_validation_counter = 0
        self._update_counter = 0
        self.do_first_validation = True
        self._checkpoint_counter = 0
        self._total_checkpoints = 0

        self.checkpoint_path = checkpoint_path
        self.validation_path = validation_path

        self.validation_envs = envs
        self.steps_per_validation = validation['frequency']
        self.steps_per_checkpoint = steps_per_checkpoint
        self.validation_seeds = validation['seeds']

    def __call__(self, policies: Dict[str, Mohito], o, a, r,
                 o_prime, run) -> Tuple[bool, Dict[str, bool]]:
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
                    self.update(policies, batch, run=run)
                    self._update_counter += 1

                self._experience_counter = 0
                did_update = True

            #validate
            if (self._validation_counter % self.steps_per_validation == 0
                    and self._validation_counter > 0):
                R = self.validate(policies, run=run)
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
                        f'{self.checkpoint_path}/{name}_{self._total_checkpoints}.pt'
                        if not any(self.has_converged.values()) else
                        f'{self.checkpoint_path}/{name}_{self._total_checkpoints}_converged.pt'
                    )
                self._checkpoint_counter = 0
                self._total_checkpoints += 1
            else:
                self._checkpoint_counter += 1

        elif self.do_first_validation:
            self.validate(policies, run=run)
            self.do_first_validation = False

        return did_update, self.has_converged

    def validate(self, policies: Dict[str, Mohito], run) -> float:
        """
        Runs validation and returns the average reward for use in convergence checks.
        """

        #override dropout
        [p.eval() for p in policies.values()]

        R_ss = []

        for env_key, validation_env in self.validation_envs.items():

            R = []
            for seed in self.validation_seeds:
                obs, info = validation_env.reset(
                    seed=seed,
                    options={
                        'log_description':
                        f'updates:{self._update_counter};validation:{self._total_validation_counter};seed:{seed};env_key:{env_key}'
                    })
                r = []
                step = 0

                while not torch.all(validation_env.finished):
                    [p.observe(obs[p.name][0]) for p in policies.values()]

                    agent_actions = {
                        agent_name:
                        policies[agent_name].act(
                            action_space=validation_env.action_space(agent_name),
                            return_logits=True)
                        for agent_name in validation_env.agents
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

                    logit_file = os.path.join(self.validation_path, 'logit.csv')

                    pd.DataFrame(
                        logits | {
                            'r_1': rewards['firefighter_1'],
                            'step': step,
                            'seed': seed,
                            'env_key': env_key,
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

        #set dropout
        [p.train() for p in policies.values()]

        return R.mean()

    def update(self, policies: Dict[str, Mohito],
               batch: List[Tuple[Dict[str, Any]]], run):
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
            p.update(batch, hyperedges=hyperedges, run=run)