from typing import List, Union, Dict, Tuple
import warnings
import torch.nn as nn
import torch
import torch_geometric as tg
from copy import deepcopy
from mohito.mohito_layers import Actor, Critic, ActorReturn, CriticReturn
from free_range_rust import Space
import random
from mohito.wrappers.mohito_wrapper import construct_critic_graph


class Mohito(nn.Module):
    """
    A GNN based Task Open Reinforcement Learning (RL) agent.
    This agent operates on 2d levi-incidence graphs (flattened hypergraphs) and uses a A2C architecture.
    """

    def __init__(self, initial_epsilon: float, epsilon_decay: float,
                 epsilon_min: float, both: dict, actor: Dict, critic: Dict,
                 name: str, gamma: float, exp_ac_softhard_logits:bool) -> object:
        """
        Args:
            initial_epsilon (float): Initial epsilon value for exploration.
            epsilon_decay (float): Decay rate for epsilon.
            epsilon_min (float): Minimum epsilon value for exploration.

            both (dict): Parameters for both actor and critic networks.
            actor (Dict): Parameters for the actor network.
            critic (Dict): Parameters for the critic network.

            name (str): Name of the agent.
            gamma (float): Discount factor for the critic network.

            exp_ac_softhard_logits (bool): experimental setting to use hard action choice but soft updates.
        Returns:
            object: MOHITO network
        """
        super(Mohito, self).__init__()
        self.name = name

        #targets and main inside actor/critic objects
        actor_params = actor | both | {'name': name}
        critic_params = critic | both | {
            'name': name
        } | {
            'actor_hidden_dim': actor['hidden_dim']
        }

        assert actor_params[
            'optimizer_fn'] == 'adam', "Optimizer function must be 'adam'."
        actor_params['optimizer_fn'] = lambda params, lr: torch.optim.Adam(
            params, lr=lr)

        assert critic_params[
            'optimizer_fn'] == 'adam', "Optimizer function must be None."
        critic_params['optimizer_fn'] = lambda params, lr: torch.optim.Adam(
            params, lr=lr)

        if exp_ac_softhard_logits:
            self.actor = Actor(**actor_params, exp_log_soft_output=True) #output updated hyperedge nodes as logits
            self.critic = Critic(**critic_params, exp_include_E_in_Gc=False) #don't embed actor outputs in critic graph (use them elsewhere)
        else:
            self.actor = Actor(**actor_params)
            self.critic = Critic(**critic_params)

        #exploration
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        self.exp_ac_softhard_logits = exp_ac_softhard_logits

    def forward(self,
                x: tg.data.Data,
                action_space=None,
                explore: bool = False) -> ActorReturn:
        """
        Forward pass through the actor network.

        Args:
            x (tg.data.Data): Input data for the actor network.
            action_space (): Action space for the agent.
            explore (bool): Whether to use exploration or not.

        Returns:
            ActorReturn: Output of the actor network.
        """
        actor_out: ActorReturn = self.actor(x, target=False)

        if explore:
            if random.random() < self.epsilon:
                actor_out.task_action = torch.tensor(
                    action_space.sample_nested())
                
                actor_out.hyperedge = actor_out.graph.x[actor_out.graph.agent_E_mask][actor_out.task_action]

            #decay epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay,
                               self.epsilon_min)

        return actor_out

    def observe(self, x: tg.data.Data) -> None:
        """
        Observe the environment and get the actor output.

        Args:
            x (tg.data.Data): Input data for the actor network.
        """
        out = self.forward(x, explore=False,
                           action_space=None)

        self._action = out.task_action
        self._hyperedges = out.logits

    def act(self, action_space: Space, return_logits: bool = False) -> torch.IntTensor:
        action = self._action.clone()
        if return_logits:
            return self._action, self._hyperedges
        else:
            return self._action

    def _batch_critic_graph(self,
                            o,
                            hyperedges,
                            key: str,
                            detach_me: bool,
                            a=None
                            ) -> Tuple[List[tg.data.Data], torch.BoolTensor]:
        """
        Creates a batch of critic graphs for the given observations and hyperedges.

        Args:
            o (List[tg.data.Data]): List of observation graphs.
            hyperedges (List[Dict[str, ActorReturn]]): List of hyperedges for each agent.
            key (str): Key to access the which set of hyperedges in the dictionary.
            detach_me (bool): Whether to detach the hyperedge or not.
            a (List[torch.FloatTensor], optional): Specify which actions to select in output mask (rather than determining automatically).
        
        Returns:
            List[tg.data.Data]: List of critic graphs for each batch.
        """
        c = []
        Sel_a = []
        for bidx, gb in enumerate(o):
            #selecting this batch hyperedges
            edmask = [
                (ha[key].graph.agent_E_mask) & (ha[key].batch_indices == bidx)
                for agent, ha in hyperedges.items()
            ]

            #generate action mask from task_action. Used to get a critic_graph sized mask over the joint action.
            if a is None:
                selected_action_mask = [
                    torch.isin(torch.arange(ha[key].graph.x[edmask[j]].shape[0]), ha[key].task_action[bidx])
                    for j, (agent, ha) in enumerate(hyperedges.items())
                ]
            else:
                selected_action_mask = [
                    torch.isin(torch.arange(ha[key].graph.x[edmask[j]].shape[0]), a[bidx][agent])
                    for j, (agent, ha) in enumerate(hyperedges.items())
                ]

            #selecting hyperedges only for this batch
            if detach_me:
                cg, sel_a = construct_critic_graph(
                    observation_graphs=list(gb.values()),
                    actor_outputs=[
                        ha[key].graph.x[edmask[j]].detach()
                        for j, (agent, ha) in enumerate(hyperedges.items())
                    ],
                    selected_action_mask=selected_action_mask)
            else:
                cg, sel_a = construct_critic_graph(
                    observation_graphs=list(gb.values()),
                    actor_outputs=[
                        ha[key].graph.x[edmask[j]].clone() if agent
                        == self.name else ha[key].graph.x[edmask[j]].detach()
                        for j, (agent, ha) in enumerate(hyperedges.items())
                    ],
                    selected_action_mask=selected_action_mask)

            c.append(cg)
            Sel_a.append(sel_a)
        return c, Sel_a

    def update_forward(self, batch: Dict) -> Dict:
        """
        MOHITO requires a two step update.
        1. Query the actor to get the hyperedges for each actor (without e-greedy exploration)

        Args:
            batch (Dict): A batch of training data.

        Returns:
            Dict: A dictionary containing the actor output.
        """

        #unspooling my batch. Yes this can be done better.
        #TODO hardcoded for parallel_env = 1
        Go = [b['o'][self.name][0] for b in batch]
        Go_prime = [b['o_prime'][self.name][0] for b in batch]

        #actor
        actor_out: ActorReturn = self.actor(Go, target=False)
        actor_out_prime: ActorReturn = self.actor(Go_prime, target=True)
        return {'actor_out': actor_out, 'actor_out_prime': actor_out_prime}

    def update(self, batch: Dict, hyperedges: List[Dict[str,
                                                        ActorReturn]]) -> None:
        """
        MOHITO requires a two step update. 
        2. Update the networks using the shared hyperedges from other agents.
        
        Args:
            batch (Dict): A batch of training data.
            hyperedges (List[Dict[str,torch.FloatTensor]]): A list of hyperedges for each agent.
                'ed': The hyperedges for the agent.
                'ed_prime': The hyperedges for the next observation.
        """

        #unpack batch
        o = [b['o'] for b in batch]
        o_prime = [b['o_prime'] for b in batch]
        a = [b['a'] for b in batch]
        r = torch.stack([b['r'][self.name] for b in batch])

        #construct critic graph
        #?we make multiple of these to ensure the correct one is detached
        #TODO this is a implmentation thing. The critic graphs are identical between agents other than what is detached.
        #gradient stops at end of critic (doesn't pass to actor)
        #the hyperedges of the actions taken in the experience
        c, sel_a = self._batch_critic_graph(o=o,
                                            hyperedges=hyperedges,
                                            key='actor_out',
                                            detach_me=True,
                                            a=a)
        c_prime, sel_a_prime = self._batch_critic_graph(o=o_prime,
                                                        hyperedges=hyperedges,
                                                        key='actor_out_prime',
                                                        detach_me=True)

        assert all([len(c) == len(k) for k in [c_prime, o, o_prime, a, r]
                    ]), "Batch sizes do not match"
        B = len(c)

        #calculating critic loss then update
        critic_target = r + self.gamma * self.critic(
            critic_graph=c_prime, target=True,
            selected_actions=sel_a_prime).q.detach()
        
        critic_main = self.critic(
            critic_graph=c, target=False, selected_actions=sel_a).q

        critic_loss = torch.F.mse_loss(critic_main, critic_target)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.update()

        #detaching the hyperedges of all other agents
        c, sel_a = self._batch_critic_graph(
            o=o,
            hyperedges=hyperedges,
            key='actor_out',
            detach_me=False,
        )

        #calculating actor loss then update
        if self.exp_ac_softhard_logits:
            #?do a stochastic policy style update -log(\pi(a|s)) * Q(s,a), but a is choosen hard
            q_values = self.critic(critic_graph=c, target=False, selected_actions=sel_a).q
            actor_logits = hyperedges[self.name]['actor_out'].logits
            actor_loss = torch.mean(-1 * actor_logits * q_values)

        else:
            actor_loss = torch.mean(-self.critic(
                critic_graph=c,
                target=False,
                selected_actions=sel_a,
            ).q)

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.update()