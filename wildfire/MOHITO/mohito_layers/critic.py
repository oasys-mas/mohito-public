from typing import Callable, List, Union
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn

import torch_geometric as tg
from torch_geometric.nn import global_mean_pool

from mohito.mohito_layers.gnn import GNN, GNNReturn
from mohito.mohito_layers.actor import ActorReturn
from mohito.mohito_layers.utils import Usqueezer


@dataclass
class CriticReturn:
    """
    Critic return type

    Args:
        x (GNNReturn): GNN output nodes
        batch_indices (torch.IntTensor): map of which batch maps to which nodes
        q (torch.FloatTensor): estimated Q value
    """
    graph: GNNReturn
    batch_indices: torch.IntTensor
    q: torch.FloatTensor


class Critic(nn.Module):
    """
    Critic module for MOHITO
    """

    def __init__(
        self,
        #hyperparameters
        optimizer_fn: Callable[[List[nn.Parameter]], torch.optim.Optimizer],
        lr: float,
        psi: float,
        K: int,
        beta: float,
        regularize: bool,
        gradient_clip: float,
        name: str,
        num_agents: int,
        actor_hidden_dim: int,

        #architectural parameters
        exp_include_E_in_Gc: bool = True,
        exp_linear_combination: bool = False,
        hyperedge_identifier: int = 5,

        #GNN parameters
        *args,
        **kwargs,
    ):
        """
        params:
            optimizer_fn: Callable[[List[nn.Parameter]], torch.optim.Optimizer] - optimizer function
            lr: float - learning rate
            psi: float - target network update rate
            K: int - number of updates between slow updates
            beta: float - entropy regularization coefficient
            regularize: bool - whether to regularize the critic
            gradient_clip: float - gradient clipping value
            name: str - name of the model
            num_agents: int - number of agents
            actor_hidden_dim: int - hidden dimension of the actor

            #?experimental parameters, False is default behavior
            exp_include_E_in_Gc: bool - if true, embed all hyperedges in the critic graph (else overwrite with constants)
            exp_linear_combination: bool - if true, use a linear combination of the hyperedges in the critic graph to get q

            hyperedge_identifier: int - identifier for "this agent" hyperedges, see hypergraph wrapper.
            
        """
        super(Critic, self).__init__()

        #hyperparameters
        self.psi = psi
        self.beta = beta
        self.regularize = regularize
        self.gradient_clip = gradient_clip
        self.name = name
        self.num_agents = num_agents
        self._step = 0
        self.K = K

        self.exp_include_E_in_Gc = exp_include_E_in_Gc
        self.exp_linear_combination = exp_linear_combination

        self.hyperedge_identifier = hyperedge_identifier
        self.actor_hidden_dim = actor_hidden_dim

        #build networks
        self.build(*args, **kwargs)
        self.optimizer = optimizer_fn(self.main.parameters(), lr=lr)

    def build(self, *args, **kwargs):
        """
        Builds the critic network

        Params:
            exp_* - See __init__

            *args: List - list of arguments to pass to the GNN
            **kwargs: Dict - dictionary of keyword arguments to pass to the GNN
        """
        gnn = GNN(*args, **kwargs)

        layers = [gnn]

        if self.exp_linear_combination:
            self.linear_combination = nn.Linear(gnn.node_feature_size,1)

        self.main = nn.ModuleList(layers)
        self.target = deepcopy(self.main)

    def __call__(self, critic_graph, selected_actions, target=False):
        return self.forward(critic_graph=critic_graph, selected_actions=selected_actions, target=target)

    def forward(self,
                critic_graph: Union[tg.data.Data, List[tg.data.Data]],
                selected_actions: torch.LongTensor,
                target: bool = False,
                **kwargs) -> torch.FloatTensor:
        """
        Forward pass of the critic

        Args:
            critic_graph (Union[tg.data.Data, List[tg.data.Data]]): tg data to pass through the GNN
            target (bool, optional): whether to use the target network
        Returns:
            torch.FloatTensor: critic output
        """
        #ensure batched input
        if isinstance(critic_graph, tg.data.Data):
            critic_graph = [critic_graph]

        batch_graph = tg.data.Batch.from_data_list(critic_graph)
        batch_indices = batch_graph.batch

        if not self.exp_include_E_in_Gc:
            #overwrite hyperedges with constants
            hyperedge_nodes = batch_graph.x[:,0] == self.hyperedge_identifier
            assert hyperedge_nodes.sum() >= self.num_agents, "Not enough hyperedge nodes in the graph, check your hypergraph wrapper. (or bad relabeling)"
            batch_graph.x[hyperedge_nodes, 1:] = -1.0

        if target:
            net = self.target
        else:
            net = self.main

        #batched graph -> updated batched graph -> <#batches, #features>
        gnn_output: torch.FloatTensor = net[0](batch_graph, **kwargs).x
        
        #calculate the Q-values.
        #? this differs from the original implementation, and is the main change over the original MOHITO where we used a MLP here and didn't inject the hyperedges into the graph.
        Q = []
        for b_sel_ed in selected_actions:

            #aggregate over features
            if self.exp_linear_combination:
                q = self.linear_combination(gnn_output[b_sel_ed]).squeeze(1)
            else:
                q = torch.mean(gnn_output[b_sel_ed], dim=1)

            q = torch.max(q)
            Q.append(q)
        Q = torch.stack(Q, dim=0) 

        

        return CriticReturn(graph=gnn_output,
                            batch_indices=batch_indices,
                            q=Q)

    def update(self) -> None:
        """
        Update the critic
        """
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.main.parameters(),
                                           self.gradient_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.K == 0 or (self._step % self.K == 0 and self._step > 0):
            self.update_target()
            self._step = 0
        else:
            self._step += 1

    def update_target(self) -> None:
        """
        Update the target network
        """
        for target_param, param in zip(self.target.parameters(),
                                       self.main.parameters()):
            target_param.data.copy_(self.psi * param.data +
                                    (1 - self.psi) * target_param.data)
