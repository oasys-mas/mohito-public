from typing import Callable, List, Union
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn

import torch_geometric as tg

from mohito.mohito_layers.gnn import GNNReturn, GNN


@dataclass
class ActorReturn:
    """
    Return type of the Actor model
    """
    graph: GNNReturn
    batch_indices: torch.IntTensor
    task_action: torch.IntTensor
    hyperedge: torch.FloatTensor
    logits : torch.FloatTensor = None

    def detach(self) -> None:
        """
        Detach the output.
        """
        assert self.task_action.grad_fn is None, "why does the discrete task_action have a gradient!?"

        graph = GNNReturn(x=self.graph.x.detach(),
                          agent_E_mask=self.graph.agent_E_mask.clone(),
                          agent_mask=self.graph.agent_mask.clone(),)

        return ActorReturn(
            graph=graph,
            batch_indices=self.batch_indices,
            task_action=self.task_action,
            hyperedge=self.hyperedge.detach(),
            logits=self.logits.detach() if self.logits is not None else None
        )


class Actor(nn.Module):
    """
    Actor module for MOHITO
    """

    def __init__(
        self,
        optimizer_fn: Callable[[List[nn.Parameter]], torch.optim.Optimizer],
        lr: float,
        psi: float,
        K: int,
        beta: float,
        regularize: bool,
        gradient_clip: float,
        name: str,

        #architectural parameters
        exp_include_E_in_Gc: bool = False,
        exp_log_soft_output: bool = False,
        exp_linear_combination: bool = False,

        #GNN parameters
        *args,
        **kwargs,
    ) -> object:
        """
        Actor module for MOHITO

        Args:
            optimizer_fn (Callable[[List[nn.Parameter]], torch.optim.Optimizer]): optimizer function
            lr (float): learning rate
            psi (float): target network update rate
            K (int): number of updates between slow updates
            beta (float): entropy regularization coefficient
            regularize (bool): whether to regularize the actor
            gradient_clip (float): gradient clipping value
            name (str): name of the model
            exp_include_E_in_Gc (bool, optional): if true, relabel hyperedge identifier
            exp_log_soft_output (bool, optional): if true, use log softmax on the output
            exp_linear_combination (bool, optional): if true, use linear combination of the output features
            args (List): list of arguments to pass to the GNN
            kwargs (Dict): dictionary of keyword arguments to pass to the GNN
        Returns:
            object: Actor object
        """
        super(Actor, self).__init__()

        #hyperparameters
        self.psi = psi
        self.beta = beta
        self.regularize = regularize
        self.gradient_clip = gradient_clip
        self.name = name
        self._step = 0
        self.K = K

        self.exp_include_E_in_Gc = exp_include_E_in_Gc
        self.exp_log_soft_output = exp_log_soft_output
        self.exp_linear_combination = exp_linear_combination

        #build networks
        self.build(*args, **kwargs)
        self.optimizer = optimizer_fn(self.main.parameters(), lr=lr)

    def build(self, *args, **kwargs) -> None:
        """
        Builds the actor network

        Args:
            *args (List): list of arguments to pass to the GNN
            **kwargs (Dict): dictionary of keyword arguments to pass to the GNN
        """
        self.main = GNN(*args, **kwargs)

        if self.exp_linear_combination:
            self.linear_combination = nn.Linear(self.main.node_feature_size,1)

        self.target = deepcopy(self.main)

    def forward(self,
                observation_graph: Union[tg.data.Data, List[tg.data.Data]],
                target: bool = False) -> ActorReturn:
        """
        Forward pass of the actor

        Args:
            observation_graph (Union[tg.data.Data, List[tg.data.Data]]): tg data to pass through the network.
            target (bool, optional): whether to use the target network.

        Returns:
            ActorReturn: output of the network.
        """

        if isinstance(observation_graph, tg.data.Data):
            observation_graph = [observation_graph]

        batched_graph = tg.data.Batch.from_data_list(observation_graph)
        batch_indices = batched_graph.batch

        if target:
            out: GNNReturn = self.target(batched_graph)
        else:
            out: GNNReturn = self.main(batched_graph)

        #?I am relabeling the first feature of the output before action selection. For consistency with the critic.
        out.x[out.agent_E_mask,0] = self.target.this_agent_hyperedge_identifier

        #Select action by batch
        selected_task_action = []
        selected_hyperedge = []
        batch_logits = []

        for b in range(len(observation_graph)):
            batch_mask = batch_indices == b
            batch_mask = batch_mask & out.agent_E_mask

            #aggregate node features
            if self.exp_linear_combination:
                summed_features = self.linear_combination(out.x[batch_mask]).squeeze(1)
            else:            
                summed_features = out.x[batch_mask].sum(dim=1)
            
            # Check for NaN values in the summed features
            assert torch.isnan(summed_features).sum() == 0, \
                "NaN values in summed features, model probably has NaNs in it."

            #do we pretend this is stochastic?
            if self.exp_log_soft_output:
                logits = torch.log_softmax(summed_features, dim=0)
            else:
                logits = summed_features

            #!shuffles hyperedges to avoid 0 index bias. Otherwise false local optima can occur.
            if self.main.ablation_ignore_task_nodes:
                #scramble the hyperedges
                new_pos = torch.randperm(logits.shape[0])
                temp_logits = logits[new_pos]
                action = torch.argmax(temp_logits)
                real_action = new_pos[action]
                selected_task_action.append(real_action)
            else:
                selected_task_action.append(torch.argmax(logits))


            selected_hyperedge.append(out.x[batch_mask][selected_task_action[-1]])
            batch_logits.append(logits[selected_task_action[-1]])

        return ActorReturn(graph=out,
                           batch_indices=batch_indices,
                           hyperedge=torch.stack(selected_hyperedge,dim=0),
                           task_action=torch.stack(selected_task_action,dim=0),
                           logits = torch.stack(batch_logits) if self.exp_log_soft_output else None)

    def update_target(self) -> None:
        """
        Update the target network
        """
        for target_param, param in zip(self.target.parameters(),
                                       self.main.parameters()):
            target_param.data.copy_(self.psi * param.data +
                                    (1 - self.psi) * target_param.data)

    def update(self) -> None:
        """
        Update the actor
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
