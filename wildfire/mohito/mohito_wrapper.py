from typing import Dict, Any, Tuple, List
from supersuit.generic_wrappers.utils.base_modifier import BaseModifier
import torch
from free_range_zoo.wrappers.wrapper_util import shared_wrapper
from free_range_zoo.utils.env import BatchedAECEnv
import torch_geometric as tg
import torch.nn.functional as F
from free_range_rust import Space
from mohito.mohito_layers.actor import ActorReturn
from torch_geometric.utils import subgraph


def construct_critic_graph(
        observation_graphs: List[tg.data.Data],
        actor_outputs: List[torch.FloatTensor],
        selected_action_mask: List[torch.BoolTensor] = None) -> tg.data.Data:
    """
    Construct a critic graph from the observation graphs and actor outputs.
    
    Args:
        observation_graphs (List[tg.data.Data]): List of observation graphs.
        actor_outputs (List[torch.FloatTensor]): List of actor outputs.
        selected_action_mask (List[torch.BoolTensor], optional): List of selected action masks.
        
    Returns:
        tg.data.Data: Critic graph.
    """
    hyperedge_number: int = 5
    task_node_number: int = 3
    action_node_number: int = 4
    agent_node_number: int = 1
    other_agent_node_number: int = 2

    # Concatenate the observation graphs and edges
    #?cloned to ensure that the original graphs are not modified
    #TODO assumes parallel_env = 1
    x = torch.cat([graph[0].x.clone() for graph in observation_graphs], dim=0)
    x_index = torch.arange(x.shape[0], dtype=torch.long)
    edge_index = torch.cat([
        graph[0].edge_index.clone() +
        sum([g[0].x.shape[0] for g in observation_graphs[0:j]])
        for j, graph in enumerate(observation_graphs)
    ],
                           dim=1)

    #replace hyperedges with the actor hyperedges
    hyperedge_mask = x[:, 0] == hyperedge_number
    actor_hyperedges = torch.cat(actor_outputs, dim=0)

    x[hyperedge_mask] = actor_hyperedges

    selected_action_mask = torch.cat(selected_action_mask, dim=0)
    assert selected_action_mask.shape[0] == x[hyperedge_mask].shape[
        0], "selected_action_mask and x must be the same size"
    selected_hyperedges = x_index[hyperedge_mask][selected_action_mask]

    #Removing observations of other agents and replacing with true agent observations
    # identify all other agent nodes.
    other_agent_mask = x[:, 0] == other_agent_node_number
    remaining_nodes = x_index[~other_agent_mask]

    #!here we could remove duplicate task / action nodes, but this is equivalent algorithmically to not doing so, so we don't.
    #for best storage #TODO remove duplicate task / action nodes.

    #safely remove other agent nodes from the edge_index
    edge_index, _ = subgraph(remaining_nodes,
                             edge_index,
                             relabel_nodes=True,
                             num_nodes=x.shape[0])
    x = x[remaining_nodes]

    #add bi directional edges between all agents
    agent_nodes = (x[:, 0] == agent_node_number).nonzero().reshape(-1)
    half_agent_connections = torch.combinations(
        agent_nodes, r=2, with_replacement=False).swapaxes(0, 1)
    agent_connections = torch.cat(
        [half_agent_connections,
         half_agent_connections.flip(0)], dim=1)
    edge_index = torch.cat([edge_index, agent_connections], dim=1)

    #add bi directional edges between all task-actions
    hyperedges = (x[:, 0] == hyperedge_number).nonzero().reshape(-1)
    half_hyperedge_connections = torch.combinations(
        hyperedges, r=2, with_replacement=False).swapaxes(0, 1)
    hyperedge_connections = torch.cat(
        [half_hyperedge_connections,
         half_hyperedge_connections.flip(0)],
        dim=1)
    edge_index = torch.cat([edge_index, hyperedge_connections], dim=1)

    # Create a new graph with the concatenated data
    return tg.data.Data(x=x, edge_index=edge_index), selected_hyperedges


class HypergraphWrapper(BaseModifier):
    """
    Wrapper for generating interaction hypergraphs (flattened).
    """
    env = True
    subject_agent = True

    def __init__(self, env: BatchedAECEnv, subject_agent: str):
        """
        Args:
            env (BatchedAECEnv): Environment to wrap.
            subject_agent (str): Name of the subject agent.
        """
        self.env = env

        # Unpack the the parallel environment if it is wrapped in one.
        if hasattr(self.env, 'aec_env'):
            self.env = self.env.aec_env
        # Unpack the order enforcing wrapper if it has one of those.
        if hasattr(self.env, 'env'):
            self.env = self.env.env

        self.subject_agent = subject_agent

        probed_observation = self.env.observe(self.subject_agent)
        self.padding_size = max([
            probed_observation['tasks'][0].shape[1],
            probed_observation['self'][0].shape[0],
            probed_observation['others'][0].shape[1],
        ]) + 1

    def modify_action_space(self, act_space) -> Space.Vector:
        """
        Modify the action space to be a hypergraph.
        
        Args: 
            act_space (Any): Action space from the environment.
        Returns:
            Any: Modified action space.
        """
        spaces = []

        for batch, space in enumerate(act_space.spaces):
            spaces.append(
                Space.Discrete(n=sum([s.n for s in space.spaces]), start=0))

        return Space.Vector(spaces=spaces)

    def modify_action(self, action: torch.IntTensor) -> torch.IntTensor:
        """
        Modify the hyperedge selected action into a <batch, [task, action]>.
        #!THIS MUST BE PERFORMED AFTER THE RELAVENT MODIFY_OBS CALL!
        Args:
            action (int): Action from the environment.
        """
        output_action = []
        for batch in range(action.shape[0]):
            try:
                output_action.append(self._wrapper_task_action_mapping[batch][action[batch]])
            except:
                print("Error in mapping actions from hyperedge back to OneOf")
                raise IndexError(
                    "Action mapping failed. Please check the action space and the mapping."
                )
        return torch.stack(output_action, dim=0)

    def modify_obs(
            self, observation: Tuple[Any,
                                     Dict[str,
                                          torch.IntTensor]]) -> tg.data.Data:
        """
        Modify the observation into  a hypergraph.
        
        Args: 
            observation (Tuple[Any, Dict[str, torch.IntTensor]]): Observation from the environment.
        Returns:
            tg.data.Data: Hypergraph data object.
        """
        #hardcoded hypergraph node prefixes
        hyperedge_number: int = 5
        task_node_number: int = 3
        action_node_number: int = 4
        agent_node_number: int = 1
        other_agent_node_number: int = 2
        null_value: int = -1.0

        #unpack the observation and the agent action mapping. (from the ActionTaskMapping wrapper)
        obs, action_map = observation

        #identify the hyperedges to create
        action_space = self.env.action_space(self.subject_agent)
        hyperedges = []
        for batch in action_space.spaces:
            hyperedges.append([[(t, a)
                                for a in range(task.start, task.start + task.n)
                                ] for t, task in enumerate(batch.spaces)])
        hyperedges = [[item for sublist in group for item in sublist]
                      for group in hyperedges]

        graphs = []

        for ed_batch, obs_batch, amap in zip(
                hyperedges, obs, action_map['agent_action_mapping']):
            #[hyperedge nodes]
            hyperedge_nodes = torch.ones((len(ed_batch), self.padding_size))
            hyperedge_nodes[:, 0] = hyperedge_number

            #<[tasks, actions, other agents, self agent], features>
            task_nodes = torch.cat([
                obs_batch['tasks'],
                torch.ones(1, obs_batch['tasks'].shape[1]) * null_value
            ],
                                   dim=0)  #add null task node
            task_nodes = torch.cat([
                torch.ones(task_nodes.shape[0], 1) * task_node_number,
                task_nodes
            ],
                                   dim=1)  #add task node label
            task_nodes = F.pad(task_nodes,
                               (0, self.padding_size - task_nodes.shape[1]),
                               value=null_value,
                               mode='constant')  #pad to max feature size

            unique_actions = list(set([h[1] for h in ed_batch
                                       ]))  #null action included here
            action_nodes = torch.tensor(unique_actions).unsqueeze(
                1)  #add feature dim
            action_nodes = torch.cat([
                torch.ones(action_nodes.shape[0], 1) * action_node_number,
                action_nodes
            ],
                                     dim=1)  #add action node label
            action_nodes = F.pad(
                action_nodes, (0, self.padding_size - action_nodes.shape[1]),
                value=null_value,
                mode='constant')  #pad to max feature size

            other_agent_nodes = torch.cat([
                torch.ones(obs_batch['others'].shape[0], 1) *
                other_agent_node_number, obs_batch['others']
            ],
                                          dim=1)  #add other agent node label
            other_agent_nodes = F.pad(
                other_agent_nodes,
                (0, self.padding_size - other_agent_nodes.shape[1]),
                value=null_value,
                mode='constant')  #pad to max feature size

            agent_node = obs_batch['self'].unsqueeze(0)  #add num_node dim
            agent_node = torch.cat([
                torch.ones(agent_node.shape[0], 1) * agent_node_number,
                agent_node
            ],
                                   dim=1)  #add agent node label
            agent_node = F.pad(agent_node,
                               (0, self.padding_size - agent_node.shape[1]),
                               value=null_value,
                               mode='constant')  #pad to max feature size
            nodes = torch.cat([
                hyperedge_nodes, task_nodes, action_nodes, other_agent_nodes,
                agent_node
            ],
                              dim=0)
            node_index = torch.arange(nodes.shape[0], dtype=torch.long)

            #<edges>
            nhe = hyperedge_nodes.shape[0]
            n_tasks = task_nodes.shape[0]
            n_actions = action_nodes.shape[0]
            n_other_agents = other_agent_nodes.shape[0]

            hyperedge_indices = torch.arange(hyperedge_nodes.shape[0],
                                             dtype=torch.long)
            self_to_hyperedge = torch.stack([
                torch.ones(hyperedge_indices.shape[0], dtype=torch.long) *
                (nodes.shape[0] - 1), hyperedge_indices
            ],
                                            dim=1)

            other_indices = torch.arange(
                n_other_agents, dtype=torch.long) + n_tasks + n_actions + nhe
            other_to_self = torch.stack([
                other_indices,
                torch.ones(other_indices.shape[0], dtype=torch.long) *
                (nodes.shape[0] - 1)
            ],
                                        dim=1)

            task_indices = torch.tensor(
                [ed_batch[i][0] for i in range(len(ed_batch))],
                dtype=torch.long)
            tasks_to_hyperedges = torch.stack(
                [task_indices + nhe, hyperedge_indices], dim=1)

            all_task_nodes = node_index[nodes[:, 0] == task_node_number]
            unseen_task_nodes = all_task_nodes[
                ~torch.isin(all_task_nodes, tasks_to_hyperedges[0, :])]
            unseen_tasks_to_self = torch.stack([
                unseen_task_nodes,
                torch.ones(unseen_task_nodes.shape[0], dtype=torch.long) *
                (nodes.shape[0] - 1)
            ],
                                               dim=1)

            action_index_mapping = {
                val: idx
                for idx, val in enumerate(unique_actions)
            }
            action_values = [ed_batch[i][1] for i in range(len(ed_batch))]
            action_indices = torch.tensor(
                [action_index_mapping[x] for x in action_values],
                dtype=torch.long)
            actions_to_hyperedges = torch.stack(
                [action_indices + nhe + n_tasks, hyperedge_indices], dim=1)

            edges = torch.cat([
                self_to_hyperedge,
                other_to_self,
                tasks_to_hyperedges,
                actions_to_hyperedges,
                unseen_tasks_to_self,
            ],
                              dim=0)
            graph = tg.data.Data(x=nodes, edge_index=edges.swapaxes(0, 1))
            graphs.append(graph)

        self._wrapper_task_action_mapping = [torch.tensor(h) for h in hyperedges]
        return graphs


def mohito_hypergraph_wrapper_v0(env) -> BatchedAECEnv:
    return shared_wrapper(
        env,
        HypergraphWrapper,
    )
