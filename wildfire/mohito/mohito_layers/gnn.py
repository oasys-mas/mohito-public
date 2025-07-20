from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict
import warnings

import torch.nn as nn
import torch
import torch_geometric as tg
from torch_geometric.nn import GATv2Conv, GATConv, GCNConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import to_undirected


def fully_connect_masked_nodes(mask: torch.Tensor,
                               batch: torch.Tensor) -> torch.Tensor:
    """
    Attaches all hyperedge nodes to each other (fully connected)

    Args:
        mask (torch.Tensor): mask of the nodes to be connected
        batch (torch.Tensor): batch index of the nodes
    
    Returns:
        torch.Tensor: edge index of the fully connected nodes
    """
    edge_index_list = []

    # Unique graph ids
    for graph_id in batch[mask].unique():
        # Get nodes in this graph and in the mask
        nodes = torch.where(mask & (batch == graph_id))[0]
        n = nodes.size(0)

        if n < 2:
            continue  # no edges possible

        # Pairwise combinations without self-loops
        src = nodes.repeat_interleave(n)
        dst = nodes.repeat(n)
        mask_no_self = src != dst
        src = src[mask_no_self]
        dst = dst[mask_no_self]

        edge_index = torch.stack([src, dst], dim=0)
        edge_index = to_undirected(edge_index)
        edge_index_list.append(edge_index)

    # Concatenate all edge indices
    if edge_index_list:
        return torch.cat(edge_index_list, dim=1)
    else:
        return torch.empty((2, 0), dtype=torch.long)


@dataclass
class GNNReturn:
    """
    Return type of the GNN model
    """
    x: torch.Tensor
    agent_E_mask: torch.Tensor
    agent_mask: torch.BoolTensor


class GNN(nn.Module):
    """
    GNN base model for the MOHITO algorithm
    """

    def __init__(
        self,
        hidden_dim: int,
        number_of_layers: int,
        relu_slope: float,
        dropout_rate: float,
        num_heads: int,
        node_feature_size: int = None,
        edge_feature_size: int = None,
        this_agent_hyperedge_identifier: int = 5,
        agent_identifier: int = 1,
        task_identifier: int = 3,
        add_self_loops: bool = True,
        use_graph_norm: bool = False,
        layer_type: str = 'gatv2',
        fc_hyperedges: bool = True,
        layer_params: Optional[Dict[str, Any]] = {'dropout': 0.0},
        device: str = 'cpu',
        #!dep inop
        swap_noop_to_zero_index: bool = None,
        modified_subset_loss: Optional[Callable[[torch.Tensor, torch.Tensor],
                                                torch.Tensor]] = None,
        ablation_ignore_task_nodes: bool = False,
    ) -> object:
        """
        Args:
            hidden_dim (int): size of the hidden layer
            number_of_layers (int): number of GAT/GCN layers in the model
            relu_slope (float): slope of the relu activation function
            dropout_rate (float): dropout rate (when mode='training')
            num_heads (int): number of attention heads
            node_feature_size (int, optional):  size of the node feature vector, x
            edge_feature_size (int, optional):  size of the edge feature vector, edge_attr

            this_agent_hyperedge_identifier (int, optional): integer prefix of node features that identify this agent's hyperedges
            agent_identifier (int, optional): integer prefix of node features that identify this agent
            task_identifier (int, optional): integer prefix of node features that identify task nodes

            add_self_loops (bool, optional): whether to add self loops to the graph
            use_graph_norm (bool, optional): whether to use graph norm
            layer_type (str, optional): - type of the layer to use (gatv2/gat/gcn)
            fc_hyperedges (bool, optional): whether to fully connect hyperedges (default: True)
            layer_params (Optional[Dict[str, Any]], optional): parameters to pass to the layer

            device (str, optional): device to use for the model (default: 'cpu')

            #!dep inop
            swap_noop_to_zero_index (bool, optional): deprecated
            modified_subset_loss (Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], optional): deprecated

            ablation_ignore_task_nodes (bool, optional): whether to ignore task nodes in the GNN. sets task_node features to 0 (for ablation studies)

            Returns:
                object: GNN object
        """

        super(GNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.number_of_layers = number_of_layers
        self.relu_slope = relu_slope
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.agent_identifier = agent_identifier
        self.this_agent_hyperedge_identifier = this_agent_hyperedge_identifier
        self.task_identifier = task_identifier
        self.ablation_ignore_task_nodes = ablation_ignore_task_nodes
        self.graph_norm = use_graph_norm
        self.add_self_loops = add_self_loops
        self.layer_params = layer_params
        self.device = device
        self.fc_hyperedges = fc_hyperedges
        
        if not self.fc_hyperedges:
            print("WARNING: fully connected hyperedges are disabled, this may impact observations")

        assert swap_noop_to_zero_index is None, "swap_noop_to_zero_index is deprecated"
        assert modified_subset_loss is None, "modified_subset_loss is deprecated"

        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.relu = torch.nn.LeakyReLU(negative_slope=self.relu_slope)

        #select the layer type
        match layer_type:
            case 'gatv2':
                self.conv = GATv2Conv
            case 'gat':
                warnings.warn(
                    "GAT faces a 'static attention problem' see https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.GATv2Conv.html."
                )
                self.conv = GATConv
            case 'gcn':
                self.conv = GCNConv
            case _:
                raise ValueError(
                    f"Invalid layer_type: {layer_type} use 'gatv2', 'gat' or 'gcn'"
                )

        self.build()

    def build(self) -> None:
        """
        Constructs network layers
        """

        #construct network  Conv/Attn -> ReLU -> Dropout > rep.....  -> Conv/Attn
        self.net = nn.ModuleList()

        self.net.append(
            self.conv(in_channels=self.node_feature_size,
                      out_channels=self.hidden_dim,
                      heads=self.num_heads,
                      add_self_loops=self.add_self_loops,
                      edge_dim=self.edge_feature_size,
                      **self.layer_params))

        if self.graph_norm:
            self.net.append(GraphNorm(self.hidden_dim * self.num_heads))

        self.net.append(self.relu)

        if self.dropout_rate > 0:
            self.net.append(self.dropout)

        for _ in range(self.number_of_layers - 2):
            self.net.append(
                self.conv(in_channels=self.hidden_dim * self.num_heads,
                          out_channels=self.hidden_dim,
                          heads=self.num_heads,
                          add_self_loops=self.add_self_loops,
                          edge_dim=self.edge_feature_size,
                          **self.layer_params))

            if self.graph_norm:
                self.net.append(GraphNorm(self.hidden_dim * self.num_heads))

            self.net.append(self.relu)

            if self.dropout_rate > 0:
                self.net.append(self.dropout)

        self.net.append(
            self.conv(in_channels=self.hidden_dim * self.num_heads,
                      out_channels=self.node_feature_size,
                      heads=1,
                      add_self_loops=self.add_self_loops,
                      edge_dim=self.edge_feature_size,
                      **self.layer_params))

    def forward(self,
                data: tg.data.Data,
                training: Optional[bool] = None,
                restyle_for: bool = False,
                hyperedge_connections: str = None) -> GNNReturn:
        """
        Forward pass of the model

        Args:
            data (tg.data.Data): input data (this should be a torch_geometric.data object already batched)
            training (Optional[bool], optional): whether the model is in training mode -DEPRECATED (use .eval() and .train() instead)
            restyle_for (bool, optional): whether to use the restyle for the hyperedge
            hyperedge_connections (str, optional): whether to use the overlapping hyperedges

        Returns:
            GNNReturn: output of the GNN
        """
        if training is not None:
            warnings.warn(
                "training is deprecated, use .eval() or .train() instead")

        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)

        if self.ablation_ignore_task_nodes:
            # Set task nodes to zero
            task_mask = x[:, 0] == self.task_identifier
            x[task_mask, 1:] = 0

        node_index = torch.arange(x.size(0), device=x.device)

        agent_E_mask = x[:, 0] == self.this_agent_hyperedge_identifier
        agent_mask = x[:, 0] == self.agent_identifier

        hyperedge_node_indices = node_index[agent_E_mask]
        connected_hyperedges = (
            torch.isin(edge_index[0], hyperedge_node_indices)
            & torch.isin(edge_index[1], hyperedge_node_indices))

        if not self.fc_hyperedges:
            edge_index = edge_index[:, ~connected_hyperedges]

        batch = data.batch if hasattr(data, 'batch') else None

        res_counter = 0

        for layer in self.net:
            if isinstance(layer, GraphNorm):
                x = layer(x, batch=batch)

            elif isinstance(layer, GATv2Conv) or isinstance(
                    layer, GATConv) or isinstance(layer, GCNConv):

                if restyle_for:
                    if res_counter % 2 == 0 and res_counter > 0:
                        new_edge_index = edge_index.clone()
                        new_edge_index = new_edge_index[:,
                                                        connected_hyperedges]
                    else:
                        new_edge_index = edge_index.clone()
                    res_counter += 1
                else:
                    new_edge_index = edge_index.clone()

                x = layer(x, new_edge_index)
            else:
                x = layer(x)

        return GNNReturn(x=x, agent_E_mask=agent_E_mask, agent_mask=agent_mask)
