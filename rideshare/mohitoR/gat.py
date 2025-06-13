#----------------------------------------------------------------------------------------------------------------------#
# Title: Graph-based Actor and Critic Networks
# Description: This file contains the Actor and Critic networks and relevant functions
# Author: Gayathri Anil 
# Version: 23.05.01
# Last updated on: 05-03-2023 
#----------------------------------------------------------------------------------------------------------------------#

import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch.distributions import Categorical
from torch.nn.functional import softmax, relu, leaky_relu
from copy import deepcopy
from rideshare.utils import add_row_to_csv

torch.autograd.set_detect_anomaly(True)

# ------------------------------------
# Actor GNN Definition

class ActorGNN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=20, heads=2):

        super(ActorGNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()

        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads))
        self.dropouts.append(torch.nn.Dropout(0.5))
        self._init_weights(self.convs[-1])

        # num_layers - 2 because we add first and last layer separately
        for _ in range(num_layers-2):  
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))  # Note the multiplication by `heads` to match dimensions
            self.dropouts.append(torch.nn.Dropout(0.5))
            self._init_weights(self.convs[-1])
        
        self.convs.append(GATConv(hidden_dim * heads, input_dim, heads=heads))  # Again note the multiplication by `heads`
        self._init_weights(self.convs[-1])

    def _init_weights(self, module):
        if hasattr(module, 'weight') and module.weight is not None:
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, data, subset_indices, training = True):

        x, edge_index = data.x, data.edge_index

        before = str(x[subset_indices].sum(dim=1).tolist())

        # apply relu and dropout to all but last layer
        for i in range(len(self.convs)-1):
            x = relu(self.convs[i](x, edge_index))
            # x = self.convs[i](x, edge_index)
            if training:
                x = self.dropouts[i](x)
        
        # last layer without relu
        x = self.convs[-1](x, edge_index)  

        # extract the subset of nodes
        subset = x[subset_indices]

        after = str(subset.sum(dim=1).tolist())
        selected_index = subset.sum(dim=1).argmax()

        # max over summed features of subset nodes
        subset_max = subset.sum(dim=1).max()

        add_row_to_csv(file_path='./forward.csv', row = [before, after, selected_index, subset_max.item()], headers = ["before", "after", "selected_index", "selected_value"])

        return subset_max, subset.sum(dim=1).argmax()

    def straightforward(self, data, training = True):
        x, edge_index = data.x, data.edge_index

        # apply relu and dropout to all but last layer
        for i in range(len(self.convs)-1):
            x = relu(self.convs[i](x, edge_index))
            # x = self.convs[i](x, edge_index)
            if training:
                x = self.dropouts[i](x)
        
        # last layer without relu
        x = self.convs[-1](x, edge_index)  
        return  x




# ------------------------------------
# Actor Network Definition

class ActorNetwork(nn.Module):
    def __init__(self, num_state_features, LR_A, BETA, hidden_dim_actor = 50, num_layers = 20, heads = 2):

        super(ActorNetwork, self).__init__()
        # main actor network
        self.main = ActorGNN(num_state_features, hidden_dim_actor, num_layers=num_layers, heads = heads)  
        # target actor network
        self.target = ActorGNN(num_state_features, hidden_dim_actor, num_layers=num_layers, heads = heads)  
        self.optimizer = optim.Adam(self.main.parameters(), lr=LR_A, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9999)
        self.BETA = BETA
        self.gradient_norms = [] 
        self.layer_grad_norms = []

    def forward(self, data, subset_nodes, network = 'main', training = True):

        if network == 'main':
            max_val, max_indx = self.main(data, subset_nodes, training = training)
        elif network == 'target':
            max_val, max_indx= self.target(data, subset_nodes, training = training)
        return max_val, max_indx

    def straightforward(self, data, network = 'main', training = True):

        if network == 'main':
            g = self.main.straightforward(data, training=training)
        elif network == 'target':
            g = self.target.straightforward(data, training=training)
        return g

    def getRandomAction(self, graph, edge_space, action_space, network = 'main', training = True):
        # generate graphs for each sample from buffer
    
        max_e_val, max_e_idx = self.randomForward(graph, edge_space, network, training=training)

        selected_action = action_space[max_e_idx.item()]

        return max_e_val, selected_action

    def getAction(self, graph, edge_space, action_space, network = 'main', training = True):
        # generate graphs for each sample from buffer

        max_e_val, max_e_idx = self.forward(graph, edge_space, network, training=training)

        selected_action = action_space[min(max_e_idx.item(), len(action_space)-1)]

        return max_e_val, selected_action

    def getBatchAction(self, graph_list, edge_space_list, action_space_list, network = 'main', training=True):
        # generate graphs for each sample from buffer
        selected_actions = []
        max_e_val = []
        
        for graph, edge_space, action_space in zip(graph_list, edge_space_list, action_space_list):
            max_e, action = self.getAction(graph, edge_space, action_space, network, training=training)

            selected_actions.append(action)
            max_e_val.append(max_e)

        return max_e_val, selected_actions

    def update(self, actor_loss):
        # reset gradients
        self.optimizer.zero_grad()  

        # compute gradients
        actor_loss.backward(retain_graph = False)  

        # record gradients
        total_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        self.gradient_norms.append(total_norm)

        # # clip gradients
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1)  # clip gradients to prevent explosion

        # update weights
        self.optimizer.step() 
        self.scheduler.step()
    
    def soft_update(self, tau = 0.001):
        for target_param, main_param in zip(self.target.parameters(), self.main.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)  

# ------------------------------------
# Critic Network 


class GNN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=20):
        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()

        self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=True))
        self.dropouts.append(torch.nn.Dropout(0.5))
        self._init_weights(self.convs[-1])

        for _ in range(num_layers-2):
            self.convs.append(GATConv(4 * hidden_dim, hidden_dim, heads=4, concat=True))
            self.dropouts.append(torch.nn.Dropout(0.5))
            self._init_weights(self.convs[-1])

        self.convs.append(GATConv(4 * hidden_dim, input_dim, heads=1, concat=False))
        self._init_weights(self.convs[-1])

    def _init_weights(self, module):
        if hasattr(module, 'weight') and module.weight is not None:
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, batch_graph, training = True):

        x, edge_index = batch_graph.x, batch_graph.edge_index
        x = x.float()

        if edge_index.max() >= x.size(0):
            print(f"Error: Maximum edge index {edge_index.max()} is out of bounds for node feature matrix shape {x.size(0)}")
            return

        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = relu(x)
            if training:
                x = self.dropouts[i](x)
        
        x = self.convs[-1](x, edge_index)

        return x

class CriticNetwork(nn.Module):
    def __init__(self, num_state_features, num_agents, LR_C, hidden_dim_critic = 50, num_layers=3, num_action_features = 1, grad_clip = 1):
        super(CriticNetwork, self).__init__()

        self.num_state_features = num_state_features
        self.num_action_features = num_action_features
        self.num_agents = num_agents
        self.grad_clip = grad_clip

        # Main Critic Network
        self.main = nn.Sequential(
            GNN(num_state_features, hidden_dim_critic, num_layers),
            nn.Linear(num_state_features + num_action_features * num_agents, hidden_dim_critic),
            nn.Linear(hidden_dim_critic, hidden_dim_critic),
            nn.Linear(hidden_dim_critic, 1),
        )

        # Target Critic Network
        self.target = nn.Sequential(
            GNN(num_state_features, hidden_dim_critic, num_layers),
            nn.Linear(num_state_features + num_action_features * num_agents, hidden_dim_critic),
            nn.Linear(hidden_dim_critic, hidden_dim_critic),
            nn.Linear(hidden_dim_critic, 1),
        )

        self.optimizer = optim.Adam(self.main.parameters(), lr=LR_C, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9999)
        self.gradient_norms = []

    def forward(self, graphs, actions_list, network='main', training = True):
        batch_graph = Batch.from_data_list(graphs)  # Create a batch from multiple graphs
        batch = batch_graph.batch  # Get the batch index tensor
        
        if network == 'main':
            state_repr = self.main[0](batch_graph, training = training)
        elif network == 'target':
            state_repr = self.target[0](batch_graph, training = training)
        
        state_repr_pooled = global_mean_pool(state_repr, batch)
        state_action_repr = torch.cat((state_repr_pooled, actions_list), dim=1)

        if network == 'main':
            output = self.main[1:](state_action_repr)
        elif network == 'target':
            output = self.target[1:](state_action_repr)
            
        return output

    def straightforward(self, graphs, actions_list, network='main', training = True):
        # batch_graph = Batch.from_data_list(graphs)  # Create a batch from multiple graphs
        # batch = batch_graph.batch  # Get the batch index tensor
        
        if network == 'main':
            state_repr = self.main[0](graphs, training = training)
        elif network == 'target':
            state_repr = self.target[0](graphs, training = training)
        
        state_repr_pooled = global_mean_pool(state_repr, graphs.batch)
        state_action_repr = torch.cat((state_repr_pooled, actions_list), dim=1)

        if network == 'main':
            output = self.main[1:](state_action_repr)
        elif network == 'target':
            output = self.target[1:](state_action_repr)
            
        return output


    def update(self, actual_q, target_q, reg_term):
        # loss
        loss = nn.MSELoss()(actual_q, target_q) + reg_term

        # reset gradients
        self.optimizer.zero_grad()

        # compute gradients
        loss.backward(retain_graph=True)

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)  

        # record gradient norms for all parameters
        total_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        self.gradient_norms.append(total_norm)

        # update weights
        self.optimizer.step()
        self.scheduler.step()

    def soft_update(self, tau = 0.001):
        for target_param, main_param in zip(self.target.parameters(), self.main.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)  
        

