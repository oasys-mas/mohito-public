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
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch.distributions import Categorical
from torch.nn.functional import softmax, relu, leaky_relu
from copy import deepcopy

torch.autograd.set_detect_anomaly(True)

# ------------------------------------
# GNN Definition

class ActorGNN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers = 20):

        super(ActorGNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList() 

        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.dropouts.append(torch.nn.Dropout(0.5))

        # num_layers - 2 because we add first and last layer separately
        for _ in range(num_layers-2):  
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.dropouts.append(torch.nn.Dropout(0.5))
        
        self.convs.append(GCNConv(hidden_dim, input_dim))


    def forward(self, data, subset_indices):

        x, edge_index = data.x, data.edge_index

        #x = x.float()
        # apply relu and dropout to all but last layer
        for i in range(len(self.convs)-1):
            x = relu(self.convs[i](x, edge_index))
            x = self.dropouts[i](x)
        
        # last layer without relu
        x = self.convs[-1](x, edge_index)  

        # extract the subset of nodes
        subset = x[subset_indices]

        # max over summed features of subset nodes
        subset_max = subset.sum(dim=1).max()

        return subset_max, subset.sum(dim=1).argmax()


    def straightforward(self, data):
        x, edge_index = data.x, data.edge_index

        # apply relu and dropout to all but last layer
        for i in range(len(self.convs)-1):
            x = relu(self.convs[i](x, edge_index))
            x = self.dropouts[i](x)
        
        # last layer without relu
        x = self.convs[-1](x, edge_index)  
        return  x


# ------------------------------------
# Actor Network Definition

class ActorNetwork(nn.Module):
    def __init__(self, num_state_features, LR_A, BETA, hidden_dim_actor = 50, num_layers = 20, grad_clip:float = 1):

        super(ActorNetwork, self).__init__()
        # main actor network
        self.main = ActorGNN(num_state_features, hidden_dim_actor, num_layers=num_layers)  
        # target actor network
        self.target = ActorGNN(num_state_features, hidden_dim_actor, num_layers=num_layers)  
        self.optimizer = optim.Adam(self.main.parameters(), lr=LR_A, weight_decay=1e-4)
        self.BETA = BETA
        self.gradient_norms = [] 
        self.layer_grad_norms = []
        self.grad_clip = grad_clip

    def forward(self, data, subset_nodes, network = 'main'):

        if network == 'main':
            max_val, max_indx = self.main(data, subset_nodes)
        elif network == 'target':
            max_val, max_indx= self.target(data, subset_nodes)
        return max_val, max_indx


    def straightforward(self, data, network = 'main'):

        if network == 'main':
            g = self.main.straightforward(data)
        elif network == 'target':
            g = self.target.straightforward(data)
        return g


    def getRandomAction(self, graph, edge_space, action_space, network = 'main'):
        # generate graphs for each sample from buffer
    
        max_e_val, max_e_idx = self.randomForward(graph, edge_space, network)

        selected_action = action_space[max_e_idx.item()]

        return max_e_val, selected_action

    def getAction(self, graph, edge_space, action_space, network = 'main'):
        # generate graphs for each sample from buffer

        max_e_val, max_e_idx = self.forward(graph, edge_space, network)

        selected_action = action_space[min(max_e_idx.item(), len(action_space)-1)]

        return max_e_val, selected_action

    def getBatchAction(self, graph_list, edge_space_list, action_space_list, network = 'main'):
        # generate graphs for each sample from buffer
        selected_actions = []
        max_e_val = []
        
        for graph, edge_space, action_space in zip(graph_list, edge_space_list, action_space_list):
            max_e, action = self.getAction(graph, edge_space, action_space, network)

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

        # clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)  # clip gradients to prevent explosion

        # update weights
        self.optimizer.step() 
    
    def soft_update(self, tau = 0.001):
        for target_param, main_param in zip(self.target.parameters(), self.main.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)  

# ------------------------------------
# Critic Network Definition

class GNN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers = 20):

        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()

        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.dropouts.append(torch.nn.Dropout(0.5))

        # num_layers - 2 because we add first and last layer separately
        for _ in range(num_layers-2): 
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.dropouts.append(torch.nn.Dropout(0.5))

        self.convs.append(GCNConv(hidden_dim, input_dim))

    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        x = x.float()

        # apply relu for all but last layer
        for i in range(len(self.convs)-1):
            x = relu(self.convs[i](x, edge_index))
            x = self.dropouts[i](x)
        
        # last layer without relu
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
        self.gradient_norms = []

    def forward(self, graphs, actions_list, network='main'):
        batch_graph = Batch.from_data_list(graphs)
        #actions = torch.tensor(actions_list, dtype=torch.float).view(-1, self.num_action_features * self.num_agents)
        batch = batch_graph.batch

        if network == 'main':
            state_repr = self.main[0](batch_graph)
            state_repr_pooled = global_mean_pool(state_repr, batch)  # global pooling
        elif network == 'target':
            state_repr = self.target[0](batch_graph)
            state_repr_pooled = global_mean_pool(state_repr, batch)  # global pooling
        else:
            raise ValueError("Invalid network choice")

        state_action_repr = torch.cat((state_repr_pooled, actions_list), dim=1)

        if network == 'main':
            output = self.main[1:](state_action_repr)
        elif network == 'target':
            output = self.target[1:](state_action_repr)
            
        return output


    def update(self, actual, target, reg_term):
        # loss
        loss = nn.MSELoss()(actual, target) # + reg_term

        # reset gradients
        self.optimizer.zero_grad()

        # compute gradients
        loss.backward(retain_graph=False)

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
        return loss

    def soft_update(self, tau = 0.001):
        for target_param, main_param in zip(self.target.parameters(), self.main.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)  
        

