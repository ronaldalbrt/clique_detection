# Description: This file contains the functions to generate the dataset of random graphs
# Author: Ronald Albert
# Last Modified: June 2023

import random

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import erdos_renyi_graph
from torch.utils.data import WeightedRandomSampler

# ------------------------------------------------------------------------------
# Generate a random G(n, p) graph
# ------------------------------------------------------------------------------
# Parameters:
# n: number of nodes
# p: probability of edge creation
# clique_size: size of the clique
# ------------------------------------------------------------------------------
# return: a Data object with the graph
# ------------------------------------------------------------------------------
def generate_graph(n, p, clique_size):
    edge_index = erdos_renyi_graph(n, edge_prob=p)
    clique_nodes = random.sample(range(n), clique_size)

    x = torch.rand(n, 1).type(torch.float32)
    class_label = torch.zeros(n).type(torch.int64)
    class_label[clique_nodes] = 1

    for i, node_i in enumerate(clique_nodes):
        for node_j in clique_nodes[i + 1:]:
            edge_to_add = torch.tensor([[node_i, node_j], [node_j, node_i]])
            edge_index = torch.cat((edge_index, edge_to_add), 1)

    return Data(x=x, adj_t=torch.transpose(edge_index, 0, 1), y=class_label)

# ------------------------------------------------------------------------------
# Generate a dataset of random graphs
# ------------------------------------------------------------------------------
# Parameters:
# n: number of nodes
# p: probability of edge creation
# clique_size: size of the clique
# train_size: number of graphs in the training set
# valid_size: number of graphs in the validation set
# test_size: number of graphs in the test set
# batch_size: batch size for the dataloaders
# ------------------------------------------------------------------------------
# return: Dataloaders for the train, validation and test sets and the class weights
# ------------------------------------------------------------------------------
def generate_dataset(n, p, clique_size, train_size, valid_size, test_size, batch_size):
    train_data = [generate_graph(n, p, clique_size) for _ in range(train_size)]
    valid_data = [generate_graph(n, p, clique_size) for _ in range(valid_size)]
    test_data = [generate_graph(n, p, clique_size) for _ in range(test_size)]

    class_weights = torch.tensor([clique_size, (n - clique_size)]).type(torch.float32)

    sampler = WeightedRandomSampler(class_weights.type('torch.DoubleTensor'), len(class_weights))

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_data, batch_size=batch_size, sampler=sampler)

    return train_loader, valid_loader, test_loader