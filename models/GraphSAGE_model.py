# # Description: GraphSAGE model for node classification with PyTorch Geometric
# Author: Ronald Albert
# Last Modified: June 2023

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, LogSoftmax
from torch_geometric.nn import SAGEConv

# ------------------------------------------------------------------------------
# GraphSAGE model for node classification
# ------------------------------------------------------------------------------
# Parameters:
# input_dim: dimension of the input features
# hidden_dim: dimension of the hidden layers
# output_dim: dimension of the output features
# num_layers: number of layers
# return_embeds: if True, the model returns the embeddings instead of the logits
# ------------------------------------------------------------------------------
# return: The GraphSAGE model
# ------------------------------------------------------------------------------
class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=False):
        super(GraphSAGE, self).__init__()

        self.convs = None
        self.softmax = None
        self.dropout = dropout

        def get_in_channels(idx):
            return hidden_dim if idx > 0 else input_dim

        def get_out_channels(idx):
            return hidden_dim if idx < num_layers - 1 else output_dim

        self.convs = torch.nn.ModuleList([
            SAGEConv(in_channels=get_in_channels(i), out_channels=get_out_channels(i))
            for i in range(num_layers)
        ])

        self.softmax = LogSoftmax(dim=1)
        self.return_embeds = return_embeds

    # ------------------------------------------------------------------------------
    # Reset the parameters of the model
    # ------------------------------------------------------------------------------
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    # ------------------------------------------------------------------------------
    # Forward pass of the model
    # ------------------------------------------------------------------------------
    # Parameters:
    # batched_data: batched data from the dataloader
    # ------------------------------------------------------------------------------
    # return: the logits or the embeddings
    # ------------------------------------------------------------------------------
    def forward(self, batched_data):
        x, adj_t = batched_data.x, torch.transpose(batched_data.adj_t,0,1)

        out = None

        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.convs[-1](x, adj_t)
        if not self.return_embeds:
            out = self.softmax(out)

        return out