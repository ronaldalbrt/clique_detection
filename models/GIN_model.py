# # Description: GIN model for node classification with PyTorch Geometric
# Author: Ronald Albert
# Last Modified: June 2023

import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv

# ------------------------------------------------------------------------------
# GIN model for node classification
# ------------------------------------------------------------------------------
# Parameters:
# input_dim: dimension of the input features
# hidden_dim: dimension of the hidden layers
# output_dim: dimension of the output features
# num_layers: number of layers
# dropout: dropout probability
# return_embeds: if True, the model returns the embeddings instead of the logits
# ------------------------------------------------------------------------------
# return: The GIN model
# ------------------------------------------------------------------------------
class GIN(torch.nn.Module):
    def __init__(self, input_dim, mlp, mlp_hidden_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        super(GIN, self).__init__()

        self.convs = None
        self.bns = None
        self.softmax = None

        def get_in_channels(idx):
            return hidden_dim if idx > 0 else input_dim

        def get_out_channels(idx):
            return hidden_dim if idx < num_layers - 1 else output_dim
                
        self.convs = torch.nn.ModuleList([
            GINConv(
                mlp(get_in_channels(i), mlp_hidden_dim, get_out_channels(i)),
            )
            for i in range(num_layers)
        ])

        self.bns = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=get_out_channels(i))
            for i in range(num_layers - 1)
        ])

        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.dropout = dropout
        self.return_embeds = return_embeds

    # ------------------------------------------------------------------------------
    # Reset the parameters of the model
    # ------------------------------------------------------------------------------
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

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

        for layer, bn in zip(self.convs[:-1], self.bns):
            x = layer(x, adj_t)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.convs[-1](x, adj_t)
        if not self.return_embeds:
            out = self.softmax(out)

        return out