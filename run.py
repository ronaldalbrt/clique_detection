from models.GCN_model import GCN
from models.GraphSAGE_model import GraphSAGE
from models.GAT_model import GAT
from models.GIN_model import GIN
from config import device, results_dir,trained_models_dir
from experiment import run_experiment
from dataset import generate_dataset

import random
import pickle
import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d

random.seed(7)

train_loader, valid_loader, test_loader = generate_dataset(100, 0.5, 5, 10000, 1000, 1000, 8)

class_weights = torch.tensor([50, 150]).type(torch.float32)

GCN_args = {
    'device': device,
    'num_layers': 3,
    'hidden_dim': 64,
    'dropout': 0.5,
    'lr': 0.01,
    'epochs': 20,
}

GraphSAGE_args = {
    'device': device,
    'num_layers': 3,
    'hidden_dim': 1,
    'dropout': 0.5,
    'lr': 0.01,
    'epochs': 20,
}

GAT_args = {
    'device': device,
    'num_layers': 3,
    'hidden_dim': 64,
    'heads': 2,
    'dropout': 0.5,
    'lr': 0.01,
    'epochs': 20
}

GIN_args = {
    'device': device,
    'num_layers': 5,
    'hidden_dim': 64,
    'dropout': 0.5,
    'lr': 0.1,
    'epochs': 100,
    'mlp_hidden_dim': 64,
    'mlp': lambda input_dim, hidden_dim, output_dim: Sequential(Linear(input_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(), Linear(hidden_dim, output_dim), ReLU())
}
num_features = 1
num_labels = 2

model = GraphSAGE(num_features, 
            GraphSAGE_args['hidden_dim'],
            num_labels, GraphSAGE_args['num_layers'],
            GraphSAGE_args['dropout']).to(device)

GraphSAGE_results, GraphSAGE_model = run_experiment(model, GraphSAGE_args, train_loader, valid_loader, test_loader, class_weights)

GraphSAGE_results_filename = 'GraphSAGE_results.pickle'
GraphSAGE_model_filename = 'GraphSAGE_model.pt'

with open(results_dir+GraphSAGE_results_filename, 'wb') as file:
    pickle.dump(GraphSAGE_results, file, protocol=pickle.HIGHEST_PROTOCOL)

torch.save(GraphSAGE_model.state_dict(), trained_models_dir+GraphSAGE_model_filename)