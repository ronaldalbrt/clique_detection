from models.GCN_model import GCN
from models.GraphSAGE_model import GraphSAGE
from models.GIN_model import GIN
from config import device, results_dir,trained_models_dir
from experiment import run_experiment
from dataset import generate_dataset

import random
import pickle
import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d

random.seed(7)

train_loader, valid_loader, test_loader, class_weights = generate_dataset(200, 0.5, 50, 5000, 1000, 1000, 32)

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
    'hidden_dim': 64,
    'dropout': 0.5,
    'lr': 0.01,
    'epochs': 100,
}

GIN_args = {
    'device': device,
    'num_layers': 5,
    'hidden_dim': 64,
    'dropout': 0.5,
    'lr': 0.01,
    'epochs': 20,
    'mlp_hidden_dim': 64,
    'mlp': lambda input_dim, hidden_dim, output_dim: Sequential(Linear(input_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(), Linear(hidden_dim, output_dim), ReLU())
}
num_features = 1
num_labels = 2

model = GIN(num_features, 
            GIN_args['mlp'], GIN_args['mlp_hidden_dim'], 
            GIN_args['hidden_dim'],
            num_labels, GIN_args['num_layers'],
            GIN_args['dropout']).to(device)

GIN_results, GIN_model = run_experiment(model, GIN_args, train_loader, valid_loader, test_loader, class_weights)

GIN_results_filename = 'GIN_results.pickle'
GIN_model_filename = 'GIN_model.pt'

with open(results_dir+GIN_results_filename, 'wb') as file:
    pickle.dump(GIN_results, file, protocol=pickle.HIGHEST_PROTOCOL)

torch.save(GIN_model.state_dict(), trained_models_dir+GIN_model_filename)