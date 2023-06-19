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

clique_sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50]

GCN_args = {
    'name': 'GCN',
    'device': device,
    'num_layers': 3,
    'hidden_dim': 64,
    'lr': 0.001,    
    'dropout': 0,
    'epochs': 100,
}

GraphSAGE_args = {
    'name': 'GraphSAGE',
    'device': device,
    'num_layers': 3,
    'hidden_dim': 64,
    'lr': 0.001,
    'dropout': 0.2,
    'epochs': 100,
}

GAT_args = {
    'name': 'GAT',
    'device': device,
    'num_layers': 2,
    'hidden_dim': 8,
    'heads': 4,
    'lr': 0.001,
    'dropout': 0.2,
    'epochs': 100
}

GIN_args = {
    'name': 'GIN',
    'device': device,
    'num_layers': 3,
    'hidden_dim': 64,
    'lr': 0.001,
    'epochs': 100,
    'mlp_hidden_dim': 64,
    'dropout': 0.2,
    'mlp': lambda input_dim, hidden_dim, output_dim: Sequential(Linear(input_dim, hidden_dim), ReLU(), Linear(hidden_dim, output_dim), ReLU())
}

args = [GCN_args, GraphSAGE_args, GAT_args, GIN_args]
for clique_size in clique_sizes:

    train_loader, valid_loader, test_loader = generate_dataset(100, 0.5, clique_size, 10000, 1000, 1000, 200)
    class_weights = torch.tensor([clique_size, 100 - clique_size]).type(torch.float32)
    for arg in args:
        num_features = 1
        num_labels = 2

        if arg['name'] == 'GCN':
            model = GCN(num_features, 
                    arg['hidden_dim'],
                    num_labels, arg['num_layers'],
                    arg['dropout']).to(device)
        elif arg['name'] == 'GraphSAGE':
            model = GraphSAGE(num_features, 
                        arg['hidden_dim'],
                        num_labels, arg['num_layers'],
                        arg['dropout']).to(device)
        elif arg['name'] == 'GAT':
            model = GAT(num_features, 
                        arg['hidden_dim'],
                        arg['heads'],
                        num_labels, arg['num_layers'],
                        arg['dropout']).to(device)
        elif arg['name'] == 'GIN':
            model = GIN(num_features, 
                        arg['mlp'], arg['mlp_hidden_dim'],
                        arg['hidden_dim'],
                        num_labels, arg['num_layers'],
                        arg['dropout']).to(device)

        results, model = run_experiment(model, arg, train_loader, valid_loader, test_loader, class_weights)

        results_filename = arg['name']+'_'+str(clique_size)+'_'+'results.pickle'
        model_filename = arg['name']+'_'+str(clique_size)+'_'+'model.pt'

        with open(results_dir+results_filename, 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

        torch.save(model.state_dict(), trained_models_dir+model_filename)