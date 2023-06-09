from models.GCN_model import GCN
from config import device, results_dir,trained_models_dir
from experiment import run_experiment
from dataset import generate_dataset

import random
import pickle
import torch

random.seed(7)

train_loader, valid_loader, test_loader, class_weights = generate_dataset(200, 0.5, 50, 5000, 1000, 1000, 32)

GCN_args = {
    'device': device,
    'num_layers': 3,
    'hidden_dim': 256,
    'dropout': 0.5,
    'lr': 0.01,
    'epochs': 20,
}

num_features = 64
num_labels = 2

model = GCN(num_features, GCN_args['hidden_dim'],
            num_labels, GCN_args['num_layers'],
            GCN_args['dropout']).to(device)

GCN_results, GCN_model = run_experiment(model, GCN_args, train_loader, valid_loader, test_loader, class_weights)

GCN_results_filename = 'GCN_results.pickle'
GCN_model_filename = 'GCN_model.pt'

with open(results_dir+GCN_results_filename, 'wb') as file:
    pickle.dump(GCN_results, file, protocol=pickle.HIGHEST_PROTOCOL)

torch.save(GCN_model.state_dict(), trained_models_dir+GCN_model_filename)