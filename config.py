# Description: Configuration file for the project
# Author: Ronald Albert
# Last Modified: June 2023

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

results_dir = 'results/'
trained_models_dir = 'trained_models/'