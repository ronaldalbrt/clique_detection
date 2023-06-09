
from config import device
from eval import train, eval

import copy
import torch
from torch.nn import CrossEntropyLoss

# ------------------------------------------------------------------------------
# Run an the experiment for a given model
# ------------------------------------------------------------------------------
# Parameters:
# model: the model to be trained
# model_args: the arguments for the model
# train_loader: the dataloader for the training set
# valid_loader: the dataloader for the validation set
# test_loader: the dataloader for the test set
# ------------------------------------------------------------------------------
# return: the best model and the best validation accuracy
# ------------------------------------------------------------------------------
def run_experiment(model, model_args, train_loader, valid_loader, test_loader, class_weights, metric='roc_auc'):
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=model_args['lr'])
    loss_fn = CrossEntropyLoss(class_weights.to(device))

    best_model = None
    best_valid_metric = 0

    results = {}

    for epoch in range(1, 1 + model_args["epochs"]):
        print('Training...')
        loss = train(model, train_loader, optimizer, loss_fn)

        print('Evaluating...')
        train_result = eval(model, train_loader)
        val_result = eval(model, valid_loader)
        test_result = eval(model, test_loader)
        
        train_metric, valid_metric, test_metric = train_result[metric], val_result[metric], test_result[metric]
        results[epoch] = {
            'train_result': train_result, 
            'validation_result': val_result, 
            'test_result':test_result
        }

        if valid_metric > best_valid_metric:
            best_valid_metric = valid_metric
            best_model = copy.deepcopy(model)
            
        print(f'Epoch: {epoch:02d}, '
                f'Loss: {loss:.4f}, '
                f'Train: {100 * train_metric:.2f}%, '
                f'Valid: {100 * valid_metric:.2f}% '
                f'Test: {100 * test_metric:.2f}%')
    
    results['best'] = {
        'train_results': eval(best_model, train_loader),
        'valid_results': eval(best_model, valid_loader),
        'test_results': eval(best_model, test_loader)
    }
        
    return results, best_model