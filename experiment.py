
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
def run_experiment(model, model_args, train_loader, valid_loader, test_loader, class_weights):
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=model_args['lr'])
    loss_fn = CrossEntropyLoss(class_weights.to(device))

    best_model = None
    best_valid_acc = 0

    results = {}

    for epoch in range(1, 1 + model_args["epochs"]):
        print('Training...')
        loss = train(model, train_loader, optimizer, loss_fn)

        print('Evaluating...')
        train_result = eval(model, train_loader)
        val_result = eval(model, valid_loader)
        test_result = eval(model, test_loader)
        
        train_acc, valid_acc, test_acc = train_result, val_result, test_result
        results[epoch] = {train_acc, valid_acc, test_acc}

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(model)
            
        print(f'Epoch: {epoch:02d}, '
                f'Loss: {loss:.4f}, '
                f'Train: {100 * train_acc:.2f}%, '
                f'Valid: {100 * valid_acc:.2f}% '
                f'Test: {100 * test_acc:.2f}%')
    
    results['best'] = {
        'train_acc': eval(best_model, train_loader),
        'valid_acc': eval(best_model, valid_loader),
        'test_acc': eval(best_model, test_loader)
    }
        
    return results