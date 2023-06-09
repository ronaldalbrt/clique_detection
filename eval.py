# Description: This file contains the training and evaluation functions for the model
# Author: Ronald Albert
# Last Modified: June 2023

import torch
from config import device

# ------------------------------------------------------------------------------
# The training function for the model
# ------------------------------------------------------------------------------
# Parameters:
# model: the model to be trained
# data_loader: the dataloader for the training set
# optimizer: the optimizer for the model
# loss_fn: the loss function for the model
# ------------------------------------------------------------------------------
# return: the loss value
# ------------------------------------------------------------------------------
def train(model, data_loader, optimizer, loss_fn):
    model.train()
    loss = 0

    for batch in data_loader:
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()

            preds = model(batch)
            labels = batch.y.to(device)
            loss = loss_fn(preds, labels.squeeze())

            loss.backward()
            optimizer.step()

    return loss.item()


# ------------------------------------------------------------------------------
# The evaluation function for the model
# ------------------------------------------------------------------------------
# Parameters:
# model: the model to be evaluated
# loader: the dataloader for the evaluation setevaluator.eval(input_dict)["acc"]
# ------------------------------------------------------------------------------
# return: the accuracy value
# ------------------------------------------------------------------------------
def eval(model, loader):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch).argmax(dim=-1, keepdim=True)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    return (y_true == y_pred).sum() / len(y_true)