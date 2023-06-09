# Description: This file contains the training and evaluation functions for the model
# Author: Ronald Albert
# Last Modified: June 2023

import torch
import torch.nn.functional as nnf
from config import device
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

# ------------------------------------------------------------------------------
# Class for metrics calculation
# ------------------------------------------------------------------------------
# Parameters:
# prob: the probability output of the model
# y: the ground truth labels
# ------------------------------------------------------------------------------
# return: the metrics
# ------------------------------------------------------------------------------
def metrics(prob, y):
    def roc_auc(prob, y):
        return roc_auc_score(y, prob)
    
    def accuracy(prob, y):
        return accuracy_score(y, prob.round())
    
    def f1(prob, y):
        return f1_score(y, prob.round())
    
    def precision(prob, y):
        return precision_score(y, prob.round(), zero_division=0)
    
    def recall(prob, y):
        return recall_score(y, prob.round())

    prob = prob.numpy()[:,1]
    y = y.numpy()

    return {
        'roc_auc': roc_auc(prob, y), 
        'acc': accuracy(prob, y), 
        'f1': f1(prob, y), 
        'precision': precision(prob, y), 
        'recall': recall(prob, y)
        }
    
    

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
    y_true = torch.tensor([])
    y_pred = torch.tensor([])

    for batch in loader:
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred_prob =  nnf.softmax(model(batch), dim=1)

            y_true = torch.cat((y_true, batch.y.view(pred_prob[:,1].shape).detach().cpu()), 0)
            y_pred = torch.cat((y_pred, pred_prob.detach().cpu()), 0)

    return metrics(y_pred, y_true)