# hamming accuracy
# zero-one accuracy

import torch


def hamming_accuracy(preds: torch.Tensor, class_labels: torch.Tensor):
    # Inverse of the hamming loss (the fraction of labels incorrectly predicted)
    accuracy = torch.mean((preds.float() == class_labels.float()).float())
    return accuracy


def aggregate_score(probs: torch.tensor):
    # average 'distance' from the predicted class
    preds = torch.round(probs).float()
    distance = torch.abs(preds - probs)
    return 1 - torch.mean(distance)


def MC_dropout_uncertainty(variational_probs):
    # aggregate over the classes, performing MC Dropout on each class treating it
    # as a binary classification problem
    raise NotImplementedError
