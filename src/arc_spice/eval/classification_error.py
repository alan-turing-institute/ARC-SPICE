# hamming accuracy
# zero-one accuracy

import torch


def hamming_accuracy(preds: torch.Tensor, class_labels: torch.Tensor):
    accuracy = torch.mean((preds.float() == class_labels.float()).float())
    return accuracy


def aggregate_score(probs: torch.tensor):
    preds = torch.round(probs).float()
    distance = torch.abs(preds - probs)
    return 1 - torch.mean(distance)
