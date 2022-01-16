import torch

# Reference: https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch


def accuracy(pred, y):
    """Calculates the accuracy rates between the prediction and the ground-truth labels"""
    return 1 - torch.linalg.norm(y - pred, "fro") / torch.linalg.norm(y, "fro")


def r2(pred, y):
    """Calculates the R2 regression rate between prediction and the ground-truth labels"""
    return 1 - torch.sum((y - pred) ** 2) / torch.sum((y - torch.mean(pred)) ** 2)


def explained_variance(pred, y):
    """Calculate the explained variance rate between prediction and the ground-truth labels"""
    return 1 - torch.var(y - pred) / torch.var(y)
