import torch

# Reference: https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch


def accuracy(pred, y):
    """Calculates the accuracy rate between prediction and actual label

    Args:
        pred: prediction
        y: ground-truth label

    Returns:
        accuracy rate
    """
    return 1 - torch.linalg.norm(y - pred, "fro") / torch.linalg.norm(y, "fro")


def r2(pred, y):
    """Calculates the R1 regression-based metric

    Args:
        pred: prediction
        y: ground-truth label

    Returns:
        r2 rate
    """
    return 1 - torch.sum((y - pred) ** 2) / torch.sum((y - torch.mean(pred)) ** 2)


def explained_variance(pred, y):
    """Calculated variance

    Args:
        pred: prediction
        y: ground-truth

    Returns:
        variance rate
    """
    return 1 - torch.var(y - pred) / torch.var(y)
