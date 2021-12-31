import torch

# Reference: https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch

def mse_with_regularizer_loss(inputs, targets, models, lamda=1.5e-3):
    """Calculates mean square error with regularizer loss

    Args:
        inputs
        targets
        models
        lamda. Defaults to 1.5e-3.

    Returns:
        mse_loss + reg_loss
    """
    reg_loss = 0.0
    for param in models.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss
    mse_loss = torch.sum((inputs - targets) ** 2) / 2
    return mse_loss + reg_loss
