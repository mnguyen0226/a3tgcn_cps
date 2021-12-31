import numpy as np
import scipy.sparse as sp
import torch
from torch.nn.functional import normalize

# Reference: https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch

def calculate_laplacian_with_self_loop(matrix):
    """Calculates the normalized laplacian with input matrix

    Args:
        matrix: input matrix

    Returns:
        normalized laplacian
    """
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian