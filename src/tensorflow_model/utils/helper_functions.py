# Reference: https://github.com/lehaifeng/T-GCN/blob/master/T-GCN/T-GCN-TensorFlow/utils.py

import tensorflow as tf
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy.linalg as la
import math


def normalized_adj(adj):
    """Normalized adjacency matrix for GCN

    Args:
        adj_matrix ([type]): adjacency matrix

    Returns:
        normalized adjacency matrix
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    normalized_adj = normalized_adj.astype(np.float32)
    return normalized_adj


def sparse_to_tuple(mx):
    """Converts sparse matrix to tuple

    Args:
        mx ([type]): matrix

    Returns:
        tuple
    """
    mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    L = tf.SparseTensor(coords, mx.data, mx.shape)
    return tf.sparse_reorder(L)


def calculate_laplacian(adj, lambda_max=1):
    adj = normalized_adj(adj + sp.eye(adj.shape[0]))
    adj = sp.csr_matrix(adj)
    adj = adj.astype(np.float32)
    return sparse_to_tuple(adj)


def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim], minval=-init_range, maxval=init_range, dtype=tf.float32
    )

    return tf.Variable(initial, name=name)


def evaluation(pred, label):
    rmse = math.sqrt(mean_squared_error(pred, label))
    mae = mean_absolute_error(pred, label)
    F_norm = la.norm(pred - label, "fro") / la.norm(pred, "fro")
    r2 = 1 - ((pred - label ** 2).sum()) / ((pred - pred.mean()) ** 2).sum()
    var = 1 - (np.var(pred - label)) / np.var(pred)
    return rmse, mae, 1 - F_norm, r2, var
