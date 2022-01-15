# Reference: https://github.com/lehaifeng/T-GCN/blob/master/T-GCN/T-GCN-TensorFlow/utils.py

import tensorflow as tf
import scipy.sparse as sp
import numpy as np


def normalized_adj(adj_matrix):
    """Normalized adjacency matrix for GCN

    Args:
        adj_matrix ([type]): adjacency matrix

    Returns:
        normalized adjacency matrix
    """
    adj_matrix = sp.coo_matrix(adj_matrix)
    rowsum = np.array(adj_matrix.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = (
        adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    )
    normalized_adj = normalized_adj.astype(np.float32)
    return normalized_adj


def sparse_to_tuple(mx):
    mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    L = tf.SparseTensor(coords, mx.data, mx.shape)
    return tf.sparse_reorder(L)


def calculate_laplacian(adj_matrix, lambda_max=1):
    adj_matrix = normalized_adj(adj_matrix + sp.eye(adj_matrix.shape[0]))
    adj_matrix = sp.csr_matrix(adj_matrix)
    adj_matrix = adj_matrix.astype(np.float32)
    return sparse_to_tuple(adj_matrix)


def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim], minval=-init_range, maxval=init_range, dtype=tf.float32
    )

    return tf.Variable(initial, name=name)
