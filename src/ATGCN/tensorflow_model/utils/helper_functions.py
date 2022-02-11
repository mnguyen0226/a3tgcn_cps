# Reference: https://github.com/lehaifeng/T-GCN/blob/master/T-GCN/T-GCN-TensorFlow/utils.py

import tensorflow as tf
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy.linalg as la
import math
from sklearn.metrics import confusion_matrix


def normalized_adj(adj):
    """Normalized adjacency matrix for GCN. This is extra features in case we have weighted graph.

    Args:
        adj_matrix: Adjacency matrix.

    Returns:
        Normalized adjacency matrix.
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
        mx: Input sparse matrix

    Returns:
        Tuple representation of sparse matrix.
    """
    mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    L = tf.SparseTensor(coords, mx.data, mx.shape)
    return tf.sparse_reorder(L)


def calculate_laplacian(adj, lambda_max=1):
    """Converts from adjacency matrix to laplacian matrix for graph representation.
    Reference: https://en.wikipedia.org/wiki/Laplacian_matrix

    Args:
        adj: Adjacency matrix.
        lambda_max (int, optional): Defaults to 1.

    Returns:
        Laplacian representation of matrix.
    """
    adj = normalized_adj(adj + sp.eye(adj.shape[0]))
    adj = sp.csr_matrix(adj)
    adj = adj.astype(np.float32)
    return sparse_to_tuple(adj)


def weight_variable_glorot(input_dim, output_dim, name=""):
    """Calculates the uniform distribution of tensor.
    Reference: https://mmuratarat.github.io/2019-02-25/xavier-glorot-he-weight-init

    Args:
        input_dim: Number of input neurons in the weight tensor.
        output_dim: Number of output neurons in the weight tensor.
        name: Defaults to "".

    Returns:
        normalized distribution.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim], minval=-init_range, maxval=init_range, dtype=tf.float32
    )

    return tf.Variable(initial, name=name)


def evaluation(pred, label):
    """Evaluates the predictions of TGCN vs ground-truth labels

    Args:
        pred: Predictions
        label: Ground-truth labels

    Returns:
        RMSE, MAE, Accuracy, Coefficient of Determination, Explained Variance Score
    """
    # root mean squared error
    rmse = math.sqrt(mean_squared_error(pred, label))

    # mean absolute error
    mae = mean_absolute_error(pred, label)

    # loss
    F_norm = la.norm(pred - label, "fro") / la.norm(pred, "fro")

    # coefficient of determination
    r2 = 1 - ((pred - label ** 2).sum()) / ((pred - pred.mean()) ** 2).sum()

    # explained variance score
    var = 1 - (np.var(pred - label)) / np.var(pred)

    return rmse, mae, 1 - F_norm, r2, var


def classification_metrics(label_arr, pred_arr):
    """Return evaluation metrics for binary classification

    Args:
        label_arr: array stored the binary labels
        pred_arr: array stored the binary predictions

    Returns:
        [type]: [description]
    """
    tn, fp, fn, tp = confusion_matrix(np.array(label_arr), np.array(pred_arr)).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)  # true positive
    f1 = (2 * precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    specificity = tn / (tn + fp)  # true negative

    return precision, recall, f1, accuracy, specificity
