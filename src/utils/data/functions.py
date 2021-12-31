import numpy as np
import pandas as pd
import torch

# Reference: https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch


def load_features(feat_path, dtype=np.float32):
    """Loads feature matrix from data path

    Args:
        feat_path: feature path
        dtype. Defaults to np.float32.

    Returns:
        feature array
    """
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    return feat


def load_adjacency_matrix(adj_path, dtype=np.float32):
    """Loads adjacency matrix from data path

    Args:
        adj_path" adjacency matrix
        dtype. Defaults to np.float32.

    Returns:
        adjacency array
    """
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj


def generate_dataset(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    """Generates training and test datasets

    Args:
        data
        seq_len
        pre_len
        time_len. Defaults to None.
        split_ratio (float, optional). Defaults to 0.8.
        normalize (bool, optional). Defaults to True.

    Returns:
        train_X, train_Y, test_X, test_Y
    """
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data / max_val
    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:time_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i : i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len : i + seq_len + pre_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i : i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pre_len]))
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)


def generate_torch_datasets(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    """Converts formulated dataset into Tensor datatype

    Args:
        data
        seq_len
        pre_len
        time_len. Defaults to None.
        split_ratio (float, optional). Defaults to 0.8.
        normalize (bool, optional). Defaults to True.

    Returns:
        train_dataset, test_dataset
    """
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )

    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset
