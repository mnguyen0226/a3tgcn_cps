import numpy as np
import pandas as pd
import torch

# Reference: https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch


def load_features(feat_path, dtype=np.float32):
    """Loads features matrix"""
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)

    # DEBUGGING
    # print(f"\n\nFeat Shape:\n {feat.shape}")
    # print(f"Feat:\n {feat}")

    return feat


def load_adjacency_matrix(adj_path, dtype=np.float32):
    """Loads adjacency matrix"""
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)

    # DEBUGGING
    # print(f"\n\Adjacency Shape:\n {adj.shape}")
    # print(f"Adjacency:\n {adj}")

    return adj


def generate_dataset(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    """Generates train_X, train_Y, test_X, test_Y"""
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

    # DEBUGGING
    # print(np.array(train_X).shape)
    # print(train_X[0])
    # print(np.array(train_Y).shape)
    # print(train_Y[0])
    # print(np.array(train_X[1]).shape)
    # print(train_X[1])
    # print(np.array(train_Y[1]).shape)
    # print(train_Y[1])

    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)


def generate_torch_datasets(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    """Transforms datasets to Tensor datatype and returns train/test datasets"""
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
