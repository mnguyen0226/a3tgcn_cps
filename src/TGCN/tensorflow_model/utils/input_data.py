# Reference: https://github.com/lehaifeng/T-GCN/blob/master/T-GCN/T-GCN-TensorFlow/input_data.py

import numpy as np
import pandas as pd
import pickle as pkl


def load_scada_data(dataset="train_eval_clean"):
    """Loads the adjacency matrix for GCN and clean time-series dataset.

    Returns:
        Loaded clean-time-series dataset and adjacency matrix
    """
    adj = pd.read_csv(r"data/processed/processed_scada_adj_matrix.csv", header=None)
    adj_matrix = np.mat(adj)
    if dataset == "train_eval_clean":
        time_series_data = pd.read_csv(
            r"data/processed/processed_clean_scada_dataset.csv"
        )
    elif dataset == "eval_poison":
        time_series_data = pd.read_csv(
            r"data/processed/processed_dataset04_origin_no_binary.csv"
        )
    elif dataset == "test":
        time_series_data = pd.read_csv(
            r"data/processed/processed_test_scada_dataset.csv"
        )
    return time_series_data, adj_matrix


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    """Preprocesses training and testing dataset into batches

    Args:
        data: Time-series dataset.
        time_len: Number of row in the matrix.
        rate: %80, 20% train, evaluate split.
        seq_len: Number of row to train =  12.
        pre_len: Number of row to predict = 3, 6, 9, or 12.

    Returns:
        Preprocessed training and testing features and labels
    """
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []

    # divides training dataset into features and label batches
    for i in range(len(train_data) - seq_len - pre_len):
        batch = train_data[i : i + seq_len + pre_len]
        trainX.append(batch[0:seq_len])
        trainY.append(batch[seq_len : seq_len + pre_len])

    # divides testing dataset into features and label batches
    for i in range(len(test_data) - seq_len - pre_len):
        batch = test_data[i : i + seq_len + pre_len]
        testX.append(batch[0:seq_len])
        testY.append(batch[seq_len : seq_len + pre_len])

    # converts lists to array
    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)

    return trainX1, trainY1, testX1, testY1
