# Reference: https://github.com/lehaifeng/T-GCN/blob/master/T-GCN/T-GCN-TensorFlow/input_data.py

import numpy as np
import pandas as pd
import pickle as pkl


def load_clean_scada_data():
    """Loads the adjacency matrix for GCN and clean time-series dataset.

    Returns:
        loaded clean-time-series dataset and adjacency matrix
    """
    adj = pd.read_csv(r"data/processed/scada_adj_matrix.csv", header=None)
    adj_matrix = np.mat(adj)
    clean_data = pd.read_csv(r"data/processed/processed_clean_scada_dataset.csv")
    return clean_data, adj_matrix


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    """Preprocesses training and testing dataset into batches

    Args:
        data ([type]): time-series dataset
        time_len ([type]): 15, 30, or 45 minutes
        rate ([type]): ?
        seq_len ([type]): ?
        pre_len ([type]): ?

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
        trainX.append(batch[0:seq_len])
        trainY.append(batch[seq_len : seq_len + pre_len])

    # convert lists to array
    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)

    return trainX1, trainY1, testX1, testY1
