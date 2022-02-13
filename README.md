# Robust Cyber-Physical Attack Detection in WaterDistribution Systems with Supervised and UnsupervisedModels: A Comparison
- Supervised Model: Attention Temporal Graph Convolutional Networks.

## 1.Attacks Detection Scheme

![alt-text](https://github.com/mnguyen0226/xtgcn_wds_cps/blob/main/docs/imgs/attack_detection_scheme.png)

## 2. Requirements

- tensorflow == 1.14 (conda install -c conda-forge tensorflow=1.14)
- python == 3.7 
- scipy (conda install -c anaconda scipy)
- numpy (conda install -c anaconda numpy)
- matplotlib (conda install -c conda-forge matplotlib)
- pandas (conda install -c anaconda pandas)
- math
- sklearn (conda install -c anaconda scikit-learn)

## 3. ATGCN Architecture Explained:

- Temporal Graph Convolutional Network (TGCN) model is a combination between Graph Convolutional Networks and Gated Recurrent Unit (GRU).
- The baseline TGCN calculate the value of at each node in the next T modments: `[Xt+1, · · · , Xt+T ] = f (G; (Xt−n, · · · , Xt−1, Xt))`.
- Loss function is regular error + L2 regularization to avoid overfitting. We use Adam optimizer to reduce the loss function.
- Metrics: 
    - Root Mean Squared Error (RMSE): is used to measure the prediction error, the smaller the value, the better the prediction effect is.
    - Mean Absolute Error (MAE): is used to measure the prediction error, the smaller the value, the better the prediction effect is.
    - Accuracy: is used to detect the prediction precision. The larger the value, the better the prediction effect is.
    - Coefficient of Determination (R2): is used to calculated the correlation coefficient, which measures the ability of the prediction result to represent the actual data. The larger the value is, the better the prediction effect is.
    - Explained Variance Score (Var): is used to calculated the correlation coefficient, which measures the ability of the prediction result to represent the actual data. The larger the value is, the better the prediction effect is.

- Hyperparameters (default, can be changed and optimized via Genetic Algorithms):
    - OP_LR = 0.001 = learning rate [0.0, 1]
    - OP_BATCH_SIZE = 32 # batch size is the number of samples that will be passed through to the network at one time (in this case, number of 12 rows/seq_len/time-series be fetched and trained in TGCN at 1 time) [4, 8, 16, 32, 64, 128, 256, 450]
    - OP_EPOCH = 3000 # number of epochs / iteration > 0
    - OP_HIDDEN_DIM = 64 # output dimension of the hidden_state in GRU. This is NOT number of GRU in 1 TGCN. [8, 16, 32, 64, 100, 128]

## 4. Results:
### Temporal Data Forecasting

### Attack Detection

### Attack Localization 
- ATTACK 8 - Alteration of L_T3 thresholds leading to underflow: `L_T3`, F_PU2, F_PU4, F_PU5, F_PU6, F_PU8, F_PU10, F_V2, P_J256, P_J289, P_J415, P_J306, P_J317, P_J422.
- ATTACK 9 - Alteration of L_T2 readings leading to overflow: L_T5, F_PU1, F_PU4, F_PU7, F_PU8, F_PU10, F_V2, P_J300, P_j289, P_J415, P_J302, P_J306, P_J307, P_J317, P_J422.
- ATTACK 10 - Activation of PU3: L_T4, L_T6, L_T7, F_PU1, F_PU2, `F_PU3`, F_PU7, F_PU8, F_PU10, F_V2, P_J280, P_J269, P_J415, P_J306, P_J317, P_J422
- ATTACK 11 - Activation of PU3: L_T4, L_T6, L_T7, F_PU1, F_PU2, `F_PU3`, F_PU4, F_PU6, F_PU7, F_PU8, F_PU10, F_V2, P_J280, P_J269, P_J256, P_J289, P_J415, P_J302 P_J306, P_J307 P_J317
- ATTACK 12 - Alteration of L_T2 readings leading to overflow: F_PU1, F_PU2, F_PU4, F_PU7, F_PU8, F_PU10, F_V2, P_J300, P_J289, P_J415, P_J306, P_J317, P_J422
- ATTACK 13 - Change the L_T7 thresholds: L_T5, F_PU2, F_PU4, F_PU7, F_PU8, F_V2, P_J415, P_J302, P_J306, P_J307, P_J317, P_J422
- ATTACK 14 - Alteration of T4 signal: L_T4, L_T7, F_PU1, F_PU2, F_PU4, F_PU7, F_PU8, F_PU10, F_V2, P_J269, P_J256m P_J289, P_J415, P_J302, P_J3206, P_J307, P_J317

### Robustness


## 5. References:

- [TGCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://github.com/lehaifeng/T-GCN)
- [A3T_GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting](https://github.com/lehaifeng/T-GCN)