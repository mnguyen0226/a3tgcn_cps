# Cyber-Physical Attack Detection, Localization, & Attribution Evaluation with Temporal Graph Neural Networks for Water Distribution Systems

Explainable TGCN for Water Distribution Systems

## 1.Overall Pipeline

![alt-text](https://github.com/mnguyen0226/xtgcn_wds_cps/blob/main/docs/imgs/pipeline.png)

## 2.Developing Pipeline

![alt-text](https://github.com/mnguyen0226/xtgcn_wds_cps/blob/main/docs/imgs/tgcn_train_pipeline.png)

## 3.Progress:

- Need to saved and loaded train model => Evaluate with poisoned dataset
- :soon: Experience with Mahalanobis Outlier => Understand the pipeline
- Provide Bench-marking results
- Option: Continue to help Namzul with GANs
- Option: Working on Attribution
- Option ATCGN

## Baseline models:

- SVM
- AE
- GCN
- GRU

## 4. Explain TGCN:

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

## 5.Tensorflow TGCN

- Note: interval option: 15 minutes, 30 minutes, 45 miutes, 60 minutes

### Requirements

- tensorflow == 1.14
- python == 3.7
- scipy
- numpy
- matplotlib
- pandas
- math
- sklearn

### Model Training

```
nguye@DESKTOP-OBHI23I MINGW64 ~/OneDrive/Desktop/Senior/ECE 4994 A3 Research/xtgcn_wds_cps/src/tensorflow_model (main)
$ python ./main.py
```

## 6.Pytorch TGCN (Ignore this)

### Requirements

- numpy
- numpy
- matplotlib
- pandas
- torch
- pytorch-lightning>=1.3.0
- torchmetrics>=0.3.0
- python-dotenv

### Model Training

```python
python train_main.py --model_name TGCN --max_epochs 1 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --hidden_dim 64 --loss mse_with_regularizer --settings supervised

python test_main.py --model_name TGCN --max_epoch 1 --batch_size 32 --loss mse_with_regularizer --settings supervised

python main.py --model_name TGCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --hidden_dim 64 --loss mse_with_regularizer --settings supervised --gpus 1
```

You can also adjust the `--data`, `--seq_len` and `--pre_len` parameters.

Run tensorboard `--logdir lightning_logs/version_0` to monitor the training progress and view the prediction results.

### References:

- [TGCN: A Temporal Graph Convolutional Network for Traffic Prediction](Reference: https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch)
