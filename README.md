# Cyber-Physical Attack Detection, Localization, & Attribution Evaluation with Temporal Graph Neural Networks for Water Distribution Systems

Explainable TGCN for Water Distribution Systems

## 1.Overall Pipeline

![alt-text](https://github.com/mnguyen0226/xtgcn_wds_cps/blob/main/docs/imgs/pipeline.png)

## 2.Developing Pipeline

![alt-text](https://github.com/mnguyen0226/xtgcn_wds_cps/blob/main/docs/imgs/tgcn_train_pipeline.png)

## 3.Progress:

- Need to saved and loaded train model => Evaluate with poisoned dataset
- :soon: Experience with Mahalanobis Outlier => Understand the pipeline
- Get training model to be 90%
- Calculate the MD between the prediction and clean validation labels.
- Calculate the MD between the prediction and the attacked labels just to make sure 
- Provide Bench-marking results
- Option: Continue to help Namzul with GANs
- Option: Working on Attribution
- Option ATCGN + Robust MD
- Dr. B Preference: Work on Seasoning Prediction: Split Clean Dataset 50/50 to train / test dataset. Train 80% of the training dataset and predict 1 days, 1 weeks, 1 months. Then validate the testing dataset.

## Mahalanobis Distance Process (in Validation phase):
- Load up trained model.
- Input the first 12 row in the validation table, predict the next row.
- Calculate the error array at time i between the predictions array at time i and the ground-truth array at time i.
- Append the newly calculated error array to the 2D ERROR array.
- Calculate the mean error array by calculate the mean of 2D Error array.
- Calculate the covariance between the mean error array and the error array at time i
- Calculate the MD_ value.
- Append the MD_ value to the MD array

=> After done, calculate the mean of MD to get the max threshold to be "normal operation"

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

- tensorflow == 1.14 (conda install -c conda-forge tensorflow=1.14)
- python == 3.7 
- scipy (conda install -c anaconda scipy)
- numpy (conda install -c anaconda numpy)
- matplotlib (conda install -c conda-forge matplotlib)
- pandas (conda install -c anaconda pandas)
- math
- sklearn (conda install -c anaconda scikit-learn)

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

### Localization
- ATTACK 8 - Alteration of L_T3 thresholds leading to underflow: `L_T3`, F_PU2, F_PU4, F_PU5, F_PU6, F_PU8, F_PU10, F_V2, P_J256, P_J289, P_J415, P_J306, P_J317, P_J422.
- ATTACK 9 - Alteration of L_T2 readings leading to overflow: L_T5, F_PU1, F_PU4, F_PU7, F_PU8, F_PU10, F_V2, P_J300, P_j289, P_J415, P_J302, P_J306, P_J307, P_J317, P_J422.
- ATTACK 10 - Activation of PU3: L_T4, L_T6, L_T7, F_PU1, F_PU2, `F_PU3`, F_PU7, F_PU8, F_PU10, F_V2, P_J280, P_J269, P_J415, P_J306, P_J317, P_J422
- ATTACK 11 - Activation of PU3: L_T4, L_T6, L_T7, F_PU1, F_PU2, `F_PU3`, F_PU4, F_PU6, F_PU7, F_PU8, F_PU10, F_V2, P_J280, P_J269, P_J256, P_J289, P_J415, P_J302 P_J306, P_J307 P_J317
- ATTACK 12 - Alteration of L_T2 readings leading to overflow: F_PU1, F_PU2, F_PU4, F_PU7, F_PU8, F_PU10, F_V2, P_J300, P_J289, P_J415, P_J306, P_J317, P_J422
- ATTACK 13 - Change the L_T7 thresholds: L_T5, F_PU2, F_PU4, F_PU7, F_PU8, F_V2, P_J415, P_J302, P_J306, P_J307, P_J317, P_J422
- ATTACK 14 - Alteration of T4 signal: `L_T4`, L_T7, F_PU1, F_PU2, F_PU4, F_PU7, F_PU8, F_PU10, F_V2, P_J269, P_J256m P_J289, P_J415, P_J302, P_J3206, P_J307, P_J317