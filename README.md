# Cyber-Physical Attack Detection, Localization, & Attribution Evaluation with Temporal Graph Neural Netorks for Water Distribution Systems

Explainable TGCN for Water Distribution Systems

## Overall Pipeline

![alt-text](https://github.com/mnguyen0226/xtgcn_wds_cps/blob/main/docs/imgs/pipeline.png)

## Developing Pipeline

![alt-text](https://github.com/mnguyen0226/xtgcn_wds_cps/blob/main/docs/imgs/tgcn_train_pipeline.png)

## Requirements

- numpy
- numpy
- matplotlib
- pandas
- torch
- pytorch-lightning>=1.3.0
- torchmetrics>=0.3.0
- python-dotenv

## Model Training

```python
python train_main.py --model_name TGCN --max_epochs 1 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --hidden_dim 64 --loss mse_with_regularizer --settings supervised

python test_main.py --model_name TGCN --max_epoch 1 --batch_size 32 --loss mse_with_regularizer --settings supervised

python main.py --model_name TGCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --hidden_dim 64 --loss mse_with_regularizer --settings supervised --gpus 1
```

You can also adjust the `--data`, `--seq_len` and `--pre_len` parameters.

Run tensorboard `--logdir lightning_logs/version_0` to monitor the training progress and view the prediction results.

## References:

- [TGCN: A Temporal Graph Convolutional Network for Traffic Prediction](Reference: https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch)
