# Attention Temporal Graph Convolutional Network for Cyber Physical Attacks Detection
- Code Implementation for "Cyber-Physical Attack Detection in Water Distribution Systems with Robust Supervised and Unsupervised DeepLearning Models: A Comparison".
- Supervised Model: Attention Temporal Graph Convolutional Networks.

## 1. Attacks Detection Scheme

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

## 3. A3T-GCN Architecture


## 4. Results

### BATADAL Ranking

| Author      | Number of Attacks Detected | S      | S_TTD | S_CM      | TPR | TNR      | 
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | 
| Housh and Ohar      | 7 | 0.97      | 0.965 | 0.975      | 0.953 | 0.997      | 
| Abokifa et al      | 7 | 0.949      | 0.958 | 0.944      | 0.921 | 0.959      | 
| **HCAE**      | **7** | **0.933**      | **0.947** | **0.918**      | **0.865** | **0.972 **     | 
| Tsiami et al      | 7 | 0.931      | 0.934 | 0.928      | 0.885 | 0.971      | 
| Giacomoni et al      | 7 | 0.927      | 0.936 | 0.917      | 0.838 | 0.997      | 
| Brentan et al      | 6 | 0.894      | 0.857 | 0.931      | 0.889 | 0.973      | 
| **A3T-GCN**     | **7** | **0.845**      | **0.839** | **0.851**      | **0.774** | **0.927**      | 
| Chandy et al      | 7 | 0.802      | 0.835 | 0.768      | 0.857 | 0.678      | 
| Pasha et al      | 7 | 0.773      | 0.885 | 0.66      | 0.329 | 0.992      | 
| Aghashahi et al      | 3 | 0.534      | 0.429 | 0.64      | 0.396 | 0.884      | 

### Forecasting Performance on Normal Dataset

|             | Baseline Model | Robust Mahalanobis Distance, Attention |
| ----------- | ----------- | ----------- |
| Minimum RMSE      | 6.858166163       | 5.960369429 | 
| Minimum MAE   | 3.3477044        | 2.7673762 |
| Maximum Accuracy   | 0.8372700512        | 0.8585200906 | 
| R2   | -0.6772449017        | -0.6805173159 |
| Variance   | 0.9530872479        | 0.9646917097 |

### Attack Detection

|             | Baseline Model | Robust Mahalanobis Distance, Attention |
| ----------- | ----------- | ----------- |
| Precision      | 0.6355932203       | 0.7208237986 | 
| Recall / True Positive Rate   | 0.5528255528       | 0.773955774 |
| F1 Score   | 0.5913272011      | 0.7464454976 | 
| Accuracy   | 0.8496131528        | 0.8965183752 |
| Specificity / True Negative   | 0.9223359422        | 0.9265502709 |



### Attack Localization 
| Attacks Labels | Attacks Description | Feature Localization of A3T-GCN | 
| ----------- | ----------- | ----------- |
| Attack 8      | Alteration of L_T3 thresholds leading to underflow | P_J256 = 11, `L_T3 = 3`, P_J289 = 2, L_T2 = 2 |
| Attack 9   | Alteration of L_T2 | P_J289 = 13, P_J422 = 13, P_J300 = 5, L_T7 = 2 |
| Attack 10   | Activation of PU3 | `F_PU3 = 38`, P_J280 = 28, L_T7 = 23,  L_T4 = 6, P_J269 = 6, F_PU1 = 8, F_PU9 = 2 |
| Attack 11   | Activation of PU3 | `F_PU3 = 36`,  P_J280 = 31, L_T7 = 23, F_PU1 = 22, L_T4 = 12, L_T6 = 11,  P_J307 = 7, P_J415 = 3, F_PU6 = 2, P_J289 = 2 |
| Attack 12   | Alteration of L_T2 readings leading to overflow | P_J289 = 7, P_J300 = 6, `L_T2 = 2` |
| Attack 13   | Change the L_T7 thresholds | L_T6 = 2 |
| Attacls 17   | Alteration of T4 signal | `L_T4 = 8`, L_T7 = 5, P_J415 = 4, L_T6 = 2 |

### Robustness with GANs


## 5. References
- [TGCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://github.com/lehaifeng/T-GCN)
- [A3T_GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting](https://github.com/lehaifeng/T-GCN)