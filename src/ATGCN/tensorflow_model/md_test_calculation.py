from cProfile import label
import numpy as np
from scipy.stats import chi2
import csv
from md_clean_calculation import data_preprocessing
import matplotlib.pyplot as plt
from md_clean_calculation import GLOBAL_MEAN_ERROR
from md_clean_calculation import L
from md_clean_calculation import UPPER_TH
from md_clean_calculation import LOWER_PLOT
from md_clean_calculation import UPPER_PLOT
import pandas as pd
import seaborn as sns
import scipy as sp
from sklearn.covariance import MinCovDet

# Before any attacks there will be a 17 hour time stamps
EVAL_POISON_LABEL_DIR = "out/tgcn/tgcn_scada_wds_lr0.005_batch128_unit64_seq8_pre1_epoch101/eval_test/eval_test_labels.csv"
EVAL_POISON_PREDS_DIR = "out/tgcn/tgcn_scada_wds_lr0.005_batch128_unit64_seq8_pre1_epoch101/eval_test/eval_test_output.csv"
EVAL_POISON_LINE_NUM = 2080  # CHANGE for each different eval_poisoned_output.csv

dataset04 = pd.read_csv(
    r"data/processed/test_scada_dataset.csv"
)  # CHANGE for each different poisoned dataset.csv

binary_arr = dataset04["ATT_FLAG"].to_list()
binary_arr = binary_arr[(L + 8) : -1]  # use 8 for prediction + L for first window size
#
convert_th_binary_arr = [LOWER_PLOT if x == 0 else UPPER_PLOT for x in binary_arr]

##########
def calculate_md_test():
    """Calculates the Mahalanobis Distance for poisoned dataset"""
    # Get lists
    df_eval_labels, df_eval_preds = data_preprocessing(
        num_line=EVAL_POISON_LINE_NUM,
        label_dataset=EVAL_POISON_LABEL_DIR,
        preds_dataset=EVAL_POISON_PREDS_DIR,
    )

    # Convert lists to numpy arrays
    df_eval_labels = np.array(df_eval_labels)
    df_eval_preds = np.array(df_eval_preds)

    # Get error array
    df_error = df_eval_labels - df_eval_preds

    # 1. Calculate the covariance matrix
    cov = np.cov(df_error, rowvar=False)

    # 2. Calculate cov^-1
    covariance_pm1 = np.linalg.matrix_power(cov, -1)

    # 3. Calculate the mean error arrayy
    global_mean_error = GLOBAL_MEAN_ERROR

    # 4. Calculate the mahalanobis distance
    distances = []
    for i, val in enumerate(df_error):
        p1 = val  # Ozone and Temp of the ith row
        p2 = global_mean_error
        distance = (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
        distances.append(distance)
        # print(f"Distance: {distance}")
    distances = np.array(distances)

    mean_batch_squared_md_arr = []

    outliers = []

    for i in range(L, (len(distances))):
        batch_squared_md = distances[i - L : i]  # take the first L batches
        mean_batch_squared_md = np.average(batch_squared_md)
        mean_batch_squared_md_arr.append(mean_batch_squared_md)
        if mean_batch_squared_md >= UPPER_TH:
            outliers.append(UPPER_PLOT)
        else:
            outliers.append(LOWER_PLOT)
            
    print(f"The Average Mean Squared Mahalanobis Distance {np.average(mean_batch_squared_md_arr)}")

    fig1 = plt.figure(figsize=(5, 3))
    plt.plot(mean_batch_squared_md_arr, label="mean squared batch squared md")
    plt.plot(convert_th_binary_arr, label="attacks labels")
    plt.title(
        "Mean Squared Robust Mahalanobis Distance Every L Hours TimeStamp - Poisoned Dataset"
    )
    plt.xlabel("Every L hours")
    plt.ylabel("Mean Squared Robust Mahalanobis Distance - To Calibrate Max Threshold")
    plt.legend()
    plt.show()

    fig1 = plt.figure(figsize=(5, 3))
    plt.plot(convert_th_binary_arr, label="attacks labels")
    plt.plot(outliers, label="attacks predictions")
    plt.legend()
    plt.show()


##########
def calculate_rmd_test():
    """Calculates the Mahalanobis Distance for poisoned dataset"""
    # Get lists
    df_eval_labels, df_eval_preds = data_preprocessing(
        num_line=EVAL_POISON_LINE_NUM,
        label_dataset=EVAL_POISON_LABEL_DIR,
        preds_dataset=EVAL_POISON_PREDS_DIR,
    )

    # Convert lists to numpy arrays
    df_eval_labels = np.array(df_eval_labels)
    df_eval_preds = np.array(df_eval_preds)

    # Get error array
    df_error = df_eval_labels - df_eval_preds

    # 1. Calculate Minimum Covariance Determinant
    rng = np.random.RandomState(0)

    # 2. Calculate the covariance matrix
    real_cov = np.cov(df_error, rowvar=False)

    # 3. Get multivariate values
    X = rng.multivariate_normal(mean=np.mean(df_error, axis=0), cov=real_cov, size=506)

    # 4. Get MCD values
    cov = MinCovDet(random_state=0).fit(X)
    mcd = cov.covariance_  # robust covariance metrics

    # 5. Calculate the invert covariance matrix
    inv_covmat = sp.linalg.inv(mcd)

    # 6. Calculate the mahalanobis distance
    distances = []
    for i, val in enumerate(df_error):
        p1 = val  # Ozone and Temp of the ith row
        p2 = GLOBAL_MEAN_ERROR
        distance = (p1 - p2).T.dot(inv_covmat).dot(p1 - p2)
        distances.append(distance)
        # print(f"Distance: {distance}")
    distances = np.array(distances)

    mean_batch_squared_rmd_arr = []

    outliers = []

    thresholds = [UPPER_TH for _ in range(len(binary_arr))]

    for i in range(L, (len(distances))):
        batch_squared_rmd = distances[i - L : i]  # take the first L batches
        mean_batch_squared_rmd = np.average(batch_squared_rmd)
        mean_batch_squared_rmd_arr.append(mean_batch_squared_rmd)
        if mean_batch_squared_rmd >= UPPER_TH:
            outliers.append(UPPER_PLOT)
        else:
            outliers.append(LOWER_PLOT)

    print(
        f"The Average Mean Squared Robust Mahalanobis Distance {np.average(mean_batch_squared_rmd_arr)}"
    )

    fig1 = plt.figure(figsize=(5, 3))
    plt.plot(mean_batch_squared_rmd_arr, label="mean squared batch squared md")
    plt.plot(convert_th_binary_arr, label="attacks labels")
    plt.plot(thresholds, label="threshold")
    plt.title(
        "Mean Squared Robust Mahalanobis Distance Every L Hours TimeStamp - Poisoned Dataset"
    )
    plt.xlabel("Every L hours")
    plt.ylabel("Mean Squared Robust Mahalanobis Distance - To Calibrate Max Threshold")
    plt.legend()
    plt.show()

    fig1 = plt.figure(figsize=(5, 3))
    plt.plot(convert_th_binary_arr, label="attacks labels")
    plt.plot(outliers, label="attacks predictions")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # calculate_md_test()
    calculate_rmd_test()
