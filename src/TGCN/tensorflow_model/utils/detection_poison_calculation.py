import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from sklearn.covariance import MinCovDet
from utils.detection_clean_calculation import data_preprocessing
from utils.detection_clean_calculation import GLOBAL_MEAN_ERROR
from utils.detection_clean_calculation import L
from utils.detection_clean_calculation import UPPER_TH
from utils.detection_clean_calculation import LOWER_PLOT
from utils.detection_clean_calculation import UPPER_PLOT


# Before any attacks there will be a 17 hour time stamps
EVAL_POISON_LABEL_DIR = "out/tgcn/tgcn_scada_wds_lr0.01_batch16_unit64_seq8_pre1_epoch101/eval_poisoned/eval_poisoned_labels.csv"
EVAL_POISON_PREDS_DIR = "out/tgcn/tgcn_scada_wds_lr0.01_batch16_unit64_seq8_pre1_epoch101/eval_poisoned/eval_poisoned_output.csv"
EVAL_POISON_LINE_NUM = 4168  # CHANGE for each different eval_poisoned_output.csv
shade_of_gray = "0.75"
shade_of_blue = "lightsteelblue"

dataset04 = pd.read_csv(
    r"data/processed/processed_dataset04_origin_binary.csv"
)  # change for each different poisoned dataset.csv

binary_arr = dataset04["ATT_FLAG"].to_list()
binary_arr = binary_arr[(L + 8) : -1]  # use 8 for prediction + L for first window size
testing_attack_labels = binary_arr  # collect binary labels
convert_th_binary_arr = [LOWER_PLOT if x == 0 else UPPER_PLOT for x in binary_arr]
thresholds = [UPPER_TH for _ in range(len(binary_arr))]


def calculate_md_poison():
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

    # Calculate the covariance matrix
    cov = np.cov(df_error, rowvar=False)

    # Calculate cov^-1
    covariance_pm1 = np.linalg.matrix_power(cov, -1)

    # Calculate the mean error arrayy
    global_mean_error = GLOBAL_MEAN_ERROR

    # Calculate the mahalanobis distance
    distances = []
    for i, val in enumerate(df_error):
        p1 = val
        p2 = global_mean_error
        distance = (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
        distances.append(distance ** 1.5)
    distances = np.array(distances)

    mean_batch_squared_md_arr = []

    outliers = []

    testing_attack_preds = []

    for i in range(L, (len(distances))):
        batch_squared_md = distances[i - L : i]  # take the first L batches
        mean_batch_squared_md = np.average(batch_squared_md)
        mean_batch_squared_md_arr.append(mean_batch_squared_md)
        if mean_batch_squared_md >= UPPER_TH:
            outliers.append(UPPER_PLOT)
            testing_attack_preds.append(1.0)
        else:
            outliers.append(LOWER_PLOT)
            testing_attack_preds.append(0.0)

    print(f"The Average Mahalanobis Distance {np.average(mean_batch_squared_md_arr)}")

    # MD Plot
    fig1 = plt.figure(figsize=(20, 8))
    plt.title("Mahalanobis Distance Of Every Hour On Poisoned Dataset")
    df_plot_labels = pd.Series((i for i in convert_th_binary_arr))
    plt.plot(convert_th_binary_arr, label="Attacks Labels")
    plt.fill_between(
        df_plot_labels.index,
        df_plot_labels.values,
        where=df_plot_labels.values <= UPPER_PLOT,
        interpolate=True,
        color=shade_of_blue,
    )
    plt.plot(mean_batch_squared_md_arr, color="black", lw=2, label="MD")
    plt.plot(thresholds, color="red", label="Threshold")

    plt.xlabel("t (h)")
    plt.ylabel("Mahalanobis Distance")
    plt.figtext(0.16, 0.24, "L = " + str(L))
    plt.figtext(0.16, 0.22, "TH = " + str(UPPER_TH))
    plt.legend(loc=2, fancybox=True, shadow=True)
    plt.show()

    # Binary Classification Plot
    fig1 = plt.figure(figsize=(20, 8))
    plt.title("Attacks Predictions vs. Ground-Truths On Poisoned Dataset")

    # Convert binary prediction to Series
    df_plot_prediction = pd.Series((i for i in testing_attack_preds))
    plt.fill_between(
        df_plot_prediction.index,
        df_plot_prediction.values,
        where=df_plot_prediction.values <= 1.0,
        interpolate=True,
        color=shade_of_gray,
    )
    plt.plot(testing_attack_preds, color=shade_of_gray, label="Attacks Predictions")
    plt.plot(
        testing_attack_labels,
        color="royalblue",
        alpha=0.85,
        lw=2,
        label="Attacks Labels",
    )
    plt.xlabel("t (h)")
    # plt.ylabel("Binary Classification")
    y_tick = ["UNDER ATTACK" if i == 1.0 else "SAFE" for i in testing_attack_preds]
    plt.yticks(testing_attack_preds, y_tick)
    plt.legend(loc=2, fancybox=True, shadow=True)
    plt.show()


def calculate_rmd_poison():
    """Calculates the Robust Mahalanobis Distance for poisoned dataset"""
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

    # Calculate Minimum Covariance Determinant
    rng = np.random.RandomState(0)

    # Calculate the covariance matrix
    real_cov = np.cov(df_error, rowvar=False)

    # Get multivariate values
    X = rng.multivariate_normal(mean=np.mean(df_error, axis=0), cov=real_cov, size=506)

    # Get MCD values
    cov = MinCovDet(random_state=0).fit(X)
    mcd = cov.covariance_  # robust covariance metrics

    # Calculate the invert covariance matrix
    inv_covmat = sp.linalg.inv(mcd)

    # Calculate the mahalanobis distance
    distances = []
    for i, val in enumerate(df_error):
        p1 = val  # Ozone and Temp of the ith row
        p2 = GLOBAL_MEAN_ERROR
        distance = (p1 - p2).T.dot(inv_covmat).dot(p1 - p2)
        distances.append(distance ** 1.5)
    distances = np.array(distances)

    mean_batch_squared_rmd_arr = []

    outliers = []

    testing_attack_preds = []

    for i in range(L, (len(distances))):
        batch_squared_md = distances[i - L : i]  # take the first L batches
        mean_batch_squared_rmd = np.average(batch_squared_md)
        mean_batch_squared_rmd_arr.append(mean_batch_squared_rmd)
        if mean_batch_squared_rmd >= UPPER_TH:
            outliers.append(UPPER_PLOT)
            testing_attack_preds.append(1.0)
        else:
            outliers.append(LOWER_PLOT)
            testing_attack_preds.append(0.0)

    print(
        f"The Average Robust Mahalanobis Distance: {np.average(mean_batch_squared_rmd_arr)}"
    )

    # Robust MD Plot
    fig1 = plt.figure(figsize=(20, 8))
    plt.title("Robust Mahalanobis Distance Of Every Hour On Testing Dataset")
    df_plot_labels = pd.Series((i for i in convert_th_binary_arr))
    plt.plot(convert_th_binary_arr, label="Attacks Labels")
    plt.fill_between(
        df_plot_labels.index,
        df_plot_labels.values,
        where=df_plot_labels.values <= UPPER_PLOT,
        interpolate=True,
        color=shade_of_blue,
    )
    plt.plot(mean_batch_squared_rmd_arr, color="black", lw=2, label="Robust MD")
    # plt.plot(first_column_dynamic_th, label="dynamic threshold")
    plt.plot(thresholds, color="red", label="Threshold")

    plt.xlabel("t (h)")
    plt.ylabel("Robust Mahalanobis Distance")
    plt.figtext(0.16, 0.24, "L = " + str(L))
    plt.figtext(0.16, 0.22, "TH = " + str(UPPER_TH))
    plt.legend(loc=2, fancybox=True, shadow=True)
    plt.show()

    # Binary Classification Plot
    fig1 = plt.figure(figsize=(20, 8))
    plt.title("Attacks Predictions vs. Ground-Truths On Testing Dataset")

    # Convert binary prediction to Series
    df_plot_prediction = pd.Series((i for i in testing_attack_preds))
    plt.fill_between(
        df_plot_prediction.index,
        df_plot_prediction.values,
        where=df_plot_prediction.values <= 1.0,
        interpolate=True,
        color=shade_of_gray,
    )
    plt.plot(testing_attack_preds, color=shade_of_gray, label="Attacks Predictions")
    plt.plot(
        testing_attack_labels,
        color="royalblue",
        alpha=0.85,
        lw=2,
        label="Attacks Labels",
    )
    plt.xlabel("t (h)")
    # plt.ylabel("Binary Classification")
    y_tick = ["UNDER ATTACK" if i == 1.0 else "SAFE" for i in testing_attack_preds]
    plt.yticks(testing_attack_preds, y_tick)
    plt.legend(loc=2, fancybox=True, shadow=True)
    plt.show()


if __name__ == "__main__":
    calculate_md_poison()
    calculate_rmd_poison()
