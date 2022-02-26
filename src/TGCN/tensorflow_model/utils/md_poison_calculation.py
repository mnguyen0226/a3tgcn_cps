import numpy as np
from scipy.stats import chi2
import csv
from TGCN.tensorflow_model.utils.md_clean_calculation import data_preprocessing
import matplotlib.pyplot as plt
from TGCN.tensorflow_model.utils.md_clean_calculation import GLOBAL_ME
import pandas as pd

# Before any attacks there will be a 17 hour time stamps
EVAL_POISON_LABEL_DIR = "out/tgcn/tgcn_scada_wds_lr0.01_batch16_unit64_seq8_pre1_epoch101/eval_poisoned/eval_poisoned_labels.csv"
EVAL_POISON_PREDS_DIR = "out/tgcn/tgcn_scada_wds_lr0.01_batch16_unit64_seq8_pre1_epoch101/eval_poisoned/eval_poisoned_output.csv"
L = 25
TH = 0
EVAL_POISON_LINE_NUM = 728

dataset04 = pd.read_csv(r"data/processed/dataset04_origin_no_binary_30_split.csv")

binary_arr = dataset04["ATT_FLAG"].to_list()
binary_arr = binary_arr[(L + 8) : -1]  # use 8 for prediction + L for first window size
print((binary_arr))

##########
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

    # 1. Calculate the covariance matrix
    cov = np.cov(df_error, rowvar=False)

    # 2. Calculate cov^-1
    covariance_pm1 = np.linalg.matrix_power(cov, -1)

    # 3. Calculate the mean error arrayy
    global_mean_error = GLOBAL_ME

    # 4. Calculate the mahalanobis distance
    distances = []
    for i, val in enumerate(df_error):
        p1 = val  # Ozone and Temp of the ith row
        p2 = global_mean_error
        distance = (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
        distances.append(distance)
        # print(f"Distance: {distance}")
    distances = np.array(distances)

    cutoff_arr = []

    for i in range(L, (len(distances))):
        batch_squared_md = distances[i - L : i]  # take the first L batches
        mean_batch_squared_md = np.average(batch_squared_md)
        # batch_cutoff = chi2.pff(0.95, 31)
        cutoff_arr.append(mean_batch_squared_md)
    print(len(cutoff_arr))

    plt.plot(cutoff_arr)
    plt.title(
        "Mean Squared Mahalanobis Distance Every L Hours TimeStamp - Poisoned Dataset"
    )
    plt.xlabel("Every L hours")
    plt.ylabel("Mean Squared Mahalanobis Distance - To Calibrate Max Threshold")
    print(f"The Average Mean Squared Mahalanobis Distance {np.average(cutoff_arr)}")
    plt.show()

    # # Check if there is any negative number in the mahalanobis distance
    # nega = [distances[i] for i in range(len(distances)) if distances[i] <= 0.0]
    # print(f"\nList of negative Mahalanobis Distance: {nega}")

    # # Calculate the average Mahalanobis Distance (not useful)
    # avg_md = np.average(distances)

    # print(f"\nThe average of the Mahalanobis Distance: {avg_md}")

    # # 5. Find the cut-off Chi-Square values. The points outside of 0.95 will be considered as outliers
    # # Note, we also set the degree of freedom values for Chi-Square. This number is equal to the number of variables in our dataset, 31
    # cutoff = chi2.ppf(
    #     0.99999999999999999, df_error.shape[1]
    # )  # THRESHOLD = 0.99999999999999999

    # # Index of outliers
    # outlier_index = np.where(distances > cutoff)

    # print("\nTIME SERIES INDEX OF OUTLIERS:")
    # print(outlier_index)

    # # print("OUTLIERS DETAILS\n")
    # # print(df_error[ distances > cutoff , :])


if __name__ == "__main__":
    calculate_md_poison()
