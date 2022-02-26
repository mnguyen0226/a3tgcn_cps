import numpy as np
from scipy.stats import chi2
import csv
from TGCN.tensorflow_model.utils.md_clean_calculation import data_preprocessing

STACKED_LABEL_DIR = "out/tgcn/tgcn_scada_wds_lr0.01_batch16_unit64_seq8_pre1_epoch101/stacked_eval/stacked_eval_labels.csv"
STACKED_PREDS_DIR = "out/tgcn/tgcn_scada_wds_lr0.01_batch16_unit64_seq8_pre1_epoch101/stacked_eval/stacked_eval_output.csv"


##########
def calculate_md_stacked():
    """Calculates the Mahalanobis Distance for clean stacked poison dataset"""
    # Get lists
    df_eval_labels, df_eval_preds = data_preprocessing(
        num_line=2227, label_dataset=STACKED_LABEL_DIR, preds_dataset=STACKED_PREDS_DIR
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
    mean_error = np.mean(df_error, axis=0)

    # 4. Calculate the mahalanobis distance
    distances = []
    for i, val in enumerate(df_error):
        p1 = val  # Ozone and Temp of the ith row
        p2 = mean_error
        distance = (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
        distances.append(distance)
        print(f"Distance: {distance}")
    distances = np.array(distances)

    # Check if there is any negative number in the mahalanobis distance
    nega = [distances[i] for i in range(len(distances)) if distances[i] <= 0.0]
    print(f"\nList of negative Mahalanobis Distance: {nega}")

    # Calculate the average Mahalanobis Distance (not useful)
    avg_md = np.average(distances)

    print(f"\nThe average of the Mahalanobis Distance: {avg_md}")

    # 5. Find the cut-off Chi-Square values. The points outside of 0.95 will be considered as outliers
    # Note, we also set the degree of freedom values for Chi-Square. This number is equal to the number of variables in our dataset, 31
    cutoff = chi2.ppf(0.85, df_error.shape[1])  # THRESHOLD = 0.99999999999999999

    # Index of outliers
    outlier_index = np.where(distances > cutoff)

    print("\nTIME SERIES INDEX OF OUTLIERS:")
    print(outlier_index)

    print("\nTIME SERIES INDEX OF OUTLIERS LENGTH:")
    print(len(outlier_index[0]))

    poisoned_dataset_detected = [i for i in outlier_index[0] if i > 1744]
    print("\nTIME SERIES INDEX OF OUTLIERS POISONED LENGTH:")
    print(len(poisoned_dataset_detected))

    # print("OUTLIERS DETAILS\n")
    # print(df_error[ distances > cutoff , :])


if __name__ == "__main__":
    calculate_md_stacked()
