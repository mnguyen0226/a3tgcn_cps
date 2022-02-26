import numpy as np
from pandas import cut
from scipy.stats import chi2
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from sklearn.covariance import MinCovDet

EVAL_CLEAN_LABEL_DIR = "out/tgcn/tgcn_scada_wds_lr0.01_batch16_unit64_seq8_pre1_epoch101/eval_clean/eval_clean_labels.csv"
EVAL_CLEAN_PREDS_DIR = "out/tgcn/tgcn_scada_wds_lr0.01_batch16_unit64_seq8_pre1_epoch101/eval_clean/eval_clean_output.csv"
### Poisoned Dataset
# L = 30
# UPPER_TH = 40.5

### Test Dataset
# 17 34.5
# 20 34.9
# 25 34.9

# distances.append(distance**1.25) L = 8, UPPER_TH = 86

L = 12
UPPER_TH = 209  # 42.5  # 34.9

UPPER_PLOT = 7700
LOWER_PLOT = 20
GLOBAL_MEAN_ERROR = np.array(  # Recall calculate everytime retrained model
    [
        0.25263925,
        1.72064805,
        -1.21583359,
        -1.09088844,
        1.8052068,
        -0.47301259,
        -0.19618666,
        0.08631689,
        0.01535588,
        0.18300351,
        -1.77710595,
        0.40690686,
        -1.50650184,
        -0.32522933,
        -0.10789579,
        0.12839262,
        -1.59134306,
        -0.87404767,
        0.21639689,
        -0.58031516,
        0.72516544,
        0.87291746,
        -1.53235269,
        0.21657286,
        -0.72567742,
        -0.53414824,
        0.1540499,
        -0.90060805,
        1.26067521,
        1.34126008,
        0.02611067,
    ]
)


def data_preprocessing(
    num_line=9, label_dataset=EVAL_CLEAN_LABEL_DIR, preds_dataset=EVAL_CLEAN_PREDS_DIR
):
    """Takes in the test_labels.csv, and test_output.csv, converts values from string to float and return the two arrays

    Args:
        num_line: Number of array. Defaults to 9.

    Returns:
        Two arrays
    """
    df_eval_labels = []
    df_eval_preds = []

    # Read in the eval ground truth
    with open(label_dataset) as label_file:

        # Read the csv file
        csv_file = csv.reader(label_file)

        # Print the first 10 lines of the csv file
        for i, line in enumerate(csv_file):
            df_eval_labels.append(line)
            if i >= num_line:
                break
    # Close ground truth file
    label_file.close()

    # Read in the eval prediction
    with open(preds_dataset) as pred_file:

        # Read tge csv file
        csv_file = csv.reader(pred_file)

        # Print the first 10 lines of the csv file
        for i, line in enumerate(csv_file):
            df_eval_preds.append(line)
            if i >= num_line:
                break
    # Close prediction file
    pred_file.close()

    # Converts variable from string to float
    for i in range(len(df_eval_labels)):
        df_eval_labels[i] = [float(x) for x in df_eval_labels[i]]
    for i in range(len(df_eval_preds)):
        df_eval_preds[i] = [float(x) for x in df_eval_preds[i]]

    return df_eval_labels, df_eval_preds


def calculate_md_clean():
    """Calculates the Mahalanobis Distance clean dataset"""
    # Get lists
    df_eval_labels, df_eval_preds = data_preprocessing(num_line=1744)

    # Convert lists to numpy arrays
    df_eval_labels = np.array(df_eval_labels)
    df_eval_preds = np.array(df_eval_preds)

    # Get error array between all labels and all predictions
    df_error = df_eval_labels - df_eval_preds

    # Calculate the covariance matrix
    cov = np.cov(df_error, rowvar=False)

    # Calculate cov^-1
    covariance_pm1 = np.linalg.matrix_power(cov, -1)

    # Calculate the global mean error arrayy
    global_mean_error = np.mean(df_error, axis=0)

    # Save the global mean error
    f = open(
        "out/tgcn/tgcn_scada_wds_lr0.01_batch16_unit64_seq8_pre1_epoch101/eval_clean/global_mean_error_clean.txt",
        "w",
    )
    f.write(np.array2string(global_mean_error, separator=","))
    f.close()

    # Calculate the mahalanobis distance
    distances = []
    for i, val in enumerate(df_error):
        p1 = val
        p2 = global_mean_error
        distance = (
            (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
        )  # squared mahalanobis distance
        distances.append(distance ** 1.5)
    distances = np.array(distances)

    mean_batch_squared_md_arr = []

    for i in range(L, (len(distances))):
        batch_squared_md = distances[i - L : i]  # take the first L batches
        mean_batch_squared_md = np.average(batch_squared_md)
        mean_batch_squared_md_arr.append(mean_batch_squared_md)

    print(f"The Average Mahalanobis Distance {np.average(mean_batch_squared_md_arr)}")

    # MD Plot
    fig1 = plt.figure(figsize=(20, 8))
    threshold_line_clean = [UPPER_TH for x in range(len(mean_batch_squared_md_arr))]

    plt.title("Mahalanobis Distance Every Hour On Clean Dataset")
    plt.plot(mean_batch_squared_md_arr, color="black", lw=2, label="MD")
    plt.plot(threshold_line_clean, color="red", label="Threshold")
    plt.xlabel("t (h)")
    plt.ylabel("Mahalanobis Distance")
    plt.figtext(0.16, 0.24, "L = " + str(L))
    plt.figtext(0.16, 0.22, "TH = " + str(UPPER_TH))
    plt.legend(loc=2, fancybox=True, shadow=True)
    plt.show()


def calculate_rmd_clean():
    """Calculates Robust Mahalanobus Distance clean dataset"""
    df_eval_labels, df_eval_preds = data_preprocessing(num_line=1744)

    # Convert lists to numpy arrays
    df_eval_labels = np.array(df_eval_labels)
    df_eval_preds = np.array(df_eval_preds)

    # Get error array between all labels and all predictions
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

    # Calculate the global mean error
    global_mean_error = cov.location_

    # Save the global mean error
    f = open(
        "out/tgcn/tgcn_scada_wds_lr0.01_batch16_unit64_seq8_pre1_epoch101/eval_clean/global_mean_error_clean.txt",
        "a",
    )
    f.write("\n\n")
    f.write(np.array2string(global_mean_error, separator=","))
    f.close()

    # Calculate the invert covariance matrix
    inv_covmat = sp.linalg.inv(mcd)

    # Calculate the mahalanobis distance
    distances = []
    for i, val in enumerate(df_error):
        p1 = val
        p2 = global_mean_error
        distance = (
            (p1 - p2).T.dot(inv_covmat).dot(p1 - p2)
        )  # squared mahalanobis distance
        distances.append(distance ** 1.5)
    distances = np.array(distances)

    mean_batch_squared_rmd_arr = []

    for i in range(L, (len(distances))):
        batch_squared_md = distances[i - L : i]  # take the first L batches
        mean_batch_squared_md = np.average(batch_squared_md)
        mean_batch_squared_rmd_arr.append(mean_batch_squared_md)

    print(
        f"The Average Robust Mahalanobis Distance: {np.average(mean_batch_squared_rmd_arr)}"
    )
    # Robust MD Plot
    fig1 = plt.figure(figsize=(20, 8))
    threshold_line_clean = [UPPER_TH for x in range(len(mean_batch_squared_rmd_arr))]

    plt.title("Robust Mahalanobis Distance Every Hour On Clean Dataset")
    plt.plot(mean_batch_squared_rmd_arr, color="black", lw=2, label="MD")
    plt.plot(threshold_line_clean, color="red", label="Threshold")
    plt.xlabel("t (h)")
    plt.ylabel("Robust Mahalanobis Distance")
    plt.figtext(0.16, 0.24, "L = " + str(L))
    plt.figtext(0.16, 0.22, "TH = " + str(UPPER_TH))
    plt.legend(loc=2, fancybox=True, shadow=True)
    plt.show()


if __name__ == "__main__":
    calculate_md_clean()
    calculate_rmd_clean()
