import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.detection_clean_calculation import L


EVAL_CLEAN_LABEL_DIR = "out/tgcn/tgcn_scada_wds_lr0.005_batch128_unit64_seq8_pre1_epoch101/eval_clean/eval_clean_labels.csv"
EVAL_CLEAN_PREDS_DIR = "out/tgcn/tgcn_scada_wds_lr0.005_batch128_unit64_seq8_pre1_epoch101/eval_clean/eval_clean_output.csv"
EVAL_TEST_LABEL_DIR = "out/tgcn/tgcn_scada_wds_lr0.005_batch128_unit64_seq8_pre1_epoch101/eval_test/eval_test_labels.csv"
EVAL_TEST_PREDS_DIR = "out/tgcn/tgcn_scada_wds_lr0.005_batch128_unit64_seq8_pre1_epoch101/eval_test/eval_test_output.csv"
LINE_COLOR = "lightgray"

attack_detection_file = pd.read_csv(
    r"out/tgcn/tgcn_scada_wds_lr0.005_batch128_unit64_seq8_pre1_epoch101/eval_test/detection_results.csv",
    header=None,
)  # change for each different poisoned dataset.csv

testing_attack_labels = attack_detection_file[0].to_list()

# dataset04 = pd.read_csv(
#     r"data/processed/test_scada_dataset.csv"
# )  # change for each different poisoned dataset.csv

# binary_arr = dataset04["ATT_FLAG"].to_list()
# testing_attack_labels = binary_arr[
#     8:-1
# ]  # use 8 for prediction + L for first window size


def localization():
    # Return index of the attacks detection => Return to which one has higher than threshold plot

    clean_eval_labels = pd.read_csv(
        EVAL_CLEAN_LABEL_DIR,
        usecols=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
        ],
        names=[
            "L_T1",
            "L_T2",
            "L_T3",
            "L_T4",
            "L_T5",
            "L_T6",
            "L_T7",
            "F_PU1",
            "F_PU2",
            "F_PU3",
            "F_PU4",
            "F_PU5",
            "F_PU6",
            "F_PU7",
            "F_PU8",
            "F_PU9",
            "F_PU10",
            "F_PU11",
            "F_V2",
            "P_J280",
            "P_J269",
            "P_J300",
            "P_J256",
            "P_J289",
            "P_J415",
            "P_J302",
            "P_J306",
            "P_J307",
            "P_J317",
            "P_J14",
            "P_J422",
        ],
        header=None,
    )
    clean_eval_labels = clean_eval_labels[(L):-1]
    clean_eval_preds = pd.read_csv(
        EVAL_CLEAN_PREDS_DIR,
        usecols=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
        ],
        names=[
            "L_T1",
            "L_T2",
            "L_T3",
            "L_T4",
            "L_T5",
            "L_T6",
            "L_T7",
            "F_PU1",
            "F_PU2",
            "F_PU3",
            "F_PU4",
            "F_PU5",
            "F_PU6",
            "F_PU7",
            "F_PU8",
            "F_PU9",
            "F_PU10",
            "F_PU11",
            "F_V2",
            "P_J280",
            "P_J269",
            "P_J300",
            "P_J256",
            "P_J289",
            "P_J415",
            "P_J302",
            "P_J306",
            "P_J307",
            "P_J317",
            "P_J14",
            "P_J422",
        ],
        header=None,
    )
    clean_eval_preds = clean_eval_preds[(L):-1]
    test_eval_labels = pd.read_csv(
        EVAL_TEST_LABEL_DIR,
        usecols=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
        ],
        names=[
            "L_T1",
            "L_T2",
            "L_T3",
            "L_T4",
            "L_T5",
            "L_T6",
            "L_T7",
            "F_PU1",
            "F_PU2",
            "F_PU3",
            "F_PU4",
            "F_PU5",
            "F_PU6",
            "F_PU7",
            "F_PU8",
            "F_PU9",
            "F_PU10",
            "F_PU11",
            "F_V2",
            "P_J280",
            "P_J269",
            "P_J300",
            "P_J256",
            "P_J289",
            "P_J415",
            "P_J302",
            "P_J306",
            "P_J307",
            "P_J317",
            "P_J14",
            "P_J422",
        ],
        header=None,
    )
    test_eval_labels = test_eval_labels[(L):-1]
    test_eval_preds = pd.read_csv(
        EVAL_TEST_PREDS_DIR,
        usecols=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
        ],
        names=[
            "L_T1",
            "L_T2",
            "L_T3",
            "L_T4",
            "L_T5",
            "L_T6",
            "L_T7",
            "F_PU1",
            "F_PU2",
            "F_PU3",
            "F_PU4",
            "F_PU5",
            "F_PU6",
            "F_PU7",
            "F_PU8",
            "F_PU9",
            "F_PU10",
            "F_PU11",
            "F_V2",
            "P_J280",
            "P_J269",
            "P_J300",
            "P_J256",
            "P_J289",
            "P_J415",
            "P_J302",
            "P_J306",
            "P_J307",
            "P_J317",
            "P_J14",
            "P_J422",
        ],
        header=None,
    )
    test_eval_preds = test_eval_preds[(L):-1]
    # Extract data from clean labels and predictions
    clean_L_T1_label = (clean_eval_labels["L_T1"]).values
    clean_L_T2_label = (clean_eval_labels["L_T2"]).values
    clean_L_T3_label = (clean_eval_labels["L_T3"]).values
    clean_L_T4_label = (clean_eval_labels["L_T4"]).values
    clean_L_T5_label = (clean_eval_labels["L_T5"]).values
    clean_L_T6_label = (clean_eval_labels["L_T6"]).values
    clean_L_T7_label = (clean_eval_labels["L_T7"]).values
    clean_F_PU1_label = (clean_eval_labels["F_PU1"]).values
    clean_F_PU2_label = (clean_eval_labels["F_PU2"]).values
    clean_F_PU3_label = (clean_eval_labels["F_PU3"]).values
    clean_F_PU4_label = (clean_eval_labels["F_PU4"]).values
    clean_F_PU5_label = (clean_eval_labels["F_PU5"]).values
    clean_F_PU6_label = (clean_eval_labels["F_PU6"]).values
    clean_F_PU7_label = (clean_eval_labels["F_PU7"]).values
    clean_F_PU8_label = (clean_eval_labels["F_PU8"]).values
    clean_F_PU9_label = (clean_eval_labels["F_PU9"]).values
    clean_F_PU10_label = (clean_eval_labels["F_PU10"]).values
    clean_F_PU11_label = (clean_eval_labels["F_PU11"]).values
    clean_F_V2_label = (clean_eval_labels["F_V2"]).values
    clean_P_J280_label = (clean_eval_labels["P_J280"]).values
    clean_P_J269_label = (clean_eval_labels["P_J269"]).values
    clean_P_J300_label = (clean_eval_labels["P_J300"]).values
    clean_P_J256_label = (clean_eval_labels["P_J256"]).values
    clean_P_J289_label = (clean_eval_labels["P_J289"]).values
    clean_P_J415_label = (clean_eval_labels["P_J415"]).values
    clean_P_J302_label = (clean_eval_labels["P_J302"]).values
    clean_P_J306_label = (clean_eval_labels["P_J306"]).values
    clean_P_J307_label = (clean_eval_labels["P_J307"]).values
    clean_P_J317_label = (clean_eval_labels["P_J317"]).values
    clean_P_J14_label = (clean_eval_labels["P_J14"]).values
    clean_P_J422_label = (clean_eval_labels["P_J422"]).values

    clean_L_T1_pred = (clean_eval_preds["L_T1"]).values
    clean_L_T2_pred = (clean_eval_preds["L_T2"]).values
    clean_L_T3_pred = (clean_eval_preds["L_T3"]).values
    clean_L_T4_pred = (clean_eval_preds["L_T4"]).values
    clean_L_T5_pred = (clean_eval_preds["L_T5"]).values
    clean_L_T6_pred = (clean_eval_preds["L_T6"]).values
    clean_L_T7_pred = (clean_eval_preds["L_T7"]).values
    clean_F_PU1_pred = (clean_eval_preds["F_PU1"]).values
    clean_F_PU2_pred = (clean_eval_preds["F_PU2"]).values
    clean_F_PU3_pred = (clean_eval_preds["F_PU3"]).values
    clean_F_PU4_pred = (clean_eval_preds["F_PU4"]).values
    clean_F_PU5_pred = (clean_eval_preds["F_PU5"]).values
    clean_F_PU6_pred = (clean_eval_preds["F_PU6"]).values
    clean_F_PU7_pred = (clean_eval_preds["F_PU7"]).values
    clean_F_PU8_pred = (clean_eval_preds["F_PU8"]).values
    clean_F_PU9_pred = (clean_eval_preds["F_PU9"]).values
    clean_F_PU10_pred = (clean_eval_preds["F_PU10"]).values
    clean_F_PU11_pred = (clean_eval_preds["F_PU11"]).values
    clean_F_V2_pred = (clean_eval_preds["F_V2"]).values
    clean_P_J280_pred = (clean_eval_preds["P_J280"]).values
    clean_P_J269_pred = (clean_eval_preds["P_J269"]).values
    clean_P_J300_pred = (clean_eval_preds["P_J300"]).values
    clean_P_J256_pred = (clean_eval_preds["P_J256"]).values
    clean_P_J289_pred = (clean_eval_preds["P_J289"]).values
    clean_P_J415_pred = (clean_eval_preds["P_J415"]).values
    clean_P_J302_pred = (clean_eval_preds["P_J302"]).values
    clean_P_J306_pred = (clean_eval_preds["P_J306"]).values
    clean_P_J307_pred = (clean_eval_preds["P_J307"]).values
    clean_P_J317_pred = (clean_eval_preds["P_J317"]).values
    clean_P_J14_pred = (clean_eval_preds["P_J14"]).values
    clean_P_J422_pred = (clean_eval_preds["P_J422"]).values

    # Return the max absolute error between the prediction and labels at each node in normal condition
    max_error_L_T1 = np.amax(np.abs(clean_L_T1_label - clean_L_T1_pred))
    max_error_L_T2 = np.amax(np.abs(clean_L_T2_label - clean_L_T2_pred))
    max_error_L_T3 = np.amax(np.abs(clean_L_T3_label - clean_L_T3_pred))
    max_error_L_T4 = np.amax(np.abs(clean_L_T4_label - clean_L_T4_pred))
    max_error_L_T5 = np.amax(np.abs(clean_L_T5_label - clean_L_T5_pred))
    max_error_L_T6 = np.amax(np.abs(clean_L_T6_label - clean_L_T6_pred))
    max_error_L_T7 = np.amax(np.abs(clean_L_T7_label - clean_L_T7_pred))
    max_error_F_PU1 = np.amax(np.abs(clean_F_PU1_label - clean_F_PU1_pred))
    max_error_F_PU2 = np.amax(np.abs(clean_F_PU2_label - clean_F_PU2_pred))
    max_error_F_PU3 = np.amax(np.abs(clean_F_PU3_label - clean_F_PU3_pred))
    max_error_F_PU4 = np.amax(np.abs(clean_F_PU4_label - clean_F_PU4_pred))
    max_error_F_PU5 = np.amax(np.abs(clean_F_PU5_label - clean_F_PU5_pred))
    max_error_F_PU6 = np.amax(np.abs(clean_F_PU6_label - clean_F_PU6_pred))
    max_error_F_PU7 = np.amax(np.abs(clean_F_PU7_label - clean_F_PU7_pred))
    max_error_F_PU8 = np.amax(np.abs(clean_F_PU8_label - clean_F_PU8_pred))
    max_error_F_PU9 = np.amax(np.abs(clean_F_PU9_label - clean_F_PU9_pred))
    max_error_F_PU10 = np.amax(np.abs(clean_F_PU10_label - clean_F_PU10_pred))
    max_error_F_PU11 = np.amax(np.abs(clean_F_PU11_label - clean_F_PU11_pred))
    max_error_F_V2 = np.amax(np.abs(clean_F_V2_label - clean_F_V2_pred))
    max_error_P_J280 = np.amax(np.abs(clean_P_J280_label - clean_P_J280_pred))
    max_error_P_J269 = np.amax(np.abs(clean_P_J269_label - clean_P_J269_pred))
    max_error_P_J300 = np.amax(np.abs(clean_P_J300_label - clean_P_J300_pred))
    max_error_P_J256 = np.amax(np.abs(clean_P_J256_label - clean_P_J256_pred))
    max_error_P_J289 = np.amax(np.abs(clean_P_J289_label - clean_P_J289_pred))
    max_error_P_J415 = np.amax(np.abs(clean_P_J415_label - clean_P_J415_pred))
    max_error_P_J302 = np.amax(np.abs(clean_P_J302_label - clean_P_J302_pred))
    max_error_P_J306 = np.amax(np.abs(clean_P_J306_label - clean_P_J306_pred))
    max_error_P_J307 = np.amax(np.abs(clean_P_J307_label - clean_P_J307_pred))
    max_error_P_J317 = np.amax(np.abs(clean_P_J317_label - clean_P_J317_pred))
    max_error_P_J14 = np.amax(np.abs(clean_P_J14_label - clean_P_J14_pred))
    max_error_P_J422 = np.amax(np.abs(clean_P_J422_label - clean_P_J422_pred))

    # Extract data from testing labels and predictions
    test_L_T1_label = (test_eval_labels["L_T1"]).values
    test_L_T2_label = (test_eval_labels["L_T2"]).values
    test_L_T3_label = (test_eval_labels["L_T3"]).values
    test_L_T4_label = (test_eval_labels["L_T4"]).values
    test_L_T5_label = (test_eval_labels["L_T5"]).values
    test_L_T6_label = (test_eval_labels["L_T6"]).values
    test_L_T7_label = (test_eval_labels["L_T7"]).values
    test_F_PU1_label = (test_eval_labels["F_PU1"]).values
    test_F_PU2_label = (test_eval_labels["F_PU2"]).values
    test_F_PU3_label = (test_eval_labels["F_PU3"]).values
    test_F_PU4_label = (test_eval_labels["F_PU4"]).values
    test_F_PU5_label = (test_eval_labels["F_PU5"]).values
    test_F_PU6_label = (test_eval_labels["F_PU6"]).values
    test_F_PU7_label = (test_eval_labels["F_PU7"]).values
    test_F_PU8_label = (test_eval_labels["F_PU8"]).values
    test_F_PU9_label = (test_eval_labels["F_PU9"]).values
    test_F_PU10_label = (test_eval_labels["F_PU10"]).values
    test_F_PU11_label = (test_eval_labels["F_PU11"]).values
    test_F_V2_label = (test_eval_labels["F_V2"]).values
    test_P_J280_label = (test_eval_labels["P_J280"]).values
    test_P_J269_label = (test_eval_labels["P_J269"]).values
    test_P_J300_label = (test_eval_labels["P_J300"]).values
    test_P_J256_label = (test_eval_labels["P_J256"]).values
    test_P_J289_label = (test_eval_labels["P_J289"]).values
    test_P_J415_label = (test_eval_labels["P_J415"]).values
    test_P_J302_label = (test_eval_labels["P_J302"]).values
    test_P_J306_label = (test_eval_labels["P_J306"]).values
    test_P_J307_label = (test_eval_labels["P_J307"]).values
    test_P_J317_label = (test_eval_labels["P_J317"]).values
    test_P_J14_label = (test_eval_labels["P_J14"]).values
    test_P_J422_label = (test_eval_labels["P_J422"]).values

    test_L_T1_pred = (test_eval_preds["L_T1"]).values
    test_L_T2_pred = (test_eval_preds["L_T2"]).values
    test_L_T3_pred = (test_eval_preds["L_T3"]).values
    test_L_T4_pred = (test_eval_preds["L_T4"]).values
    test_L_T5_pred = (test_eval_preds["L_T5"]).values
    test_L_T6_pred = (test_eval_preds["L_T6"]).values
    test_L_T7_pred = (test_eval_preds["L_T7"]).values
    test_F_PU1_pred = (test_eval_preds["F_PU1"]).values
    test_F_PU2_pred = (test_eval_preds["F_PU2"]).values
    test_F_PU3_pred = (test_eval_preds["F_PU3"]).values
    test_F_PU4_pred = (test_eval_preds["F_PU4"]).values
    test_F_PU5_pred = (test_eval_preds["F_PU5"]).values
    test_F_PU6_pred = (test_eval_preds["F_PU6"]).values
    test_F_PU7_pred = (test_eval_preds["F_PU7"]).values
    test_F_PU8_pred = (test_eval_preds["F_PU8"]).values
    test_F_PU9_pred = (test_eval_preds["F_PU9"]).values
    test_F_PU10_pred = (test_eval_preds["F_PU10"]).values
    test_F_PU11_pred = (test_eval_preds["F_PU11"]).values
    test_F_V2_pred = (test_eval_preds["F_V2"]).values
    test_P_J280_pred = (test_eval_preds["P_J280"]).values
    test_P_J269_pred = (test_eval_preds["P_J269"]).values
    test_P_J300_pred = (test_eval_preds["P_J300"]).values
    test_P_J256_pred = (test_eval_preds["P_J256"]).values
    test_P_J289_pred = (test_eval_preds["P_J289"]).values
    test_P_J415_pred = (test_eval_preds["P_J415"]).values
    test_P_J302_pred = (test_eval_preds["P_J302"]).values
    test_P_J306_pred = (test_eval_preds["P_J306"]).values
    test_P_J307_pred = (test_eval_preds["P_J307"]).values
    test_P_J317_pred = (test_eval_preds["P_J317"]).values
    test_P_J14_pred = (test_eval_preds["P_J14"]).values
    test_P_J422_pred = (test_eval_preds["P_J422"]).values

    # Return the max absolute error between the prediction and labels at each node in testing condition
    test_error_L_T1 = np.abs(test_L_T1_label - test_L_T1_pred)
    test_error_L_T2 = np.abs(test_L_T2_label - test_L_T2_pred)
    test_error_L_T3 = np.abs(test_L_T3_label - test_L_T3_pred)
    test_error_L_T4 = np.abs(test_L_T4_label - test_L_T4_pred)
    test_error_L_T5 = np.abs(test_L_T5_label - test_L_T5_pred)
    test_error_L_T6 = np.abs(test_L_T6_label - test_L_T6_pred)
    test_error_L_T7 = np.abs(test_L_T7_label - test_L_T7_pred)
    test_error_F_PU1 = np.abs(test_F_PU1_label - test_F_PU1_pred)
    test_error_F_PU2 = np.abs(test_F_PU2_label - test_F_PU2_pred)
    test_error_F_PU3 = np.abs(test_F_PU3_label - test_F_PU3_pred)
    test_error_F_PU4 = np.abs(test_F_PU4_label - test_F_PU4_pred)
    test_error_F_PU5 = np.abs(test_F_PU5_label - test_F_PU5_pred)
    test_error_F_PU6 = np.abs(test_F_PU6_label - test_F_PU6_pred)
    test_error_F_PU7 = np.abs(test_F_PU7_label - test_F_PU7_pred)
    test_error_F_PU8 = np.abs(test_F_PU8_label - test_F_PU8_pred)
    test_error_F_PU9 = np.abs(test_F_PU9_label - test_F_PU9_pred)
    test_error_F_PU10 = np.abs(test_F_PU10_label - test_F_PU10_pred)
    test_error_F_PU11 = np.abs(test_F_PU11_label - test_F_PU11_pred)
    test_error_F_V2 = np.abs(test_F_V2_label - test_F_V2_pred)
    test_error_P_J280 = np.abs(test_P_J280_label - test_P_J280_pred)
    test_error_P_J269 = np.abs(test_P_J269_label - test_P_J269_pred)
    test_error_P_J300 = np.abs(test_P_J300_label - test_P_J300_pred)
    test_error_P_J256 = np.abs(test_P_J256_label - test_P_J256_pred)
    test_error_P_J289 = np.abs(test_P_J289_label - test_P_J289_pred)
    test_error_P_J415 = np.abs(test_P_J415_label - test_P_J415_pred)
    test_error_P_J302 = np.abs(test_P_J302_label - test_P_J302_pred)
    test_error_P_J306 = np.abs(test_P_J306_label - test_P_J306_pred)
    test_error_P_J307 = np.abs(test_P_J307_label - test_P_J307_pred)
    test_error_P_J317 = np.abs(test_P_J317_label - test_P_J317_pred)
    test_error_P_J14 = np.abs(test_P_J14_label - test_P_J14_pred)
    test_error_P_J422 = np.abs(test_P_J422_label - test_P_J422_pred)

    # Detection for all note
    detection_L_T1 = test_error_L_T1 > max_error_L_T1
    detection_L_T1 = np.array(list(map(int, detection_L_T1)))
    detection_L_T1 = np.array([float("nan") if x == 0 else x for x in detection_L_T1])

    detection_L_T2 = test_error_L_T2 > max_error_L_T2
    detection_L_T2 = np.array(list(map(int, detection_L_T2)))
    detection_L_T2 = np.array([float("nan") if x == 0 else x for x in detection_L_T2])

    detection_L_T3 = test_error_L_T3 > max_error_L_T3
    detection_L_T3 = np.array(list(map(int, detection_L_T3)))
    detection_L_T3 = np.array([float("nan") if x == 0 else x for x in detection_L_T3])

    detection_L_T4 = test_error_L_T4 > max_error_L_T4
    detection_L_T4 = np.array(list(map(int, detection_L_T4)))
    detection_L_T4 = np.array([float("nan") if x == 0 else x for x in detection_L_T4])

    detection_L_T5 = test_error_L_T5 > max_error_L_T5
    detection_L_T5 = np.array(list(map(int, detection_L_T5)))
    detection_L_T5 = np.array([float("nan") if x == 0 else x for x in detection_L_T5])

    detection_L_T6 = test_error_L_T6 > max_error_L_T6
    detection_L_T6 = np.array(list(map(int, detection_L_T6)))
    detection_L_T6 = np.array([float("nan") if x == 0 else x for x in detection_L_T6])

    detection_L_T7 = test_error_L_T7 > max_error_L_T7
    detection_L_T7 = np.array(list(map(int, detection_L_T7)))
    detection_L_T7 = np.array([float("nan") if x == 0 else x for x in detection_L_T7])

    detection_F_PU1 = test_error_F_PU1 > max_error_F_PU1
    detection_F_PU1 = np.array(list(map(int, detection_F_PU1)))
    detection_F_PU1 = np.array([float("nan") if x == 0 else x for x in detection_F_PU1])

    detection_F_PU2 = test_error_F_PU2 > max_error_F_PU2
    detection_F_PU2 = np.array(list(map(int, detection_F_PU2)))
    detection_F_PU2 = np.array([float("nan") if x == 0 else x for x in detection_F_PU2])

    detection_F_PU3 = test_error_F_PU3 > max_error_F_PU3
    detection_F_PU3 = np.array(list(map(int, detection_F_PU3)))
    detection_F_PU3 = np.array([float("nan") if x == 0 else x for x in detection_F_PU3])

    detection_F_PU4 = test_error_F_PU4 > max_error_F_PU4
    detection_F_PU4 = np.array(list(map(int, detection_F_PU4)))
    detection_F_PU4 = np.array([float("nan") if x == 0 else x for x in detection_F_PU4])

    detection_F_PU5 = test_error_F_PU5 > max_error_F_PU5
    detection_F_PU5 = np.array(list(map(int, detection_F_PU5)))
    detection_F_PU5 = np.array([float("nan") if x == 0 else x for x in detection_F_PU5])

    detection_F_PU6 = test_error_F_PU6 > max_error_F_PU6
    detection_F_PU6 = np.array(list(map(int, detection_F_PU6)))
    detection_F_PU6 = np.array([float("nan") if x == 0 else x for x in detection_F_PU6])

    detection_F_PU7 = test_error_F_PU7 > max_error_F_PU7
    detection_F_PU7 = np.array(list(map(int, detection_F_PU7)))
    detection_F_PU7 = np.array([float("nan") if x == 0 else x for x in detection_F_PU7])

    detection_F_PU8 = test_error_F_PU8 > max_error_F_PU8
    detection_F_PU8 = np.array(list(map(int, detection_F_PU8)))
    detection_F_PU8 = np.array([float("nan") if x == 0 else x for x in detection_F_PU8])

    detection_F_PU9 = test_error_F_PU9 > max_error_F_PU9
    detection_F_PU9 = np.array(list(map(int, detection_F_PU9)))
    detection_F_PU9 = np.array([float("nan") if x == 0 else x for x in detection_F_PU9])

    detection_F_PU10 = test_error_F_PU10 > max_error_F_PU10
    detection_F_PU10 = np.array(list(map(int, detection_F_PU10)))
    detection_F_PU10 = np.array(
        [float("nan") if x == 0 else x for x in detection_F_PU10]
    )

    detection_F_PU11 = test_error_F_PU11 > max_error_F_PU11
    detection_F_PU11 = np.array(list(map(int, detection_F_PU11)))
    detection_F_PU11 = np.array(
        [float("nan") if x == 0 else x for x in detection_F_PU11]
    )

    detection_F_V2 = test_error_F_V2 > max_error_F_V2
    detection_F_V2 = np.array(list(map(int, detection_F_V2)))
    detection_F_V2 = np.array([float("nan") if x == 0 else x for x in detection_F_V2])

    detection_P_J280 = test_error_P_J280 > max_error_P_J280
    detection_P_J280 = np.array(list(map(int, detection_P_J280)))
    detection_P_J280 = np.array(
        [float("nan") if x == 0 else x for x in detection_P_J280]
    )

    detection_P_J269 = test_error_P_J269 > max_error_P_J269
    detection_P_J269 = np.array(list(map(int, detection_P_J269)))
    detection_P_J269 = np.array(
        [float("nan") if x == 0 else x for x in detection_P_J269]
    )

    detection_P_J300 = test_error_P_J300 > max_error_P_J300
    detection_P_J300 = np.array(list(map(int, detection_P_J300)))
    detection_P_J300 = np.array(
        [float("nan") if x == 0 else x for x in detection_P_J300]
    )

    detection_P_J256 = test_error_P_J256 > max_error_P_J256
    detection_P_J256 = np.array(list(map(int, detection_P_J256)))
    detection_P_J256 = np.array(
        [float("nan") if x == 0 else x for x in detection_P_J256]
    )

    detection_P_J289 = test_error_P_J289 > max_error_P_J289
    detection_P_J289 = np.array(list(map(int, detection_P_J289)))
    detection_P_J289 = np.array(
        [float("nan") if x == 0 else x for x in detection_P_J289]
    )

    detection_P_J415 = test_error_P_J415 > max_error_P_J415
    detection_P_J415 = np.array(list(map(int, detection_P_J415)))
    detection_P_J415 = np.array(
        [float("nan") if x == 0 else x for x in detection_P_J415]
    )

    detection_P_J302 = test_error_P_J302 > max_error_P_J302
    detection_P_J302 = np.array(list(map(int, detection_P_J302)))
    detection_P_J302 = np.array(
        [float("nan") if x == 0 else x for x in detection_P_J302]
    )

    detection_P_J306 = test_error_P_J306 > max_error_P_J306
    detection_P_J306 = np.array(list(map(int, detection_P_J306)))
    detection_P_J306 = np.array(
        [float("nan") if x == 0 else x for x in detection_P_J306]
    )

    detection_P_J307 = test_error_P_J307 > max_error_P_J307
    detection_P_J307 = np.array(list(map(int, detection_P_J307)))
    detection_P_J307 = np.array(
        [float("nan") if x == 0 else x for x in detection_P_J307]
    )

    detection_P_J317 = test_error_P_J317 > max_error_P_J317
    detection_P_J317 = np.array(list(map(int, detection_P_J317)))
    detection_P_J317 = np.array(
        [float("nan") if x == 0 else x for x in detection_P_J317]
    )

    detection_P_J14 = test_error_P_J14 > max_error_P_J14
    detection_P_J14 = np.array(list(map(int, detection_P_J14)))
    detection_P_J14 = np.array([float("nan") if x == 0 else x for x in detection_P_J14])

    detection_P_J422 = test_error_P_J422 > max_error_P_J422
    detection_P_J422 = np.array(list(map(int, detection_P_J422)))
    detection_P_J422 = np.array(
        [float("nan") if x == 0 else x for x in detection_P_J422]
    )

    straight_line = np.array([1 for _ in range(len(detection_L_T2))])
    fig1 = plt.figure(figsize=(28, 8))
    plt.title("Cyber-Physical Attacks Localization For Water Distribution Systems")

    plt.plot(2 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(4 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(6 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(8 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(10 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(12 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(14 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(16 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(18 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(20 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(22 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(24 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(26 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(28 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(30 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(32 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(34 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(36 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(38 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(40 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(42 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(44 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(46 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(48 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(50 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(52 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(54 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(56 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(58 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(60 / 65 * straight_line, color=LINE_COLOR)
    plt.plot(62 / 65 * straight_line, color=LINE_COLOR)

    plt.plot(detection_L_T1 * 2 / 65, "x", label="L_T1")
    plt.plot(detection_L_T2 * 4 / 65, "x", label="L_T2")
    plt.plot(detection_L_T3 * 6 / 65, "x", label="L_T3")
    plt.plot(detection_L_T4 * 8 / 65, "x", label="L_T4")
    plt.plot(detection_L_T5 * 10 / 65, "x", label="L_T5")
    plt.plot(detection_L_T6 * 12 / 65, "x", label="L_T6")
    plt.plot(detection_L_T7 * 14 / 65, "x", label="L_T7")
    plt.plot(detection_F_PU1 * 16 / 65, "x", label="F_PU1")
    plt.plot(detection_F_PU2 * 18 / 65, "x", label="F_PU2")
    plt.plot(detection_F_PU3 * 20 / 65, "x", label="F_PU3")
    plt.plot(detection_F_PU4 * 22 / 65, "x", label="F_PU4")
    plt.plot(detection_F_PU5 * 24 / 65, "x", label="F_PU5")
    plt.plot(detection_F_PU6 * 26 / 65, "x", label="F_PU6")
    plt.plot(detection_F_PU7 * 28 / 65, "x", label="F_PU7")
    plt.plot(detection_F_PU8 * 30 / 65, "x", label="F_PU8")
    plt.plot(detection_F_PU9 * 32 / 65, "x", label="F_PU9")
    plt.plot(detection_F_PU10 * 34 / 65, "x", label="F_PU10")
    plt.plot(detection_F_PU11 * 36 / 65, "x", label="F_PU11")
    plt.plot(detection_F_V2 * 38 / 65, "x", label="F_V2")
    plt.plot(detection_P_J280 * 40 / 65, "x", label="P_J280")
    plt.plot(detection_P_J269 * 42 / 65, "x", label="P_J269")
    plt.plot(detection_P_J300 * 44 / 65, "x", label="P_J300")
    plt.plot(detection_P_J256 * 46 / 65, "x", label="P_J256")
    plt.plot(detection_P_J289 * 48 / 65, "x", label="P_J289")
    plt.plot(detection_P_J415 * 50 / 65, "x", label="P_J415")
    plt.plot(detection_P_J302 * 52 / 65, "x", label="P_J302")
    plt.plot(detection_P_J306 * 54 / 65, "x", label="P_J306")
    plt.plot(detection_P_J307 * 56 / 65, "x", label="P_J307")
    plt.plot(detection_P_J317 * 58 / 65, "x", label="P_J317")
    plt.plot(detection_P_J14 * 60 / 65, "x", label="P_J14")
    plt.plot(detection_P_J422 * 62 / 65, "x", label="P_J422")

    plt.xlabel("t (h)")
    y_tick = ["UNDER ATTACK" if i == 1.0 else "SAFE" for i in testing_attack_labels]
    plt.yticks(testing_attack_labels, y_tick)

    df_plot_labels = pd.Series((i for i in testing_attack_labels))
    plt.fill_between(
        df_plot_labels.index,
        df_plot_labels.values,
        where=df_plot_labels.values <= 1.0,
        interpolate=True,
        color="lightsteelblue",
    )

    # Put a legend to the right of the current axis
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
    plt.show()


if __name__ == "__main__":
    localization()
