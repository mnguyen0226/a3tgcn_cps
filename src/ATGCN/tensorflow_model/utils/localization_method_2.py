import pandas as pd
import numpy as np
from utils.detection_clean_calculation import L
from sklearn.metrics import mean_squared_error

EVAL_CLEAN_LABEL_DIR = "out/tgcn/tgcn_scada_wds_lr0.005_batch128_unit64_seq8_pre1_epoch101/eval_clean/eval_clean_labels.csv"
EVAL_CLEAN_PREDS_DIR = "out/tgcn/tgcn_scada_wds_lr0.005_batch128_unit64_seq8_pre1_epoch101/eval_clean/eval_clean_output.csv"
EVAL_TEST_LABEL_DIR = "out/tgcn/tgcn_scada_wds_lr0.005_batch128_unit64_seq8_pre1_epoch101/eval_test/eval_test_labels.csv"
EVAL_TEST_PREDS_DIR = "out/tgcn/tgcn_scada_wds_lr0.005_batch128_unit64_seq8_pre1_epoch101/eval_test/eval_test_output.csv"
LINE_COLOR = "lightgray"
APPEARANCE_TH = 1

attack_detection_file = pd.read_csv(
    r"out/tgcn/tgcn_scada_wds_lr0.005_batch128_unit64_seq8_pre1_epoch101/eval_test/detection_results.csv",
    header=None,
)  # change for each different poisoned dataset.csv

testing_attack_preds = attack_detection_file[0].to_list()

# ATTACK_1_PRED = testing_attack_preds[285:355] # center_plot_index = 312
# ATTACK_2_PRED = testing_attack_preds[612:690] # center_plot_index = 644
# ATTACK_3_PRED = testing_attack_preds[839:894] # center_plot_index = 862
# ATTACK_4_PRED = testing_attack_preds[918:963] # center_plot_index = 932
# ATTACK_5_PRED = testing_attack_preds[1218:1317] # center_plot_index = 1258
# ATTACK_6_PRED = testing_attack_preds[1572:1637] # center_plot_index = 1593
# ATTACK_7_PRED = testing_attack_preds[1929:1950] # center_plot_index = 1934

dataset04 = pd.read_csv(
    r"data/processed/test_scada_dataset.csv"
)  # change for each different poisoned dataset.csv

binary_arr = dataset04["ATT_FLAG"].to_list()
testing_attack_labels = binary_arr[
    (L + 8) : -1
]  # use 8 for prediction + L for first window size


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

    # Compares the MSE of the prediction at each node to the max absolute error in attack 1
    # mse_L_T1_a1 = (
    #     mean_squared_error(test_L_T1_label[285:355], test_L_T1_pred[285:355])
    #     > max_error_L_T1
    # )
    # mse_L_T2_a1 = (
    #     mean_squared_error(test_L_T2_label[285:355], test_L_T2_pred[285:355])
    #     > max_error_L_T2
    # )
    # mse_L_T3_a1 = (
    #     mean_squared_error(test_L_T3_label[285:355], test_L_T3_pred[285:355])
    #     > max_error_L_T3
    # )
    # mse_L_T4_a1 = (
    #     mean_squared_error(test_L_T4_label[285:355], test_L_T4_pred[285:355])
    #     > max_error_L_T4
    # )
    # mse_L_T5_a1 = (
    #     mean_squared_error(test_L_T5_label[285:355], test_L_T5_pred[285:355])
    #     > max_error_L_T5
    # )
    # mse_L_T6_a1 = (
    #     mean_squared_error(test_L_T6_label[285:355], test_L_T6_pred[285:355])
    #     > max_error_L_T6
    # )
    # mse_L_T7_a1 = (
    #     mean_squared_error(test_L_T7_label[285:355], test_L_T7_pred[285:355])
    #     > max_error_L_T7
    # )
    # mse_F_PU1_a1 = (
    #     mean_squared_error(test_F_PU1_label[285:355], test_F_PU1_pred[285:355])
    #     > max_error_F_PU1
    # )
    # mse_F_PU2_a1 = (
    #     mean_squared_error(test_F_PU2_label[285:355], test_F_PU2_pred[285:355])
    #     > max_error_F_PU2
    # )
    # mse_F_PU3_a1 = (
    #     mean_squared_error(test_F_PU3_label[285:355], test_F_PU3_pred[285:355])
    #     > max_error_F_PU3
    # )
    # mse_F_PU4_a1 = (
    #     mean_squared_error(test_F_PU4_label[285:355], test_F_PU4_pred[285:355])
    #     > max_error_F_PU4
    # )
    # mse_F_PU5_a1 = (
    #     mean_squared_error(test_F_PU5_label[285:355], test_F_PU5_pred[285:355])
    #     > max_error_F_PU5
    # )
    # mse_F_PU6_a1 = (
    #     mean_squared_error(test_F_PU6_label[285:355], test_F_PU6_pred[285:355])
    #     > max_error_F_PU6
    # )
    # mse_F_PU7_a1 = (
    #     mean_squared_error(test_F_PU7_label[285:355], test_F_PU7_pred[285:355])
    #     > max_error_F_PU7
    # )
    # mse_F_PU8_a1 = (
    #     mean_squared_error(test_F_PU8_label[285:355], test_F_PU8_pred[285:355])
    #     > max_error_F_PU8
    # )
    # mse_F_PU9_a1 = (
    #     mean_squared_error(test_F_PU9_label[285:355], test_F_PU9_pred[285:355])
    #     > max_error_F_PU9
    # )
    # mse_F_PU10_a1 = (
    #     mean_squared_error(test_F_PU10_label[285:355], test_F_PU10_pred[285:355])
    #     > max_error_F_PU10
    # )
    # mse_F_PU11_a1 = (
    #     mean_squared_error(test_F_PU11_label[285:355], test_F_PU11_pred[285:355])
    #     > max_error_F_PU11
    # )
    # mse_F_V2_a1 = (
    #     mean_squared_error(test_F_V2_label[285:355], test_F_V2_pred[285:355])
    #     > max_error_F_V2
    # )
    # mse_P_J280_a1 = (
    #     mean_squared_error(test_P_J280_label[285:355], test_P_J280_pred[285:355])
    #     > max_error_P_J280
    # )
    # mse_P_J269_a1 = (
    #     mean_squared_error(test_P_J269_label[285:355], test_P_J269_pred[285:355])
    #     > max_error_P_J269
    # )
    # mse_P_J300_a1 = (
    #     mean_squared_error(test_P_J300_label[285:355], test_P_J300_pred[285:355])
    #     > max_error_P_J300
    # )
    # mse_P_J256_a1 = (
    #     mean_squared_error(test_P_J256_label[285:355], test_P_J256_pred[285:355])
    #     > max_error_P_J256
    # )
    # mse_P_J289_a1 = (
    #     mean_squared_error(test_P_J289_label[285:355], test_P_J289_pred[285:355])
    #     > max_error_P_J289
    # )
    # mse_P_J415_a1 = (
    #     mean_squared_error(test_P_J415_label[285:355], test_P_J415_pred[285:355])
    #     > max_error_P_J415
    # )
    # mse_P_J302_a1 = (
    #     mean_squared_error(test_P_J302_label[285:355], test_P_J302_pred[285:355])
    #     > max_error_P_J302
    # )
    # mse_P_J306_a1 = (
    #     mean_squared_error(test_P_J306_label[285:355], test_P_J306_pred[285:355])
    #     > max_error_P_J306
    # )
    # mse_P_J307_a1 = (
    #     mean_squared_error(test_P_J307_label[285:355], test_P_J307_pred[285:355])
    #     > max_error_P_J307
    # )
    # mse_P_J317_a1 = (
    #     mean_squared_error(test_P_J317_label[285:355], test_P_J317_pred[285:355])
    #     > max_error_P_J317
    # )
    # mse_P_J14_a1 = (
    #     mean_squared_error(test_P_J14_label[285:355], test_P_J14_pred[285:355])
    #     > max_error_P_J14
    # )
    # mse_P_J422_a1 = (
    #     mean_squared_error(test_P_J422_label[285:355], test_P_J422_pred[285:355])
    #     > max_error_P_J422
    # )

    # Count the number of absolute abnormal error vs max normal error in attack 1
    check_L_T1_a1 = (
        np.count_nonzero(
            np.abs(test_L_T1_label[285:355] - test_L_T1_pred[285:355]) > max_error_L_T1
        )
        > APPEARANCE_TH
    )
    check_L_T2_a1 = (
        np.count_nonzero(
            np.abs(test_L_T2_label[285:355] - test_L_T2_pred[285:355]) > max_error_L_T2
        )
        > APPEARANCE_TH
    )
    check_L_T3_a1 = (
        np.count_nonzero(
            np.abs(test_L_T3_label[285:355] - test_L_T3_pred[285:355]) > max_error_L_T3
        )
        > APPEARANCE_TH
    )
    check_L_T4_a1 = (
        np.count_nonzero(
            np.abs(test_L_T4_label[285:355] - test_L_T4_pred[285:355]) > max_error_L_T4
        )
        > APPEARANCE_TH
    )
    check_L_T5_a1 = (
        np.count_nonzero(
            np.abs(test_L_T5_label[285:355] - test_L_T5_pred[285:355]) > max_error_L_T5
        )
        > APPEARANCE_TH
    )
    check_L_T6_a1 = (
        np.count_nonzero(
            np.abs(test_L_T6_label[285:355] - test_L_T6_pred[285:355]) > max_error_L_T6
        )
        > APPEARANCE_TH
    )
    check_L_T7_a1 = (
        np.count_nonzero(
            np.abs(test_L_T7_label[285:355] - test_L_T7_pred[285:355]) > max_error_L_T7
        )
        > APPEARANCE_TH
    )
    check_F_PU1_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU1_label[285:355] - test_F_PU1_pred[285:355])
            > max_error_F_PU1
        )
        > APPEARANCE_TH
    )
    check_F_PU2_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU2_label[285:355] - test_F_PU2_pred[285:355])
            > max_error_F_PU2
        )
        > APPEARANCE_TH
    )
    check_F_PU3_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU3_label[285:355] - test_F_PU3_pred[285:355])
            > max_error_F_PU3
        )
        > APPEARANCE_TH
    )
    check_F_PU4_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU4_label[285:355] - test_F_PU4_pred[285:355])
            > max_error_F_PU4
        )
        > APPEARANCE_TH
    )
    check_F_PU5_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU5_label[285:355] - test_F_PU5_pred[285:355])
            > max_error_F_PU5
        )
        > APPEARANCE_TH
    )
    check_F_PU6_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU6_label[285:355] - test_F_PU6_pred[285:355])
            > max_error_F_PU6
        )
        > APPEARANCE_TH
    )
    check_F_PU7_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU7_label[285:355] - test_F_PU7_pred[285:355])
            > max_error_F_PU7
        )
        > APPEARANCE_TH
    )
    check_F_PU8_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU8_label[285:355] - test_F_PU8_pred[285:355])
            > max_error_F_PU8
        )
        > APPEARANCE_TH
    )
    check_F_PU9_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU9_label[285:355] - test_F_PU9_pred[285:355])
            > max_error_F_PU9
        )
        > APPEARANCE_TH
    )
    check_F_PU10_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU10_label[285:355] - test_F_PU10_pred[285:355])
            > max_error_F_PU10
        )
        > APPEARANCE_TH
    )
    check_F_PU11_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU11_label[285:355] - test_F_PU11_pred[285:355])
            > max_error_F_PU11
        )
        > APPEARANCE_TH
    )
    check_F_V2_a1 = (
        np.count_nonzero(
            np.abs(test_F_V2_label[285:355] - test_F_V2_pred[285:355]) > max_error_F_V2
        )
        > APPEARANCE_TH
    )
    check_P_J280_a1 = (
        np.count_nonzero(
            np.abs(test_P_J280_label[285:355] - test_P_J280_pred[285:355])
            > max_error_P_J280
        )
        > APPEARANCE_TH
    )
    check_P_J269_a1 = (
        np.count_nonzero(
            np.abs(test_P_J269_label[285:355] - test_P_J269_pred[285:355])
            > max_error_P_J269
        )
        > APPEARANCE_TH
    )
    check_P_J300_a1 = (
        np.count_nonzero(
            np.abs(test_P_J300_label[285:355] - test_P_J300_pred[285:355])
            > max_error_P_J300
        )
        > APPEARANCE_TH
    )
    check_P_J256_a1 = (
        np.count_nonzero(
            np.abs(test_P_J256_label[285:355] - test_P_J256_pred[285:355])
            > max_error_P_J256
        )
        > APPEARANCE_TH
    )
    check_P_J289_a1 = (
        np.count_nonzero(
            np.abs(test_P_J289_label[285:355] - test_P_J289_pred[285:355])
            > max_error_P_J289
        )
        > APPEARANCE_TH
    )
    check_P_J415_a1 = (
        np.count_nonzero(
            np.abs(test_P_J415_label[285:355] - test_P_J415_pred[285:355])
            > max_error_P_J415
        )
        > APPEARANCE_TH
    )
    check_P_J302_a1 = (
        np.count_nonzero(
            np.abs(test_P_J302_label[285:355] - test_P_J302_pred[285:355])
            > max_error_P_J302
        )
        > APPEARANCE_TH
    )
    check_P_J306_a1 = (
        np.count_nonzero(
            np.abs(test_P_J306_label[285:355] - test_P_J306_pred[285:355])
            > max_error_P_J306
        )
        > APPEARANCE_TH
    )
    check_P_J307_a1 = (
        np.count_nonzero(
            np.abs(test_P_J307_label[285:355] - test_P_J307_pred[285:355])
            > max_error_P_J307
        )
        > APPEARANCE_TH
    )
    check_P_J317_a1 = (
        np.count_nonzero(
            np.abs(test_P_J317_label[285:355] - test_P_J317_pred[285:355])
            > max_error_P_J317
        )
        > APPEARANCE_TH
    )
    check_P_J14_a1 = (
        np.count_nonzero(
            np.abs(test_P_J14_label[285:355] - test_P_J14_pred[285:355])
            > max_error_P_J14
        )
        > APPEARANCE_TH
    )
    check_P_J422_a1 = (
        np.count_nonzero(
            np.abs(test_P_J422_label[285:355] - test_P_J422_pred[285:355])
            > max_error_P_J422
        )
        > APPEARANCE_TH
    )

    print("\n----------------------------------------------")
    print("ATTACK 1 COMPROMISED SENSORS DETECTION REPORT:")
    print(
        f"TANKS: L_T1 Compromised? {check_L_T1_a1} || L_T2 Compromised? {check_L_T2_a1} || L_T3 Compromised? {check_L_T3_a1} || L_T4 Compromised? {check_L_T4_a1} || L_T5 Compromised? {check_L_T5_a1} || L_T6 Compromised? {check_L_T6_a1} || L_T7 Compromised? {check_L_T7_a1}"
    )
    print(
        f"PUMPS: F_PU1 Compromised? {check_F_PU1_a1} || F_PU2 Compromised? {check_F_PU2_a1} || F_PU3 Compromised? {check_F_PU3_a1} || F_PU4 Compromised? {check_F_PU4_a1} || F_PU5 Compromised? {check_F_PU5_a1} || F_PU6 Compromised? {check_F_PU6_a1} || F_PU7 Compromised? {check_F_PU7_a1} || F_PU8 Compromised? {check_F_PU8_a1} || F_PU9 Compromised? {check_F_PU9_a1} || F_PU10 Compromised? {check_F_PU10_a1} || F_PU11 Compromised? {check_F_PU11_a1}"
    )
    print(f"VALVE: F_V2 Compromised? {check_F_V2_a1}")
    print(
        f"PRESSURE PUMPS: P_J280 Compromised? {check_P_J280_a1} || P_J269 Compromised? {check_P_J269_a1} || P_J300 Compromised? {check_P_J300_a1} || P_J256 Compromised? {check_P_J256_a1} || P_J289 Compromised? {check_P_J289_a1} || P_J415 Compromised? {check_P_J415_a1} || P_J302 Compromised? {check_P_J302_a1} || P_J306 Compromised? {check_P_J306_a1} || P_J307 Compromised? {check_P_J307_a1} || P_J317 Compromised? {check_P_J317_a1} || P_J14 Compromised? {check_P_J14_a1} || P_J422 Compromised? {check_P_J422_a1}"
    )

    # Count the number of absolute abnormal error vs max normal error in attack 2
    check_L_T1_a1 = (
        np.count_nonzero(
            np.abs(test_L_T1_label[612:690] - test_L_T1_pred[612:690]) > max_error_L_T1
        )
        > APPEARANCE_TH
    )
    check_L_T2_a1 = (
        np.count_nonzero(
            np.abs(test_L_T2_label[612:690] - test_L_T2_pred[612:690]) > max_error_L_T2
        )
        > APPEARANCE_TH
    )
    check_L_T3_a1 = (
        np.count_nonzero(
            np.abs(test_L_T3_label[612:690] - test_L_T3_pred[612:690]) > max_error_L_T3
        )
        > APPEARANCE_TH
    )
    check_L_T4_a1 = (
        np.count_nonzero(
            np.abs(test_L_T4_label[612:690] - test_L_T4_pred[612:690]) > max_error_L_T4
        )
        > APPEARANCE_TH
    )
    check_L_T5_a1 = (
        np.count_nonzero(
            np.abs(test_L_T5_label[612:690] - test_L_T5_pred[612:690]) > max_error_L_T5
        )
        > APPEARANCE_TH
    )
    check_L_T6_a1 = (
        np.count_nonzero(
            np.abs(test_L_T6_label[612:690] - test_L_T6_pred[612:690]) > max_error_L_T6
        )
        > APPEARANCE_TH
    )
    check_L_T7_a1 = (
        np.count_nonzero(
            np.abs(test_L_T7_label[612:690] - test_L_T7_pred[612:690]) > max_error_L_T7
        )
        > APPEARANCE_TH
    )
    check_F_PU1_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU1_label[612:690] - test_F_PU1_pred[612:690])
            > max_error_F_PU1
        )
        > APPEARANCE_TH
    )
    check_F_PU2_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU2_label[612:690] - test_F_PU2_pred[612:690])
            > max_error_F_PU2
        )
        > APPEARANCE_TH
    )
    check_F_PU3_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU3_label[612:690] - test_F_PU3_pred[612:690])
            > max_error_F_PU3
        )
        > APPEARANCE_TH
    )
    check_F_PU4_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU4_label[612:690] - test_F_PU4_pred[612:690])
            > max_error_F_PU4
        )
        > APPEARANCE_TH
    )
    check_F_PU5_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU5_label[612:690] - test_F_PU5_pred[612:690])
            > max_error_F_PU5
        )
        > APPEARANCE_TH
    )
    check_F_PU6_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU6_label[612:690] - test_F_PU6_pred[612:690])
            > max_error_F_PU6
        )
        > APPEARANCE_TH
    )
    check_F_PU7_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU7_label[612:690] - test_F_PU7_pred[612:690])
            > max_error_F_PU7
        )
        > APPEARANCE_TH
    )
    check_F_PU8_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU8_label[612:690] - test_F_PU8_pred[612:690])
            > max_error_F_PU8
        )
        > APPEARANCE_TH
    )
    check_F_PU9_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU9_label[612:690] - test_F_PU9_pred[612:690])
            > max_error_F_PU9
        )
        > APPEARANCE_TH
    )
    check_F_PU10_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU10_label[612:690] - test_F_PU10_pred[612:690])
            > max_error_F_PU10
        )
        > APPEARANCE_TH
    )
    check_F_PU11_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU11_label[612:690] - test_F_PU11_pred[612:690])
            > max_error_F_PU11
        )
        > APPEARANCE_TH
    )
    check_F_V2_a1 = (
        np.count_nonzero(
            np.abs(test_F_V2_label[612:690] - test_F_V2_pred[612:690]) > max_error_F_V2
        )
        > APPEARANCE_TH
    )
    check_P_J280_a1 = (
        np.count_nonzero(
            np.abs(test_P_J280_label[612:690] - test_P_J280_pred[612:690])
            > max_error_P_J280
        )
        > APPEARANCE_TH
    )
    check_P_J269_a1 = (
        np.count_nonzero(
            np.abs(test_P_J269_label[612:690] - test_P_J269_pred[612:690])
            > max_error_P_J269
        )
        > APPEARANCE_TH
    )
    check_P_J300_a1 = (
        np.count_nonzero(
            np.abs(test_P_J300_label[612:690] - test_P_J300_pred[612:690])
            > max_error_P_J300
        )
        > APPEARANCE_TH
    )
    check_P_J256_a1 = (
        np.count_nonzero(
            np.abs(test_P_J256_label[612:690] - test_P_J256_pred[612:690])
            > max_error_P_J256
        )
        > APPEARANCE_TH
    )
    check_P_J289_a1 = (
        np.count_nonzero(
            np.abs(test_P_J289_label[612:690] - test_P_J289_pred[612:690])
            > max_error_P_J289
        )
        > APPEARANCE_TH
    )
    check_P_J415_a1 = (
        np.count_nonzero(
            np.abs(test_P_J415_label[612:690] - test_P_J415_pred[612:690])
            > max_error_P_J415
        )
        > APPEARANCE_TH
    )
    check_P_J302_a1 = (
        np.count_nonzero(
            np.abs(test_P_J302_label[612:690] - test_P_J302_pred[612:690])
            > max_error_P_J302
        )
        > APPEARANCE_TH
    )
    check_P_J306_a1 = (
        np.count_nonzero(
            np.abs(test_P_J306_label[612:690] - test_P_J306_pred[612:690])
            > max_error_P_J306
        )
        > APPEARANCE_TH
    )
    check_P_J307_a1 = (
        np.count_nonzero(
            np.abs(test_P_J307_label[612:690] - test_P_J307_pred[612:690])
            > max_error_P_J307
        )
        > APPEARANCE_TH
    )
    check_P_J317_a1 = (
        np.count_nonzero(
            np.abs(test_P_J317_label[612:690] - test_P_J317_pred[612:690])
            > max_error_P_J317
        )
        > APPEARANCE_TH
    )
    check_P_J14_a1 = (
        np.count_nonzero(
            np.abs(test_P_J14_label[612:690] - test_P_J14_pred[612:690])
            > max_error_P_J14
        )
        > APPEARANCE_TH
    )
    check_P_J422_a1 = (
        np.count_nonzero(
            np.abs(test_P_J422_label[612:690] - test_P_J422_pred[612:690])
            > max_error_P_J422
        )
        > APPEARANCE_TH
    )

    print("\n----------------------------------------------")
    print("ATTACK 2 COMPROMISED SENSORS DETECTION REPORT:")
    print(
        f"TANKS: L_T1 Compromised? {check_L_T1_a1} || L_T2 Compromised? {check_L_T2_a1} || L_T3 Compromised? {check_L_T3_a1} || L_T4 Compromised? {check_L_T4_a1} || L_T5 Compromised? {check_L_T5_a1} || L_T6 Compromised? {check_L_T6_a1} || L_T7 Compromised? {check_L_T7_a1}"
    )
    print(
        f"PUMPS: F_PU1 Compromised? {check_F_PU1_a1} || F_PU2 Compromised? {check_F_PU2_a1} || F_PU3 Compromised? {check_F_PU3_a1} || F_PU4 Compromised? {check_F_PU4_a1} || F_PU5 Compromised? {check_F_PU5_a1} || F_PU6 Compromised? {check_F_PU6_a1} || F_PU7 Compromised? {check_F_PU7_a1} || F_PU8 Compromised? {check_F_PU8_a1} || F_PU9 Compromised? {check_F_PU9_a1} || F_PU10 Compromised? {check_F_PU10_a1} || F_PU11 Compromised? {check_F_PU11_a1}"
    )
    print(f"VALVE: F_V2 Compromised? {check_F_V2_a1}")
    print(
        f"PRESSURE PUMPS: P_J280 Compromised? {check_P_J280_a1} || P_J269 Compromised? {check_P_J269_a1} || P_J300 Compromised? {check_P_J300_a1} || P_J256 Compromised? {check_P_J256_a1} || P_J289 Compromised? {check_P_J289_a1} || P_J415 Compromised? {check_P_J415_a1} || P_J302 Compromised? {check_P_J302_a1} || P_J306 Compromised? {check_P_J306_a1} || P_J307 Compromised? {check_P_J307_a1} || P_J317 Compromised? {check_P_J317_a1} || P_J14 Compromised? {check_P_J14_a1} || P_J422 Compromised? {check_P_J422_a1}"
    )

    # Count the number of absolute abnormal error vs max normal error in attack 3
    check_L_T1_a1 = (
        np.count_nonzero(
            np.abs(test_L_T1_label[839:894] - test_L_T1_pred[839:894]) > max_error_L_T1
        )
        > APPEARANCE_TH
    )
    check_L_T2_a1 = (
        np.count_nonzero(
            np.abs(test_L_T2_label[839:894] - test_L_T2_pred[839:894]) > max_error_L_T2
        )
        > APPEARANCE_TH
    )
    check_L_T3_a1 = (
        np.count_nonzero(
            np.abs(test_L_T3_label[839:894] - test_L_T3_pred[839:894]) > max_error_L_T3
        )
        > APPEARANCE_TH
    )
    check_L_T4_a1 = (
        np.count_nonzero(
            np.abs(test_L_T4_label[839:894] - test_L_T4_pred[839:894]) > max_error_L_T4
        )
        > APPEARANCE_TH
    )
    check_L_T5_a1 = (
        np.count_nonzero(
            np.abs(test_L_T5_label[839:894] - test_L_T5_pred[839:894]) > max_error_L_T5
        )
        > APPEARANCE_TH
    )
    check_L_T6_a1 = (
        np.count_nonzero(
            np.abs(test_L_T6_label[839:894] - test_L_T6_pred[839:894]) > max_error_L_T6
        )
        > APPEARANCE_TH
    )
    check_L_T7_a1 = (
        np.count_nonzero(
            np.abs(test_L_T7_label[839:894] - test_L_T7_pred[839:894]) > max_error_L_T7
        )
        > APPEARANCE_TH
    )
    check_F_PU1_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU1_label[839:894] - test_F_PU1_pred[839:894])
            > max_error_F_PU1
        )
        > APPEARANCE_TH
    )
    check_F_PU2_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU2_label[839:894] - test_F_PU2_pred[839:894])
            > max_error_F_PU2
        )
        > APPEARANCE_TH
    )
    check_F_PU3_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU3_label[839:894] - test_F_PU3_pred[839:894])
            > max_error_F_PU3
        )
        > APPEARANCE_TH
    )
    check_F_PU4_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU4_label[839:894] - test_F_PU4_pred[839:894])
            > max_error_F_PU4
        )
        > APPEARANCE_TH
    )
    check_F_PU5_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU5_label[839:894] - test_F_PU5_pred[839:894])
            > max_error_F_PU5
        )
        > APPEARANCE_TH
    )
    check_F_PU6_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU6_label[839:894] - test_F_PU6_pred[839:894])
            > max_error_F_PU6
        )
        > APPEARANCE_TH
    )
    check_F_PU7_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU7_label[839:894] - test_F_PU7_pred[839:894])
            > max_error_F_PU7
        )
        > APPEARANCE_TH
    )
    check_F_PU8_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU8_label[839:894] - test_F_PU8_pred[839:894])
            > max_error_F_PU8
        )
        > APPEARANCE_TH
    )
    check_F_PU9_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU9_label[839:894] - test_F_PU9_pred[839:894])
            > max_error_F_PU9
        )
        > APPEARANCE_TH
    )
    check_F_PU10_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU10_label[839:894] - test_F_PU10_pred[839:894])
            > max_error_F_PU10
        )
        > APPEARANCE_TH
    )
    check_F_PU11_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU11_label[839:894] - test_F_PU11_pred[839:894])
            > max_error_F_PU11
        )
        > APPEARANCE_TH
    )
    check_F_V2_a1 = (
        np.count_nonzero(
            np.abs(test_F_V2_label[839:894] - test_F_V2_pred[839:894]) > max_error_F_V2
        )
        > APPEARANCE_TH
    )
    check_P_J280_a1 = (
        np.count_nonzero(
            np.abs(test_P_J280_label[839:894] - test_P_J280_pred[839:894])
            > max_error_P_J280
        )
        > APPEARANCE_TH
    )
    check_P_J269_a1 = (
        np.count_nonzero(
            np.abs(test_P_J269_label[839:894] - test_P_J269_pred[839:894])
            > max_error_P_J269
        )
        > APPEARANCE_TH
    )
    check_P_J300_a1 = (
        np.count_nonzero(
            np.abs(test_P_J300_label[839:894] - test_P_J300_pred[839:894])
            > max_error_P_J300
        )
        > APPEARANCE_TH
    )
    check_P_J256_a1 = (
        np.count_nonzero(
            np.abs(test_P_J256_label[839:894] - test_P_J256_pred[839:894])
            > max_error_P_J256
        )
        > APPEARANCE_TH
    )
    check_P_J289_a1 = (
        np.count_nonzero(
            np.abs(test_P_J289_label[839:894] - test_P_J289_pred[839:894])
            > max_error_P_J289
        )
        > APPEARANCE_TH
    )
    check_P_J415_a1 = (
        np.count_nonzero(
            np.abs(test_P_J415_label[839:894] - test_P_J415_pred[839:894])
            > max_error_P_J415
        )
        > APPEARANCE_TH
    )
    check_P_J302_a1 = (
        np.count_nonzero(
            np.abs(test_P_J302_label[839:894] - test_P_J302_pred[839:894])
            > max_error_P_J302
        )
        > APPEARANCE_TH
    )
    check_P_J306_a1 = (
        np.count_nonzero(
            np.abs(test_P_J306_label[839:894] - test_P_J306_pred[839:894])
            > max_error_P_J306
        )
        > APPEARANCE_TH
    )
    check_P_J307_a1 = (
        np.count_nonzero(
            np.abs(test_P_J307_label[839:894] - test_P_J307_pred[839:894])
            > max_error_P_J307
        )
        > APPEARANCE_TH
    )
    check_P_J317_a1 = (
        np.count_nonzero(
            np.abs(test_P_J317_label[839:894] - test_P_J317_pred[839:894])
            > max_error_P_J317
        )
        > APPEARANCE_TH
    )
    check_P_J14_a1 = (
        np.count_nonzero(
            np.abs(test_P_J14_label[839:894] - test_P_J14_pred[839:894])
            > max_error_P_J14
        )
        > APPEARANCE_TH
    )
    check_P_J422_a1 = (
        np.count_nonzero(
            np.abs(test_P_J422_label[839:894] - test_P_J422_pred[839:894])
            > max_error_P_J422
        )
        > APPEARANCE_TH
    )

    print("\n----------------------------------------------")
    print("ATTACK 3 COMPROMISED SENSORS DETECTION REPORT:")
    print(
        f"TANKS: L_T1 Compromised? {check_L_T1_a1} || L_T2 Compromised? {check_L_T2_a1} || L_T3 Compromised? {check_L_T3_a1} || L_T4 Compromised? {check_L_T4_a1} || L_T5 Compromised? {check_L_T5_a1} || L_T6 Compromised? {check_L_T6_a1} || L_T7 Compromised? {check_L_T7_a1}"
    )
    print(
        f"PUMPS: F_PU1 Compromised? {check_F_PU1_a1} || F_PU2 Compromised? {check_F_PU2_a1} || F_PU3 Compromised? {check_F_PU3_a1} || F_PU4 Compromised? {check_F_PU4_a1} || F_PU5 Compromised? {check_F_PU5_a1} || F_PU6 Compromised? {check_F_PU6_a1} || F_PU7 Compromised? {check_F_PU7_a1} || F_PU8 Compromised? {check_F_PU8_a1} || F_PU9 Compromised? {check_F_PU9_a1} || F_PU10 Compromised? {check_F_PU10_a1} || F_PU11 Compromised? {check_F_PU11_a1}"
    )
    print(f"VALVE: F_V2 Compromised? {check_F_V2_a1}")
    print(
        f"PRESSURE PUMPS: P_J280 Compromised? {check_P_J280_a1} || P_J269 Compromised? {check_P_J269_a1} || P_J300 Compromised? {check_P_J300_a1} || P_J256 Compromised? {check_P_J256_a1} || P_J289 Compromised? {check_P_J289_a1} || P_J415 Compromised? {check_P_J415_a1} || P_J302 Compromised? {check_P_J302_a1} || P_J306 Compromised? {check_P_J306_a1} || P_J307 Compromised? {check_P_J307_a1} || P_J317 Compromised? {check_P_J317_a1} || P_J14 Compromised? {check_P_J14_a1} || P_J422 Compromised? {check_P_J422_a1}"
    )

    # Count the number of absolute abnormal error vs max normal error in attack 4
    check_L_T1_a1 = (
        np.count_nonzero(
            np.abs(test_L_T1_label[918:963] - test_L_T1_pred[918:963]) > max_error_L_T1
        )
        > APPEARANCE_TH
    )
    check_L_T2_a1 = (
        np.count_nonzero(
            np.abs(test_L_T2_label[918:963] - test_L_T2_pred[918:963]) > max_error_L_T2
        )
        > APPEARANCE_TH
    )
    check_L_T3_a1 = (
        np.count_nonzero(
            np.abs(test_L_T3_label[918:963] - test_L_T3_pred[918:963]) > max_error_L_T3
        )
        > APPEARANCE_TH
    )
    check_L_T4_a1 = (
        np.count_nonzero(
            np.abs(test_L_T4_label[918:963] - test_L_T4_pred[918:963]) > max_error_L_T4
        )
        > APPEARANCE_TH
    )
    check_L_T5_a1 = (
        np.count_nonzero(
            np.abs(test_L_T5_label[918:963] - test_L_T5_pred[918:963]) > max_error_L_T5
        )
        > APPEARANCE_TH
    )
    check_L_T6_a1 = (
        np.count_nonzero(
            np.abs(test_L_T6_label[918:963] - test_L_T6_pred[918:963]) > max_error_L_T6
        )
        > APPEARANCE_TH
    )
    check_L_T7_a1 = (
        np.count_nonzero(
            np.abs(test_L_T7_label[918:963] - test_L_T7_pred[918:963]) > max_error_L_T7
        )
        > APPEARANCE_TH
    )
    check_F_PU1_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU1_label[918:963] - test_F_PU1_pred[918:963])
            > max_error_F_PU1
        )
        > APPEARANCE_TH
    )
    check_F_PU2_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU2_label[918:963] - test_F_PU2_pred[918:963])
            > max_error_F_PU2
        )
        > APPEARANCE_TH
    )
    check_F_PU3_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU3_label[918:963] - test_F_PU3_pred[918:963])
            > max_error_F_PU3
        )
        > APPEARANCE_TH
    )
    check_F_PU4_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU4_label[918:963] - test_F_PU4_pred[918:963])
            > max_error_F_PU4
        )
        > APPEARANCE_TH
    )
    check_F_PU5_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU5_label[918:963] - test_F_PU5_pred[918:963])
            > max_error_F_PU5
        )
        > APPEARANCE_TH
    )
    check_F_PU6_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU6_label[918:963] - test_F_PU6_pred[918:963])
            > max_error_F_PU6
        )
        > APPEARANCE_TH
    )
    check_F_PU7_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU7_label[918:963] - test_F_PU7_pred[918:963])
            > max_error_F_PU7
        )
        > APPEARANCE_TH
    )
    check_F_PU8_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU8_label[918:963] - test_F_PU8_pred[918:963])
            > max_error_F_PU8
        )
        > APPEARANCE_TH
    )
    check_F_PU9_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU9_label[918:963] - test_F_PU9_pred[918:963])
            > max_error_F_PU9
        )
        > APPEARANCE_TH
    )
    check_F_PU10_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU10_label[918:963] - test_F_PU10_pred[918:963])
            > max_error_F_PU10
        )
        > APPEARANCE_TH
    )
    check_F_PU11_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU11_label[918:963] - test_F_PU11_pred[918:963])
            > max_error_F_PU11
        )
        > APPEARANCE_TH
    )
    check_F_V2_a1 = (
        np.count_nonzero(
            np.abs(test_F_V2_label[918:963] - test_F_V2_pred[918:963]) > max_error_F_V2
        )
        > APPEARANCE_TH
    )
    check_P_J280_a1 = (
        np.count_nonzero(
            np.abs(test_P_J280_label[918:963] - test_P_J280_pred[918:963])
            > max_error_P_J280
        )
        > APPEARANCE_TH
    )
    check_P_J269_a1 = (
        np.count_nonzero(
            np.abs(test_P_J269_label[918:963] - test_P_J269_pred[918:963])
            > max_error_P_J269
        )
        > APPEARANCE_TH
    )
    check_P_J300_a1 = (
        np.count_nonzero(
            np.abs(test_P_J300_label[918:963] - test_P_J300_pred[918:963])
            > max_error_P_J300
        )
        > APPEARANCE_TH
    )
    check_P_J256_a1 = (
        np.count_nonzero(
            np.abs(test_P_J256_label[918:963] - test_P_J256_pred[918:963])
            > max_error_P_J256
        )
        > APPEARANCE_TH
    )
    check_P_J289_a1 = (
        np.count_nonzero(
            np.abs(test_P_J289_label[918:963] - test_P_J289_pred[918:963])
            > max_error_P_J289
        )
        > APPEARANCE_TH
    )
    check_P_J415_a1 = (
        np.count_nonzero(
            np.abs(test_P_J415_label[918:963] - test_P_J415_pred[918:963])
            > max_error_P_J415
        )
        > APPEARANCE_TH
    )
    check_P_J302_a1 = (
        np.count_nonzero(
            np.abs(test_P_J302_label[918:963] - test_P_J302_pred[918:963])
            > max_error_P_J302
        )
        > APPEARANCE_TH
    )
    check_P_J306_a1 = (
        np.count_nonzero(
            np.abs(test_P_J306_label[918:963] - test_P_J306_pred[918:963])
            > max_error_P_J306
        )
        > APPEARANCE_TH
    )
    check_P_J307_a1 = (
        np.count_nonzero(
            np.abs(test_P_J307_label[918:963] - test_P_J307_pred[918:963])
            > max_error_P_J307
        )
        > APPEARANCE_TH
    )
    check_P_J317_a1 = (
        np.count_nonzero(
            np.abs(test_P_J317_label[918:963] - test_P_J317_pred[918:963])
            > max_error_P_J317
        )
        > APPEARANCE_TH
    )
    check_P_J14_a1 = (
        np.count_nonzero(
            np.abs(test_P_J14_label[918:963] - test_P_J14_pred[918:963])
            > max_error_P_J14
        )
        > APPEARANCE_TH
    )
    check_P_J422_a1 = (
        np.count_nonzero(
            np.abs(test_P_J422_label[918:963] - test_P_J422_pred[918:963])
            > max_error_P_J422
        )
        > APPEARANCE_TH
    )

    print("\n----------------------------------------------")
    print("ATTACK 4 COMPROMISED SENSORS DETECTION REPORT:")
    print(
        f"TANKS: L_T1 Compromised? {check_L_T1_a1} || L_T2 Compromised? {check_L_T2_a1} || L_T3 Compromised? {check_L_T3_a1} || L_T4 Compromised? {check_L_T4_a1} || L_T5 Compromised? {check_L_T5_a1} || L_T6 Compromised? {check_L_T6_a1} || L_T7 Compromised? {check_L_T7_a1}"
    )
    print(
        f"PUMPS: F_PU1 Compromised? {check_F_PU1_a1} || F_PU2 Compromised? {check_F_PU2_a1} || F_PU3 Compromised? {check_F_PU3_a1} || F_PU4 Compromised? {check_F_PU4_a1} || F_PU5 Compromised? {check_F_PU5_a1} || F_PU6 Compromised? {check_F_PU6_a1} || F_PU7 Compromised? {check_F_PU7_a1} || F_PU8 Compromised? {check_F_PU8_a1} || F_PU9 Compromised? {check_F_PU9_a1} || F_PU10 Compromised? {check_F_PU10_a1} || F_PU11 Compromised? {check_F_PU11_a1}"
    )
    print(f"VALVE: F_V2 Compromised? {check_F_V2_a1}")
    print(
        f"PRESSURE PUMPS: P_J280 Compromised? {check_P_J280_a1} || P_J269 Compromised? {check_P_J269_a1} || P_J300 Compromised? {check_P_J300_a1} || P_J256 Compromised? {check_P_J256_a1} || P_J289 Compromised? {check_P_J289_a1} || P_J415 Compromised? {check_P_J415_a1} || P_J302 Compromised? {check_P_J302_a1} || P_J306 Compromised? {check_P_J306_a1} || P_J307 Compromised? {check_P_J307_a1} || P_J317 Compromised? {check_P_J317_a1} || P_J14 Compromised? {check_P_J14_a1} || P_J422 Compromised? {check_P_J422_a1}"
    )

    # Count the number of absolute abnormal error vs max normal error in attack 5
    check_L_T1_a1 = (
        np.count_nonzero(
            np.abs(test_L_T1_label[1218:1317] - test_L_T1_pred[1218:1317])
            > max_error_L_T1
        )
        > APPEARANCE_TH
    )
    check_L_T2_a1 = (
        np.count_nonzero(
            np.abs(test_L_T2_label[1218:1317] - test_L_T2_pred[1218:1317])
            > max_error_L_T2
        )
        > APPEARANCE_TH
    )
    check_L_T3_a1 = (
        np.count_nonzero(
            np.abs(test_L_T3_label[1218:1317] - test_L_T3_pred[1218:1317])
            > max_error_L_T3
        )
        > APPEARANCE_TH
    )
    check_L_T4_a1 = (
        np.count_nonzero(
            np.abs(test_L_T4_label[1218:1317] - test_L_T4_pred[1218:1317])
            > max_error_L_T4
        )
        > APPEARANCE_TH
    )
    check_L_T5_a1 = (
        np.count_nonzero(
            np.abs(test_L_T5_label[1218:1317] - test_L_T5_pred[1218:1317])
            > max_error_L_T5
        )
        > APPEARANCE_TH
    )
    check_L_T6_a1 = (
        np.count_nonzero(
            np.abs(test_L_T6_label[1218:1317] - test_L_T6_pred[1218:1317])
            > max_error_L_T6
        )
        > APPEARANCE_TH
    )
    check_L_T7_a1 = (
        np.count_nonzero(
            np.abs(test_L_T7_label[1218:1317] - test_L_T7_pred[1218:1317])
            > max_error_L_T7
        )
        > APPEARANCE_TH
    )
    check_F_PU1_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU1_label[1218:1317] - test_F_PU1_pred[1218:1317])
            > max_error_F_PU1
        )
        > APPEARANCE_TH
    )
    check_F_PU2_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU2_label[1218:1317] - test_F_PU2_pred[1218:1317])
            > max_error_F_PU2
        )
        > APPEARANCE_TH
    )
    check_F_PU3_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU3_label[1218:1317] - test_F_PU3_pred[1218:1317])
            > max_error_F_PU3
        )
        > APPEARANCE_TH
    )
    check_F_PU4_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU4_label[1218:1317] - test_F_PU4_pred[1218:1317])
            > max_error_F_PU4
        )
        > APPEARANCE_TH
    )
    check_F_PU5_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU5_label[1218:1317] - test_F_PU5_pred[1218:1317])
            > max_error_F_PU5
        )
        > APPEARANCE_TH
    )
    check_F_PU6_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU6_label[1218:1317] - test_F_PU6_pred[1218:1317])
            > max_error_F_PU6
        )
        > APPEARANCE_TH
    )
    check_F_PU7_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU7_label[1218:1317] - test_F_PU7_pred[1218:1317])
            > max_error_F_PU7
        )
        > APPEARANCE_TH
    )
    check_F_PU8_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU8_label[1218:1317] - test_F_PU8_pred[1218:1317])
            > max_error_F_PU8
        )
        > APPEARANCE_TH
    )
    check_F_PU9_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU9_label[1218:1317] - test_F_PU9_pred[1218:1317])
            > max_error_F_PU9
        )
        > APPEARANCE_TH
    )
    check_F_PU10_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU10_label[1218:1317] - test_F_PU10_pred[1218:1317])
            > max_error_F_PU10
        )
        > APPEARANCE_TH
    )
    check_F_PU11_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU11_label[1218:1317] - test_F_PU11_pred[1218:1317])
            > max_error_F_PU11
        )
        > APPEARANCE_TH
    )
    check_F_V2_a1 = (
        np.count_nonzero(
            np.abs(test_F_V2_label[1218:1317] - test_F_V2_pred[1218:1317])
            > max_error_F_V2
        )
        > APPEARANCE_TH
    )
    check_P_J280_a1 = (
        np.count_nonzero(
            np.abs(test_P_J280_label[1218:1317] - test_P_J280_pred[1218:1317])
            > max_error_P_J280
        )
        > APPEARANCE_TH
    )
    check_P_J269_a1 = (
        np.count_nonzero(
            np.abs(test_P_J269_label[1218:1317] - test_P_J269_pred[1218:1317])
            > max_error_P_J269
        )
        > APPEARANCE_TH
    )
    check_P_J300_a1 = (
        np.count_nonzero(
            np.abs(test_P_J300_label[1218:1317] - test_P_J300_pred[1218:1317])
            > max_error_P_J300
        )
        > APPEARANCE_TH
    )
    check_P_J256_a1 = (
        np.count_nonzero(
            np.abs(test_P_J256_label[1218:1317] - test_P_J256_pred[1218:1317])
            > max_error_P_J256
        )
        > APPEARANCE_TH
    )
    check_P_J289_a1 = (
        np.count_nonzero(
            np.abs(test_P_J289_label[1218:1317] - test_P_J289_pred[1218:1317])
            > max_error_P_J289
        )
        > APPEARANCE_TH
    )
    check_P_J415_a1 = (
        np.count_nonzero(
            np.abs(test_P_J415_label[1218:1317] - test_P_J415_pred[1218:1317])
            > max_error_P_J415
        )
        > APPEARANCE_TH
    )
    check_P_J302_a1 = (
        np.count_nonzero(
            np.abs(test_P_J302_label[1218:1317] - test_P_J302_pred[1218:1317])
            > max_error_P_J302
        )
        > APPEARANCE_TH
    )
    check_P_J306_a1 = (
        np.count_nonzero(
            np.abs(test_P_J306_label[1218:1317] - test_P_J306_pred[1218:1317])
            > max_error_P_J306
        )
        > APPEARANCE_TH
    )
    check_P_J307_a1 = (
        np.count_nonzero(
            np.abs(test_P_J307_label[1218:1317] - test_P_J307_pred[1218:1317])
            > max_error_P_J307
        )
        > APPEARANCE_TH
    )
    check_P_J317_a1 = (
        np.count_nonzero(
            np.abs(test_P_J317_label[1218:1317] - test_P_J317_pred[1218:1317])
            > max_error_P_J317
        )
        > APPEARANCE_TH
    )
    check_P_J14_a1 = (
        np.count_nonzero(
            np.abs(test_P_J14_label[1218:1317] - test_P_J14_pred[1218:1317])
            > max_error_P_J14
        )
        > APPEARANCE_TH
    )
    check_P_J422_a1 = (
        np.count_nonzero(
            np.abs(test_P_J422_label[1218:1317] - test_P_J422_pred[1218:1317])
            > max_error_P_J422
        )
        > APPEARANCE_TH
    )

    print("\n----------------------------------------------")
    print("ATTACK 5 COMPROMISED SENSORS DETECTION REPORT:")
    print(
        f"TANKS: L_T1 Compromised? {check_L_T1_a1} || L_T2 Compromised? {check_L_T2_a1} || L_T3 Compromised? {check_L_T3_a1} || L_T4 Compromised? {check_L_T4_a1} || L_T5 Compromised? {check_L_T5_a1} || L_T6 Compromised? {check_L_T6_a1} || L_T7 Compromised? {check_L_T7_a1}"
    )
    print(
        f"PUMPS: F_PU1 Compromised? {check_F_PU1_a1} || F_PU2 Compromised? {check_F_PU2_a1} || F_PU3 Compromised? {check_F_PU3_a1} || F_PU4 Compromised? {check_F_PU4_a1} || F_PU5 Compromised? {check_F_PU5_a1} || F_PU6 Compromised? {check_F_PU6_a1} || F_PU7 Compromised? {check_F_PU7_a1} || F_PU8 Compromised? {check_F_PU8_a1} || F_PU9 Compromised? {check_F_PU9_a1} || F_PU10 Compromised? {check_F_PU10_a1} || F_PU11 Compromised? {check_F_PU11_a1}"
    )
    print(f"VALVE: F_V2 Compromised? {check_F_V2_a1}")
    print(
        f"PRESSURE PUMPS: P_J280 Compromised? {check_P_J280_a1} || P_J269 Compromised? {check_P_J269_a1} || P_J300 Compromised? {check_P_J300_a1} || P_J256 Compromised? {check_P_J256_a1} || P_J289 Compromised? {check_P_J289_a1} || P_J415 Compromised? {check_P_J415_a1} || P_J302 Compromised? {check_P_J302_a1} || P_J306 Compromised? {check_P_J306_a1} || P_J307 Compromised? {check_P_J307_a1} || P_J317 Compromised? {check_P_J317_a1} || P_J14 Compromised? {check_P_J14_a1} || P_J422 Compromised? {check_P_J422_a1}"
    )

    # Count the number of absolute abnormal error vs max normal error in attack 6
    check_L_T1_a1 = (
        np.count_nonzero(
            np.abs(test_L_T1_label[1572:1637] - test_L_T1_pred[1572:1637])
            > max_error_L_T1
        )
        > APPEARANCE_TH
    )
    check_L_T2_a1 = (
        np.count_nonzero(
            np.abs(test_L_T2_label[1572:1637] - test_L_T2_pred[1572:1637])
            > max_error_L_T2
        )
        > APPEARANCE_TH
    )
    check_L_T3_a1 = (
        np.count_nonzero(
            np.abs(test_L_T3_label[1572:1637] - test_L_T3_pred[1572:1637])
            > max_error_L_T3
        )
        > APPEARANCE_TH
    )
    check_L_T4_a1 = (
        np.count_nonzero(
            np.abs(test_L_T4_label[1572:1637] - test_L_T4_pred[1572:1637])
            > max_error_L_T4
        )
        > APPEARANCE_TH
    )
    check_L_T5_a1 = (
        np.count_nonzero(
            np.abs(test_L_T5_label[1572:1637] - test_L_T5_pred[1572:1637])
            > max_error_L_T5
        )
        > APPEARANCE_TH
    )
    check_L_T6_a1 = (
        np.count_nonzero(
            np.abs(test_L_T6_label[1572:1637] - test_L_T6_pred[1572:1637])
            > max_error_L_T6
        )
        > APPEARANCE_TH
    )
    check_L_T7_a1 = (
        np.count_nonzero(
            np.abs(test_L_T7_label[1572:1637] - test_L_T7_pred[1572:1637])
            > max_error_L_T7
        )
        > APPEARANCE_TH
    )
    check_F_PU1_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU1_label[1572:1637] - test_F_PU1_pred[1572:1637])
            > max_error_F_PU1
        )
        > APPEARANCE_TH
    )
    check_F_PU2_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU2_label[1572:1637] - test_F_PU2_pred[1572:1637])
            > max_error_F_PU2
        )
        > APPEARANCE_TH
    )
    check_F_PU3_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU3_label[1572:1637] - test_F_PU3_pred[1572:1637])
            > max_error_F_PU3
        )
        > APPEARANCE_TH
    )
    check_F_PU4_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU4_label[1572:1637] - test_F_PU4_pred[1572:1637])
            > max_error_F_PU4
        )
        > APPEARANCE_TH
    )
    check_F_PU5_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU5_label[1572:1637] - test_F_PU5_pred[1572:1637])
            > max_error_F_PU5
        )
        > APPEARANCE_TH
    )
    check_F_PU6_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU6_label[1572:1637] - test_F_PU6_pred[1572:1637])
            > max_error_F_PU6
        )
        > APPEARANCE_TH
    )
    check_F_PU7_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU7_label[1572:1637] - test_F_PU7_pred[1572:1637])
            > max_error_F_PU7
        )
        > APPEARANCE_TH
    )
    check_F_PU8_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU8_label[1572:1637] - test_F_PU8_pred[1572:1637])
            > max_error_F_PU8
        )
        > APPEARANCE_TH
    )
    check_F_PU9_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU9_label[1572:1637] - test_F_PU9_pred[1572:1637])
            > max_error_F_PU9
        )
        > APPEARANCE_TH
    )
    check_F_PU10_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU10_label[1572:1637] - test_F_PU10_pred[1572:1637])
            > max_error_F_PU10
        )
        > APPEARANCE_TH
    )
    check_F_PU11_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU11_label[1572:1637] - test_F_PU11_pred[1572:1637])
            > max_error_F_PU11
        )
        > APPEARANCE_TH
    )
    check_F_V2_a1 = (
        np.count_nonzero(
            np.abs(test_F_V2_label[1572:1637] - test_F_V2_pred[1572:1637])
            > max_error_F_V2
        )
        > APPEARANCE_TH
    )
    check_P_J280_a1 = (
        np.count_nonzero(
            np.abs(test_P_J280_label[1572:1637] - test_P_J280_pred[1572:1637])
            > max_error_P_J280
        )
        > APPEARANCE_TH
    )
    check_P_J269_a1 = (
        np.count_nonzero(
            np.abs(test_P_J269_label[1572:1637] - test_P_J269_pred[1572:1637])
            > max_error_P_J269
        )
        > APPEARANCE_TH
    )
    check_P_J300_a1 = (
        np.count_nonzero(
            np.abs(test_P_J300_label[1572:1637] - test_P_J300_pred[1572:1637])
            > max_error_P_J300
        )
        > APPEARANCE_TH
    )
    check_P_J256_a1 = (
        np.count_nonzero(
            np.abs(test_P_J256_label[1572:1637] - test_P_J256_pred[1572:1637])
            > max_error_P_J256
        )
        > APPEARANCE_TH
    )
    check_P_J289_a1 = (
        np.count_nonzero(
            np.abs(test_P_J289_label[1572:1637] - test_P_J289_pred[1572:1637])
            > max_error_P_J289
        )
        > APPEARANCE_TH
    )
    check_P_J415_a1 = (
        np.count_nonzero(
            np.abs(test_P_J415_label[1572:1637] - test_P_J415_pred[1572:1637])
            > max_error_P_J415
        )
        > APPEARANCE_TH
    )
    check_P_J302_a1 = (
        np.count_nonzero(
            np.abs(test_P_J302_label[1572:1637] - test_P_J302_pred[1572:1637])
            > max_error_P_J302
        )
        > APPEARANCE_TH
    )
    check_P_J306_a1 = (
        np.count_nonzero(
            np.abs(test_P_J306_label[1572:1637] - test_P_J306_pred[1572:1637])
            > max_error_P_J306
        )
        > APPEARANCE_TH
    )
    check_P_J307_a1 = (
        np.count_nonzero(
            np.abs(test_P_J307_label[1572:1637] - test_P_J307_pred[1572:1637])
            > max_error_P_J307
        )
        > APPEARANCE_TH
    )
    check_P_J317_a1 = (
        np.count_nonzero(
            np.abs(test_P_J317_label[1572:1637] - test_P_J317_pred[1572:1637])
            > max_error_P_J317
        )
        > APPEARANCE_TH
    )
    check_P_J14_a1 = (
        np.count_nonzero(
            np.abs(test_P_J14_label[1572:1637] - test_P_J14_pred[1572:1637])
            > max_error_P_J14
        )
        > APPEARANCE_TH
    )
    check_P_J422_a1 = (
        np.count_nonzero(
            np.abs(test_P_J422_label[1572:1637] - test_P_J422_pred[1572:1637])
            > max_error_P_J422
        )
        > APPEARANCE_TH
    )

    print("\n----------------------------------------------")
    print("ATTACK 6 COMPROMISED SENSORS DETECTION REPORT:")
    print(
        f"TANKS: L_T1 Compromised? {check_L_T1_a1} || L_T2 Compromised? {check_L_T2_a1} || L_T3 Compromised? {check_L_T3_a1} || L_T4 Compromised? {check_L_T4_a1} || L_T5 Compromised? {check_L_T5_a1} || L_T6 Compromised? {check_L_T6_a1} || L_T7 Compromised? {check_L_T7_a1}"
    )
    print(
        f"PUMPS: F_PU1 Compromised? {check_F_PU1_a1} || F_PU2 Compromised? {check_F_PU2_a1} || F_PU3 Compromised? {check_F_PU3_a1} || F_PU4 Compromised? {check_F_PU4_a1} || F_PU5 Compromised? {check_F_PU5_a1} || F_PU6 Compromised? {check_F_PU6_a1} || F_PU7 Compromised? {check_F_PU7_a1} || F_PU8 Compromised? {check_F_PU8_a1} || F_PU9 Compromised? {check_F_PU9_a1} || F_PU10 Compromised? {check_F_PU10_a1} || F_PU11 Compromised? {check_F_PU11_a1}"
    )
    print(f"VALVE: F_V2 Compromised? {check_F_V2_a1}")
    print(
        f"PRESSURE PUMPS: P_J280 Compromised? {check_P_J280_a1} || P_J269 Compromised? {check_P_J269_a1} || P_J300 Compromised? {check_P_J300_a1} || P_J256 Compromised? {check_P_J256_a1} || P_J289 Compromised? {check_P_J289_a1} || P_J415 Compromised? {check_P_J415_a1} || P_J302 Compromised? {check_P_J302_a1} || P_J306 Compromised? {check_P_J306_a1} || P_J307 Compromised? {check_P_J307_a1} || P_J317 Compromised? {check_P_J317_a1} || P_J14 Compromised? {check_P_J14_a1} || P_J422 Compromised? {check_P_J422_a1}"
    )

    # Count the number of absolute abnormal error vs max normal error in attack 7
    check_L_T1_a1 = (
        np.count_nonzero(
            np.abs(test_L_T1_label[1929:1950] - test_L_T1_pred[1929:1950])
            > max_error_L_T1
        )
        > APPEARANCE_TH
    )
    check_L_T2_a1 = (
        np.count_nonzero(
            np.abs(test_L_T2_label[1929:1950] - test_L_T2_pred[1929:1950])
            > max_error_L_T2
        )
        > APPEARANCE_TH
    )
    check_L_T3_a1 = (
        np.count_nonzero(
            np.abs(test_L_T3_label[1929:1950] - test_L_T3_pred[1929:1950])
            > max_error_L_T3
        )
        > APPEARANCE_TH
    )
    check_L_T4_a1 = (
        np.count_nonzero(
            np.abs(test_L_T4_label[1929:1950] - test_L_T4_pred[1929:1950])
            > max_error_L_T4
        )
        > APPEARANCE_TH
    )
    check_L_T5_a1 = (
        np.count_nonzero(
            np.abs(test_L_T5_label[1929:1950] - test_L_T5_pred[1929:1950])
            > max_error_L_T5
        )
        > APPEARANCE_TH
    )
    check_L_T6_a1 = (
        np.count_nonzero(
            np.abs(test_L_T6_label[1929:1950] - test_L_T6_pred[1929:1950])
            > max_error_L_T6
        )
        > APPEARANCE_TH
    )
    check_L_T7_a1 = (
        np.count_nonzero(
            np.abs(test_L_T7_label[1929:1950] - test_L_T7_pred[1929:1950])
            > max_error_L_T7
        )
        > APPEARANCE_TH
    )
    check_F_PU1_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU1_label[1929:1950] - test_F_PU1_pred[1929:1950])
            > max_error_F_PU1
        )
        > APPEARANCE_TH
    )
    check_F_PU2_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU2_label[1929:1950] - test_F_PU2_pred[1929:1950])
            > max_error_F_PU2
        )
        > APPEARANCE_TH
    )
    check_F_PU3_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU3_label[1929:1950] - test_F_PU3_pred[1929:1950])
            > max_error_F_PU3
        )
        > APPEARANCE_TH
    )
    check_F_PU4_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU4_label[1929:1950] - test_F_PU4_pred[1929:1950])
            > max_error_F_PU4
        )
        > APPEARANCE_TH
    )
    check_F_PU5_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU5_label[1929:1950] - test_F_PU5_pred[1929:1950])
            > max_error_F_PU5
        )
        > APPEARANCE_TH
    )
    check_F_PU6_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU6_label[1929:1950] - test_F_PU6_pred[1929:1950])
            > max_error_F_PU6
        )
        > APPEARANCE_TH
    )
    check_F_PU7_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU7_label[1929:1950] - test_F_PU7_pred[1929:1950])
            > max_error_F_PU7
        )
        > APPEARANCE_TH
    )
    check_F_PU8_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU8_label[1929:1950] - test_F_PU8_pred[1929:1950])
            > max_error_F_PU8
        )
        > APPEARANCE_TH
    )
    check_F_PU9_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU9_label[1929:1950] - test_F_PU9_pred[1929:1950])
            > max_error_F_PU9
        )
        > APPEARANCE_TH
    )
    check_F_PU10_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU10_label[1929:1950] - test_F_PU10_pred[1929:1950])
            > max_error_F_PU10
        )
        > APPEARANCE_TH
    )
    check_F_PU11_a1 = (
        np.count_nonzero(
            np.abs(test_F_PU11_label[1929:1950] - test_F_PU11_pred[1929:1950])
            > max_error_F_PU11
        )
        > APPEARANCE_TH
    )
    check_F_V2_a1 = (
        np.count_nonzero(
            np.abs(test_F_V2_label[1929:1950] - test_F_V2_pred[1929:1950])
            > max_error_F_V2
        )
        > APPEARANCE_TH
    )
    check_P_J280_a1 = (
        np.count_nonzero(
            np.abs(test_P_J280_label[1929:1950] - test_P_J280_pred[1929:1950])
            > max_error_P_J280
        )
        > APPEARANCE_TH
    )
    check_P_J269_a1 = (
        np.count_nonzero(
            np.abs(test_P_J269_label[1929:1950] - test_P_J269_pred[1929:1950])
            > max_error_P_J269
        )
        > APPEARANCE_TH
    )
    check_P_J300_a1 = (
        np.count_nonzero(
            np.abs(test_P_J300_label[1929:1950] - test_P_J300_pred[1929:1950])
            > max_error_P_J300
        )
        > APPEARANCE_TH
    )
    check_P_J256_a1 = (
        np.count_nonzero(
            np.abs(test_P_J256_label[1929:1950] - test_P_J256_pred[1929:1950])
            > max_error_P_J256
        )
        > APPEARANCE_TH
    )
    check_P_J289_a1 = (
        np.count_nonzero(
            np.abs(test_P_J289_label[1929:1950] - test_P_J289_pred[1929:1950])
            > max_error_P_J289
        )
        > APPEARANCE_TH
    )
    check_P_J415_a1 = (
        np.count_nonzero(
            np.abs(test_P_J415_label[1929:1950] - test_P_J415_pred[1929:1950])
            > max_error_P_J415
        )
        > APPEARANCE_TH
    )
    check_P_J302_a1 = (
        np.count_nonzero(
            np.abs(test_P_J302_label[1929:1950] - test_P_J302_pred[1929:1950])
            > max_error_P_J302
        )
        > APPEARANCE_TH
    )
    check_P_J306_a1 = (
        np.count_nonzero(
            np.abs(test_P_J306_label[1929:1950] - test_P_J306_pred[1929:1950])
            > max_error_P_J306
        )
        > APPEARANCE_TH
    )
    check_P_J307_a1 = (
        np.count_nonzero(
            np.abs(test_P_J307_label[1929:1950] - test_P_J307_pred[1929:1950])
            > max_error_P_J307
        )
        > APPEARANCE_TH
    )
    check_P_J317_a1 = (
        np.count_nonzero(
            np.abs(test_P_J317_label[1929:1950] - test_P_J317_pred[1929:1950])
            > max_error_P_J317
        )
        > APPEARANCE_TH
    )
    check_P_J14_a1 = (
        np.count_nonzero(
            np.abs(test_P_J14_label[1929:1950] - test_P_J14_pred[1929:1950])
            > max_error_P_J14
        )
        > APPEARANCE_TH
    )
    check_P_J422_a1 = (
        np.count_nonzero(
            np.abs(test_P_J422_label[1929:1950] - test_P_J422_pred[1929:1950])
            > max_error_P_J422
        )
        > APPEARANCE_TH
    )

    print("\n----------------------------------------------")
    print("ATTACK 7 COMPROMISED SENSORS DETECTION REPORT:")
    print(
        f"TANKS: L_T1 Compromised? {check_L_T1_a1} || L_T2 Compromised? {check_L_T2_a1} || L_T3 Compromised? {check_L_T3_a1} || L_T4 Compromised? {check_L_T4_a1} || L_T5 Compromised? {check_L_T5_a1} || L_T6 Compromised? {check_L_T6_a1} || L_T7 Compromised? {check_L_T7_a1}"
    )
    print(
        f"PUMPS: F_PU1 Compromised? {check_F_PU1_a1} || F_PU2 Compromised? {check_F_PU2_a1} || F_PU3 Compromised? {check_F_PU3_a1} || F_PU4 Compromised? {check_F_PU4_a1} || F_PU5 Compromised? {check_F_PU5_a1} || F_PU6 Compromised? {check_F_PU6_a1} || F_PU7 Compromised? {check_F_PU7_a1} || F_PU8 Compromised? {check_F_PU8_a1} || F_PU9 Compromised? {check_F_PU9_a1} || F_PU10 Compromised? {check_F_PU10_a1} || F_PU11 Compromised? {check_F_PU11_a1}"
    )
    print(f"VALVE: F_V2 Compromised? {check_F_V2_a1}")
    print(
        f"PRESSURE PUMPS: P_J280 Compromised? {check_P_J280_a1} || P_J269 Compromised? {check_P_J269_a1} || P_J300 Compromised? {check_P_J300_a1} || P_J256 Compromised? {check_P_J256_a1} || P_J289 Compromised? {check_P_J289_a1} || P_J415 Compromised? {check_P_J415_a1} || P_J302 Compromised? {check_P_J302_a1} || P_J306 Compromised? {check_P_J306_a1} || P_J307 Compromised? {check_P_J307_a1} || P_J317 Compromised? {check_P_J317_a1} || P_J14 Compromised? {check_P_J14_a1} || P_J422 Compromised? {check_P_J422_a1}"
    )


if __name__ == "__main__":
    localization()
