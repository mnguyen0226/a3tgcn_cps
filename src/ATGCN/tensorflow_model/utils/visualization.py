# Reference: https://github.com/lehaifeng/T-GCN/blob/master/T-GCN/T-GCN-TensorFlow/visualization.py

import matplotlib.pyplot as plt


def plot_result_tank(test_result, test_label, path, hour=24 * 7):
    """Plots test datasets ground truth labels vs predictions on ground truth features

    Args:
        test_result: Predictions on testing features.
        test_label: Testing labels.
        path: Saving image results in path.
    """
    # plot 1 day for all water levels from tank 1-7
    fig1, axs = plt.subplots(4, 2)
    hour_cal = float(hour) / 24.0
    fig_name = "Water Levels of Tanks # 1-7 in " + str(hour_cal) + " Day(s)"
    fig1.suptitle(fig_name)

    # tank 1
    tank1_pred = test_result[0:hour, 0]  # 1 days for 1 cols
    tank1_true = test_label[0:hour, 0]  # 1 days for 1 cols
    axs[0, 0].plot(tank1_pred, "r-", label="tank # 1 prediction")
    axs[0, 0].plot(tank1_true, "b-", label="tank # 1 label")
    axs[0, 0].legend(loc="upper left", fontsize=10)
    sub_hour_1 = "Water Levels of Tank # 1 - L_T1 in " + str(hour) + " Hours"
    axs[0, 0].set_title(sub_hour_1)

    # tank 2
    tank2_pred = test_result[0:hour, 1]  # 1 days for 1 cols
    tank2_true = test_label[0:hour, 1]  # 1 days for 1 cols
    axs[1, 0].plot(tank2_pred, "r-", label="tank # 2 prediction")
    axs[1, 0].plot(tank2_true, "b-", label="tank # 2 label")
    axs[1, 0].legend(loc="upper left", fontsize=10)
    sub_hour_2 = "Water Levels of Tank # 2 - L_T2 in " + str(hour) + " Hours"
    axs[1, 0].set_title(sub_hour_2)

    # tank 3
    tank3_pred = test_result[0:hour, 2]  # 1 days for 1 cols
    tank3_true = test_label[0:hour, 2]  # 1 days for 1 cols
    axs[2, 0].plot(tank3_pred, "r-", label="tank # 3 prediction")
    axs[2, 0].plot(tank3_true, "b-", label="tank # 3 label")
    axs[2, 0].legend(loc="upper left", fontsize=10)
    sub_hour_3 = "Water Levels of Tank # 3 - L_T3 in " + str(hour) + " Hours"
    axs[2, 0].set_title(sub_hour_3)

    # tank 4
    tank4_pred = test_result[0:hour, 3]  # 1 days for 1 cols
    tank4_true = test_label[0:hour, 3]  # 1 days for 1 cols
    axs[3, 0].plot(tank4_pred, "r-", label="tank # 4 prediction")
    axs[3, 0].plot(tank4_true, "b-", label="tank # 4 label")
    axs[3, 0].legend(loc="upper left", fontsize=10)
    sub_hour_4 = "Water Levels of Tank # 4 - L_T4 in" + str(hour) + " Hours"
    axs[3, 0].set_title(sub_hour_4)

    # tank 5
    tank5_pred = test_result[0:hour, 4]  # 1 days for 1 cols
    tank5_true = test_label[0:hour, 4]  # 1 days for 1 cols
    axs[0, 1].plot(tank5_pred, "r-", label="tank # 5 prediction")
    axs[0, 1].plot(tank5_true, "b-", label="tank # 5 label")
    axs[0, 1].legend(loc="upper left", fontsize=10)
    sub_hour_5 = "Water Levels of Tank # 5 - L_T5 in " + str(hour) + " Hours"
    axs[0, 1].set_title(sub_hour_5)

    # tank 6
    tank6_pred = test_result[0:hour, 5]  # 1 days for 1 cols
    tank6_true = test_label[0:hour, 5]  # 1 days for 1 cols
    axs[1, 1].plot(tank6_pred, "r-", label="tank # 6 prediction")
    axs[1, 1].plot(tank6_true, "b-", label="tank # 6 label")
    axs[1, 1].legend(loc="upper left", fontsize=10)
    sub_hour_6 = "Water Levels of Tank # 6 - L_T6 in " + str(hour) + " Hours"
    axs[1, 1].set_title(sub_hour_6)

    # tank 7
    tank7_pred = test_result[0:hour, 6]  # 1 days for 1 cols
    tank7_true = test_label[0:hour, 6]  # 1 days for 1 cols
    axs[2, 1].plot(tank7_pred, "r-", label="tank # 7 prediction")
    axs[2, 1].plot(tank7_true, "b-", label="tank # 7 label")
    axs[2, 1].legend(loc="upper left", fontsize=10)
    sub_hour_7 = "Water Levels of Tank # 7 L_T7 in " + str(hour) + " Hours"
    axs[2, 1].set_title(sub_hour_7)

    for ax in axs.flat:
        ax.set(xlabel="Number of Hours", ylabel="Water Level (m)")

    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()

    # set the spacing between subplots
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.9
    )
    name = "/test_" + str(hour) + "_tank_water_level.png"
    plt.savefig(path + name)
    plt.show()


def plot_result_pump(test_result, test_label, path, hour=24 * 7):
    """Plots test datasets ground truth labels vs predictions on ground truth features

    Args:
        test_result: Predictions on testing features.
        test_label: Testing labels.
        path: Saving image results in path.
    """
    # plot 1 day for all flowrate from pump 8-18
    fig1, axs = plt.subplots(6, 2)
    hour_cal = float(hour) / 24.0
    fig_name = "Flow Rates of Pumps # 1-11 in " + str(hour_cal) + " Day(s)"
    fig1.suptitle(fig_name)

    # pump 1
    pump1_pred = test_result[0:hour, 7]  # 1 days for 1 cols
    pump1_true = test_label[0:hour, 7]  # 1 days for 1 cols
    axs[0, 0].plot(pump1_pred, "r-", label="pump # 1 prediction")
    axs[0, 0].plot(pump1_true, "b-", label="pump # 1 label")
    axs[0, 0].legend(loc="upper left", fontsize=10)
    sub_hour_1 = "Flow Rates of Pump # 1 - F_PU1 in " + str(hour) + " Hours"
    axs[0, 0].set_title(sub_hour_1)

    # pump 2
    pump2_pred = test_result[0:hour, 8]  # 1 days for 1 cols
    pump2_true = test_label[0:hour, 8]  # 1 days for 1 cols
    axs[1, 0].plot(pump2_pred, "r-", label="pump # 2 prediction")
    axs[1, 0].plot(pump2_true, "b-", label="pump # 2 label")
    axs[1, 0].legend(loc="upper left", fontsize=10)
    sub_hour_2 = "Flow Rates of Pump # 2 - F_PU2 in " + str(hour) + " Hours"
    axs[1, 0].set_title(sub_hour_2)

    # pump 3
    pump3_pred = test_result[0:hour, 9]  # 1 days for 1 cols
    pump3_true = test_label[0:hour, 9]  # 1 days for 1 cols
    axs[2, 0].plot(pump3_pred, "r-", label="pump # 3 prediction")
    axs[2, 0].plot(pump3_true, "b-", label="pump # 3 label")
    axs[2, 0].legend(loc="upper left", fontsize=10)
    sub_hour_3 = "Flow Rates of Pump # 3 - F_PU3 in " + str(hour) + " Hours"
    axs[2, 0].set_title(sub_hour_3)

    # pump 4
    pump4_pred = test_result[0:hour, 10]  # 1 days for 1 cols
    pump4_true = test_label[0:hour, 10]  # 1 days for 1 cols
    axs[3, 0].plot(pump4_pred, "r-", label="pump # 4 prediction")
    axs[3, 0].plot(pump4_pred, "b-", label="pump # 4 label")
    axs[3, 0].legend(loc="upper left", fontsize=10)
    sub_hour_4 = "Flow Rates of Pump # 4 - F_PU4 in " + str(hour) + " Hours"
    axs[3, 0].set_title(sub_hour_4)

    # pump 5
    pump5_pred = test_result[0:hour, 11]  # 1 days for 1 cols
    pump5_true = test_label[0:hour, 11]  # 1 days for 1 cols
    axs[4, 0].plot(pump5_pred, "r-", label="pump # 5 prediction")
    axs[4, 0].plot(pump5_true, "b-", label="pump # 5 label")
    axs[4, 0].legend(loc="upper left", fontsize=10)
    sub_hour_5 = "Flow Rates of Pump # 5 - F_PU5 in " + str(hour) + " Hours"
    axs[4, 0].set_title(sub_hour_5)

    # pump 6
    pump6_pred = test_result[0:hour, 12]  # 1 days for 1 cols
    pump6_true = test_label[0:hour, 12]  # 1 days for 1 cols
    axs[5, 0].plot(pump6_pred, "r-", label="pump # 6 prediction")
    axs[5, 0].plot(pump6_true, "b-", label="pump # 6 label")
    axs[5, 0].legend(loc="upper left", fontsize=10)
    sub_hour_6 = "Flow Rates of Pump # 6 - F_PU6 in " + str(hour) + " Hours"
    axs[5, 0].set_title(sub_hour_6)

    # pump 7
    pump7_pred = test_result[0:hour, 13]  # 1 days for 1 cols
    pump7_true = test_label[0:hour, 13]  # 1 days for 1 cols
    axs[0, 1].plot(pump7_pred, "r-", label="pump # 7 prediction")
    axs[0, 1].plot(pump7_true, "b-", label="pump # 7 label")
    axs[0, 1].legend(loc="upper left", fontsize=10)
    sub_hour_7 = "Flow Rates of Pump # 7 - F_PU7 in " + str(hour) + " Hours"
    axs[0, 1].set_title(sub_hour_7)

    # pump 8
    pump8_pred = test_result[0:hour, 14]  # 1 days for 1 cols
    pump8_true = test_label[0:hour, 14]  # 1 days for 1 cols
    axs[1, 1].plot(pump8_pred, "r-", label="pump # 8 prediction")
    axs[1, 1].plot(pump8_true, "b-", label="pump # 8 label")
    axs[1, 1].legend(loc="upper left", fontsize=10)
    sub_hour_8 = "Flow Rates of Pump # 8 - F_PU8 in " + str(hour) + " Hours"
    axs[1, 1].set_title(sub_hour_8)

    # pump 9
    pump9_pred = test_result[0:hour, 15]  # 1 days for 1 cols
    pump9_true = test_label[0:hour, 15]  # 1 days for 1 cols
    axs[2, 1].plot(pump9_pred, "r-", label="pump # 9 prediction")
    axs[2, 1].plot(pump9_true, "b-", label="pump # 9 label")
    axs[2, 1].legend(loc="upper left", fontsize=10)
    sub_hour_9 = "Flow Rates of Pump # 9- F_PU9 in " + str(hour) + " Hours"
    axs[2, 1].set_title(sub_hour_9)

    # pump 10
    pump10_pred = test_result[0:hour, 16]  # 1 days for 1 cols
    pump10_true = test_label[0:hour, 16]  # 1 days for 1 cols
    axs[3, 1].plot(pump10_pred, "r-", label="pump # 10 prediction")
    axs[3, 1].plot(pump10_true, "b-", label="pump # 10 label")
    axs[3, 1].legend(loc="upper left", fontsize=10)
    sub_hour_10 = "Flow Rates of Pump # 10 - F_PU10 in " + str(hour) + " Hours"
    axs[3, 1].set_title(sub_hour_10)

    # pump 11
    pump11_pred = test_result[0:hour, 17]  # 1 days for 1 cols
    pump11_true = test_label[0:hour, 17]  # 1 days for 1 cols
    axs[4, 1].plot(pump11_pred, "r-", label="pump # 11 prediction")
    axs[4, 1].plot(pump11_true, "b-", label="pump # 11 label")
    axs[4, 1].legend(loc="upper left", fontsize=10)
    sub_hour_11 = "Flow Rates of Pump # 11 - F_PU11 in " + str(hour) + " Hours"
    axs[4, 1].set_title(sub_hour_11)

    for ax in axs.flat:
        ax.set(xlabel="Number of Hours", ylabel="Flow Rate (L/s)")

    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()

    # set the spacing between subplots
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.9
    )

    name = "/test_" + str(hour) + "_pump_flow_rate.png"
    plt.savefig(path + name)
    plt.show()


def plot_result_valve(test_result, test_label, path, hour=24 * 7):
    """Plots test datasets ground truth labels vs predictions on ground truth features

    Args:
        test_result: Predictions on testing features.
        test_label: Testing labels.
        path: Saving image results in path.
    """
    fig1 = plt.figure(figsize=(7, 1.5))

    valve_pred = test_result[:hour, 18]  # all rows of the first cols
    valve_true = test_label[:hour, 18]  # all rows of the first cols
    plt.plot(valve_pred, "r-", label="valve # 2 prediction")
    plt.plot(valve_true, "b-", label="valve # 2 label")
    plt.legend(loc="upper left", fontsize=10)
    hour_cal = float(hour) / 24.0
    fig_name = "Flow Rates of Valve # 2 - F_V2 in " + str(hour_cal) + " Day(s)"
    plt.title(fig_name)
    plt.xlabel("Number of Hours")
    plt.ylabel("Flow Rate (L/s)")

    name = "/test_" + str(hour) + "_valve_flow_rate.png"

    plt.savefig(path + name)
    plt.show()


def plot_result_junction(test_result, test_label, path, hour=24 * 7):
    """Plots test datasets ground truth labels vs predictions on ground truth features

    Args:
        test_result: Predictions on testing features.
        test_label: Testing labels.
        path: Saving image results in path.
    """
    # plot 1 day for all pressure from junction 20-31
    fig1, axs = plt.subplots(6, 2)
    hour_cal = float(hour) / 24.0
    fig_name = (
        "Inlet & Outlet Pressure of Junctions # 1-12 in " + str(hour_cal) + " Day(s)"
    )
    fig1.suptitle(fig_name)

    # junction 1
    junction1_pred = test_result[0:hour, 19]  # 1 days for 1 cols
    junction2_true = test_label[0:hour, 19]  # 1 days for 1 cols
    axs[0, 0].plot(junction1_pred, "r-", label="junction # 1 prediction")
    axs[0, 0].plot(junction2_true, "b-", label="junction # 1 label")
    axs[0, 0].legend(loc="upper left", fontsize=10)
    sub_hour_1 = "Pressures of junction # 1 - P_J280 in " + str(hour) + " Hours"
    axs[0, 0].set_title(sub_hour_1)

    # junction 2
    junction2_pred = test_result[0:hour, 20]  # 1 days for 1 cols
    junction2_true = test_label[0:hour, 20]  # 1 days for 1 cols
    axs[1, 0].plot(junction2_pred, "r-", label="junction # 2 prediction")
    axs[1, 0].plot(junction2_true, "b-", label="junction # 2 label")
    axs[1, 0].legend(loc="upper left", fontsize=10)
    sub_hour_2 = "Pressures of junction # 2 - P_J269 in " + str(hour) + " Hours"
    axs[1, 0].set_title(sub_hour_2)

    # junction 3
    junction3_pred = test_result[0:hour, 21]  # 1 days for 1 cols
    junction3_true = test_label[0:hour, 21]  # 1 days for 1 cols
    axs[2, 0].plot(junction3_pred, "r-", label="junction # 3 prediction")
    axs[2, 0].plot(junction3_true, "b-", label="junction # 3 label")
    axs[2, 0].legend(loc="upper left", fontsize=10)
    sub_hour_3 = "Pressures of junction # 3 - P_J300 in " + str(hour) + " Hours"
    axs[2, 0].set_title(sub_hour_3)

    # junction 4
    junction4_pred = test_result[0:hour, 22]  # 1 days for 1 cols
    junction4_true = test_label[0:hour, 22]  # 1 days for 1 cols
    axs[3, 0].plot(junction4_pred, "r-", label="junction # 4 prediction")
    axs[3, 0].plot(junction4_true, "b-", label="junction # 4 label")
    axs[3, 0].legend(loc="upper left", fontsize=10)
    sub_hour_4 = "Pressures of junction # 4 - P_J256 in " + str(hour) + " Hours"
    axs[3, 0].set_title(sub_hour_4)

    # junction 5
    junction5_pred = test_result[0:hour, 23]  # 1 days for 1 cols
    junction5_true = test_label[0:hour, 23]  # 1 days for 1 cols
    axs[4, 0].plot(junction5_pred, "r-", label="junction # 5 prediction")
    axs[4, 0].plot(junction5_true, "b-", label="junction # 5 label")
    axs[4, 0].legend(loc="upper left", fontsize=10)
    sub_hour_5 = "Pressures of junction # 5 - P_J289 in " + str(hour) + " Hours"
    axs[4, 0].set_title(sub_hour_5)

    # junction 6
    junction6_pred = test_result[0:hour, 24]  # 1 days for 1 cols
    junction6_true = test_label[0:hour, 24]  # 1 days for 1 cols
    axs[5, 0].plot(junction6_pred, "r-", label="junction # 6 prediction")
    axs[5, 0].plot(junction6_true, "b-", label="junction # 6 label")
    axs[5, 0].legend(loc="upper left", fontsize=10)
    sub_hour_6 = "Pressures of junction # 6 - P_J415 in " + str(hour) + " Hours"
    axs[5, 0].set_title(sub_hour_6)

    # junction 7
    junction7_pred = test_result[0:hour, 25]  # 1 days for 1 cols
    junction7_true = test_label[0:hour, 25]  # 1 days for 1 cols
    axs[0, 1].plot(junction7_pred, "r-", label="junction # 7 prediction")
    axs[0, 1].plot(junction7_true, "b-", label="junction # 7 label")
    axs[0, 1].legend(loc="upper left", fontsize=10)
    sub_hour_7 = "Pressures of junction # 7 - P_J302 in " + str(hour) + " Hours"
    axs[0, 1].set_title(sub_hour_7)

    # junction 8
    junction8_pred = test_result[0:hour, 26]  # 1 days for 1 cols
    junction8_true = test_label[0:hour, 26]  # 1 days for 1 cols
    axs[1, 1].plot(junction8_pred, "r-", label="junction # 8 prediction")
    axs[1, 1].plot(junction8_true, "b-", label="junction # 8 label")
    axs[1, 1].legend(loc="upper left", fontsize=10)
    sub_hour_8 = "Pressures of junction # 8 - P_J306 in " + str(hour) + " Hours"
    axs[1, 1].set_title(sub_hour_8)

    # junction 9
    junction9_pred = test_result[0:hour, 27]  # 1 days for 1 cols
    junction9_true = test_label[0:hour, 27]  # 1 days for 1 cols
    axs[2, 1].plot(junction9_pred, "r-", label="junction # 9 prediction")
    axs[2, 1].plot(junction9_true, "b-", label="junction # 9 label")
    axs[2, 1].legend(loc="upper left", fontsize=10)
    sub_hour_9 = "Pressures of junction # 9 - P_J307 in " + str(hour) + " Hours"
    axs[2, 1].set_title(sub_hour_9)

    # junction 10
    junction10_pred = test_result[0:hour, 28]  # 1 days for 1 cols
    junction10_true = test_label[0:hour, 28]  # 1 days for 1 cols
    axs[3, 1].plot(junction10_pred, "r-", label="junction # 10 prediction")
    axs[3, 1].plot(junction10_true, "b-", label="junction # 10 label")
    axs[3, 1].legend(loc="upper left", fontsize=10)
    sub_hour_10 = "Pressures of junction # 10 - P_J317 in " + str(hour) + " Hours"
    axs[3, 1].set_title(sub_hour_10)

    # junction 11
    junction11_pred = test_result[0:hour, 29]  # 1 days for 1 cols
    junction11_true = test_label[0:hour, 29]  # 1 days for 1 cols
    axs[4, 1].plot(junction11_pred, "r-", label="junction # 11 prediction")
    axs[4, 1].plot(junction11_true, "b-", label="junction # 11 label")
    axs[4, 1].legend(loc="upper left", fontsize=10)
    sub_hour_11 = "Pressures of junction # 11 - P_J14 in " + str(hour) + " Hours"
    axs[4, 1].set_title(sub_hour_11)

    # junction 12
    junction12_pred = test_result[0:hour, 30]  # 1 days for 1 cols
    junction12_true = test_label[0:hour, 30]  # 1 days for 1 cols
    axs[5, 1].plot(junction12_pred, "r-", label="junction # 12 prediction")
    axs[5, 1].plot(junction12_true, "b-", label="junction # 12 label")
    axs[5, 1].legend(loc="upper left", fontsize=10)
    sub_hour_12 = "Pressures of junction # 12 - P_J422 in " + str(hour) + " Hours"
    axs[5, 1].set_title(sub_hour_12)

    for ax in axs.flat:
        ax.set(xlabel="Number of Hours", ylabel="Pressure (m)")

    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()

    # set the spacing between subplots
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.9
    )

    name = "/test_" + str(hour) + "_junction_pressure.png"
    plt.savefig(path + name)
    plt.show()


def plot_error(
    train_rmse, train_loss, test_rmse, test_acc, test_mae, path, plot_eval=False
):
    """Plots rmse, loss, accuracy, mae

    Args:
        train_rmse: List of root mean squared error on training dataset.
        train_loss: List of training loss.
        test_rmse: List of root-mean-squared error on testing dataset.
        test_acc: List of accuracy on testing dataset.
        test_mae: List of mean absolute error on testing dataset.
        path: Saving image results in path.
    """
    if plot_eval == False:
        # train_rmse & test_rmse
        fig1 = plt.figure(figsize=(5, 3))
        plt.plot(train_rmse, "r-", label="train_rmse")
        plt.plot(test_rmse, "b-", label="test_rmse")
        plt.legend(loc="best", fontsize=10)
        plt.savefig(path + "/rmse.png")
        plt.show()

        # train_loss & train_rmse
        fig1 = plt.figure(figsize=(5, 3))
        plt.plot(train_loss, "b-", label="train_loss")
        plt.legend(loc="best", fontsize=10)
        plt.savefig(path + "/train_loss.png")
        plt.show()

        fig1 = plt.figure(figsize=(5, 3))
        plt.plot(train_rmse, "b-", label="train_rmse")
        plt.legend(loc="best", fontsize=10)
        plt.savefig(path + "/train_rmse.png")
        plt.show()

        # test accuracy
        fig1 = plt.figure(figsize=(5, 3))
        plt.plot(test_acc, "b-", label="test_acc")
        plt.legend(loc="best", fontsize=10)
        plt.savefig(path + "/test_acc.png")
        plt.show()

        # test rmse
        fig1 = plt.figure(figsize=(5, 3))
        plt.plot(test_rmse, "b-", label="test_rmse")
        plt.legend(loc="best", fontsize=10)
        plt.savefig(path + "/test_rmse.png")
        plt.show()

        # test mae
        fig1 = plt.figure(figsize=(5, 3))
        plt.plot(test_mae, "b-", label="test_mae")
        plt.legend(loc="best", fontsize=10)
        plt.savefig(path + "/test_mae.png")
        plt.show()
    elif plot_eval == True:
        # train accuracy
        fig1 = plt.figure(figsize=(5, 3))
        plt.plot(test_acc, "b-", label="test_acc")
        plt.legend(loc="best", fontsize=10)
        plt.savefig(path + "/test_acc.png")
        plt.show()

        # train rmse
        fig1 = plt.figure(figsize=(5, 3))
        plt.plot(test_rmse, "b-", label="test_rmse")
        plt.legend(loc="best", fontsize=10)
        plt.savefig(path + "/test_rmse.png")
        plt.show()

        # train mae
        fig1 = plt.figure(figsize=(5, 3))
        plt.plot(test_mae, "b-", label="test_mae")
        plt.legend(loc="best", fontsize=10)
        plt.savefig(path + "/test_mae.png")
        plt.show()
