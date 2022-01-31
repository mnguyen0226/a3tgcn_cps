# Reference: https://github.com/lehaifeng/T-GCN/blob/master/T-GCN/T-GCN-TensorFlow/visualization.py

import matplotlib.pyplot as plt


# def plot_result(test_result, test_label, path):
#     """Plots test datasets ground truth labels vs predictions on ground truth features

#     Args:
#         test_result: Predictions on testing features.
#         test_label: Testing labels.
#         path: Saving image results in path.
#     """
#     # all test result visualization
#     # fig1 = plt.figure(figsize=(7, 1.5))
#     fig1 = plt.figure()

#     #    ax1 = fig1.add_subplot(1,1,1)
#     a_pred = test_result[:, 0] # all rows of the first cols
#     a_true = test_label[:, 0]  # all rows of the first cols
#     plt.plot(a_pred, "r-", label="prediction")
#     plt.plot(a_true, "b-", label="true")
#     plt.legend(loc="best", fontsize=10)
#     plt.savefig(path + "/test_all.png")
#     plt.show()

#     # oneday test result visualization
#     fig1 = plt.figure(figsize=(7, 1.5))

#     #    ax1 = fig1.add_subplot(1,1,1)
#     a_pred = test_result[0:96, 0] # 1 days for 1 cols
#     a_true = test_label[0:96, 0]  # 1 days for 1 cols
#     plt.plot(a_pred, "r-", label="prediction")
#     plt.plot(a_true, "b-", label="true")
#     plt.legend(loc="best", fontsize=10)
#     plt.savefig(path + "/test_oneday.png")
#     plt.show()

def plot_result_tank(test_result, test_label, path, hour = 24):
    """Plots test datasets ground truth labels vs predictions on ground truth features

    Args:
        test_result: Predictions on testing features.
        test_label: Testing labels.
        path: Saving image results in path.
    """
    # plot 1 day for all water levels from tank 1-7
    fig1, axs = plt.subplots(4,2)
    hour_cal = float(hour) / 24.0
    fig_name = "Water Levels of Tanks # 1-7 in " + str(hour_cal) + " Day(s)"
    fig1.suptitle(fig_name)
    
    # tank 1
    tank1_pred = test_result[0:hour, 0] # 1 days for 1 cols
    tank1_true = test_label[0:hour, 0]  # 1 days for 1 cols
    axs[0, 0].plot(tank1_pred, "r-", label="tank # 1 prediction")
    axs[0, 0].plot(tank1_true, "b-", label="tank # 1 label")
    axs[0, 0].legend(loc="upper left", fontsize=10)
    sub_hour_1 = "Water Levels of Tank # 1 in " + str(hour) + " Hours"
    axs[0, 0].set_title(sub_hour_1)
    
    # tank 2
    tank2_pred = test_result[0:hour, 1] # 1 days for 1 cols
    tank2_true = test_label[0:hour, 1]  # 1 days for 1 cols
    axs[1, 0].plot(tank2_pred, "r-", label="tank # 2 prediction")
    axs[1, 0].plot(tank2_true, "b-", label="tank # 2 label")
    axs[1, 0].legend(loc="upper left", fontsize=10)
    sub_hour_2 = "Water Levels of Tank # 2 in " + str(hour) + " Hours"
    axs[1, 0].set_title(sub_hour_2)

    # tank 3
    tank3_pred = test_result[0:hour, 2] # 1 days for 1 cols
    tank3_true = test_label[0:hour, 2]  # 1 days for 1 cols
    axs[2, 0].plot(tank3_pred, "r-", label="tank # 3 prediction")
    axs[2, 0].plot(tank3_true, "b-", label="tank # 3 label")
    axs[2, 0].legend(loc="upper left", fontsize=10)
    sub_hour_3 = "Water Levels of Tank # 3 in " + str(hour) + " Hours"
    axs[2, 0].set_title(sub_hour_3)
    
    # tank 4
    tank4_pred = test_result[0:hour, 3] # 1 days for 1 cols
    tank4_true = test_label[0:hour, 3]  # 1 days for 1 cols
    axs[3, 0].plot(tank4_pred, "r-", label="tank # 4 prediction")
    axs[3, 0].plot(tank4_true, "b-", label="tank # 4 label")
    axs[3, 0].legend(loc="upper left", fontsize=10)
    sub_hour_4 = "Water Levels of Tank # 4 in " + str(hour) + " Hours"
    axs[3, 0].set_title(sub_hour_4)
    
    # tank 5
    tank5_pred = test_result[0:hour, 4] # 1 days for 1 cols
    tank5_true = test_label[0:hour, 4]  # 1 days for 1 cols
    axs[0, 1].plot(tank5_pred, "r-", label="tank # 5 prediction")
    axs[0, 1].plot(tank5_true, "b-", label="tank # 5 label")
    axs[0, 1].legend(loc="upper left", fontsize=10)
    sub_hour_5 = "Water Levels of Tank # 5 in " + str(hour) + " Hours"
    axs[0, 1].set_title(sub_hour_5)
    
    # tank 6
    tank6_pred = test_result[0:hour, 5] # 1 days for 1 cols
    tank6_true = test_label[0:hour, 5]  # 1 days for 1 cols
    axs[1, 1].plot(tank6_pred, "r-", label="tank # 6 prediction")
    axs[1, 1].plot(tank6_true, "b-", label="tank # 6 label")
    axs[1, 1].legend(loc="upper left", fontsize=10)
    sub_hour_6 = "Water Levels of Tank # 6 in " + str(hour) + " Hours"
    axs[1, 1].set_title(sub_hour_6)
    
    # tank 7
    tank7_pred = test_result[0:hour, 6] # 1 days for 1 cols
    tank7_true = test_label[0:hour, 6]  # 1 days for 1 cols
    axs[2, 1].plot(tank7_pred, "r-", label="tank # 7 prediction")
    axs[2, 1].plot(tank7_true, "b-", label="tank # 7 label")
    axs[2, 1].legend(loc="upper left", fontsize=10)
    sub_hour_7 = "Water Levels of Tank # 7 in " + str(hour) + " Hours"
    axs[2, 1].set_title(sub_hour_7)
    
    # tank 8
    tank8_pred = test_result[0:hour, 7] # 1 days for 1 cols
    tank8_true = test_label[0:hour, 7]  # 1 days for 1 cols
    axs[3, 1].plot(tank8_pred, "r-", label="tank # 8 prediction")
    axs[3, 1].plot(tank8_true, "b-", label="tank # 8 label")
    axs[3, 1].legend(loc="upper left", fontsize=10)
    sub_hour_8 = "Water Levels of Tank # 8 in " + str(hour) + " Hours"
    axs[3, 1].set_title(sub_hour_8)

    for ax in axs.flat:
        ax.set(xlabel='Number of Hours', ylabel='Water Level (m)')
    
    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()
      
    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.1, 
                        hspace=0.9)  
    
    plt.savefig(path + "/test_oneday_water_level.png")
    plt.show()

def plot_result_pump(test_result, test_label, path, hour = 24):
    """Plots test datasets ground truth labels vs predictions on ground truth features

    Args:
        test_result: Predictions on testing features.
        test_label: Testing labels.
        path: Saving image results in path.
    """
    # plot 1 day for all flowrate from pump 8-18
    fig1, axs = plt.subplots(5,2)
    hour_cal = float(hour) / 24.0
    fig_name = "Flow Rates of Pumps # 8-18 in " + str(hour_cal) + " Day(s)"
    fig1.suptitle(fig_name)
    
    # pump 1
    pump1_pred = test_result[0:hour, 8] # 1 days for 1 cols
    pump1_true = test_label[0:hour, 8]  # 1 days for 1 cols
    axs[0, 0].plot(pump1_pred, "r-", label="pump # 1 prediction")
    axs[0, 0].plot(pump1_true, "b-", label="pump # 1 label")
    axs[0, 0].legend(loc="upper left", fontsize=10)
    sub_hour_1 = "Flow Rates of Pump # 1 in " + str(hour) + " Hours"
    axs[0, 0].set_title(sub_hour_1)
    
    # pump 2
    pump2_pred = test_result[0:hour, 9] # 1 days for 1 cols
    pump2_true = test_label[0:hour, 9]  # 1 days for 1 cols
    axs[1, 0].plot(pump2_pred, "r-", label="pump # 2 prediction")
    axs[1, 0].plot(pump2_true, "b-", label="pump # 2 label")
    axs[1, 0].legend(loc="upper left", fontsize=10)
    sub_hour_2 = "Flow Rates of Pump # 2 in " + str(hour) + " Hours"
    axs[1, 0].set_title(sub_hour_2)

    # pump 3
    pump3_pred = test_result[0:hour, 10] # 1 days for 1 cols
    pump3_true = test_label[0:hour, 10]  # 1 days for 1 cols
    axs[2, 0].plot(pump3_pred, "r-", label="pump # 3 prediction")
    axs[2, 0].plot(pump3_true, "b-", label="pump # 3 label")
    axs[2, 0].legend(loc="upper left", fontsize=10)
    sub_hour_3 = "Flow Rates of Pump # 3 in " + str(hour) + " Hours"
    axs[2, 0].set_title(sub_hour_3)
    
    # pump 4
    pump4_pred = test_result[0:hour, 11] # 1 days for 1 cols
    pump4_true = test_label[0:hour, 11]  # 1 days for 1 cols
    axs[3, 0].plot(pump4_pred, "r-", label="pump # 4 prediction")
    axs[3, 0].plot(pump4_true, "b-", label="pump # 4 label")
    axs[3, 0].legend(loc="upper left", fontsize=10)
    sub_hour_4 = "Flow Rates of Pump # 4 in " + str(hour) + " Hours"
    axs[3, 0].set_title(sub_hour_4)
    
    # pump 5
    pump5_pred = test_result[0:hour, 12] # 1 days for 1 cols
    pump5_true = test_label[0:hour, 12]  # 1 days for 1 cols
    axs[4, 0].plot(pump5_pred, "r-", label="pump # 5 prediction")
    axs[4, 0].plot(pump5_true, "b-", label="pump # 5 label")
    axs[4, 0].legend(loc="upper left", fontsize=10)
    sub_hour_5 = "Flow Rates of Pump # 5 in " + str(hour) + " Hours"
    axs[4, 0].set_title(sub_hour_5)
    
    # pump 6
    pump6_pred = test_result[0:hour, 13] # 1 days for 1 cols
    pump6_true = test_label[0:hour, 13]  # 1 days for 1 cols
    axs[0, 1].plot(pump6_pred, "r-", label="pump # 6 prediction")
    axs[0, 1].plot(pump6_true, "b-", label="pump # 6 label")
    axs[0, 1].legend(loc="upper left", fontsize=10)
    sub_hour_6 = "Flow Rates of Pump # 6 in " + str(hour) + " Hours"
    axs[0, 1].set_title(sub_hour_6)
    
    # pump 7
    pump7_pred = test_result[0:hour, 14] # 1 days for 1 cols
    pump7_true = test_label[0:hour, 14]  # 1 days for 1 cols
    axs[1, 1].plot(pump7_pred, "r-", label="pump # 7 prediction")
    axs[1, 1].plot(pump7_true, "b-", label="pump # 7 label")
    axs[1, 1].legend(loc="upper left", fontsize=10)
    sub_hour_7 = "Flow Rates of Pump # 7 in " + str(hour) + " Hours"
    axs[1, 1].set_title(sub_hour_7)
    
    # pump 8
    pump8_pred = test_result[0:hour, 15] # 1 days for 1 cols
    pump8_true = test_label[0:hour, 15]  # 1 days for 1 cols
    axs[2, 1].plot(pump8_pred, "r-", label="pump # 8 prediction")
    axs[2, 1].plot(pump8_true, "b-", label="pump # 8 label")
    axs[2, 1].legend(loc="upper left", fontsize=10)
    sub_hour_8 = "Flow Rates of Pump # 8 in " + str(hour) + " Hours"
    axs[2, 1].set_title(sub_hour_8)

    # pump 9
    pump7_pred = test_result[0:hour, 16] # 1 days for 1 cols
    pump7_true = test_label[0:hour, 16]  # 1 days for 1 cols
    axs[3, 1].plot(pump7_pred, "r-", label="pump # 7 prediction")
    axs[3, 1].plot(pump7_true, "b-", label="pump # 7 label")
    axs[3, 1].legend(loc="upper left", fontsize=10)
    sub_hour_7 = "Flow Rates of Pump # 7 in " + str(hour) + " Hours"
    axs[3, 1].set_title(sub_hour_7)
    
    # pump 10
    pump8_pred = test_result[0:hour, 17] # 1 days for 1 cols
    pump8_true = test_label[0:hour, 17]  # 1 days for 1 cols
    axs[4, 1].plot(pump8_pred, "r-", label="pump # 8 prediction")
    axs[4, 1].plot(pump8_true, "b-", label="pump # 8 label")
    axs[4, 1].legend(loc="upper left", fontsize=10)
    sub_hour_8 = "Flow Rates of Pump # 8 in " + str(hour) + " Hours"
    axs[4, 1].set_title(sub_hour_8)

    for ax in axs.flat:
        ax.set(xlabel='Number of Hours', ylabel='Water Level (m)')
    
    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()
      
    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.1, 
                        hspace=0.9)  
    
    plt.savefig(path + "/test_oneday_flow_rate.png")
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
