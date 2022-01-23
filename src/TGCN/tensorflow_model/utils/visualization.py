# Reference: https://github.com/lehaifeng/T-GCN/blob/master/T-GCN/T-GCN-TensorFlow/visualization.py

import matplotlib.pyplot as plt


def plot_result(test_result, test_label, path):
    """Plots test datasets ground truth labels vs predictions on ground truth features

    Args:
        test_result: Predictions on testing features.
        test_label: Testing labels.
        path: Saving image results in path.
    """
    # all test result visualization
    fig1 = plt.figure(figsize=(7, 1.5))

    #    ax1 = fig1.add_subplot(1,1,1)
    a_pred = test_result[:, 0]
    a_true = test_label[:, 0]
    plt.plot(a_pred, "r-", label="prediction")
    plt.plot(a_true, "b-", label="true")
    plt.legend(loc="best", fontsize=10)
    plt.savefig(path + "/test_all.png")
    plt.show()

    # oneday test result visualization
    fig1 = plt.figure(figsize=(7, 1.5))

    #    ax1 = fig1.add_subplot(1,1,1)
    a_pred = test_result[0:96, 0]
    a_true = test_label[0:96, 0]
    plt.plot(a_pred, "r-", label="prediction")
    plt.plot(a_true, "b-", label="true")
    plt.legend(loc="best", fontsize=10)
    plt.savefig(path + "/test_oneday.png")
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
