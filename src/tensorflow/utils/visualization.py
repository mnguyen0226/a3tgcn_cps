# Reference: https://github.com/lehaifeng/T-GCN/blob/master/T-GCN/T-GCN-TensorFlow/visualization.py

import matplotlib.pyplot as plt


def plot_result(test_result, test_label, path="results"):
    """Plots test datasets ground truth labels vs predictions on ground truth features

    Args:
        test_result ([type]): predictions on testing features
        test_label ([type]): testing labels
        path ([type]): saving image results
    """
    # all test result visualization
    fig1 = plt.figure(figsize=(7, 1.5))
    #    ax1 = fig1.add_subplot(1,1,1)
    a_pred = test_result[:, 0]
    a_true = test_label[:, 0]
    plt.plot(a_pred, "r-", label="prediction")
    plt.plot(a_true, "b-", label="true")
    plt.legend(loc="best", fontsize=10)
    plt.savefig(path + "/test_all.jpg")
    plt.show()

    # oneday test result visualization
    fig1 = plt.figure(figsize=(7, 1.5))
    #    ax1 = fig1.add_subplot(1,1,1)
    a_pred = test_result[0:96, 0]
    a_true = test_label[0:96, 0]
    plt.plot(a_pred, "r-", label="prediction")
    plt.plot(a_true, "b-", label="true")
    plt.legend(loc="best", fontsize=10)
    plt.savefig(path + "/test_oneday.jpg")
    plt.show()


def plot_error(train_rmse, train_loss, test_rmse, test_acc, test_mae, path="results"):
    """Plots RMSE, Loss, accuracy, mae

    Args:
        train_rmse ([type]): root-mean-squared error on training dataset
        train_loss ([type]): training loss
        test_rmse ([type]): root-mean-squared error on testing dataset
        test_acc ([type]): testing accuracy
        test_mae ([type]): mean, absolute error
        path ([type]): saving image results
    """
    # train_rmse & test_rmse
    fig1 = plt.figure(figsize=(5, 3))
    plt.plot(train_rmse, "r-", label="train_rmse")
    plt.plot(test_rmse, "b-", label="test_rmse")
    plt.legend(loc="best", fontsize=10)
    plt.savefig(path + "/rmse.jpg")
    plt.show()

    # train_loss & train_rmse
    fig1 = plt.figure(figsize=(5, 3))
    plt.plot(train_loss, "b-", label="train_loss")
    plt.legend(loc="best", fontsize=10)
    plt.savefig(path + "/train_loss.jpg")
    plt.show()

    fig1 = plt.figure(figsize=(5, 3))
    plt.plot(train_rmse, "b-", label="train_rmse")
    plt.legend(loc="best", fontsize=10)
    plt.savefig(path + "/train_rmse.jpg")
    plt.show()

    # accuracy
    fig1 = plt.figure(figsize=(5, 3))
    plt.plot(test_acc, "b-", label="test_acc")
    plt.legend(loc="best", fontsize=10)
    plt.savefig(path + "/test_acc.jpg")
    plt.show()

    # rmse
    fig1 = plt.figure(figsize=(5, 3))
    plt.plot(test_rmse, "b-", label="test_rmse")
    plt.legend(loc="best", fontsize=10)
    plt.savefig(path + "/test_rmse.jpg")
    plt.show()

    # mae
    fig1 = plt.figure(figsize=(5, 3))
    plt.plot(test_mae, "b-", label="test_mae")
    plt.legend(loc="best", fontsize=10)
    plt.savefig(path + "/test_mae.jpg")
    plt.show()
