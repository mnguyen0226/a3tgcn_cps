# Reference: https://github.com/lehaifeng/T-GCN/blob/master/T-GCN/T-GCN-TensorFlow/main.py

import pickle as plk
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from utils import preprocess_data
from utils import load_clean_scada_data
from models import TGCNCell
from utils import plot_error
from utils import plot_result
from utils import evaluation
import time

### Global variables for Optimization (Ashita)
OP_LR = 0.001
OP_EPOCH = 10
OP_BATCH_SIZE = 32

### Parse settings from command line
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rate", OP_LR, "Initial learning rate.")
flags.DEFINE_integer("training_epoch", OP_EPOCH, "Number of epoch to train.")
flags.DEFINE_integer("gru_units", 64, "hidden_units of gru")
flags.DEFINE_integer("seq_len", 12, "time length of inputs time series.")
flags.DEFINE_integer("pre_len", 3, "time length of prediction.")
flags.DEFINE_float("train_rate", 0.8, "rate of training set: 80% train, 20% validate.")
flags.DEFINE_integer("batch_size", OP_BATCH_SIZE, "batch size.")

### Global variables
TRAIN_RATE = FLAGS.train_rate
SEQ_LEN = FLAGS.seq_len
OUTPUT_DIM = PRE_LEN = FLAGS.pre_len
BATCH_SIZE = FLAGS.batch_size
LR = FLAGS.learning_rate
TRAINING_EPOCH = FLAGS.training_epoch
GRU_UNITS = FLAGS.gru_units
MODEL_NAME = "tgcn"
DATA_NAME = "scada_wds"

# Loads data
data, adj = load_clean_scada_data()

time_len = data.shape[0]
num_nodes = data.shape[1]
data_maxtrix = np.mat(data, dtype=np.float32)

# Normalizes data
max_value = np.max(data_maxtrix)
data_maxtrix = data_maxtrix / max_value
trainX, trainY, testX, testY = preprocess_data(
    data=data_maxtrix,
    time_len=time_len,
    rate=TRAIN_RATE,
    seq_len=SEQ_LEN,
    pre_len=PRE_LEN,
)

# Gets number of batches
total_batch = int(trainX.shape[0] / BATCH_SIZE)
training_data_count = len(trainX)


def TGCN(_X, _weights, _biases):
    """Temporal Graph Convolutional Network = GCN + RNN

    Args:
        _X ([type]): adj matrix, time series
        _weights ([type]): weights
        _biases ([type]): biases
    """
    cell_1 = TGCNCell(num_units=GRU_UNITS, adj=adj, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
    m = []
    for i in outputs:
        o = tf.reshape(i, shape=[-1, num_nodes, GRU_UNITS])
        o = tf.reshape(o, shape=[-1, GRU_UNITS])
        m.append(o)
    last_output = m[-1]
    output = tf.matmul(last_output, _weights["out"]) + _biases["out"]
    output = tf.reshape(output, shape=[-1, num_nodes, PRE_LEN])
    output = tf.transpose(output, perm=[0, 2, 1])
    output = tf.reshape(output, shape=[-1, num_nodes])
    return output, m, states


### Place holders
inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, SEQ_LEN, num_nodes])
labels = tf.compat.v1.placeholder(tf.float32, shape=[None, PRE_LEN, num_nodes])

### Graph weights
weights = {
    "out": tf.Variable(
        tf.random.normal([GRU_UNITS, PRE_LEN], mean=1.0), name="weight_o"
    )
}
biases = {"out": tf.Variable(tf.random.normal([PRE_LEN]), name="bias_o")}

y_pred, ttts, ttto = TGCN(inputs, weights, biases)

### Optimizer
lambda_loss = 0.0015
L_reg = lambda_loss * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.compat.v1.trainable_variables()
)
label = tf.reshape(labels, [-1, num_nodes])

### Losses
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred - label) + L_reg)
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred - label)))
optimizer = tf.compat.v1.train.AdamOptimizer(LR).minimize(loss)


def train_and_eval():
    print("Start the training process")
    time_start = time.time()

    ### Initializes session
    variables = tf.global_variables()
    saver = tf.compat.v1.train.Saver(tf.global_variables())
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    out = "out/%s" % (MODEL_NAME)
    path1 = "%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r" % (
        MODEL_NAME,
        DATA_NAME,
        LR,
        BATCH_SIZE,
        GRU_UNITS,
        SEQ_LEN,
        PRE_LEN,
        TRAINING_EPOCH,
    )

    path = os.path.join(out, path1)
    if not os.path.exists(path):
        os.makedirs(path)

    batch_loss, batch_rmse = [], []
    test_loss, test_rmse, test_mae, test_acc, test_r2, test_var, test_pred = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    print(
        "-----------------------------------------\nResults of training and testing results:"
    )
    result_file = open(path + "/summary.txt", "a")

    # writes results to files
    result_file.write(
        "-----------------------------------------\nResults of training and testing results:\n"
    )

    for epoch in range(TRAINING_EPOCH):
        for m in range(total_batch):
            mini_batch = trainX[m * BATCH_SIZE : (m + 1) * BATCH_SIZE]
            mini_label = trainY[m * BATCH_SIZE : (m + 1) * BATCH_SIZE]
            _, loss1, rmse1, train_output = sess.run(
                [optimizer, loss, error, y_pred],
                feed_dict={inputs: mini_batch, labels: mini_label},
            )
            batch_loss.append(loss1)
            batch_rmse.append(rmse1 * max_value)

        # Tests completely at every epoch
        loss2, rmse2, test_output = sess.run(
            [loss, error, y_pred], feed_dict={inputs: testX, labels: testY}
        )

        test_label = np.reshape(testY, [-1, num_nodes])
        rmse, mae, acc, r2_score, var_score = evaluation(test_label, test_output)
        test_label1 = test_label * max_value
        test_output1 = test_output * max_value
        test_loss.append(loss2)
        test_rmse.append(rmse * max_value)
        test_mae.append(mae * max_value)
        test_acc.append(acc)
        test_r2.append(r2_score)
        test_var.append(var_score)
        test_pred.append(test_output1)

        print("----------\nIter/Epoch #: {}".format(epoch))
        print("Train_rmse: {:.4}".format(batch_rmse[-1]))
        print("Test_loss: {:.4}".format(loss2))
        print("Test_rmse: {:.4}".format(rmse))
        print("Test_acc: {:.4}\n".format(acc))

        # writes results to files
        result_file.write("----------\nIter/Epoch #: {}\n".format(epoch))
        result_file.write("Train_rmse: {:.4}\n".format(batch_rmse[-1]))
        result_file.write("Test_loss: {:.4}\n".format(loss2))
        result_file.write("Test_rmse: {:.4}\n".format(rmse))
        result_file.write("Test_acc: {:.4}\n\n".format(acc))

        if epoch % 500 == 0:
            saver.save(sess, path + "/model_100/TGCN_pre_%r" % epoch, global_step=epoch)

    time_end = time.time()
    print(f"Training Time: {time_end - time_start} sec")

    # writes results to files
    result_file.write(f"Training Time: {time_end - time_start} sec.\n")

    ### Visualization
    b = int(len(batch_rmse) / total_batch)
    batch_rmse1 = [i for i in batch_rmse]
    train_rmse = [
        (sum(batch_rmse1[i * total_batch : (i + 1) * total_batch]) / total_batch)
        for i in range(b)
    ]
    batch_loss1 = [i for i in batch_loss]
    train_loss = [
        (sum(batch_loss1[i * total_batch : (i + 1) * total_batch]) / total_batch)
        for i in range(b)
    ]

    index = test_rmse.index(np.min(test_rmse))
    test_result = test_pred[index]
    var = pd.DataFrame(test_result)
    var.to_csv(path + "/test_result.csv", index=False, header=False)
    plot_result(test_result, test_label1, path)
    plot_error(train_rmse, train_loss, test_rmse, test_acc, test_mae, path)

    print("-----------------------------------------\nEvaluation Metrics:")
    print("min_rmse: %r" % (np.min(test_rmse)))
    print("min_mae: %r" % (test_mae[index]))
    print("max_acc: %r" % (test_acc[index]))
    print("r2: %r" % (test_r2[index]))
    print("var: %r" % test_var[index])

    # writes results to files
    result_file.write("-----------------------------------------\nEvaluation Metrics:")
    result_file.write("min_rmse: %r\n" % (np.min(test_rmse)))
    result_file.write("min_mae: %r\n" % (test_mae[index]))
    result_file.write("max_acc: %r\n" % (test_acc[index]))
    result_file.write("r2: %r\n" % (test_r2[index]))
    result_file.write("var: %r\n" % test_var[index])

    result_file.close()


if __name__ == "__main__":
    train_and_eval()