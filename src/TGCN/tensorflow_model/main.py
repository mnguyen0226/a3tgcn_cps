import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from models import TGCNCell
from utils import preprocess_data
from utils import load_scada_data
from utils import plot_error
from utils import plot_result_tank
from utils import plot_result_pump
from utils import plot_result_valve
from utils import plot_result_junction
from utils import evaluation
from utils import calculate_rmd_clean
from utils import calculate_rmd_poison
from utils import calculate_rmd_test
from utils import calculate_md_clean
from utils import calculate_md_poison
from utils import calculate_md_test

# Sets time for saving different trained time model
local_time = time.asctime(time.localtime(time.time()))

# Global variables
OP_LR = 0.01
OP_EPOCH = 101
OP_BATCH_SIZE = 16
OP_HIDDEN_DIM = 64  # [8, 16, 32, 64, 100, 128]
MODEL_NAME = "tgcn"
DATA_NAME = "scada_wds"

# Parses settings from command line
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rate", OP_LR, "Initial learning rate.")
flags.DEFINE_integer("training_epoch", OP_EPOCH, "Number of epoch to train.")
flags.DEFINE_integer("gru_units", OP_HIDDEN_DIM, "hidden_units of gru")
flags.DEFINE_integer("seq_len", 8, "time length of inputs time series.")
flags.DEFINE_integer("pre_len", 1, "time length of prediction.")
flags.DEFINE_float("train_rate", 0.8, "rate of training set: 80% train, 20% validate.")
flags.DEFINE_float("poison_eval_rate", 0, "100% of evaluation rate.")
flags.DEFINE_integer("batch_size", OP_BATCH_SIZE, "batch size.")

# Global variables set based on flags
EVAL_RATE = FLAGS.poison_eval_rate
TRAIN_RATE = FLAGS.train_rate
SEQ_LEN = FLAGS.seq_len
OUTPUT_DIM = PRE_LEN = FLAGS.pre_len
BATCH_SIZE = FLAGS.batch_size
LR = FLAGS.learning_rate
TRAINING_EPOCH = FLAGS.training_epoch
GRU_UNITS = FLAGS.gru_units
MODEL_NAME = "tgcn"
DATA_NAME = "scada_wds"
SAVING_STEP = 10

# Preprocess clean dataset for training and evaluating process
clean_data, adj = load_scada_data(dataset="train_eval_clean")
time_len = clean_data.shape[0]
num_nodes = clean_data.shape[1]
data_maxtrix = np.mat(clean_data, dtype=np.float32)

# Normalizes data
max_value = np.max(data_maxtrix)
data_maxtrix = data_maxtrix / max_value
train_X_clean, train_Y_clean, eval_X_clean, eval_Y_clean = preprocess_data(
    data=data_maxtrix,
    time_len=time_len,
    rate=TRAIN_RATE,
    seq_len=SEQ_LEN,
    pre_len=PRE_LEN,
)

# Gets number of batches of clean dataset
total_clean_batch = int(train_X_clean.shape[0] / BATCH_SIZE)

# Preprocess poisoned dataset for evaluating process
poisoned_data, _ = load_scada_data(dataset="eval_poison")
p_time_len = poisoned_data.shape[0]
p_data_maxtrix = np.mat(poisoned_data, dtype=np.float32)

# Normalizes data
p_max_value = np.max(p_data_maxtrix)
p_data_maxtrix = p_data_maxtrix / p_max_value
_, _, eval_X_poison, eval_Y_poison = preprocess_data(
    data=p_data_maxtrix,
    time_len=p_time_len,
    rate=EVAL_RATE,
    seq_len=SEQ_LEN,
    pre_len=PRE_LEN,
)

# Preprocess test dataset for testing process
test_data, _ = load_scada_data(dataset="test")
t_time_len = test_data.shape[0]
t_data_maxtrix = np.mat(test_data, dtype=np.float32)

# Normalizes data
t_max_value = np.max(t_data_maxtrix)
t_data_maxtrix = t_data_maxtrix / t_max_value
_, _, test_X, test_Y = preprocess_data(
    data=t_data_maxtrix,
    time_len=t_time_len,
    rate=EVAL_RATE,
    seq_len=SEQ_LEN,
    pre_len=PRE_LEN,
)

########## TGCN
def TGCN(_X, _weights, _biases, reuse=None):
    """TGCN model for scada batadal datasets, including multiple TGCNCell(s)

    Args:
        _X: Adjacency matrix, time series.
        _weights: Weights.
        _biases: Biases.
    """
    cell_1 = TGCNCell(num_units=GRU_UNITS, adj=adj, num_nodes=num_nodes, reuse=reuse)
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
    output = tf.reshape(
        output, shape=[-1, num_nodes], name="op_to_restore"
    )  # name for restoration
    return output, m, states


# Prepares to feed input to model
inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, SEQ_LEN, num_nodes])
labels = tf.compat.v1.placeholder(tf.float32, shape=[None, PRE_LEN, num_nodes])

# Graph weights and biases initialization of all neurons and layers
weights = {
    "out": tf.Variable(
        tf.random.normal([GRU_UNITS, PRE_LEN], mean=1.0), name="weight_o"
    )
}
biases = {"out": tf.Variable(tf.random.normal([PRE_LEN]), name="bias_o")}

# Define TGCN model
y_pred, _, _ = TGCN(inputs, weights, biases)

# Optimizer
lambda_loss = 0.0015

# L2 regularization to avoid over fit
L_reg = lambda_loss * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.compat.v1.trainable_variables()
)
label = tf.reshape(labels, [-1, num_nodes])

# Initializes losses
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred - label) + L_reg)
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred - label)))
optimizer = tf.compat.v1.train.AdamOptimizer(LR).minimize(loss)

# Initializes hyperparameter
init = tf.global_variables_initializer()

# 'saver' op to save and restore all the variables
saver = tf.train.Saver()


def train_and_eval():
    """Trains and evaluates TGCN on the clean_scada time-series dataset."""
    print("Start the training and saving process")
    time_start = time.time()

    # Checks for GPU
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    # Setups training session
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init)

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
    eval_loss, eval_rmse, eval_mae, eval_acc, eval_r2, eval_var, eval_pred = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    print(
        "-----------------------------------------------\nResults of training and evaluating results:"
    )
    result_file = open(path + "/summary.txt", "a")

    # Logs in time
    result_file.write(
        "----------------------------------------------------------------------------------------------\n"
    )
    result_file.write("TIME LOG ----------------\n")
    result_file.write(local_time + "\n")

    # Writes results to files
    result_file.write(
        "-----------------------------------------------\n\nResults of training and evaluating results:\n"
    )

    for epoch in range(TRAINING_EPOCH):
        for m in range(total_clean_batch):
            mini_batch = train_X_clean[m * BATCH_SIZE : (m + 1) * BATCH_SIZE]
            mini_label = train_Y_clean[m * BATCH_SIZE : (m + 1) * BATCH_SIZE]
            _, loss1, rmse1, train_output = sess.run(
                [optimizer, loss, error, y_pred],
                feed_dict={inputs: mini_batch, labels: mini_label},
            )
            batch_loss.append(loss1)
            batch_rmse.append(rmse1 * max_value)

        # Evaluates completely at every epoch
        loss2, rmse2, eval_output = sess.run(
            [loss, error, y_pred],
            feed_dict={inputs: eval_X_clean, labels: eval_Y_clean},
        )

        eval_label = np.reshape(eval_Y_clean, [-1, num_nodes])
        rmse, mae, acc, r2_score, var_score = evaluation(eval_label, eval_output)
        eval_label1 = eval_label * max_value
        eval_output1 = eval_output * max_value
        eval_loss.append(loss2)
        eval_rmse.append(rmse * max_value)
        eval_mae.append(mae * max_value)
        eval_acc.append(acc)
        eval_r2.append(r2_score)
        eval_var.append(var_score)
        eval_pred.append(eval_output1)

        print("-------------------------\nIter/Epoch #: {}".format(epoch))
        print("Train_rmse: {:.4}".format(batch_rmse[-1]))
        print("Eval_loss: {:.4}".format(loss2))
        print("Eval_rmse: {:.4}".format(rmse))
        print("Eval_acc: {:.4}\n".format(acc))

        # Writes results to files
        result_file.write("-------------------------\nIter/Epoch #: {}\n".format(epoch))
        result_file.write("Train_rmse: {:.4}\n".format(batch_rmse[-1]))
        result_file.write("Eval_loss: {:.4}\n".format(loss2))
        result_file.write("Eval_rmse: {:.4}\n".format(rmse))
        result_file.write("Eval_acc: {:.4}\n\n".format(acc))

        if epoch % SAVING_STEP == 0:
            # Saves model every SAVING_STEP epoch
            saver.save(sess, path + "/model_100/TGCN_pre_%r" % epoch, global_step=epoch)

    time_end = time.time()
    print(f"Training Time: {time_end - time_start} sec")

    # Writes results to files
    result_file.write(f"Training Time: {time_end - time_start} sec.\n")

    # Visualization
    b = int(len(batch_rmse) / total_clean_batch)
    batch_rmse1 = [i for i in batch_rmse]
    train_rmse = [
        (
            sum(batch_rmse1[i * total_clean_batch : (i + 1) * total_clean_batch])
            / total_clean_batch
        )
        for i in range(b)
    ]
    batch_loss1 = [i for i in batch_loss]
    train_loss = [
        (
            sum(batch_loss1[i * total_clean_batch : (i + 1) * total_clean_batch])
            / total_clean_batch
        )
        for i in range(b)
    ]

    index = eval_rmse.index(np.min(eval_rmse))
    eval_result = eval_pred[index]
    var = pd.DataFrame(eval_result)  # gets the prediction to unnormalized result
    var.to_csv(path + "/eval_result.csv", index=False, header=False)
    plot_result_tank(eval_result, eval_label1, path)
    plot_error(train_rmse, train_loss, eval_rmse, eval_acc, eval_mae, path)

    print("-----------------------------------------------\nEvaluation Metrics:")
    print("min_rmse: %r" % (np.min(eval_rmse)))
    print("min_mae: %r" % (eval_mae[index]))
    print("max_acc: %r" % (eval_acc[index]))
    print("r2: %r" % (eval_r2[index]))
    print("var: %r" % eval_var[index])

    # Writes results to files
    result_file.write(
        "-----------------------------------------------\nEvaluation Metrics:"
    )
    result_file.write("min_rmse: %r\n" % (np.min(eval_rmse)))
    result_file.write("min_mae: %r\n" % (eval_mae[index]))
    result_file.write("max_acc: %r\n" % (eval_acc[index]))
    result_file.write("r2: %r\n" % (eval_r2[index]))
    result_file.write("var: %r\n" % eval_var[index])

    result_file.close()


def load_and_eval_clean_dataset():
    """Loads and evaluates trained model clean dataset"""

    print("Start the loading and evaluating process")
    time_start = time.time()

    # Checks for GPU
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    # Setups traininng session
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init)

    # Chooses trained model path (CHANGE)
    saved_path = "out/tgcn/tgcn_scada_wds_lr0.01_batch16_unit64_seq8_pre1_epoch101/model_100/TGCN_pre_100-100"

    # Loads model from trained path
    load_path = saver.restore(sess, saved_path)

    # Initializes the array for evaluating results
    eval_loss, eval_rmse, eval_mae, eval_acc, eval_r2, eval_var, eval_pred = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # Evals completely at every epoch
    loss2, rmse2, eval_output = sess.run(
        [loss, error, y_pred], feed_dict={inputs: eval_X_clean, labels: eval_Y_clean}
    )

    # Provides evaluating results
    eval_label = np.reshape(eval_Y_clean, [-1, num_nodes])

    rmse, mae, acc, r2_score, var_score = evaluation(eval_label, eval_output)
    eval_label1 = eval_label * max_value
    eval_output1 = eval_output * max_value
    eval_loss.append(loss2)
    eval_rmse.append(rmse * max_value)
    eval_mae.append(mae * max_value)
    eval_acc.append(acc)
    eval_r2.append(r2_score)
    eval_var.append(var_score)
    eval_pred.append(eval_output1)

    # Sets index and provides eval results
    index = eval_rmse.index(np.min(eval_rmse))
    eval_result = eval_pred[index]

    # Create a evaluation path
    eval_path = (
        "out/tgcn/tgcn_scada_wds_lr0.01_batch16_unit64_seq8_pre1_epoch101/eval_clean"
    )

    var_eval_output = pd.DataFrame(
        eval_output * max_value
    )  # eval_result, make this unnormalize
    var_eval_output.to_csv(
        eval_path + "/eval_clean_output.csv", index=False, header=False
    )

    var_eval_label = pd.DataFrame(eval_label * max_value)
    var_eval_label.to_csv(
        eval_path + "/eval_clean_labels.csv", index=False, header=False
    )

    # Plots results
    plot_result_tank(eval_result, eval_label1, eval_path, hour=168)
    plot_result_pump(eval_result, eval_label1, eval_path, hour=168)
    plot_result_valve(eval_result, eval_label1, eval_path, hour=168)
    plot_result_junction(eval_result, eval_label1, eval_path, hour=168)

    # Prints out evaluates results
    print("-----------------------------------------------\nEvaluation Metrics:")
    print("min_rmse: %r" % (np.min(eval_rmse)))
    print("min_mae: %r" % (eval_mae[index]))
    print("max_acc: %r" % (eval_acc[index]))
    print("r2: %r" % (eval_r2[index]))
    print("var: %r" % eval_var[index])

    time_end = time.time()
    print(f"Training Time: {time_end - time_start} sec")


def load_and_eval_poisoned_dataset():
    """Loads and evaluates trained model poisoned dataset"""

    print("Start the loading and evaluating process")
    time_start = time.time()

    # Checks for GPU
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    # Setups traininng session
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init)

    # Chooses trained model path (CHANGE)
    saved_path = "out/tgcn/tgcn_scada_wds_lr0.01_batch16_unit64_seq8_pre1_epoch101/model_100/TGCN_pre_100-100"

    # Loads model from trained path
    load_path = saver.restore(sess, saved_path)

    # Initializes the array for evaluating results
    eval_loss, eval_rmse, eval_mae, eval_acc, eval_r2, eval_var, eval_pred = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # Evals completely at every epoch
    loss2, rmse2, eval_output = sess.run(
        [loss, error, y_pred], feed_dict={inputs: eval_X_poison, labels: eval_Y_poison}
    )

    # Provides evaluating results
    eval_label = np.reshape(eval_Y_poison, [-1, num_nodes])

    rmse, mae, acc, r2_score, var_score = evaluation(eval_label, eval_output)
    eval_label1 = eval_label * p_max_value
    eval_output1 = eval_output * p_max_value
    eval_loss.append(loss2)
    eval_rmse.append(rmse * p_max_value)
    eval_mae.append(mae * p_max_value)
    eval_acc.append(acc)
    eval_r2.append(r2_score)
    eval_var.append(var_score)
    eval_pred.append(eval_output1)

    # Sets index and provides eval results
    index = eval_rmse.index(np.min(eval_rmse))
    eval_result = eval_pred[index]

    # Create a evaluation path
    eval_path = (
        "out/tgcn/tgcn_scada_wds_lr0.01_batch16_unit64_seq8_pre1_epoch101/eval_poisoned"
    )

    var_eval_output = pd.DataFrame(
        eval_output * p_max_value
    )  # eval_result, make this unnormalize
    var_eval_output.to_csv(
        eval_path + "/eval_poisoned_output.csv", index=False, header=False
    )

    var_eval_label = pd.DataFrame(eval_label * p_max_value)
    var_eval_label.to_csv(
        eval_path + "/eval_poisoned_labels.csv", index=False, header=False
    )

    # Plots results
    plot_result_tank(eval_result, eval_label1, eval_path, hour=738)
    plot_result_pump(eval_result, eval_label1, eval_path, hour=738)
    plot_result_valve(eval_result, eval_label1, eval_path, hour=738)
    plot_result_junction(eval_result, eval_label1, eval_path, hour=738)

    # Prints out evaluates results
    print("-----------------------------------------------\nEvaluation Metrics:")
    print("min_rmse: %r" % (np.min(eval_rmse)))
    print("min_mae: %r" % (eval_mae[index]))
    print("max_acc: %r" % (eval_acc[index]))
    print("r2: %r" % (eval_r2[index]))
    print("var: %r" % eval_var[index])

    time_end = time.time()
    print(f"Training Time: {time_end - time_start} sec")


def load_and_eval_test_dataset():
    """Loads and evaluates trained model testing dataset"""

    print("Start the loading and evaluating process")
    time_start = time.time()

    # Checks for GPU
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    # Setups traininng session
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init)

    # Chooses trained model path
    saved_path = "out/tgcn/tgcn_scada_wds_lr0.01_batch16_unit64_seq8_pre1_epoch101/model_100/TGCN_pre_100-100"

    # Loads model from trained path
    load_path = saver.restore(sess, saved_path)

    # Initializes the array for evaluating results
    eval_loss, eval_rmse, eval_mae, eval_acc, eval_r2, eval_var, eval_pred = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # Evals completely at every epoch
    loss2, rmse2, eval_output = sess.run(
        [loss, error, y_pred], feed_dict={inputs: test_X, labels: test_Y}
    )

    # Provides evaluating results
    eval_label = np.reshape(test_Y, [-1, num_nodes])

    rmse, mae, acc, r2_score, var_score = evaluation(eval_label, eval_output)
    eval_label1 = eval_label * p_max_value
    eval_output1 = eval_output * p_max_value
    eval_loss.append(loss2)
    eval_rmse.append(rmse * p_max_value)
    eval_mae.append(mae * p_max_value)
    eval_acc.append(acc)
    eval_r2.append(r2_score)
    eval_var.append(var_score)
    eval_pred.append(eval_output1)

    # Sets index and provides eval results
    index = eval_rmse.index(np.min(eval_rmse))
    eval_result = eval_pred[index]

    # Create a evaluation path
    eval_path = (
        "out/tgcn/tgcn_scada_wds_lr0.01_batch16_unit64_seq8_pre1_epoch101/eval_test"
    )

    var_eval_output = pd.DataFrame(
        eval_output * p_max_value
    )  # eval_result, make this unnormalize
    var_eval_output.to_csv(
        eval_path + "/eval_test_output.csv", index=False, header=False
    )

    var_eval_label = pd.DataFrame(eval_label * p_max_value)
    var_eval_label.to_csv(
        eval_path + "/eval_test_labels.csv", index=False, header=False
    )

    # Plots results
    plot_result_tank(eval_result, eval_label1, eval_path, hour=720)
    plot_result_pump(eval_result, eval_label1, eval_path, hour=720)
    plot_result_valve(eval_result, eval_label1, eval_path, hour=720)
    plot_result_junction(eval_result, eval_label1, eval_path, hour=720)

    # Prints out evaluates results
    print("-----------------------------------------------\nEvaluation Metrics:")
    print("min_rmse: %r" % (np.min(eval_rmse)))
    print("min_mae: %r" % (eval_mae[index]))
    print("max_acc: %r" % (eval_acc[index]))
    print("r2: %r" % (eval_r2[index]))
    print("var: %r" % eval_var[index])

    time_end = time.time()
    print(f"Training Time: {time_end - time_start} sec")


def main():
    """User Interface"""
    # train_and_eval()
    # load_and_eval_clean_dataset()
    # load_and_eval_poisoned_dataset()
    # load_and_eval_test_dataset()

    ### Traditional Mahalanobis Distance
    # calculate_md_clean()
    # calculate_md_poison()
    calculate_md_test()

    ### Robust Mahalanobis Distance
    # calculate_rmd_clean()
    # calculate_rmd_poison()
    # calculate_rmd_test()


if __name__ == "__main__":
    main()
