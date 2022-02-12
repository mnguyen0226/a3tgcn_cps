# Plot out data and see the distribution after normalization
# Do the split: Get rid of the first month: Split half: Train 3.5 months, Predict next 2 months
# Do seasonal prediction

import tensorflow as tf
import numpy as np
import time
import os
import pandas as pd
from models import TGCNCell
import matplotlib.pyplot as plt
from utils import load_scada_data
from utils import preprocess_data
from utils import preprocess_data
from utils import load_scada_data
from utils import plot_error
from utils import plot_result_tank
from utils import plot_result_pump
from utils import plot_result_valve
from utils import plot_result_junction
from utils import evaluation

# Sets time for saving different trained time model
local_time = time.asctime(time.localtime(time.time()))

# Global variables
OP_LR = 0.005
OP_EPOCH = 101
OP_BATCH_SIZE = 1  # just 1 big batch
OP_HIDDEN_DIM = 64  # [8, 16, 32, 64, 100, 128]
MODEL_NAME = "tgcn"
DATA_NAME = "scada_wds"
SAVING_STEP = 50
LAMBDA_LOSS = 0.0015

# Parses settings from command line
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rate", OP_LR, "Initial learning rate.")
flags.DEFINE_integer("training_epoch", OP_EPOCH, "Number of epoch to train.")
flags.DEFINE_integer("gru_units", OP_HIDDEN_DIM, "Hidden_units of gru")
flags.DEFINE_integer("seq_len", 3504, "Time length of inputs time series.")
flags.DEFINE_integer("pre_len", 720, "Time length of prediction.")
flags.DEFINE_float("train_rate", 0.5, "Rate of training set: 50% train, 50% testing.")
flags.DEFINE_float("poison_eval_rate", 0, "100% of evaluation rate.")
flags.DEFINE_integer("batch_size", OP_BATCH_SIZE, "Batch size.")

# Global variables set based on flags
EVAL_RATE = FLAGS.poison_eval_rate
TRAIN_RATE = FLAGS.train_rate
SEQ_LEN = FLAGS.seq_len
OUTPUT_DIM = PRE_LEN = FLAGS.pre_len
BATCH_SIZE = FLAGS.batch_size
LR = FLAGS.learning_rate
TRAINING_EPOCH = FLAGS.training_epoch
GRU_UNITS = FLAGS.gru_units

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


def self_attention(x, weight_att, bias_att):
    """Constructs self-attention mechanism for TGCN
    # Reference: https://github.com/lehaifeng/T-GCN/blob/master/A3T-GCN/A3T-GCN.py

    Args:
        x: Input.
        weight_att: Weights attribute.
        bias_att: Biases attribute.
    """
    x = tf.matmul(tf.reshape(x, [-1, GRU_UNITS]), weight_att["w1"]) + bias_att["b1"]
    f = tf.matmul(tf.reshape(x, [-1, num_nodes]), weight_att["w2"]) + bias_att["b2"]
    g = tf.matmul(tf.reshape(x, [-1, num_nodes]), weight_att["w2"]) + bias_att["b2"]
    h = tf.matmul(tf.reshape(x, [-1, num_nodes]), weight_att["w2"]) + bias_att["b2"]

    f1 = tf.reshape(f, [-1, SEQ_LEN])
    g1 = tf.reshape(g, [-1, SEQ_LEN])
    h1 = tf.reshape(h, [-1, SEQ_LEN])
    s = g1 * f1
    print("s", s)

    beta = tf.nn.softmax(s, dim=-1)  # attention map
    print("beta", beta)
    context = tf.expand_dims(beta, 2) * tf.reshape(x, [-1, SEQ_LEN, num_nodes])

    context = tf.transpose(context, perm=[0, 2, 1])
    print("context", context)
    return context, beta


def ATGCN(_X, weights, biases):
    """Attention - TGCN model for scada batadal datasets, including multiple TGCNCell(s)
    # Reference: https://github.com/lehaifeng/T-GCN/blob/master/A3T-GCN/A3T-GCN.py

    Args:
        _X: Adjacency matrix, time series.
        weights: Weights.
        biases: Biases.
    """
    cell_1 = TGCNCell(GRU_UNITS, adj, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)

    out = tf.concat(outputs, axis=0)
    out = tf.reshape(out, shape=[SEQ_LEN, -1, num_nodes, GRU_UNITS])
    out = tf.transpose(out, perm=[1, 0, 2, 3])

    last_output, alpha = self_attention(out, weight_att, bias_att)

    output = tf.reshape(last_output, shape=[-1, SEQ_LEN])
    output = tf.matmul(output, weights["out"]) + biases["out"]
    output = tf.reshape(output, shape=[-1, num_nodes, PRE_LEN])
    output = tf.transpose(output, perm=[0, 2, 1])
    output = tf.reshape(output, shape=[-1, num_nodes])

    return output, outputs, states, alpha


# Prepares to feed input to model
inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, SEQ_LEN, num_nodes])
labels = tf.compat.v1.placeholder(tf.float32, shape=[None, PRE_LEN, num_nodes])

# Graph weights and biases initialization of all neurons and layers
weights = {
    "out": tf.Variable(tf.random.normal([SEQ_LEN, PRE_LEN], mean=1.0), name="weight_o")
}
biases = {"out": tf.Variable(tf.random.normal([PRE_LEN]), name="bias_o")}

# Attention weights and bias initialization
weight_att = {
    "w1": tf.Variable(tf.random_normal([GRU_UNITS, 1], stddev=0.1), name="att_w1"),
    "w2": tf.Variable(tf.random_normal([num_nodes, 1], stddev=0.1), name="att_w2"),
}
bias_att = {
    "b1": tf.Variable(tf.random_normal([1]), name="att_b1"),
    "b2": tf.Variable(tf.random_normal([1]), name="att_b2"),
}

# Defines A-TGCN model
y_pred, _, _, alpha = ATGCN(inputs, weights, biases)

# L2 regularization to avoid over fit
L_reg = LAMBDA_LOSS * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.compat.v1.trainable_variables()
)

# Reshapes labels
label = tf.reshape(labels, [-1, num_nodes])

# Initializes losses and Optimizer
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred - label) + L_reg)
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred - label)))
optimizer = tf.compat.v1.train.AdamOptimizer(LR).minimize(loss)

# Initializes hyperparameter
init = tf.global_variables_initializer()

# 'saver' op to save and restore all the variables
saver = tf.train.Saver()


def train_and_eval():
    """Trains and evaluates ATGCN on the clean_scada time-series dataset."""
    print("Start the training, evaluating, and saving process")
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
            _, loss1, rmse1, train_output, alpha1 = sess.run(
                [optimizer, loss, error, y_pred, alpha],
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

    fig1 = plt.figure(figsize=(7, 3))
    ax1 = fig1.add_subplot(1, 1, 1)
    plt.plot(np.sum(alpha1, 0))
    plt.savefig(path + "/alpha1.png", dpi=500)
    plt.show()

    plt.imshow(np.mat(np.sum(alpha1, 0)))
    plt.savefig(path + "/alpha2.png", dpi=500)
    plt.show()

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


def plot_data():
    L_T1_norm = data_maxtrix[:, 0]
    L_T2_norm = data_maxtrix[:, 1]
    L_T3_norm = data_maxtrix[:, 2]
    L_T4_norm = data_maxtrix[:, 3]
    L_T5_norm = data_maxtrix[:, 4]
    L_T6_norm = data_maxtrix[:, 5]
    L_T7_norm = data_maxtrix[:, 6]
    F_PU1_norm = data_maxtrix[:, 7]
    F_PU2_norm = data_maxtrix[:, 8]
    F_PU3_norm = data_maxtrix[:, 9]
    F_PU4_norm = data_maxtrix[:, 10]
    F_PU5_norm = data_maxtrix[:, 11]
    F_PU6_norm = data_maxtrix[:, 12]
    F_PU7_norm = data_maxtrix[:, 13]
    F_PU8_norm = data_maxtrix[:, 14]
    F_PU9_norm = data_maxtrix[:, 15]
    F_PU10_norm = data_maxtrix[:, 16]
    F_PU11_norm = data_maxtrix[:, 17]
    F_V2_norm = data_maxtrix[:, 18]
    P_J280_norm = data_maxtrix[:, 19]
    P_J269_norm = data_maxtrix[:, 20]
    P_J300_norm = data_maxtrix[:, 21]
    P_J256_norm = data_maxtrix[:, 22]
    P_J289_norm = data_maxtrix[:, 23]
    P_J415_norm = data_maxtrix[:, 24]
    P_J302_norm = data_maxtrix[:, 25]
    P_J306_norm = data_maxtrix[:, 26]
    P_J307_norm = data_maxtrix[:, 27]
    P_J317_norm = data_maxtrix[:, 28]
    P_J14_norm = data_maxtrix[:, 29]
    P_J422_norm = data_maxtrix[:, 30]

    # Try Exponential Moving Average
    ema_L_T1 = pd.DataFrame(L_T1_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_L_T1 = ema_L_T1.iloc[:, 0]
    ema_L_T1 = ema_L_T1.to_numpy()

    ema_L_T2 = pd.DataFrame(L_T2_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_L_T2 = ema_L_T2.iloc[:, 0]
    ema_L_T2 = ema_L_T2.to_numpy()

    ema_L_T3 = pd.DataFrame(L_T3_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_L_T3 = ema_L_T3.iloc[:, 0]
    ema_L_T3 = ema_L_T3.to_numpy()

    ema_L_T4 = pd.DataFrame(L_T4_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_L_T4 = ema_L_T4.iloc[:, 0]
    ema_L_T4 = ema_L_T4.to_numpy()

    ema_L_T5 = pd.DataFrame(L_T5_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_L_T5 = ema_L_T5.iloc[:, 0]
    ema_L_T5 = ema_L_T5.to_numpy()

    ema_L_T6 = pd.DataFrame(L_T6_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_L_T6 = ema_L_T6.iloc[:, 0]
    ema_L_T6 = ema_L_T6.to_numpy()

    ema_L_T7 = pd.DataFrame(L_T7_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_L_T7 = ema_L_T7.iloc[:, 0]
    ema_L_T7 = ema_L_T7.to_numpy()

    ema_F_PU1 = pd.DataFrame(F_PU1_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_F_PU1 = ema_F_PU1.iloc[:, 0]
    ema_F_PU1 = ema_F_PU1.to_numpy()

    ema_F_PU2 = pd.DataFrame(F_PU2_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_F_PU2 = ema_F_PU2.iloc[:, 0]
    ema_F_PU2 = ema_F_PU2.to_numpy()

    ema_F_PU3 = pd.DataFrame(F_PU2_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_F_PU3 = ema_F_PU3.iloc[:, 0]
    ema_F_PU3 = ema_F_PU3.to_numpy()

    ema_F_PU4 = pd.DataFrame(F_PU4_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_F_PU4 = ema_F_PU4.iloc[:, 0]
    ema_F_PU4 = ema_F_PU4.to_numpy()

    ema_F_PU5 = pd.DataFrame(F_PU5_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_F_PU5 = ema_F_PU5.iloc[:, 0]
    ema_F_PU5 = ema_F_PU5.to_numpy()

    ema_F_PU6 = pd.DataFrame(F_PU6_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_F_PU6 = ema_F_PU6.iloc[:, 0]
    ema_F_PU6 = ema_F_PU6.to_numpy()

    ema_F_PU7 = pd.DataFrame(F_PU7_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_F_PU7 = ema_F_PU7.iloc[:, 0]
    ema_F_PU7 = ema_F_PU7.to_numpy()

    ema_F_PU8 = pd.DataFrame(F_PU8_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_F_PU8 = ema_F_PU8.iloc[:, 0]
    ema_F_PU8 = ema_F_PU8.to_numpy()

    ema_F_PU9 = pd.DataFrame(F_PU9_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_F_PU9 = ema_F_PU9.iloc[:, 0]
    ema_F_PU9 = ema_F_PU9.to_numpy()

    ema_F_PU10 = pd.DataFrame(F_PU10_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_F_PU10 = ema_F_PU10.iloc[:, 0]
    ema_F_PU10 = ema_F_PU10.to_numpy()

    ema_F_PU11 = pd.DataFrame(F_PU11_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_F_PU11 = ema_F_PU11.iloc[:, 0]
    ema_F_PU11 = ema_F_PU11.to_numpy()

    ema_F_V2 = pd.DataFrame(F_V2_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_F_V2 = ema_F_V2.iloc[:, 0]
    ema_F_V2 = ema_F_V2.to_numpy()

    ema_P_J280 = pd.DataFrame(P_J280_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_P_J280 = ema_P_J280.iloc[:, 0]
    ema_P_J280 = ema_P_J280.to_numpy()

    ema_P_J269 = pd.DataFrame(P_J269_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_P_J269 = ema_P_J269.iloc[:, 0]
    ema_P_J269 = ema_P_J269.to_numpy()

    ema_P_J300 = pd.DataFrame(P_J300_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_P_J300 = ema_P_J300.iloc[:, 0]
    ema_P_J300 = ema_P_J300.to_numpy()

    ema_P_J256 = pd.DataFrame(P_J256_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_P_J256 = ema_P_J256.iloc[:, 0]
    ema_P_J256 = ema_P_J256.to_numpy()

    ema_P_J289 = pd.DataFrame(P_J289_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_P_J289 = ema_P_J289.iloc[:, 0]
    ema_P_J289 = ema_P_J289.to_numpy()

    ema_P_J415 = pd.DataFrame(P_J415_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_P_J415 = ema_P_J415.iloc[:, 0]
    ema_P_J415 = ema_P_J415.to_numpy()

    ema_P_J302 = pd.DataFrame(P_J302_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_P_J302 = ema_P_J302.iloc[:, 0]
    ema_P_J302 = ema_P_J302.to_numpy()

    ema_P_J306 = pd.DataFrame(P_J306_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_P_J306 = ema_P_J306.iloc[:, 0]
    ema_P_J306 = ema_P_J306.to_numpy()

    ema_P_J307 = pd.DataFrame(P_J307_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_P_J307 = ema_P_J307.iloc[:, 0]
    ema_P_J307 = ema_P_J307.to_numpy()

    ema_P_J317 = pd.DataFrame(P_J317_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_P_J317 = ema_P_J317.iloc[:, 0]
    ema_P_J317 = ema_P_J317.to_numpy()

    ema_P_J14 = pd.DataFrame(P_J14_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_P_J14 = ema_P_J14.iloc[:, 0]
    ema_P_J14 = ema_P_J14.to_numpy()

    ema_P_J422 = pd.DataFrame(P_J422_norm).ewm(alpha=0.01, adjust=False).mean()
    ema_P_J422 = ema_P_J422.iloc[:, 0]
    ema_P_J422 = ema_P_J422.to_numpy()

    fig1 = plt.figure(figsize=(25, 8))
    plt.title("Normalized Annual Values Of All 31 Sensors")
    plt.plot(ema_L_T1, label="L_T1 EMA")
    plt.plot(ema_L_T2, label="L_T2 EMA")
    plt.plot(ema_L_T3, label="L_T3 EMA")
    plt.plot(ema_L_T4, label="L_T4 EMA")
    plt.plot(ema_L_T5, label="L_T5 EMA")
    plt.plot(ema_L_T6, label="L_T6 EMA")
    plt.plot(ema_L_T7, label="L_T7 EMA")
    plt.plot(ema_F_PU1, label="F_PU1 EMA")
    plt.plot(ema_F_PU2, label="F_PU2 EMA")
    plt.plot(ema_F_PU3, label="F_PU3 EMA")
    plt.plot(ema_F_PU4, label="F_PU4 EMA")
    plt.plot(ema_F_PU5, label="F_PU5 EMA")
    plt.plot(ema_F_PU6, label="F_PU6 EMA")
    plt.plot(ema_F_PU7, label="F_PU7 EMA")
    plt.plot(ema_F_PU8, label="F_PU8 EMA")
    plt.plot(ema_F_PU9, label="F_PU9 EMA")
    plt.plot(ema_F_PU10, label="F_PU10 EMA")
    plt.plot(ema_F_PU11, label="F_PU11 EMA")
    plt.plot(ema_F_V2, label="F_V2 EMA")
    plt.plot(ema_P_J280, label="P_J280 EMA")
    plt.plot(ema_P_J269, label="P_J269 EMA")
    plt.plot(ema_P_J300, label="P_J300 EMA")
    plt.plot(ema_P_J256, label="P_J256 EMA")
    plt.plot(ema_P_J289, label="P_J289 EMA")
    plt.plot(ema_P_J415, label="P_J415 EMA")
    plt.plot(ema_P_J302, label="P_J302 EMA")
    plt.plot(ema_P_J306, label="P_J306 EMA")
    plt.plot(ema_P_J307, label="P_J307 EMA")
    plt.plot(ema_P_J317, label="P_J317 EMA")
    plt.plot(ema_P_J14, label="P_J14 EMA")
    plt.plot(ema_P_J422, label="P_J422 EMA")

    plt.xlabel("t (h)")
    plt.ylabel("Normalized Sensors Values")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
    plt.show()


def main():
    print("Running")


if __name__ == "__main__":
    plot_data()
