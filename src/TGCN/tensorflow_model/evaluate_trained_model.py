# script load and test prediction on each node with poison dataset.
# run 1 iteration and provide results on that interation.
# write in path model_evaluation

import tensorflow as tf
import time
from utils import load_scada_data
from utils import preprocess_data
from utils import evaluation
import numpy as np

### Global variables
SEQ_LEN = 12  # input 12 rows in time-series dataset.
PRE_LEN = 3  # evaluate the next 3 rows in time-series dataset.
BATCH_SIZE = 32

# imports time-series data and adjacency matrix
data, adj = load_scada_data(dataset="train_mix")

time_len = data.shape[0]
num_nodes = data.shape[1]
data_maxtrix = np.mat(data, dtype=np.float32)

# normalizes data
max_value = np.max(data_maxtrix)
data_maxtrix = data_maxtrix / max_value
_, _, evalX, evalY = preprocess_data(
    data=data_maxtrix,
    time_len=time_len,
    rate=1,  # using 100% of dataset for testing
    seq_len=SEQ_LEN,
    pre_len=PRE_LEN,
)

# Gets number of batches
total_batch = int(evalX.shape[0] / BATCH_SIZE)
evaluating_data_count = len(evalX)

# loads saved model's path
saved_model_path = (
    "out/tgcn/tgcn_scada_wds_lr0.001_batch32_unit64_seq12_pre3_epoch10/model_100/"
)


def evaluate():
    print("-----------------------------------------\nRun evaluation on trained model")

    # checks for GPU
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(saved_model_path + "TGCN_pre_0-0.meta")
    saver.restore(sess, tf.train.latest_checkpoint(saved_model_path))

    print("MODEL IS RESTORED!!!")

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

    # access and create a place holder variables and create a feed-dict to feed new data
    inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, SEQ_LEN, num_nodes])
    labels = tf.compat.v1.placeholder(tf.float32, shape=[None, PRE_LEN, num_nodes])

    # evaluating process
    for m in range(total_batch):
        mini_batch = evalX[m * BATCH_SIZE : (m + 1) * BATCH_SIZE]
        mini_label = evalY[m * BATCH_SIZE : (m + 1) * BATCH_SIZE]
        _, loss1, rmse1, train_output = sess.run(
                feed_dict={inputs: mini_batch, labels: mini_label},
        )
        batch_loss.append(loss1)
        batch_rmse.append(rmse1 * max_value)
    
    # Tests completely at every epoch
    loss2, rmse2, test_output = sess.run(
        feed_dict={inputs: evalX, labels: evalY}
    )

    test_label = np.reshape(evalY, [-1, num_nodes])
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
    
    
    print("----------\n One Evaluation Iteration")
    print("Train_rmse: {:.4}".format(batch_rmse[-1]))
    print("Test_loss: {:.4}".format(loss2))
    print("Test_rmse: {:.4}".format(rmse))
    print("Test_acc: {:.4}\n".format(acc))
    
    index = test_rmse.index(np.min(test_rmse))
    print("-----------------------------------------\nEvaluation Metrics:")
    print("min_rmse: %r" % (np.min(test_rmse)))
    print("min_mae: %r" % (test_mae[index]))
    print("max_acc: %r" % (test_acc[index]))
    print("r2: %r" % (test_r2[index]))
    print("var: %r" % test_var[index])
        
    print("Complete")


if __name__ == "__main__":
    evaluate()
