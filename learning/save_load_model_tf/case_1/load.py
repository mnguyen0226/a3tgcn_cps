import tensorflow as tf


def load_model():
    print("Load trained model:")

    sess = tf.Session()

    # First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph("trained_models/my_test_model.meta")
    saver.restore(sess, tf.train.latest_checkpoint("./trained_models/"))

    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data

    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name("w1:0")
    w2 = graph.get_tensor_by_name("w2:0")
    feed_dict = {w1: 13.0, w2: 17.0}  # fetch in new data to "test"

    # Now, access the op that you want to run.
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

    print(sess.run(op_to_restore, feed_dict))
    # This will print 60 which is calculated
    # using new values of w1 and w2 and saved value of b1.

    print("Complete.")


if __name__ == "__main__":
    load_model()
