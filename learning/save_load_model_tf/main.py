# Reference: https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

import tensorflow as tf

def main():
    w1 = tf.Variable(tf.random_normal(shape=[2]), name = 'w1')
    w2 =tf.Variable(tf.random_normal(shape=[5]), name = 'w2')
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, 'trained_models/my_test_model') # save trained model after every 1000 iteration.
    
    
    
    
if __name__ == "__main__":
    main()