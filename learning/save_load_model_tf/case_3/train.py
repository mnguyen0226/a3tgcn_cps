# Reference: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/4_Utils/save_restore_model.ipynb
# File trains simple neural network in MNIST and saves model locally.

from pickletools import optimize
import tensorflow as tf

# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# Sets parameters
learning_rate = 0.001
batch_size = 100
display_step = 4
model_path = "trained_model/model.ckpt"

# Network parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img_shape = 28x28)
n_classes = 10 # MNIST total classes (0-9 digits)

# Tf graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Creates model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weights and biases
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# 'Saver" op to save and restore all the variables
saver = tf.train.Saver()
            
def train():
    # Run first session
    print("Start 1st session...")
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)
        
        # Training cycles
        for epoch in range(3):
            avg_loss = 0.0
            total_batch = int(mnist.train.num_examples/batch_size)
            
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                
                # Run optimization op (back_prop) and cost op (to get loss value)
                _, c = sess.run([optimizer, loss], feed_dict = {x: batch_x, y: batch_y})
                
                # Compute average loss
                avg_loss += c/total_batch
            
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch: ", '%04d' % (epoch+1), "; Loss: ", "{:.9f}".format(avg_loss))
            
        print("First optimization finished!")
        
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
        # Save model weights to disk
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)
    
if __name__ == "__main__":
    train()