# Reference: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/4_Utils/save_restore_model.ipynb

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
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2'], biases['b2']))
    layer_2 = tf.nn.relu(layer_2)
    
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out'], biases['out'])
    return out_layer

# Store layers weights and biases

    
def train():
    print("Start Training.")
    
    
if __name__ == "__main__":
    train()