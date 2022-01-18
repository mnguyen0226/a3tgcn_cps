# Reference: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/4_Utils/save_restore_model.ipynb

import tensorflow as tf

# import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

def train():
    print("Start Training.")
    
if __name__ == "__main__":
    train()