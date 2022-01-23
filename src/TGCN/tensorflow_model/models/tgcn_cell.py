# Reference: https://github.com/lehaifeng/T-GCN/blob/master/T-GCN/T-GCN-TensorFlow/tgcn.py

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from utils import calculate_laplacian


class TGCNCell(RNNCell):
    """One cell of temporal convolutional neural network architecture including graph convolution and GRU"""

    def call(self, inputs, **kwargs):
        pass

    def __init__(
        self, num_units, adj, num_nodes, input_size=None, act=tf.nn.tanh, reuse=None
    ):
        """Constructors of one TGCN Cell.

        Args:
            num_units: Number of units.
            adj: Adjacency matrix.
            num_nodes: Number of nodes
            input_size: Input size. Defaults to None.
            act: Logistic regression function for supervised learning. Defaults to tf.nn.tanh.
            reuse: Defaults to None.
        """

        super(TGCNCell, self).__init__(_reuse=reuse)
        self._act = act
        self._nodes = num_nodes
        self._units = num_units
        self._adj = []
        self._adj.append(calculate_laplacian(adj))

    @property  # getter and setter in python class
    def state_size(self):
        return self._nodes * self._units

    @property  # getter and setter in python class
    def output_size(self):
        return self._units

    def __call__(self, inputs, state, scope=None):
        """Built-in methods in Python enable to write classes where the instances behave like function and can be called as function.
        Reference: https://www.geeksforgeeks.org/__call__-in-python/

        These are functions in 1 TGCN cells: https://arxiv.org/pdf/1811.05320.pdf

        Args:
            inputs: Inputs.
            state: State.
            scope: Defaults to None.

        Returns:
            [type]: [description]
        """
        with tf.variable_scope(scope or "tgcn"):
            with tf.variable_scope("gates"):
                # ut = σ(Wu [f(A, Xt), ht−1] + bu)
                # rt = σ(Wr [f(A, Xt), ht−1] + br)
                value = tf.nn.sigmoid(
                    self._gc(inputs, state, 2 * self._units, bias=1.0, scope=scope)
                )
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)

            with tf.variable_scope("candidate"):
                # ct = tanh(Wc [f(A, Xt),(rt ∗ ht−1)] + bc)
                r_state = r * state
                c = self._act(self._gc(inputs, r_state, self._units, scope=scope))

            # ht = ut ∗ ht−1 + (1 − ut) ∗ ct
            new_h = u * state + (1 - u) * c

        return new_h, new_h

    def _gc(self, inputs, state, output_size, bias=0.0, scope=None):
        """Graph Convolution cell.

        Args:
            inputs: Inputs.
            state: States.
            output_size: Output size.
            bias: Bias rate. Defaults to 0.0
            scope: Defaults to None.

        """
        ## inputs:(-1,num_nodes)
        inputs = tf.expand_dims(inputs, 2)

        ## state:(batch,num_node,gru_units)
        state = tf.reshape(state, (-1, self._nodes, self._units))

        ## concat
        x_s = tf.concat([inputs, state], axis=2)
        input_size = x_s.get_shape()[2].value

        ## (num_node,input_size,-1)
        x0 = tf.transpose(x_s, perm=[1, 2, 0])
        x0 = tf.reshape(x0, shape=[self._nodes, -1])

        # scope = tf.get_variable_scope()
        scope = tf.compat.v1.get_variable_scope()
        with tf.variable_scope(scope):
            for m in self._adj:
                x1 = tf.sparse_tensor_dense_matmul(m, x0)

            #                print(x1)
            x = tf.reshape(x1, shape=[self._nodes, input_size, -1])
            x = tf.transpose(x, perm=[2, 0, 1])
            x = tf.reshape(x, shape=[-1, input_size])
            weights = tf.get_variable(
                "weights",
                [input_size, output_size],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            x = tf.matmul(x, weights)  # (batch_size * self._nodes, output_size)
            biases = tf.get_variable(
                "biases",
                [output_size],
                initializer=tf.constant_initializer(bias, dtype=tf.float32),
            )
            x = tf.nn.bias_add(x, biases)
            x = tf.reshape(x, shape=[-1, self._nodes, output_size])
            x = tf.reshape(x, shape=[-1, self._nodes * output_size])
        return x
