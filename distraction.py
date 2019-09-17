import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
linear = tf.contrib.layers.fully_connected

class DistractionGRUCell_soft(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, input_size=None, activation=tanh):
        if input_size is not None:
          print("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation
    
    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units        

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
          with vs.variable_scope("Gates"):  # Reset gate and update gate.
            # We start with bias of 1.0 to not reset and not update.
            r = linear(tf.concat((inputs, state), 1), self._num_units, activation_fn=None)
            g = linear(tf.concat((inputs, state), 1), self._num_units, activation_fn=None)
            u = linear(tf.concat((inputs, state), 1), self._num_units, activation_fn=None)
            r, u, g = tf.cast(sigmoid(r), tf.float32), tf.cast(sigmoid(u), tf.float32), tf.cast(sigmoid(g), tf.float32)
          print ("R SHAPE: ", r)
          print ("STATE: ", state)
          print ("INPUTS: ", inputs)
          print ("CONCAT: ", tf.concat((inputs, r * state), 1))
          with vs.variable_scope("Candidate"):
            c = self._activation(linear(tf.concat((inputs, r * state), 1), self._num_units, activation_fn=None))
          new_h = u * state + (1 - u) * c
          eps = 1e-13
          temp = math_ops.div(math_ops.reduce_sum(math_ops.mul(new_h, state),1), math_ops.reduce_sum(math_ops.mul(state,state),1) + eps)
          m = array_ops.transpose(g)
          t1 = math_ops.mul(m , temp)
          t1 = array_ops.transpose(t1) 
          distract_h = new_h  -  state * t1

        return distract_h, distract_h