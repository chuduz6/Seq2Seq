from tensorflow.contrib import rnn as rnn_cell
from tensorflow.python.ops import rnn
import tensorflow as tf

def highway_layer(inputs, embedding_size, scope):
    with tf.variable_scope(scope or "highway_layer"):
        d = embedding_size
        #print ("HIGHWAY LAYER: ", inputs)
        trans = fc_layer(inputs, d, scope='trans', activation_fn=None)
        trans = tf.nn.relu(trans)
        gate = fc_layer(inputs, d, scope='gate', activation_fn=None)
        gate = tf.nn.sigmoid(gate)
        out = gate * trans + (1 - gate) * inputs
        return out


def highway_network(inputs, num_layers, embedding_size, scope=None):
    with tf.variable_scope(scope or "highway_network"):
        prev = inputs
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev, embedding_size, scope="layer_{}".format(layer_idx))
            prev = cur
        return cur

          '''
            print ("BEFORE HIGHWAY: ", encoder_query_inputs_emb)
            
            with tf.variable_scope("highway"):
                encoder_query_inputs_emb = highway_network(encoder_query_inputs_emb, args.highway_num_layers, self.embedding_size)
                tf.get_variable_scope().reuse_variables()
                encoder_doc_inputs_emb = highway_network(encoder_doc_inputs_emb, args.highway_num_layers, self.embedding_size)

            print ("AFTER HIGHWAY: ", encoder_query_inputs_emb)
            '''