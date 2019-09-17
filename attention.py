import tensorflow as tf

fc_layer = tf.contrib.layers.fully_connected
initializer = tf.contrib.layers.xavier_initializer()

def softsel(target, logits, scope=None):
    with tf.name_scope(scope or "Softsel"):
        a = tf.nn.softmax(logits)
        out = a*target
        return out

def bi_attention(h, u, scope=None):
    with tf.variable_scope(scope or "bi_attention"):
        output_size = int(h.get_shape()[-1])
        u_logits = fc_layer(tf.concat((h,u,h*u), 2), output_size, activation_fn=None)
        print ("U_LOGITS: ", u_logits)
        u_a = softsel(u, u_logits)  
        h_a = softsel(h, u_logits)  
        print ("SHAPE OF U_A: ", u_a)
        p0 = fc_layer(tf.concat([h, u_a, h * u_a, h * h_a], 2), output_size)
        print ("PO: ", p0)
        return p0
        
def self_attention(inputs, time_major=False, return_alphas=False):

    hidden_size = int(inputs.get_shape()[-1])
    attention_size = int (hidden_size/3)
    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    

    if not return_alphas:
        return output
    else:
        return output, alphas   