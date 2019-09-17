 # coding: utf-8
import tensorflow as tf
from helper import get_init_embedding
#from args import load_args
#from vocabulary import load_dict, create_dict
from attention import bi_attention, self_attention
from distraction import DistractionGRUCell_soft

MARK_PAD = "<PAD>"
MARK_UNK = "<UNK>"
MARK_EOS = "<EOS>"
MARK_GO = "<GO>"
MARKS = [MARK_PAD, MARK_UNK, MARK_EOS, MARK_GO]
ID_PAD = 0
ID_UNK = 1
ID_EOS = 2
ID_GO = 3

fc_layer = tf.contrib.layers.fully_connected

def highway_layer(inputs, scope):
    with tf.variable_scope(scope or "highway_layer"):
        d = int(inputs.get_shape()[-1])
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
            cur = highway_layer(prev, scope="layer_{}".format(layer_idx))
            prev = cur
        return cur

class BiGRUModel(object):

    def __init__(self,doc_dict, sum_dict, args, batch_size, forward_only=False, dtype=tf.float32):
        self.dtype = dtype
        self.source_vocab_size = args.doc_vocab_size
        self.target_vocab_size = args.sum_vocab_size
        self.batch_size = batch_size
        self.beam_width = args.beam_width
        self.learning_rate = args.learning_rate
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.state_size = args.size
        self.sum_dict = sum_dict
        self.doc_dict = doc_dict
        self.embedding_size = args.embsize
        self.num_layers = args.num_layers

        self.encoder_query_inputs = tf.placeholder( tf.int32, shape=[self.batch_size, None], name='qI')
        self.encoder_doc_inputs = tf.placeholder( tf.int32, shape=[self.batch_size, None], name='docI')
        self.decoder_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, None], name='decI')
        self.decoder_targets = tf.placeholder(tf.int32, shape=[self.batch_size, None], name='decT')
        self.encoder_query_len = tf.placeholder(tf.int32, shape=[self.batch_size], name='qL')
        self.encoder_doc_len = tf.placeholder(tf.int32, shape=[self.batch_size], name='docL')
        self.decoder_len = tf.placeholder(tf.int32, shape=[self.batch_size], name='decL')


        with tf.variable_scope("seq2seq", dtype=dtype):
            
            with tf.variable_scope("decoder/projection"):
                self.projection_layer = tf.layers.Dense(self.target_vocab_size, use_bias=False)
                
            
            with tf.variable_scope("embedding"):                
                if not forward_only and args.load_pretrained_embeddings and args.pretrained_embeddings_vec_path is not None:
                    init_embeddings_doc = tf.constant(get_init_embedding('doc', args.pretrained_embeddings_vec_path, doc_dict, args), dtype=tf.float32)
                    init_embeddings_sum = tf.constant(get_init_embedding('sum', args.pretrained_embeddings_vec_path, sum_dict, args), dtype=tf.float32)
                else:
                    init_embeddings_doc = tf.random_uniform([self.source_vocab_size, self.embedding_size], -1.0, 1.0)
                    init_embeddings_sum = tf.random_uniform([self.target_vocab_size, self.embedding_size], -1.0, 1.0)                
                encoder_query_emb = tf.get_variable("embedding_query", initializer=init_embeddings_doc)
                encoder_doc_emb = tf.get_variable("embedding_doc", initializer=init_embeddings_doc)
                decoder_emb = tf.get_variable("embedding", initializer= init_embeddings_sum)
                encoder_query_inputs_emb = tf.nn.embedding_lookup(encoder_query_emb, self.encoder_query_inputs)
                encoder_doc_inputs_emb = tf.nn.embedding_lookup(encoder_doc_emb, self.encoder_doc_inputs)
                decoder_inputs_emb = tf.nn.embedding_lookup(decoder_emb, self.decoder_inputs)
            
            print ("BEFORE HIGHWAY: ", encoder_query_inputs_emb)
            
            with tf.variable_scope("highway"):
                encoder_query_inputs_emb = highway_network(encoder_query_inputs_emb, args.highway_num_layers, self.embedding_size)
                tf.get_variable_scope().reuse_variables()
                encoder_doc_inputs_emb = highway_network(encoder_doc_inputs_emb, args.highway_num_layers, self.embedding_size)
                decoder_inputs_emb = highway_network(decoder_inputs_emb, args.highway_num_layers, self.embedding_size)
                              
            with tf.name_scope("encoder"):
                self.encoder_query_outputs, self.encoder_query_states = self.bi_gru_layer(encoder_query_inputs_emb, self.encoder_query_len, args.query_encoder_dropout_flag, args.query_encoder_keep_prob, args.stack_resolve, forward_only, 'encoder_query')
                print ("ENCODER QUERY STATES: ", self.encoder_query_states)
                
                self.encoder_doc_outputs, self.encoder_doc_states = self.bi_gru_layer(encoder_doc_inputs_emb, self.encoder_doc_len, args.doc_encoder_dropout_flag, args.doc_encoder_keep_prob, args.stack_resolve, forward_only, 'encoder_doc')   
                print ("ENCODER DOC OUTPUTS: ", self.encoder_doc_outputs)
        
    
            with tf.name_scope("bi_attention"):
                bi_context_vectors = bi_attention(self.encoder_doc_outputs, self.encoder_query_outputs, scope='attention')
                print ("BI CONTEXT VECTORS: ", bi_context_vectors)
                
            with tf.name_scope("self_attention"):
                outputs_sa, states_sa = self.bi_gru_layer(bi_context_vectors, self.encoder_doc_len, args.query_encoder_dropout_flag, args.query_encoder_keep_prob, args.stack_resolve, forward_only, 'encoding_self_attention')
                print ("OUTPUTS: ", outputs_sa)
                self_attended_vectors = self_attention(outputs_sa)
                self_attended_vectors_reshape = fc_layer(self_attended_vectors, 2*self.state_size, activation_fn=None)
                self_attended_vectors_reshape = tf.expand_dims(self_attended_vectors_reshape, 1)

                print ("SELF ATTENTED VECTORS RESHAPE: ", self_attended_vectors_reshape)
                
                all_vec = tf.reduce_sum((bi_context_vectors, self_attended_vectors_reshape), 0)
                print ("BEFORE SQUEEZE: ", all_vec)
                all_vec = tf.squeeze(all_vec)
                print ("ALL VEC: ", all_vec)
                
                distract_state  = fc_layer(tf.concat((self.encoder_query_states, self.encoder_doc_states), 1), 2*self.state_size, activation_fn=None)
                print ("DISTRACT STATE: ", distract_state)
                distraction_cell = DistractionGRUCell_soft(2*self.state_size)
                distract_output, distract_state = distraction_cell(all_vec, distract_state)  
                
                ''''
                self_attention_query = self_attention(self.encoder_query_outputs)
                self_attention_doc = self_attention(self.encoder_doc_outputs)
                print ("SELF ATTENTION QUERY: ", self_attention_query)
                '''
                             
            with tf.name_scope("decoder"), tf.variable_scope("decoder"):       
                decoder_cell = tf.contrib.rnn.GRUCell(self.state_size)
                if(args.decoder_dropout_flag):
                    if not forward_only:
                        decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, output_keep_prob=args.decoder_keep_prob)                
                
                if not forward_only:
                    
                    self.att_states_query = self.encoder_query_outputs
                    self.att_states_query.set_shape([self.batch_size, None, self.state_size*2])
                    
                    self.att_states_doc = self.encoder_doc_outputs
                    self.att_states_doc.set_shape([self.batch_size, None, self.state_size*2])
                    
                    #QUERY ATTENTION
                    attention_mechanism_query = tf.contrib.seq2seq.BahdanauAttention(self.state_size, self.att_states_query, self.encoder_query_len)
                    attention_wrapper_query = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism_query, self.state_size * 2) 

                    #encoder_combined_states = fc_layer(tf.concat((encoder_query_states_fw, encoder_query_states_bw), 1), self.state_size)
                    #print ("ENCODER COMBINED STATES: ", encoder_combined_states)
                    
                    attention_wrapper_query_zero = attention_wrapper_query.zero_state(dtype=tf.float32, batch_size=self.batch_size)
                    initial_state_query = attention_wrapper_query_zero.clone(cell_state=self.encoder_query_states)
                    #initial_state = tf.contrib.seq2seq.AttentionWrapperState(cell_state = self.encoder_query_states, attention = attention_wrapper_query_zero, attention_state = attention_wrapper_query, time = 0 ,alignments=None , alignment_history=())

                    #DOCUMENT ATTENTION
                    attention_mechanism_doc = tf.contrib.seq2seq.BahdanauAttention(self.state_size, self.att_states_doc, self.encoder_doc_len)                
                    attention_wrapper_doc = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism_doc, self.state_size * 2)

                    attention_wrapper_doc_zero = attention_wrapper_doc.zero_state(dtype=tf.float32, batch_size=self.batch_size)
                    initial_state_doc = attention_wrapper_doc_zero.clone(cell_state=self.encoder_doc_states)
                    #attention_wrapper_state_doc = tf.contrib.seq2seq.AttentionWrapperState(cell_state = self.encoder_doc_states, attention = attention_wrapper_doc_zero, time = 0, attention_state = attention_wrapper_doc, alignments=None , alignment_history=())

                    initial_state = initial_state_query, initial_state_doc
                    attention_wrapper_combined = attention_wrapper_query, attention_wrapper_doc
                    
                    #self.combined_final_states = (encoder_doc_states_fw, encoder_doc_states_fw)
                    #print ("COMBINED FINAL STATES: ", self.combined_final_states)
                    #initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
                    #initial_state = initial_state.clone(cell_state=self.combined_final_states)
                    
                    decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(distract_output, self.target_vocab_size)

                    helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_emb, self.decoder_len)
                    decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper, initial_state=initial_state_doc)
                    outputs = tf.contrib.seq2seq.dynamic_decode(decoder)
                    outputs_logits = outputs[0].rnn_output
                    self.outputs = outputs_logits    

                else:
                    
                    self.att_states_query = self.encoder_query_outputs
                    self.att_states_query.set_shape([self.batch_size, None, self.state_size*2])
                    tiled_att_states_query = tf.contrib.seq2seq.tile_batch(self.att_states_query, multiplier=self.beam_width)
                    tiled_encoder_query_len = tf.contrib.seq2seq.tile_batch(self.encoder_query_len, multiplier=self.beam_width)


                    self.att_states_doc = self.encoder_doc_outputs
                    self.att_states_doc.set_shape([self.batch_size, None, self.state_size*2])
                    tiled_att_states_doc = tf.contrib.seq2seq.tile_batch(self.att_states_doc, multiplier=self.beam_width)
                    tiled_encoder_doc_len= tf.contrib.seq2seq.tile_batch(self.encoder_doc_len, multiplier=self.beam_width)


                    attention_mechanism_query = tf.contrib.seq2seq.BahdanauAttention(self.state_size, tiled_att_states_query, tiled_encoder_query_len)
                    attention_state_query = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism_query, self.state_size * 2)              

                    attention_mechanism_doc = tf.contrib.seq2seq.BahdanauAttention(self.state_size, tiled_att_states_doc, tiled_encoder_doc_len)                
                    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism_doc, self.state_size * 2)               

                    self.combined_final_states = tf.concat((encoder_doc_states_fw, encoder_doc_states_fw), 1)
                    print ("COMBINED FINAL STATES: ", self.combined_final_states)
                    self.combined_final_states = tf.contrib.seq2seq.tile_batch(self.combined_final_states, multiplier=self.beam_width)
                    print ("COMBINED FINAL STATES: ", self.combined_final_states)
                    initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size*self.beam_width)
                    #initial_state = initial_state.clone(cell_state=self.combined_final_states)


                    decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, self.target_vocab_size)


                    decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                               embedding=decoder_emb,
                                                               start_tokens=tf.fill([self.batch_size], tf.constant(ID_GO)),
                                                               end_token=tf.constant(ID_EOS), 
                                                               initial_state=initial_state,
                                                               beam_width=self.beam_width,
                                                               output_layer=None)
                    print ("DECODER: ", decoder)
                    outputs = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=args.beam_max_iterations)
                    self.predictions = outputs[0].predicted_ids

                
            with tf.variable_scope("loss"):
                if not forward_only:
                    #print ("SHAPE OF OUTPUTS: ", tf.shape(outputs_logits, out_type=tf.int32 ))
                    #print ("SHAPE OF TARGETS: ", tf.shape(self.decoder_targets, out_type=tf.int32))
                    weights = tf.sequence_mask(self.decoder_len, dtype=tf.float32)
                    loss_t = tf.contrib.seq2seq.sequence_loss(outputs_logits, self.decoder_targets, weights, average_across_timesteps=False, average_across_batch=False)
                    self.loss = tf.reduce_sum(loss_t)/self.batch_size



                    predictions = tf.cast(tf.argmax(outputs_logits, axis=2), tf.int32) 
                    self.accuracy = tf.contrib.metrics.accuracy(predictions, self.decoder_targets)



                    params = tf.trainable_variables()
                    opt = tf.train.AdadeltaOptimizer(self.learning_rate, epsilon=1e-4)
                    gradients = tf.gradients(self.loss, params)
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients, args.max_gradient)
                    self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
                    tf.summary.scalar('loss', self.loss)
                    tf.summary.scalar('accuracy', self.accuracy)

        
        self.saver = tf.train.Saver(max_to_keep=args.max_to_keep)
        self.summary_merge = tf.summary.merge_all()

    def bi_gru_layer(self, inputs_emb, sequence_length, dropout_flag, keep_prob, stack_resolve, forward_only, scope):
        with tf.variable_scope(scope):                                   
            fw_cells = [tf.contrib.rnn.GRUCell(self.state_size) for _ in range(self.num_layers)]
            bw_cells = [tf.contrib.rnn.GRUCell(self.state_size) for _ in range(self.num_layers)]
            if(dropout_flag):
                if not forward_only:
                    fw_cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob) for cell in fw_cells]
                    bw_cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob) for cell in bw_cells]
            outputs, states_fw, states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cells, bw_cells, inputs_emb, sequence_length=sequence_length, dtype=self.dtype)
            print ("FW: ", states_fw)
            if(stack_resolve == 'sum'):
                states_fw = tf.reduce_sum(states_fw, axis=0)
                states_bw = tf.reduce_sum(states_bw, axis=0)
            else:
                states_fw = states_fw[self.num_layers-1]
                states_bw = states_bw[self.num_layers-1]
            states = fc_layer(tf.concat((states_fw, states_bw), 1), self.state_size)
            outputs = tf.concat(outputs, 2)
            return outputs, states

    
        
    def step(self,session,encoder_doc_inputs,encoder_query_inputs,decoder_inputs,encoder_doc_len,encoder_query_len,decoder_len,forward_only,summary_writer=None):

        # dim fit is important for sequence_mask
        # TODO better way to use sequence_mask
        if encoder_query_inputs.shape[1] != max(encoder_query_len):
            raise ValueError("encoder_query_inputs and encoder_query_len does not fit")
        if encoder_doc_inputs.shape[1] != max(encoder_doc_len):
            raise ValueError("encoder_doc_inputs and encoder_doc_len does not fit")
        if not forward_only and             decoder_inputs.shape[1] != max(decoder_len) + 1:
            raise ValueError("decoder_inputs and decoder_len does not fit")
            
        input_feed = {}
        input_feed[self.encoder_query_inputs] = encoder_query_inputs
        input_feed[self.encoder_doc_inputs] = encoder_doc_inputs
        input_feed[self.decoder_inputs] = decoder_inputs[:, :-1]
        input_feed[self.decoder_targets] = decoder_inputs[:, 1:]
        input_feed[self.encoder_query_len] = encoder_query_len
        input_feed[self.encoder_doc_len] = encoder_doc_len
        input_feed[self.decoder_len] = decoder_len

        output_feed = [self.loss, self.accuracy, self.updates]
        
        if summary_writer:
            output_feed += [self.summary_merge, self.global_step]

        outputs = session.run(output_feed, input_feed)

        if summary_writer:
            summary_writer.add_summary(outputs[3], outputs[4])
        return outputs[:3]    

    
    def step_beam(self, session, encoder_doc_inputs, encoder_query_inputs, encoder_doc_len, encoder_query_len, beam_size):
        
  
        #print ("ENCODER QUERY SHAPE: ", session.run(tf.shape(encoder_query_len)) )  
        
        input_feed = {}
        input_feed[self.encoder_query_inputs] = encoder_query_inputs
        input_feed[self.encoder_doc_inputs] = encoder_doc_inputs
        print ()
        input_feed[self.encoder_query_len] = encoder_query_len
        input_feed[self.encoder_doc_len] = encoder_doc_len
        
        output_feed = [self.predictions]
        outputs = session.run(output_feed, input_feed)
        return outputs



'''
with tf.Session() as sess:
    args = load_args()
    doc_dict = load_dict(args.doc_dict_path, args.doc_vocab_size)
    sum_dict = load_dict(args.sum_dict_path, args.sum_vocab_size)
    create_model(sess, doc_dict, sum_dict, args.train_batch_size, False, False, args)
'''        