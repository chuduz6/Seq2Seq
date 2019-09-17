
# coding: utf-8

# In[1]:


from __future__ import print_function
import logging
import json
import io
#import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import re
import string
import random
import numpy as np
import tensorflow as tf
import math
import os
import sys
import time
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import pickle
from sklearn.model_selection import train_test_split


# In[2]:


# this loop is needed to reset the flags to that notebook won't throw duplicate flags error
from absl import flags
for name in list(flags.FLAGS):
  delattr(flags.FLAGS, name)

# Dictionary parameters
tf.app.flags.DEFINE_string("doc_dict_path", "doc_dict.txt", "Document Dictionary output.")
tf.app.flags.DEFINE_string("sum_dict_path", "sum_dict.txt", "Summary Dictionary output.")
tf.app.flags.DEFINE_boolean("create_dict_flag", False, "Whether to create new dictionary or not ")
tf.app.flags.DEFINE_integer("doc_vocab_size", 30000, "Document vocabulary size.")
tf.app.flags.DEFINE_integer("sum_vocab_size", 10000, "Summary vocabulary size.")
tf.app.flags.DEFINE_float("train_test_split", 0.33, "Test Split ratio")
tf.app.flags.DEFINE_boolean("pretrained_embeddings", True, "Whether to look up pre-trained embedding for not ")
tf.app.flags.DEFINE_string("embedding_path", "glove.twitter.27B.100d.txt", "Embedding path")


# needed to get rid of missing f parameter
tf.app.flags.DEFINE_string('f', '', 'kernel')

# Optimization Parameters
tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_integer("size", 400, "Size of hidden layers.")
tf.app.flags.DEFINE_integer("embsize", 100, "Size of embedding.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_float("max_gradient", 1.0, "Clip gradients l2 norm to this range.")
tf.app.flags.DEFINE_integer("batch_size", 5, "Batch size")
tf.app.flags.DEFINE_integer("beam_width_size", 2, "beam size")

tf.app.flags.DEFINE_integer("max_epochs", 1, "Maximum training Epochs.")
tf.app.flags.DEFINE_float("doc_encoder_keep_prob", 0.5, "doc keep prob")
tf.app.flags.DEFINE_float("query_encoder_keep_prob", 0.5, "query keep prob")
tf.app.flags.DEFINE_float("decoder_keep_prob", 0.5, "decoder keep prob")


tf.app.flags.DEFINE_boolean("doc_encoder_dropout_flag", True, "Whether to drop out doc encoder cell")
tf.app.flags.DEFINE_boolean("query_encoder_dropout_flag", True, "Whether to drop out query encoder cell")
tf.app.flags.DEFINE_boolean("decoder_dropout_flag", True, "Whether to drop out query encoder cell")


# Data Directory Paramters
tf.app.flags.DEFINE_string("data_dir", "data_sample_train.json", "Data directory")
tf.app.flags.DEFINE_string("test_file", "data_sample_test.txt", "Test filename.")

# Output Data Directory Parameters
tf.app.flags.DEFINE_string("test_output", "test_output.txt", "Test output.")
tf.app.flags.DEFINE_string("train_dir", "model", "Training directory.")
tf.app.flags.DEFINE_string("tfboard", "tfboard", "Tensorboard log directory.")
tf.app.flags.DEFINE_integer("steps_per_print", 50, "Training steps between printing.")
tf.app.flags.DEFINE_integer("steps_per_validation", 1000, "Training steps between validations.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 750, "Training steps between checkpoints.")
tf.app.flags.DEFINE_boolean("load_checkpoint", False, "Flag to whether load the checkpoint or not")


# Progam Running Mode: Train or decode
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for testing.")
tf.app.flags.DEFINE_boolean("geneos", True, "Do not generate EOS. ")

tf.app.flags.DEFINE_integer('seed',           3435, 'random number generator seed')
FLAGS = tf.app.flags.FLAGS

print ("DONE")


# In[3]:


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
#logging.basicConfig(level=print,format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",datefmt='%b %d %H:%M')


# In[4]:


MARK_PAD = "<PAD>"
MARK_UNK = "<UNK>"
MARK_EOS = "<EOS>"
MARK_GO = "<GO>"
MARKS = [MARK_PAD, MARK_UNK, MARK_EOS, MARK_GO]
ID_PAD = 0
ID_UNK = 1
ID_EOS = 2
ID_GO = 3


# In[5]:


def load_dict(dict_path, max_vocab=None):
    print("Try load dict from {}.".format(dict_path))
    try:
        dict_file = open(dict_path)
        dict_data = dict_file.readlines()
        dict_file.close()
    except:
        print("Load dict {dict} failed, create later.".format(dict=dict_path))
        return None

    dict_data = list(map(lambda x: x.split(), dict_data))
    if max_vocab:
        dict_data = list(filter(lambda x: int(x[0]) < max_vocab, dict_data))
    tok2id = dict(map(lambda x: (x[1], int(x[0])), dict_data))
    id2tok = dict(map(lambda x: (int(x[0]), x[1]), dict_data))
    print("Load dict {} with {} words.".format(dict_path, len(tok2id)))
    return (tok2id, id2tok)


# In[6]:


def create_dict(dict_path, corpus, max_vocab=None):
    print("Create dict {}.".format(dict_path))
    counter = {}
    counter2 = 0
    for line in corpus:
        for word in line:
            try:
                counter[word] += 1
            except:
                counter[word] = 1

    for mark_t in MARKS:
        if mark_t in counter:
            del counter[mark_t]
            logging.warning("{} appears in corpus.".format(mark_t))

    counter = list(counter.items())
    counter.sort(key=lambda x: -x[1])
    words = list(map(lambda x: x[0], counter))
    words = [MARK_PAD, MARK_UNK, MARK_EOS, MARK_GO] + words
    if max_vocab:
        words = words[:max_vocab]

    tok2id = dict()
    id2tok = dict()
    with open(dict_path, 'w') as dict_file:
        for idx, tok in enumerate(words):
            print(idx, tok, file=dict_file)
            tok2id[tok] = idx
            id2tok[idx] = tok

    print("Create dict {} with {} words.".format(dict_path, len(words)))
    return (tok2id, id2tok)


# In[7]:


def corpus_map2id(data, tok2id):
    ret = []
    unk = 0
    tot = 0
    for doc in data:
        tmp = []
        for word in doc:
            tot += 1
            try:
                tmp.append(tok2id[word])
            except:
                tmp.append(ID_UNK)
                unk +=1
        ret.append(tmp)
    print ("TOTAL :", tot, " UNK :", unk)
    return ret, (tot - unk)/tot


# In[8]:


def sen_map2tok(sen, id2tok):
    return list(map(lambda x: id2tok[x], sen))


# In[9]:



print("Load document from {}.".format(FLAGS.data_dir))
with io.open(FLAGS.data_dir, 'r', encoding='ascii', errors='ignore') as input_file:
    data = json.loads(input_file.read())
    docs = []
    summaries = []
    queries = []
    for i in range (len(data['passages'])):
        docs.append(' '.join([data['passages'][str(i)][j]['passage_text'] for j in range (len(data['passages'][str(i)]))]))
        summaries.append(''.join(data['answers'][str(i)]))
        queries.append(data['query'][str(i)])

assert     len(docs) == len(queries)
print (len(docs))


# In[10]:


#print ("DOCS :", docs[0])
print ("QUERY :", queries[0])
print ("ANSWER :", summaries[0])


# In[11]:


with tf.device('/gpu:1'):
    print ("Splitting docs...")

    #print ("BEFORE SPLITTING: ", docs[0])
    now = time.time()
    docs_splitted = list(map(lambda x: word_tokenize(x), docs))
    docs_splitted = [[word.lower()for word in doc] for doc in docs_splitted]
    del docs
    print ("TIME TAKEN TO SPLIT DOCS: ", time.time()-now)
    #print ("AFTER SPLITTING: ", docs[0])
    print ("DONE")


# In[12]:


print ("Splitting queries...")
queries_splitted = list(map(lambda x: [y.lower() for y in word_tokenize(x)], queries)) 
del queries
print ("DONE")
print (len(queries_splitted))


# In[13]:


print ("Splitting summaries...")
summaries_splitted = list(map(lambda x: [y.lower() for y in word_tokenize(x)], summaries)) 
del summaries
print ("DONE")


# In[14]:


print ("Working on dictionary...")
if(FLAGS.create_dict_flag):
    doc_dict = create_dict(FLAGS.doc_dict_path, docs_splitted+queries_splitted, FLAGS.doc_vocab_size)
    sum_dict = create_dict(FLAGS.sum_dict_path, summaries_splitted, FLAGS.sum_vocab_size)
else:
    doc_dict = load_dict(FLAGS.doc_dict_path, FLAGS.doc_vocab_size)
    sum_dict = load_dict(FLAGS.sum_dict_path, FLAGS.sum_vocab_size)

print ("DONE")
print (len(doc_dict))
print (len(sum_dict))
#print (sum_dict[0])


# In[15]:


print ("Converting to ids...")
docid, cover = corpus_map2id(docs_splitted, doc_dict[0])
print("Doc dict covers {:.2f}% words.".format(cover * 100))

sumid, cover = corpus_map2id(summaries_splitted, sum_dict[0])
print("Sum dict covers {:.2f}% words.".format(cover * 100))

queryid, cover = corpus_map2id(queries_splitted, doc_dict[0])
print("Query dict covers {:.2f}% words.".format(cover * 100))

print ("DONE")


# In[16]:


print (len(queryid[0]))
print (queryid[0])
print (" ".join(sen_map2tok(queryid[0], doc_dict[1])))


# In[17]:



print ("DONE")


# In[18]:


def get_word_vectors():
    glove_file = FLAGS.embedding_path
    word2vec_file = get_tmpfile("word2vec_format.vec")
    glove2word2vec(glove_file, word2vec_file)
    print("Loading Glove vectors...")
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)
    return word_vectors
#word_vectors = get_word_vectors()
print ("DONE")


# In[19]:


'''
with open('word_vectors.pickle', 'wb') as handle:
    pickle.dump(word_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''
with open('word_vectors.pickle', 'rb') as handle:
    word_vectors = pickle.load(handle)


# In[20]:


def get_init_embedding(word_vectors, word_vocab, vocab_size):
    word_vec_list = list()
    success_count = 0
    failure_count = 0
    for word, _ in word_vocab[0].items():
        try:
            word_vec = word_vectors.word_vec(word)
            success_count += 1
        except KeyError:
            word_vec = np.zeros([FLAGS.embsize], dtype=np.float32)
            failure_count += 1
        word_vec_list.append(word_vec)

    word_vec_list[2] = np.random.normal(0, 1, FLAGS.embsize)
    word_vec_list[3] = np.random.normal(0, 1, FLAGS.embsize)
    print ("SUCCESS COUNT: ", success_count, " FAILURE COUNT: ", failure_count)
    return np.array(word_vec_list)
print ("DONE")


# In[21]:


start_time = time.time()
if FLAGS.pretrained_embeddings:
    init_embeddings_doc = tf.constant(get_init_embedding(word_vectors, doc_dict, FLAGS.doc_vocab_size), dtype=tf.float32)
    init_embeddings_sum = tf.constant(get_init_embedding(word_vectors, sum_dict, FLAGS.sum_vocab_size), dtype=tf.float32)
print ("LOADED GLOVE VECTORS IN TIME: ", time.time() - start_time)


# In[22]:


fc_layer = tf.contrib.layers.fully_connected


# In[34]:


class BiGRUModel(object):

    def __init__(self,doc_dict, sum_dict, source_vocab_size,target_vocab_size, buckets, state_size, num_layers, embedding_size, max_gradient, batch_size, learning_rate, forward_only=False, beam_width=1, dtype=tf.float32):

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.beam_width = beam_width
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.state_size = state_size
        self.sum_dict = sum_dict
        self.doc_dict = doc_dict
        self.embedding_size = embedding_size

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
                if not forward_only and FLAGS.pretrained_embeddings is not None:
                    init_embeddings_doc = tf.constant(get_init_embedding(word_vectors, doc_dict, FLAGS.doc_vocab_size), dtype=tf.float32)
                    init_embeddings_sum = tf.constant(get_init_embedding(word_vectors, sum_dict, FLAGS.sum_vocab_size), dtype=tf.float32)
                else:
                    init_embeddings_doc = tf.random_uniform([self.source_vocab_size, self.embedding_size], -1.0, 1.0)
                    init_embeddings_sum = tf.random_uniform([self.target_vocab_size, self.embedding_size], -1.0, 1.0)                
                encoder_query_emb = tf.get_variable("embedding_query", initializer=init_embeddings_doc)
                encoder_doc_emb = tf.get_variable("embedding_doc", initializer=init_embeddings_doc)
                decoder_emb = tf.get_variable("embedding", initializer= init_embeddings_sum)
                encoder_query_inputs_emb = tf.nn.embedding_lookup(encoder_query_emb, self.encoder_query_inputs)
                encoder_doc_inputs_emb = tf.nn.embedding_lookup(encoder_doc_emb, self.encoder_doc_inputs)
                decoder_inputs_emb = tf.nn.embedding_lookup(decoder_emb, self.decoder_inputs)

                
            with tf.variable_scope("encoder_query"):                                   
                encoder_fw_cells_query = [tf.contrib.rnn.GRUCell(state_size) for _ in range(FLAGS.num_layers)]
                encoder_bw_cells_query = [tf.contrib.rnn.GRUCell(state_size) for _ in range(FLAGS.num_layers)]
                if(FLAGS.query_encoder_dropout_flag):
                    if not forward_only:
                        encoder_fw_cells_query = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=FLAGS.query_encoder_keep_prob) for cell in encoder_fw_cells_query]
                        encoder_bw_cells_query = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=FLAGS.query_encoder_keep_prob) for cell in encoder_bw_cells_query]
                encoder_query_outputs, encoder_query_states_fw, encoder_query_states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(encoder_fw_cells_query, encoder_bw_cells_query, encoder_query_inputs_emb, sequence_length=self.encoder_query_len, dtype=dtype)
                print ("FW: ", encoder_query_states_fw)
                self.encoder_query_states = (encoder_query_states_fw, encoder_query_states_bw)
                self.encoder_query_outputs = tf.concat(encoder_query_outputs, 2)
                

                #print ("ENCODER QUERY STATES: ", self.encoder_query_states)

            with tf.variable_scope("encoder_doc"):                
                encoder_fw_cells_doc = [tf.contrib.rnn.GRUCell(state_size) for _ in range(FLAGS.num_layers)]
                encoder_bw_cells_doc = [tf.contrib.rnn.GRUCell(state_size) for _ in range(FLAGS.num_layers)]
                if(FLAGS.doc_encoder_dropout_flag):
                    if not forward_only:
                        encoder_fw_cells_doc = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=FLAGS.doc_encoder_keep_prob) for cell in encoder_fw_cells_doc]
                        encoder_bw_cells_doc = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=FLAGS.doc_encoder_keep_prob) for cell in encoder_bw_cells_doc]
                encoder_doc_outputs, encoder_doc_states_fw, encoder_doc_states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(encoder_fw_cells_doc, encoder_bw_cells_doc, encoder_doc_inputs_emb, sequence_length=self.encoder_doc_len, dtype=dtype)
                
                self.encoder_doc_states = (encoder_doc_states_fw, encoder_doc_states_fw)
                self.encoder_doc_outputs = tf.concat(encoder_doc_outputs, 2)
                
                #print ("ENCODER DOC STATES: ", self.encoder_doc_states)
                
                    
            with tf.name_scope("decoder"), tf.variable_scope("decoder"):       
                decoder_cell = tf.contrib.rnn.GRUCell(state_size)
                if(FLAGS.decoder_dropout_flag):
                    if not forward_only:
                        decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, output_keep_prob=FLAGS.decoder_keep_prob)                
                
                if not forward_only:
                    
                    self.att_states_query = self.encoder_query_outputs
                    self.att_states_query.set_shape([self.batch_size, None, state_size*2])
                    
                    self.att_states_doc = self.encoder_doc_outputs
                    self.att_states_doc.set_shape([self.batch_size, None, state_size*2])

                    attention_mechanism_query = tf.contrib.seq2seq.BahdanauAttention(state_size, self.att_states_query, self.encoder_query_len)
                    attention_state_query = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism_query, state_size * 2)              

                    attention_mechanism_doc = tf.contrib.seq2seq.BahdanauAttention(state_size, self.att_states_doc, self.encoder_doc_len)                
                    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism_doc, state_size * 2)               

                    self.combined_final_states = (encoder_doc_states_fw, encoder_doc_states_fw)
                    #print ("COMBINED FINAL STATES: ", self.combined_final_states)
                    initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
                    #initial_state = initial_state.clone(cell_state=self.combined_final_states)

                    decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, target_vocab_size)
                    helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_emb, self.decoder_len)
                    decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper, initial_state=initial_state)
                    outputs = tf.contrib.seq2seq.dynamic_decode(decoder)
                    outputs_logits = outputs[0].rnn_output
                    self.outputs = outputs_logits      

                else:
                    
                    self.att_states_query = self.encoder_query_outputs
                    self.att_states_query.set_shape([self.batch_size, None, state_size*2])
                    tiled_att_states_query = tf.contrib.seq2seq.tile_batch(self.att_states_query, multiplier=self.beam_width)
                    tiled_encoder_query_len = tf.contrib.seq2seq.tile_batch(self.encoder_query_len, multiplier=self.beam_width)


                    self.att_states_doc = self.encoder_doc_outputs
                    self.att_states_doc.set_shape([self.batch_size, None, state_size*2])
                    tiled_att_states_doc = tf.contrib.seq2seq.tile_batch(self.att_states_doc, multiplier=self.beam_width)
                    tiled_encoder_doc_len= tf.contrib.seq2seq.tile_batch(self.encoder_doc_len, multiplier=self.beam_width)


                    attention_mechanism_query = tf.contrib.seq2seq.BahdanauAttention(state_size, tiled_att_states_query, tiled_encoder_query_len)
                    attention_state_query = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism_query, state_size * 2)              

                    attention_mechanism_doc = tf.contrib.seq2seq.BahdanauAttention(state_size, tiled_att_states_doc, tiled_encoder_doc_len)                
                    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism_doc, state_size * 2)               

                    self.combined_final_states = tf.concat((encoder_doc_states_fw, encoder_doc_states_fw), 1)
                    print ("COMBINED FINAL STATES: ", self.combined_final_states)
                    self.combined_final_states = tf.contrib.seq2seq.tile_batch(self.combined_final_states, multiplier=self.beam_width)
                    print ("COMBINED FINAL STATES: ", self.combined_final_states)
                    initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size*self.beam_width)
                    #initial_state = initial_state.clone(cell_state=self.combined_final_states)


                    decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, target_vocab_size)


                    decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                               embedding=decoder_emb,
                                                               start_tokens=tf.fill([self.batch_size], tf.constant(ID_GO)),
                                                               end_token=tf.constant(ID_EOS), 
                                                               initial_state=initial_state,
                                                               beam_width=self.beam_width,
                                                               output_layer=None)
                    print ("DECODER: ", decoder)
                    outputs = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=50)
                    self.predictions = outputs[0].predicted_ids

                
            with tf.variable_scope("loss"):
                if not forward_only:
                    #print ("SHAPE OF OUTPUTS: ", tf.shape(outputs_logits, out_type=tf.int32 ))
                    #print ("SHAPE OF TARGETS: ", tf.shape(self.decoder_targets, out_type=tf.int32))
                    weights = tf.sequence_mask(self.decoder_len, dtype=tf.float32)
                    loss_t = tf.contrib.seq2seq.sequence_loss(outputs_logits, self.decoder_targets, weights, average_across_timesteps=False, average_across_batch=False)
                    self.loss = tf.reduce_sum(loss_t)/FLAGS.batch_size



                    predictions = tf.cast(tf.argmax(outputs_logits, axis=2), tf.int32) 
                    self.accuracy = tf.contrib.metrics.accuracy(predictions, self.decoder_targets)



                    params = tf.trainable_variables()
                    opt = tf.train.AdadeltaOptimizer(self.learning_rate, epsilon=1e-4)
                    gradients = tf.gradients(self.loss, params)
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient)
                    self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
                    tf.summary.scalar('loss', self.loss)
                    tf.summary.scalar('accuracy', self.accuracy)

        
        self.saver = tf.train.Saver(max_to_keep=20)
        self.summary_merge = tf.summary.merge_all()    

    
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



    def add_pad(self, data, fixlen):
        data = map(lambda x: x + [ID_PAD] * (fixlen - len(x)), data)
        data = list(data)
        return np.asarray(data)
    
    def batchify (self, data_set, _buckets, batch_size):
        print ("BATCH SIZE IS: ", batch_size)
        batched_data_set = []
        encoder_query_inputs, encoder_doc_inputs, decoder_inputs = [], [], []
        encoder_query_len, encoder_doc_len, decoder_len = [], [], []
        num_data = 0
        counter = 0
        for bucket_id in range (len(_buckets)):
            if(len(data_set[bucket_id])==0):
                continue
            for j in range(len(data_set[bucket_id])):
                counter += 1
                encoder_doc_input, encoder_query_input, decoder_input = data_set[bucket_id][j]
                encoder_doc_inputs.append(encoder_doc_input)
                encoder_doc_len.append(len(encoder_doc_input))            
                encoder_query_inputs.append(encoder_query_input)
                encoder_query_len.append(len(encoder_query_input))
                decoder_inputs.append(decoder_input)
                decoder_len.append(len(decoder_input))
                num_data += 1
                
                if(num_data == batch_size):
                    num_data = 0
                    batch_enc_doc_len = max(encoder_doc_len)
                    batch_enc_query_len = max(encoder_query_len)
                    batch_dec_len = max(decoder_len)
                    encoder_doc_inputs = self.add_pad(encoder_doc_inputs, batch_enc_doc_len)
                    encoder_query_inputs = self.add_pad(encoder_query_inputs, batch_enc_query_len)
                    decoder_inputs = self.add_pad(decoder_inputs, batch_dec_len)
                    encoder_doc_len = np.asarray(encoder_doc_len)
                    encoder_query_len = np.asarray(encoder_query_len)
                    decoder_len = np.asarray(decoder_len) - 1
                    
                    batched_data_set.append([encoder_doc_inputs, encoder_query_inputs, decoder_inputs, encoder_doc_len, encoder_query_len, decoder_len])
                    
                    encoder_query_inputs, encoder_doc_inputs, decoder_inputs = [], [], []
                    encoder_query_len, encoder_doc_len, decoder_len = [], [], []
        print ("BATCHED COUNTER: ", counter)
        print ("BATCHED LENGTH: ", len(batched_data_set))
        return batched_data_set
    
    
    

print ("DONE")   


# In[35]:


# We use a number of buckets for sampling
_buckets = [(300,5,15), (300,10,20), (300,15,25), (300,20,40), (300,40,50), (350,5,15), (350,10,20), (350,15,25), (350,20,40), (350,40,50), (450,5,15), (450,10,30), (450,15,50), (450,20,100), (450,40,150), (550,5,15), (550,10,30), (550,15,60), (550,20,100), (550,40,150), (650,5,15), (650,10,30), (650,15,60), (650,20,100), (650,40,150), (750,5,15), (750,10,30), (750,15,60), (750,20,100), (750,40,240), (850,5,15), (850,10,30), (850,15,60), (850,20,100), (850,40,240), (1050,5,15), (1050,10,30), (1050,15,60), (1050,20,100), (1050,40,240), (1500,5,15), (1500,10,30), (1500,15,60), (1500,20,100), (1500,40,300), ]


print ("DONE")


# In[36]:


def create_bucket(source, query, target):
    totalDocs = len(source)
    print ("TOTAL DOCS BEFORE BUCKETS: ", totalDocs)
    data_set = [[] for _ in _buckets]
    for s, q, t in zip(source, query, target):
        t = [ID_GO] + t + [ID_EOS]
        found = False
        for bucket_id, (s_size, q_size, t_size) in enumerate(_buckets):
            if len(s) <= s_size and len(q) <= q_size and len(t) <= t_size:
                data_set[bucket_id].append([s, q, t])
                found = True
                break
        if(found != True):
            print ("Didn't find bucket for {}, {}, {}".format(len(s), len(q), len(t)))
    return data_set


def create_model(session, doc_dict, sum_dict, forward_only):
    """Create model and initialize or load parameters in session."""
    dtype = tf.float32
    model = BiGRUModel(doc_dict, sum_dict, FLAGS.doc_vocab_size, FLAGS.sum_vocab_size, _buckets, FLAGS.size,  FLAGS.num_layers, FLAGS.embsize, FLAGS.max_gradient,
        FLAGS.batch_size,     FLAGS.learning_rate,       forward_only, FLAGS.beam_width_size,       dtype=dtype)
    print ("Loading Checkpoint: ", FLAGS.load_checkpoint)
    if (FLAGS.load_checkpoint):        
        ckpt = tf.train.latest_checkpoint(FLAGS.train_dir)
        if ckpt:
            #ckpt = ckpt.model_checkpoint_path
            if ckpt and tf.train.checkpoint_exists(ckpt):
                print("Reading model parameters from %s" % ckpt)
                model.saver.restore(session, ckpt)
                print ("DONE RESTORING CHECKPOINT")
            else:
                logging.error("Don't have any checkpoints to load: %s" % ckpt)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


print ("DONE")


# In[26]:


print ("In Train")
try:
    os.makedirs(FLAGS.train_dir)
except:
    pass

print("Preparing summarization data.")

train_docid, val_docid, train_queryid, val_queryid, train_sumid, val_sumid = train_test_split(docid, queryid, sumid, test_size=FLAGS.train_test_split, shuffle=False, random_state=42)


tf.reset_default_graph()
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)    
# please do not use the totality of the GPU memory
config.gpu_options.per_process_gpu_memory_fraction = 0.90
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
with tf.Graph().as_default(), tf.Session(config=config) as sess:
    # tensorflow seed must be inside graph
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(seed=FLAGS.seed)

    # Create model.
    print("Creating %d layers of %d units." %
                 (FLAGS.num_layers, FLAGS.size))
    train_writer = tf.summary.FileWriter(FLAGS.tfboard+'/train', sess.graph)
    model = create_model(sess, doc_dict, sum_dict, False)

    # Read data into buckets and compute their sizes.
    print("Create buckets.")
    dev_set = create_bucket(val_docid, val_queryid, val_sumid)
    train_set = create_bucket(train_docid, train_queryid, train_sumid)

    train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in range(len(train_bucket_sizes))]

    for (s_size, q_size, t_size), nsample in zip(_buckets, train_bucket_sizes):
        print("Train set bucket ({}, {}, {}) has {} samples.".format(
            s_size, q_size, t_size, nsample))
    batched_train_set = model.batchify(train_set, _buckets, FLAGS.batch_size)
    batched_dev_set = model.batchify(dev_set, _buckets, FLAGS.batch_size)
    # This is the training loop.
    step_time, train_acc, train_loss = 0.0, 0.0, 0.0
    step_start_time = 0
    num_epoch = 0
    step_time = 0
    while num_epoch <= FLAGS.max_epochs:
        epoch_train_loss = 0.0 
        epoch_train_acc = 0.0
        current_train_step = 0
        epoch_start_time = time.time()

        for batch_train in batched_train_set:
            
            
            step_start_time = time.time()                
            encoder_doc_inputs, encoder_query_inputs, decoder_inputs, encoder_doc_len, encoder_query_len, decoder_len = batch_train
            print ("ENCODER QUERY LEN: ", encoder_query_len)
            step_train_loss, step_train_acc, _ = model.step(sess, encoder_doc_inputs, encoder_query_inputs, decoder_inputs,
                encoder_doc_len, encoder_query_len, decoder_len, False, train_writer)
            
            step_time = time.time() - step_start_time
            #print ("CURRENT STEP: ", current_train_step, " STEP TIME: ", step_time)
    
            step_train_loss =  (step_train_loss * FLAGS.batch_size)/np.sum(decoder_len)
            epoch_train_loss += step_train_loss
            epoch_train_acc += step_train_acc      
            current_train_step += 1
            
            # Once in a while, we save checkpoint.
            if current_train_step % FLAGS.steps_per_checkpoint == 0:
                # Save checkpoint and zero timer and loss.
                save_time_start = time.time()
                checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                time_taken_to_save = time.time() - save_time_start
                print("Time taken to save checkpoint: ", time_taken_to_save)

            # Once in a while, we print statistics and run evals.
            if current_train_step % FLAGS.steps_per_print == 0:
                # Print statistics for the previous epoch.
                print ("Epoch: %d, GlobalStep: %d, step-time %.2f, Acc: %.4f, Loss: %.4f, Perpxty: %.2f" % (num_epoch, model.global_step.eval(), 
                                               step_time, 
                                               step_train_acc, 
                                               step_train_loss, 
                                               np.exp(float(step_train_loss))))
                step_time, train_acc, train_loss = 0.0, 0.0, 0.0   
            
            

        #epoch_train_loss, epoch_train_acc, current_train_step = 1., 2., 15
        epoch_eval_loss, epoch_eval_acc = 0.0, 0.0
        current_eval_step = 0
        for batch_dev in batched_dev_set:
            
            encoder_doc_inputs, encoder_query_inputs, decoder_inputs, encoder_doc_len, encoder_query_len, decoder_len = batch_dev
            step_eval_loss, step_eval_acc, _ = model.step(sess, encoder_doc_inputs,encoder_query_inputs,
                                        decoder_inputs, encoder_doc_len, encoder_query_len,
                                        decoder_len, True)
            step_eval_loss = (step_eval_loss * FLAGS.batch_size) / np.sum(decoder_len)
            epoch_eval_loss += step_eval_loss
            epoch_eval_acc += step_eval_acc
            current_eval_step += 1
                
        print("at the end of epoch:", num_epoch)
        print("Average train loss = %6.8f, Average perplexity = %6.8f" % (epoch_train_loss/current_train_step, np.exp(epoch_train_loss/current_train_step)))
        print("Average train acc = %6.8f" % (epoch_train_acc/current_train_step))
        print("validation loss = %6.8f, perplexity = %6.8f" % (epoch_eval_loss/current_eval_step, np.exp(epoch_eval_loss/current_eval_step)))
        print("Average Validation acc = %6.8f" % (epoch_eval_acc/current_eval_step))

        # Save checkpoint and zero timer and loss.
        save_time_start = time.time()
        checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        time_taken_to_save = time.time() - save_time_start
        print("Time taken to save checkpoint: ", time_taken_to_save)
        num_epoch += 1
            
    sys.stdout.flush()

print ("DONE")



# In[27]:


test_data = (val_docid, val_queryid, val_sumid)

with open('test_data.pickle', 'wb') as handle:
    pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('test_data.pickle', 'rb') as handle:
    loaded_test_data = pickle.load(handle)

print (test_data == loaded_test_data)


# In[37]:


def batch_iter(docid, queryid, sumid, batch_size, num_epochs):

    docid = np.array(docid)
    queryid = np.array(queryid)
    queryid = np.array(queryid)

    num_batches_per_epoch = (len(docid) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(docid))
            yield docid[start_index:end_index], queryid[start_index:end_index], sumid[start_index:end_index]

def decode(docid, queryid, sumid):
    print ("In Decode")
    FLAGS.load_checkpoint = True
    FLAGS.batch_size = 1
    # Load vocabularies.
    doc_dict = load_dict(FLAGS.doc_dict_path)
    sum_dict = load_dict(FLAGS.sum_dict_path)
    if doc_dict is None or sum_dict is None:
        logging.warning("Dict not found.")   
    
    
    tf.reset_default_graph()
    with tf.Session() as sess2:
        # Create model and load parameters.
        print("Creating %d layers of %d units." %
                     (FLAGS.num_layers, FLAGS.size))        
        
        model2 = create_model(sess2, doc_dict, sum_dict, True)
        
        with open('test_data.pickle', 'rb') as handle:
            test_data = pickle.load(handle)
            docid, queryid, sumid = test_data
            dev_set = create_bucket(val_docid, val_queryid, val_sumid)
            batched_test_data = model2.batchify (dev_set, _buckets, FLAGS.batch_size)

        result = []
        idx = 0
        for batch_test in batched_test_data:
            idx += 1
            encoder_doc_inputs, encoder_query_inputs, decoder_inputs, encoder_doc_len, encoder_query_len, decoder_len = batch_test
            print ("OUTSIDE STEP: ", encoder_query_len)

            outputs = model2.step_beam(sess2, encoder_doc_inputs, encoder_query_inputs, encoder_doc_len, encoder_query_len, FLAGS.beam_width_size)
            outputs = np.array(outputs).flatten()
            print ("OUTPUT FROM BEAM SEARCH DECODER", outputs)
            # If there is an EOS symbol in outputs, cut them at that point.
            if ID_EOS in outputs:
                outputs = outputs[:outputs.index(ID_EOS)]
            gen_sum = " ".join(sen_map2tok(outputs, sum_dict[1]))
            result.append(gen_sum)
            print("Finish {} samples. :: {}".format(idx, gen_sum[:75]))
        with open(FLAGS.test_output, "w") as f:
            for item in result:
                print(item, file=f)

decode(val_docid, val_queryid, val_sumid)
print ("DONE")


# In[ ]:




