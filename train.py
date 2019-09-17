 # coding: utf-8

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
from data_loader import load_data, create_bucket, batchify, batch_iter, _buckets
from sklearn.model_selection import train_test_split
from args import load_args
import pickle
from vocabulary import load_dict
from Model import BiGRUModel



def store_test_dataid_pickle(docid, queryid, sumid):
    with open("test_dataid.pickle", "wb") as f:
        pickle.dump((docid, queryid, sumid), f)
        
def load_test_dataid_pickle():
    with open("test_dataid.pickle", "rb") as f:
        docid, queryid, sumid = pickle.load(f)
    return docid, queryid, sumid

def store_train_dataid_pickle(docid, queryid, sumid):
    with open("train_dataid.pickle", "wb") as f:
        pickle.dump((docid, queryid, sumid), f)
        
def load_train_dataid_pickle():
    with open("train_dataid.pickle", "rb") as f:
        docid, queryid, sumid = pickle.load(f)
    return docid, queryid, sumid


def create_model(session, doc_dict, sum_dict, batch_size, load_checkpoint, forward_only, args):
    dtype = tf.float32
    model = BiGRUModel(doc_dict, sum_dict, args, batch_size, forward_only, dtype=dtype)
    print ("Loading Checkpoint: ", load_checkpoint)
    if (load_checkpoint):        
        ckpt = tf.train.latest_checkpoint(args.train_dir)
        if ckpt:
            #ckpt = ckpt.model_checkpoint_path
            if ckpt and tf.train.checkpoint_exists(ckpt):
                print("Reading model parameters from %s" % ckpt)
                model.saver.restore(session, ckpt)
                print ("DONE RESTORING CHECKPOINT")
            else:
                raise Exception("Don't have any checkpoints to load: %s" % ckpt)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model
    
        
def run_training():
    print ("In Train")
    try:
        os.makedirs(args.train_dir)
    except:
        pass

    print("Preparing summarization data.")
    
    args = load_args()
    
    if (args.reload_data or args.reload_all):
        docid, queryid, sumid, doc_dict, sum_dict = load_data(args, args.full_data_dir)
        train_docid, val_docid, train_queryid, val_queryid, train_sumid, val_sumid = train_test_split(docid, queryid, sumid, test_size=args.train_test_split, shuffle=False, random_state=42)
        store_train_dataid_pickle(train_docid, train_queryid, train_sumid)
        store_test_dataid_pickle(val_docid, val_queryid, val_sumid)
    else:
        train_docid, train_queryid, train_sumid = load_train_dataid_pickle()
        val_docid, val_queryid, val_sumid = load_test_dataid_pickle()
        doc_dict = load_dict(args.doc_dict_path, args.doc_vocab_size)
        sum_dict = load_dict(args.sum_dict_path, args.sum_vocab_size)
        

    tf.reset_default_graph()
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)    
    # please do not use the totality of the GPU memory
    config.gpu_options.per_process_gpu_memory_fraction = 0.90
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        # tensorflow seed must be inside graph
        tf.set_random_seed(args.seed)
        np.random.seed(seed=args.seed)

        # Create model.
        print("Creating %d layers of %d units." %
                     (args.num_layers, args.size))
        train_writer = tf.summary.FileWriter(args.tfboard+'/train', sess.graph)
        model = create_model(sess, doc_dict, sum_dict, args.train_batch_size, args.train_load_checkpoint, args.train_forward_only, args)

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
        batched_train_set = batchify(train_set, _buckets, args.train_batch_size)
        batched_dev_set = batchify(dev_set, _buckets, args.train_batch_size)
        # This is the training loop.
        step_time, train_acc, train_loss = 0.0, 0.0, 0.0
        step_start_time = 0
        num_epoch = 0
        step_time = 0
        while num_epoch <= args.max_epochs:
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
        
                step_train_loss =  (step_train_loss * args.train_batch_size)/np.sum(decoder_len)
                epoch_train_loss += step_train_loss
                epoch_train_acc += step_train_acc      
                current_train_step += 1
                
                # Once in a while, we save checkpoint.
                if current_train_step % args.steps_per_checkpoint == 0:
                    # Save checkpoint and zero timer and loss.
                    save_time_start = time.time()
                    checkpoint_path = os.path.join(args.train_dir, "model.ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                    time_taken_to_save = time.time() - save_time_start
                    print("Time taken to save checkpoint: ", time_taken_to_save)

                # Once in a while, we print statistics and run evals.
                if current_train_step % args.steps_per_print == 0:
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
                step_eval_loss = (step_eval_loss * args.train_batch_size) / np.sum(decoder_len)
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
            checkpoint_path = os.path.join(args.train_dir, "model.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            time_taken_to_save = time.time() - save_time_start
            print("Time taken to save checkpoint: ", time_taken_to_save)
            num_epoch += 1
                
        sys.stdout.flush()

if __name__== "__main__":
    run_training()