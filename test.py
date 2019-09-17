# coding: utf-8
from vocabulary import load_dict, sen_map2tok
from train import load_test_dataid_pickle, create_model
from args import load_args
from data_loader import create_bucket, batchify, _buckets
import tensorflow as tf
import numpy as np


MARK_PAD = "<PAD>"
MARK_UNK = "<UNK>"
MARK_EOS = "<EOS>"
MARK_GO = "<GO>"
MARKS = [MARK_PAD, MARK_UNK, MARK_EOS, MARK_GO]
ID_PAD = 0
ID_UNK = 1
ID_EOS = 2
ID_GO = 3


def run_inference_decoding():
    print ("In Decode")
    # Load vocabularies.
    args = load_args()
    doc_dict = load_dict(args.doc_dict_path)
    sum_dict = load_dict(args.sum_dict_path)
    if doc_dict is None or sum_dict is None:
        logging.warning("Dict not found.")   
    
    val_docid, val_queryid, val_sumid = load_test_dataid_pickle()
    dev_set = create_bucket(val_docid, val_queryid, val_sumid)
    batched_test_data = batchify (dev_set, _buckets, args.test_batch_size)
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        # Create model and load parameters.
        print("Creating %d layers of %d units." %
                     (args.num_layers, args.size))        
        
        model = create_model(sess, doc_dict, sum_dict, args.test_batch_size, args.test_load_checkpoint, args.test_forward_only, args)
        
        result = []
        idx = 0
        for batch_test in batched_test_data:
            idx += 1
            encoder_doc_inputs, encoder_query_inputs, decoder_inputs, encoder_doc_len, encoder_query_len, decoder_len = batch_test
            #encoder_doc_inputs = list(map(lambda x: x if x!=0, encoder_doc_inputs))

            print ("OUTSIDE STEP: ", encoder_query_len)

            outputs = model.step_beam(sess, encoder_doc_inputs, encoder_query_inputs, encoder_doc_len, encoder_query_len, args.beam_width)
            outputs = np.array(outputs).flatten()
            #print ("OUTPUT FROM BEAM SEARCH DECODER", outputs)
            # If there is an EOS symbol in outputs, cut them at that point.
            if ID_EOS in outputs:
                outputs = outputs[:outputs.index(ID_EOS)]
            gen_sum = " ".join(sen_map2tok(outputs, sum_dict[1]))
            result.append(gen_sum)
            print("Finish {} samples. :: {}".format(idx, gen_sum))
        with open(args.test_output, "w") as f:
            for item in result:
                print(item, file=f)

if __name__== "__main__":
    run_inference_decoding()