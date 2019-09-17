# coding: utf-8

import json
import io
#import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from args import load_args
from vocabulary import corpus_map2id, create_dict, load_dict
import pickle
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

# We use a number of buckets for sampling
_buckets = [(300,5,15), (300,10,20), (300,15,25), (300,20,40), (300,40,50), (350,5,15), (350,10,20), (350,15,25), (350,20,40), (350,40,50), (450,5,15), (450,10,30), (450,15,50), (450,20,100), (450,40,150), (550,5,15), (550,10,30), (550,15,60), (550,20,100), (550,40,150), (650,5,15), (650,10,30), (650,15,60), (650,20,100), (650,40,150), (750,5,15), (750,10,30), (750,15,60), (750,20,100), (750,40,240), (850,5,15), (850,10,30), (850,15,60), (850,20,100), (850,40,240), (1050,5,15), (1050,10,30), (1050,15,60), (1050,20,100), (1050,40,240), (1500,5,15), (1500,10,30), (1500,15,60), (1500,20,100), (1500,40,300), ]

def create_bucket(source, query, target):
    totalDocs = len(source)
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
    
    
def load_data(args, data_dir):

    if(args.reload_data or args.reload_all):
        print("Reloading document from {}.....".format(data_dir))
        with io.open(data_dir, 'r', encoding='ascii', errors='ignore') as input_file:
            data = json.loads(input_file.read())
            docs = []
            summaries = []
            queries = []
            for i in range (len(data['passages'])):
                docs.append(' '.join([data['passages'][str(i)][j]['passage_text'] for j in range (len(data['passages'][str(i)]))]))
                summaries.append(''.join(data['answers'][str(i)]))
                queries.append(data['query'][str(i)])

        assert     len(docs) == len(queries)
        print ("Number of Data Loaded: ", len(docs))
        
        docs = list(map(lambda x: [y.lower() for y in word_tokenize(x)], docs))
        queries = list(map(lambda x: [y.lower() for y in word_tokenize(x)], queries)) 
        summaries = list(map(lambda x: [y.lower() for y in word_tokenize(x)], summaries))
        
        store_tokenized_data(docs, queries, summaries)
    else:
        print ("Using saved data from ", data_dir)
        docs, queries, summaries = load_tokenized_data()

    if(args.create_dict_flag or args.reload_all):
        doc_dict = create_dict(args.doc_dict_path, docs+queries, args.doc_vocab_size)
        sum_dict = create_dict(args.sum_dict_path, summaries, args.sum_vocab_size)
    else:
        doc_dict = load_dict(args.doc_dict_path, args.doc_vocab_size)
        sum_dict = load_dict(args.sum_dict_path, args.sum_vocab_size)

    print ("Converting to ids...")
    docid, cover = corpus_map2id(docs, doc_dict[0])
    print("Doc dict covers {:.2f}% words.".format(cover * 100))
    
    queryid, cover = corpus_map2id(queries, doc_dict[0])
    print("Query dict covers {:.2f}% words.".format(cover * 100))

    sumid, cover = corpus_map2id(summaries, sum_dict[0])
    print("Sum dict covers {:.2f}% words.".format(cover * 100))    
    
    store_dataid_pickle(docid, queryid, sumid)
    
    return docid, queryid, sumid, doc_dict, sum_dict

def store_tokenized_data(docs, queries, summaries):
    with open("tokenized_data.pickle", "wb") as f:
        pickle.dump((docs, queries, summaries), f)
        print ("Done Dumping Tokenized Data Pickle")

def load_tokenized_data():
    with open('tokenized_data.pickle', 'rb') as f:
        docs, queries, summaries = pickle.load(f) 
    return docs, queries, summaries
    
def store_dataid_pickle(docid, queryid, sumid):
    with open("dataid.pickle", "wb") as f:
        pickle.dump((docid, queryid, sumid), f)
        #print ("Done Dumping DataId Pickle")

def load_dataid_pickle():
    with open('dataid.pickle', 'rb') as f:
        docid, queryid, sumid = pickle.load(f)    
    return docid, queryid, sumid
    
#load_data(load_args())

def add_pad(data, fixlen):
    data = map(lambda x: x + [ID_PAD] * (fixlen - len(x)), data)
    data = list(data)
    return np.asarray(data)

def batchify ( data_set, _buckets, batch_size):
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
                encoder_doc_inputs = add_pad(encoder_doc_inputs, batch_enc_doc_len)
                encoder_query_inputs = add_pad(encoder_query_inputs, batch_enc_query_len)
                decoder_inputs = add_pad(decoder_inputs, batch_dec_len)
                encoder_doc_len = np.asarray(encoder_doc_len)
                encoder_query_len = np.asarray(encoder_query_len)
                decoder_len = np.asarray(decoder_len) - 1
                
                batched_data_set.append([encoder_doc_inputs, encoder_query_inputs, decoder_inputs, encoder_doc_len, encoder_query_len, decoder_len])
                
                encoder_query_inputs, encoder_doc_inputs, decoder_inputs = [], [], []
                encoder_query_len, encoder_doc_len, decoder_len = [], [], []
    print ("BATCHED COUNTER: ", counter)
    print ("BATCHED LENGTH: ", len(batched_data_set))
    return batched_data_set
    
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