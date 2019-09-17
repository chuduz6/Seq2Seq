 # coding: utf-8
from gensim.models.keyedvectors import KeyedVectors
from args import load_args
from vocabulary import load_dict
import numpy as np
import tensorflow as tf
import pickle

        
def store_doc_word_vector_list(word_vec_list):
    with open('doc_word_vector_list.pickle', 'wb') as f:
        pickle.dump(word_vec_list, f)
        
def store_sum_word_vector_list(word_vec_list):
    with open('sum_word_vector_list.pickle', 'wb') as f:
        pickle.dump(word_vec_list, f)
        
def load_doc_word_vector_list():
    with open('doc_word_vector_list.pickle', 'rb') as f:
        word_vec_list = pickle.load(f)
    return np.array(word_vec_list)
    
def load_sum_word_vector_list():
    with open('sum_word_vector_list.pickle', 'rb') as f:
        word_vec_list = pickle.load(f)
    return np.array(word_vec_list)

def store_word_vectors_pickle(word_vectors):
    with open('word_vectors.pickle', 'wb') as f:
        pickle.dump(word_vectors, f)
        
def load_word_vectors_pickle():
    with open('word_vectors.pickle', 'rb') as f:
        word_vectors = pickle.load(f)
    return word_vectors


def get_init_embedding(type, word2vec_file, word_vocab, args):
    if(args.reload_word_vectors_list or args.reload_all):
        if(args.reload_all):
            word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)
            store_word_vectors_pickle(word_vectors)
        word_vectors = load_word_vectors_pickle()
        word_vec_list = list()
        success_count = 0
        failure_count = 0
        for word, _ in word_vocab[0].items():
            try:
                word_vec = word_vectors.word_vec(word)
                success_count += 1
            except KeyError:
                word_vec = np.zeros([args.embsize], dtype=np.float32)
                failure_count += 1
            word_vec_list.append(word_vec)

        word_vec_list[2] = np.random.normal(0, 1, args.embsize)
        word_vec_list[3] = np.random.normal(0, 1, args.embsize)
        print ("SUCCESS COUNT: ", success_count, " FAILURE COUNT: ", failure_count)
        
        if(type == 'doc'):
            store_doc_word_vector_list(np.array(word_vec_list))
        elif(type == 'sum'):
            store_sum_word_vector_list(np.array(word_vec_list))
        else:
            raise NotImplementedError
    else:
        if(type == 'doc'):
            word_vec_list = load_doc_word_vector_list()
        elif(type == 'sum'):
            word_vec_list = load_sum_word_vector_list()
        else:
            raise NotImplementedError
        print ("Done Loading Word Vectors List for ", type)
    return np.array(word_vec_list)
    

args = load_args()
sum_dict = load_dict(args.sum_dict_path, args.sum_vocab_size)
get_init_embedding('sum', args.pretrained_embeddings_vec_path, sum_dict, args)



