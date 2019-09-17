# coding: utf-8


MARK_PAD = "<PAD>"
MARK_UNK = "<UNK>"
MARK_EOS = "<EOS>"
MARK_GO = "<GO>"
MARKS = [MARK_PAD, MARK_UNK, MARK_EOS, MARK_GO]
ID_PAD = 0
ID_UNK = 1
ID_EOS = 2
ID_GO = 3

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


def sen_map2tok(sen, id2tok):
    return list(map(lambda x: id2tok[x], sen))
    
