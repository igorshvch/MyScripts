import numpy as np
import random as rd
import copy
from scipy.spatial.distance import cosine as sp_cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from writer import writer

#token_pattern='\w+'

counter = 0

alph = [chr(i) for i in range(97, 102, 1)]

def create_rand_seq(char_quant, seq_len):
    holder = [''.join(rd.choices(alph, k=4)) for i in range(seq_len)]
    string = ' '.join(holder)
    return string

def iter_create_rand_seq(iter_num, char_quant, seq_len1, seq_len2):
    vocab = create_rand_seq(char_quant, seq_len1)
    holder = []
    for i in range(iter_num):
        holder.append(create_rand_seq(char_quant, seq_len2))
    return {'vocab':[vocab], 'sents':holder}

def create_mtrx_from_voc(seq_dict):
    vect = TfidfVectorizer(token_pattern='\w+')
    vect.fit(seq_dict['vocab'])
    compressed_mtrx = vect.transform(seq_dict['sents'])
    result_mtrx = compressed_mtrx.toarray()
    global counter
    writer(result_mtrx, 'result_mtrx_{}'.format(counter), mode='w')
    counter+=1
    return result_mtrx

def glue_together_two_first(seq_dict):
    dt = copy.deepcopy(seq_dict)
    holder = dt['sents']
    one=holder.pop(0)
    two=holder.pop(0)
    assert one == seq_dict['sents'][0]
    assert two == seq_dict['sents'][1]
    el1 = one+' '+two
    holder = [el1]+holder
    assert el1 == holder[0]
    assert el1 == seq_dict['sents'][0]+' '+seq_dict['sents'][1]
    dt['sents'] = holder
    return dt



