import re
import numpy as np
from scipy.spatial.distance import cosine as sp_cosine

PATH = r'C:\Users\EA-ShevchenkoIS\sentences.txt'

VOCAB = dict()

def load_text(path):
    with open(path) as file:
        sent_lst = file.readlines()
    return sent_lst

def tokenise(text):
    text = text.lower()
    text = re.split('[^a-z]', text)
    return [w for w in text if w]

def iterate_tokenisation(sent_lst):
    return [tokenise(sent) for sent in sent_lst]

def words_index(tokens, vocab):
    for word in tokens:
        if word not in vocab:
            index = len(vocab)
            vocab[word] = index

def iterate_wrd_indexing(tokens_lst, vocab):
    for sent in tokens_lst:
        words_index(sent, vocab)

def create_mtrx(sent_lst, vocab):
    mtrx = np.zeros((len(sent_lst), len(vocab)))
    print(mtrx.shape)
    return mtrx

def fullfil_matrix(mtrx, row_n, toks_lst, vocab):
    row = 0
    for sent in toks_lst:
        for word in sent:
            col = vocab[word]
            mtrx[row, col]+=1
        row+=1
    return mtrx

def create_full_mtrx(path=PATH, vocab=VOCAB):
    sent_lst = load_text(path)
    toks_lst  = iterate_tokenisation(sent_lst)
    iterate_wrd_indexing(toks_lst, vocab)
    mtrx = create_mtrx(sent_lst, vocab)
    mtrx = fullfil_matrix(mtrx, len(sent_lst), toks_lst, vocab)
    return mtrx

def eval_cos_dist(mtrx, reverses=False):
    base = mtrx[0,:]
    holder = []
    for i in range(1,22,1):
        dist = sp_cosine(base, mtrx[i,:])
        holder.append((i, dist))
    return sorted(holder, key=lambda x: x[1], reverse=reverses)

