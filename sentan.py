import re
import numpy as np
from scipy.spatial.distance import cosine as sp_cosine

PATH = r'C:\Users\EA-ShevchenkoIS\sentences.txt'

VOCAB = dict()

class CosDistEval():
    def __init__(self):
        print('CosDistEval class instance was created')

    def load_text(self, path):
        with open(path) as file:
            sent_lst = file.readlines()
        return sent_lst

    def tokenise(self, text):
        text = text.lower()
        text = re.split('[^a-z]', text)
        return [w for w in text if w]

    def iterate_tokenisation(self, sent_lst):
        return [self.tokenise(sent) for sent in sent_lst]

    def words_index(self, tokens, vocab):
        for word in tokens:
            if word not in vocab:
                index = len(vocab)
                vocab[word] = index

    def iterate_wrd_indexing(self, tokens_lst, vocab):
        for sent in tokens_lst:
            self.words_index(sent, vocab)

    def create_mtrx(self, sent_lst, vocab):
        mtrx = np.zeros((len(sent_lst), len(vocab)))
        print(mtrx.shape)
        return mtrx

    def fullfill_matrix(self, mtrx, row_n, toks_lst, vocab):
        row = 0
        for sent in toks_lst:
            for word in sent:
                col = vocab[word]
                mtrx[row, col]+=1
            row+=1
        return mtrx

    def create_full_mtrx(self, path=PATH, vocab=VOCAB):
        sent_lst = self.load_text(path)
        toks_lst  = self.iterate_tokenisation(sent_lst)
        self.iterate_wrd_indexing(toks_lst, vocab)
        mtrx = self.create_mtrx(sent_lst, vocab)
        mtrx = self.fullfill_matrix(mtrx, len(sent_lst), toks_lst, vocab)
        return mtrx

    def eval_cos_dist(self, mtrx, reverses=False):
        base = mtrx[0,:]
        holder = []
        for i in range(1,22,1):
            dist = sp_cosine(base, mtrx[i,:])
            holder.append((i, dist))
        return sorted(holder, key=lambda x: x[1], reverse=reverses)

