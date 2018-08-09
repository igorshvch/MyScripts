from time import time
from math import (
    log10 as math_log,
    exp as math_exp
)
from sentan import mysqlite
from sentan.lowlevel import rwtool
from sentan.stringbreakers import (
    DCTITM_B, DCTKEY_B, RAWPAR_B
)
from sentan.lowlevel.texttools import (
    create_bigrams as crtbgr,
    string_to_indexdct as str_to_indct,
    DataStore
)

__version__ = 0.1

###Content=====================================================================
VOCAB_NW = rwtool.load_pickle(
    r'C:\Users\EA-ShevchenkoIS\TextProcessing\StatData\vocab_nw'
)
TOTAL_PARS = rwtool.load_pickle(
    r'C:\Users\EA-ShevchenkoIS\TextProcessing\StatData\total_lem_pars'
)

def extract_pairs_term_freq(word1, word2, info):
    key_pref = 'total'+DCTKEY_B
    counter = 0
    if word1 in info and word2 in info:
        order1 = info[word1]
        order2 = info[word2]
        if info[key_pref+word1]<info[key_pref+word2]:
            min_order = order1
            max_order = order2
        else:
            min_order = order2
            max_order = order1
        for place in min_order:
            if (place+1) in max_order:
                counter+=1
            elif (place+2) in max_order:
                counter+=0.5
            elif (place-1) in max_order:
                counter+=0.5
        return counter
    else:
        return 0

def extract_p_ontopic(word, vocab, total_parts):
    D = total_parts
    CF = vocab.get(word)
    if CF:
        return 1-math_exp(-1.5*CF/D)
    else:
        return 0

def w_single(word, info, vocab, total_parts):
    delim = DCTKEY_B
    DL = info['total']
    TF = info.get('total'+delim+word)
    if not TF:
        return 0
    else:
        p_ontopic = extract_p_ontopic(word, vocab, total_parts)
        return math_log(p_ontopic)*(TF/(TF+1+(1/350)+DL))

def w_pair(word1, word2, info, vocab, total_parts):
    pTF = extract_pairs_term_freq(word1, word2, info)
    if not pTF:
        return 0
    p_ontopic1 = extract_p_ontopic(word1, vocab, total_parts)
    p_ontopic2 = extract_p_ontopic(word2, vocab, total_parts)
    return (
        0.3*(math_log(p_ontopic1)+math_log(p_ontopic2))*pTF/(1+pTF)
    )

def w_allwords(phrase_words, act, vocab, total_parts):
    act_words = set(act) ###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    mis_count = 0
    for w in phrase_words:
        if w not in act_words:
            mis_count += 1
    p_ontopics = [
        extract_p_ontopic(w, vocab, total_parts)
        for w in phrase_words
        if w in act_words
    ]
    log_ps = [math_log(p) for p in p_ontopics]
    return 0.2*sum(log_ps)*0.03**mis_count

def score(phrase_words, act, info, vocab=VOCAB_NW, total_parts=TOTAL_PARS):
    local_crtbgr = crtbgr
    w_singles = [w_single(w, info, vocab, total_parts) for w in phrase_words]
    #print(w_singles)
    pairs = local_crtbgr(phrase_words)
    #print(pairs)
    w_pairs = [
        w_pair(*pair, info, vocab)
        for pair in pairs
    ]
    #print(w_pairs)
    res_w_allwords = w_allwords(phrase_words, act, vocab, total_parts)
    #print(w_allwords)
    return abs(sum(w_singles)+sum(w_pairs)+res_w_allwords)

class Scorer():
    def __init__(self):
        self.total_parts = TOTAL_PARS
        self.vocab_nw = VOCAB_NW
        self.DB_load = mysqlite.DataBase(
            raw_path = r'C:\Users\EA-ShevchenkoIS\TextProcessing\TNBI',
            base_name='TNBI',
            tb_name=True
        )
        self.total_acts = self.DB_load.total_rows()
        #DB_load.close()
    
    def score_acts(self, concl):
        #Initialize local vars:
        t0=time()
        sep_keyval = DCTITM_B
        TA = self.total_acts
        OUTPUT = TA//10 if TA > 10 else TA//2
        acts_gen = self.DB_load.iterate_row_retr(length=TA, output=OUTPUT)
        counter = 1
        #Initialize local funcs:
        local_scorer = score
        local_str_to_indct = str_to_indct
        print('Start corpus scoring!')
        for batch in acts_gen:
            t1 = time()
            print(
                '\tStarting new batch! Batch # {}.'.format(counter)
                +'{:4.5f}'.format(time()-t0)
            )
            for row in batch:
                _, court, req, rawpars, _, _, index_act, _ = row
                index_dct = local_str_to_indct(index_act.split(sep_keyval))
                sc = local_scorer(concl, )
                


###Testing=====================================================================
if __name__ == '__main__':
    import sys
    try:
        sys.argv[1]
        if sys.argv[1] == '-v':
            print('Module name: {}'.format(sys.argv[0]))
            print('Version info:', __version__)
        elif sys.argv[1] == '-t':
            print('Testing mode!')
            print('Not implemented!')
        else:
            print('Not implemented!')
    except IndexError:
        print('Mode var wasn\'t passed!')