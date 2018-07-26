#built-in modules
from collections import Counter
from time import time
#my modules
from sentan.stringbreakers import (
    BGRINR_B, DCTKEY_B, KEYVAL_B, INDEXP_B
)

__version__ = 0.1

class DataStore():
    '''
    Storage for different text information units with arbitrary structure
    '''
    def __init__(self):
        self.vocab = Counter()
    
    def words_count(self, act):
        voc = self.vocab
        for par in act:
            voc.update(par)

def words_count(acts_gen):
        t0 = time()
        vocab = Counter()
        for act in acts_gen:
            for par in act:
                vocab.update(par)
        t1 = time()
        print('Words were counted in {} seconds'.format(t1-t0))
        return vocab
    
def collect_all_words(act):
    return [
        w
        for par in act
        for w in par
    ]
    
def create_bigrams(par):
    separator = BGRINR_B
    holder=[]
    holder = [
        separator.join((par[i-1], par[i])) for i in range(1, len(par), 1)
    ]
    return holder
    
def create_indexdct_from_par(par_list):
    separator = DCTKEY_B
    index_dct = {word:set() for word in par_list}
    counter = 0
    for word in par_list:
        index_dct[word].add(counter)
        counter+=1
    index_dct['total'] = {len(par_list)}
    for word in par_list:
        index_dct['total'+separator+word]={len(index_dct[word])}
    return index_dct

def indexdct_to_string(index_dct):
    sep_keyval = KEYVAL_B
    sep_index = INDEXP_B
    holder = [
        key+sep_keyval+sep_index.join([str(item) for item in val])
        for key,val in index_dct.items()
    ]
    return holder

def string_to_indexdct(list_of_strings):
    sep_keyval = KEYVAL_B
    sep_index = INDEXP_B
    par_dict = {}
    for string in list_of_strings:
        key, vals = string.split(sep_keyval)
        par_dict[key] = set([int(i) for i in vals.split(sep_index)])
    return par_dict


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