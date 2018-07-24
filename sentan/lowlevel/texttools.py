from collections import Counter
from time import time

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
    holder=[]
    lp = len(par)
    assert lp > 1
    for i in range(1, lp, 1):
        holder.append('#'.join((par[i-1], par[i])))
    return holder
    
def position_search(words):
    d = {word:set() for word in words}
    counter = 0
    for word in words:
        d[word].add(counter)
        counter+=1
    d['total'] = len(words)
    for word in words:
        d['total#'+word]=len(d[word])
    return d

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