import pymorphy2
import re
from collections import Counter
from time import time

__version__ = 0.1

###pymoprhy2 analyzer instance=================================================
MORPH = pymorphy2.MorphAnalyzer()
parser_options = {
    'parser1': (
        lambda x: MORPH.parse(x)[0][2]
    ),
    'parser2': (
        lambda x:
        MORPH.parse(x)[0].inflect({'sing', 'nomn'}).word
    )
}
PAR_TYPE = 'parser1'
PARSER = parser_options[PAR_TYPE]

###Content=====================================================================
class ParsDataStore():
    '''
    Storage for different text information units with arbitrary structure
    '''
    def __init__(self):
        self.vocab = Counter()
        self.lem_map = None
    
    def words_count(self, act):
        voc = self.vocab
        for par in act:
            voc.update(par)

    def create_lem_map(self):
        t0 = time()
        lemz = PARSER
        voc = self.vocab
        dct = {key:lemz(key) for key in voc}
        self.lem_map = dct
        print('Lem_map was created in {:4.5f} seconds'.format(time()-t0))

def tokenize(text, threshold=0):
    text = text.lower().strip()
    if threshold:
        return [
            token for token in re.split('\W', text) if len(token)>threshold
        ]
    else:
        return [token for token in re.split('\W', text) if token]

def lemmatize(tokens_list):
    local_parser = PARSER
    return [local_parser(token) for token in tokens_list]

def change_parser():
    global PAR_TYPE
    global PARSER
    if PAR_TYPE == 'parser1':
        PAR_TYPE = 'parser2'
        PARSER = parser_options[PAR_TYPE]
    elif PAR_TYPE == 'parser2':
        PAR_TYPE = 'parser1'
        PARSER = parser_options[PAR_TYPE]
    print('Parser was changed to {}'.format(PAR_TYPE))

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
        elif sys.argv[1] == '-par_type':
            print('Parser type: {}'.format(PAR_TYPE))
        else:
            print('Not implemented!')
    except IndexError:
        print('Mode var wasn\'t passed!')