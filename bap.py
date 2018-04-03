import re
import shelve
from string import punctuation
from nltk.corpus import stopwords
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense

BASE_PATH = r'C:\Users\EA-ShevchenkoIS\Python36\AP\SimpleBase'

class BD_Connection():
    '''
    Connection class. Provide simple interface to data_storage.
    All loaded data is stored to .data attribute.
    Class provides with sevral methods:
        .print_base_info(info=..., key=...) - print to console
        basic information about stored data
        .extract_data(data_key) - load data specified by data_key
        argument to .data attribute
    '''
    def __init__(self, path=BASE_PATH):
        self.path = path
        self.data = {}
    
    def print_base_info(self, info='type', key=None):
        with shelve.open(self.path) as data_base:
            if key:
                if info == 'info':
                    return(type(data_base[key]))
                else:
                    print(sorted(data_base[key].keys()))
            else:
                print(sorted(data_base.keys()))
    
    def extract_data(self, data_key):
        data_base = shelve.open(self.path)
        instances = {
            'tags': sorted(data_base['tags'].keys()),
            'marked_acts': list(data_base['marked_acts'].values()),
            'opened_acts': 0,
            'acts_keys': 0,
            'all_acts': 0,
            'processed_acts': 0
        }
        data = instances[data_key]
        data_base.close()
        self.data = data
        print('Data \'{}\' is stored to .data attribute'.format(data_key))

class TextCleaner():
    def __init__(self):
        self.re_search_obj = None
        self.vocab = None

    def srch_lims(self, start, stop, inclusion=True):
        '''
        Method returns reg.exp object with setted search borders
        '''
        if inclusion:
            pattern = r'(?={0}).+(?<={1})'.format(start, stop)
        else:
            pattern = r'(?<={0}).+(?={1})'.format(start, stop)
        self.re_search_obj = re.compile(pattern, flags=re.DOTALL)

    def text_cleaning(self, text, mode='raw'):
        stp_w = set(stopwords.words('russian'))
        if mode == 'raw':
            re_punc = re.compile('[{}]'.format(re.escape(punctuation)))
            text = re_punc.sub('', text)

    def add_to_vocab(self, text, min_ocur=2):
        pass
    
    def load_text(self, text):
        if isinstance(text, str):
            pass
        if isinstance(text, list):
            pass