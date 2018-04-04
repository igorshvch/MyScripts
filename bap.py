import re
import shelve
import pymorphy2
from string import punctuation as string_punc
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from collections import Counter
from textextrconst import RU_word_strip_pattern as tec_rwsp
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense

BASE_PATH = r'C:\Users\EA-ShevchenkoIS\Python36\AP\SimpleBase'

class DB_Con():
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

class BTO():
    '''
    BasicTextOperations API
    '''
    def __init__(self):
        self.re_pd_obj = None

    def srch_lims(self, start, stop, inclusion=True):
        '''
        Method returns reg.exp object with setted search borders
        to .re_search_obj attribute
        '''
        if inclusion:
            pattern = r'(?={0}).+(?<={1})'.format(start, stop)
        else:
            pattern = r'(?<={0}).+(?={1})'.format(start, stop)
        return re.compile(pattern, flags=re.DOTALL)

    def punc_del(self, punc_signs=string_punc):
        self.re_pd_obj = re.compile('[{}]'.format(re.escape(punc_signs)))
        

class ATO(BTO):
    '''
    AdvancedTextOperations API
    '''
    def __init__(self, punc_signs=string_punc):
        BTO.__init__(self)
        self.punc_del(punc_signs=punc_signs)
        self.stp_w = set(stopwords.words('russian'))

    def find_tagged_part(self, tag, text):
        re_obj = self.srch_lims(tag, ('/'+tag), inclusion=False)
        text_part = re_obj.search(text).group(0)
        return text_part

    def sent_tok(self, text, lower=True):
        if lower:
            return sent_tokenize(text.lower())
        else:
            return sent_tokenize(text)

    def text_to_tokens(self,
                       text,
                       morph_strip=False,
                       pattern='rwsp',
                       stop_words=True):
        text = text.lower()
        if pattern != 'rwsp':
            tokens = text.split()
            tokens = [self.re_pd_obj.sub('', w) for w in tokens]
        else:
            tokens = re.findall(tec_rwsp, text)
        if stop_words:
            tokens = [w for w in tokens if w not in self.stp_w and len(w)>1]
        if morph_strip:
            morph = pymorphy2.MorphAnalyzer()
            parser = lambda x: morph.parse(x)[0].inflect({'sing', 'nomn'}).word
            tokens = [parser(w) for w in tokens]
        return tokens
    
    def v_update(self, vocab, tokens, **kwargs):
        return vocab.update(tokens)
    
    def text_to_lines(self, vocab, text, min_ocur = 1):
        vocab = [k for k,v in vocab.items() if v > min_ocur]
        vocab = set(vocab)
        tokens = self.text_to_tokens(text)
        tokens = [w for w in tokens if w in vocab]
        line = ' '.join(tokens)
        return line

class DocIndex():
    def __init__(self):
        self.tokenizer = None

    def create_tokenizer(self, list_of_texts):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(list_of_texts)
        self.tokenizer = tokenizer

    def update_tokenizer(self, list_of_texts):
        self.tokenizer.fit_on_texts(list_of_texts)

    def convert(self, list_of_texts, mode='binary'):
        return self.tokenizer.texts_to_matrix(list_of_texts, mode=mode)

class NN_Model():
    def __init__(self):
        self.model = None
    
    def define_model(self, input_layer_length):
        model = Sequential()
        model.add(Dense(50,
                        input_shape=(input_layer_length,),
                        activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        self.model = model
    
    def training(self, xtrain, ytrain):
        self.model.fit(xtrain, ytrain, epochs=10, verbose=2)
    
    def test(self, xtest, ytest):
        _, acc = self.model.evaluate(xtest, ytest, verbose=0)
        print('Test Accuracy: %f' % (acc*100))
    
    def predict(self, data):
        result = self.model.predict(data, verbose=0)
        percent_pos = result[0,0]
        if round(percent_pos) == 0:
            return (1-percent_pos), 'NEGATIVE prediction'
        else:
            return percent_pos, 'POSITIVE prediction'






