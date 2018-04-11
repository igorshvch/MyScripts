import re
import shelve
import pymorphy2
import numpy
from string import punctuation as string_punc
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from collections import Counter
from textextrconst import RU_word_strip_pattern as tec_rwsp
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from writer import writer

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
        self.data = None
    
    def print_base_info(self, info=True, key=None):
        '''
        Print info about DB internals. If none arguments passed
        return all DB keys.
        Select stored data object by the key argument.
        If info argument is not set to True return sorted keys of 
        data object.
        Otherwise return type(data_object) value.
        '''
        with shelve.open(self.path) as data_base:
            if key:
                if info:
                    return(type(data_base[key]))
                else:
                    print(sorted(data_base[key].keys()))
            else:
                print(sorted(data_base.keys()))
    
    def extract_data(self, data_key):
        '''
        Extract data from DB by the data_key argument
        and save data to .data attribute.
        Accepted data_key arguments:
        'tags', 'marked_acts'.
        '''
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
        return data
        #print('Data \'{}\' is stored to .data attribute'.format(data_key))

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
        '''
        Method creates reg.exp ogject to delete all punctuation
        chars from the string. The object is saved to .re_pd_obj attribute.
        Accept custom punctuation chars as string.
        '''
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
        '''
        Split text into sentences
        '''
        if lower:
            return sent_tokenize(text.lower())
        else:
            return sent_tokenize(text)

    def text_to_tokens(self,
                       text,
                       morph_strip=False,
                       pattern='rwsp',
                       stop_words=True):
        '''
        Split text into tokens
        '''
        text = text.lower()
        if pattern != 'rwsp':
            #splitting and deletinig punctuation chars (including hyhense)
            tokens = text.split()
            tokens = [self.re_pd_obj.sub('', w) for w in tokens]
        else:
            #custom word splitting without punctuation. Hyphenes are saved
            tokens = re.findall(tec_rwsp, text)
        if stop_words:
            #filtering tokens from stop words
            tokens = [w for w in tokens if w not in self.stp_w and len(w)>1]
        if morph_strip:
            #lemmatization step; pymorphy2 is used
            morph = pymorphy2.MorphAnalyzer()
            parser = lambda x: morph.parse(x)[0].inflect({'sing', 'nomn'}).word
            tokens = [parser(w) for w in tokens]
        return tokens
    
    def text_to_lines(self, vocab, text, min_ocur=None, most_com=None):
        if min_ocur:
            vocab = [k for k,v in vocab.items() if v > min_ocur]
        elif most_com:
            vocab = vocab.most_common(most_com)
            vocab = [item[0] for item in vocab]
        else:
            raise TypeError #just for fun
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
    
    def predict(self, data, threshold=0.85):
        result = self.model.predict(data, verbose=0)
        percent_pos = result[0,0]
        if round(percent_pos) == 0:
            return '  N' #(1-percent_pos), '# NEGATIVE'
        elif percent_pos >= threshold:
            return percent_pos, '# POSITIVE'


class Constructor():
    def __init__(self):
        self.DBC = DB_Con()
        self.ATO = ATO()
        self.DI = DocIndex()
        self.NNM = NN_Model()
        self.count = Counter()
        self.acts_list = self.DBC.extract_data('marked_acts')
    
    def load_acts_parts(self, tag):
        acts_parts = [self.ATO.find_tagged_part(tag, act)
                      for act in self.acts_list]
        return acts_parts
    
    def parts_to_tokens(self, acts_parts):
        tokens_list = [self.ATO.text_to_tokens(part)
                  for part in acts_parts]
        return tokens_list
    
    def parts_to_lines(self, acts_parts, vocab, ocur_com=(1, None)):
        lines = [self.ATO.text_to_lines(vocab, part,
                                        min_ocur=ocur_com[0],
                                        most_com=ocur_com[1])
                 for part in acts_parts]
        return lines
    
    def text_to_lines(self, tag1, tag2, ocur_com=(1,None)):
        right_parts = self.load_acts_parts(tag1)
        wrong_parts = self.load_acts_parts(tag2)
        print(len(right_parts), len(wrong_parts))
        right_toks = self.parts_to_tokens(right_parts)
        wrong_toks = self.parts_to_tokens(wrong_parts)
        print(len(right_toks), len(wrong_toks))
        self.count.update(right_toks)
        self.count.update(wrong_toks)
        print(len(self.count)) #tags '02_DEM', '07_REASON', len = 2314
        right_lines = self.parts_to_lines(right_parts,
                                          self.count,
                                          ocur_com=ocur_com)
        wrong_lines = self.parts_to_lines(wrong_parts,
                                          self.count,
                                          ocur_com=ocur_com)
        print(len(right_lines), len(wrong_lines))
        return right_lines+wrong_lines
    
    def labbeling(self, lines):
        l2 = len(lines)//2
        pos = [1 for i in range(l2)]
        neg = [0 for i in range(l2)]
        return pos+neg
    
    def document_index(self, lines):
        self.DI.create_tokenizer(lines)
        return self.DI.convert(lines)
    
    def training(self, x_data, y_labels):
        print(len(x_data), len(y_labels))
        print(x_data.shape) #tags '02_DEM', '07_REASON', .shape = (202, 981)
        self.NNM.define_model(x_data.shape[1])
        self.NNM.training(x_data, y_labels)
    
    def prediction(self,
                   raw_text,
                   threshold=0.85,
                   ocur_com=(1, None),
                   write=True):
        splitted = raw_text.split('\n')
        lines = [self.ATO.text_to_lines(self.count, par,
                                        min_ocur=ocur_com[0],
                                        most_com=ocur_com[1])
                for par in splitted]
        print(len(lines))
        data = [self.DI.convert(line)
                for line in lines]
        print(data.shape)
        predictions = [self.NNM.predict(vect, threshold)
                       for vect in data]
        results = [(splitted[i], predictions[i])
                   for i in range(len(splitted))]
        if write:
            from datetime import datetime as dt
            now = str(dt.now())[:-6]
            writer(results, now)
        else:
            return results










