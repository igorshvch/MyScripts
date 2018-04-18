import re
import numpy as np
import pymorphy2
import os
import csv
import pickle
from time import time, strftime
from scipy.spatial.distance import cosine as sp_cosine
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter, deque
from nltk.corpus import stopwords

# Term 'bc' in comments below means 'backward compatibility'

# Each class method with leading single underscore in its title
# needs to be modified or removed or was provided for bc reasons.
# All such methods are error prone and not recomended for usage

#acceptable data format:
#file_name.txt:
#----whole file
#    (string_obj)
#file_name.txt:
#----whole file wrapper:
#    (list_obj)
#------------whole paragraph wrapper:
#            (list_obj)
#--------------------paragraph
#                    (string_obj)

PATTERN_ACT_CLEAN = (
    r'-{66}\nКонсультантПлюс.+?-{66}\n'
)
PATTERN_ACT_SEP = (
    r'\n\n\n-{66}\n\n\n'
)

MORPH = pymorphy2.MorphAnalyzer()


PATH_TO_ACTS = (
    r'C:\Users\EA-ShevchenkoIS\Documents'+
    r'\Рабочая\2018-04\Обработка текстов'+
    r'\выборка\viborka_aktov'
)
PATH_TO_CONCLUSIONS = (
    r'C:\Users\EA-ShevchenkoIS\Documents'+
    r'\Рабочая\2018-04\Обработка текстов'+
    r'\выводы 16.04'
)
PATH_TO_OUTPUT = (
    r'C:\Users\EA-ShevchenkoIS\Documents'+
    r'\Рабочая\2018-04\Обработка текстов'
)


class ReadWriteTool():
    '''
    Class provides API to reading and writing options.
    '''
    def __init__(self, enc='cp1251'):
        self.enc=enc
        print('RWT class created')

    def load_text(self, path):
        with open(path, mode='r', encoding=self.enc) as fle:
            text = fle.read()
        return text
    
    def _back_comp_load_text(self, path):
        with open(path, mode='r', encoding=self.enc) as fle:
            text = fle.read()
        return text[1:-1]

    def create_text_loading_paths_list(self, top_dir):
        return [os.path.join(root_dir, fle)
                for root_dir, _, fles in os.walk(top_dir)
                for fle in fles]
        
    def iterate_text_loading(self, top_dir, bc=False):
        '''
        Return generator object iterating over all text files
        in the top_dir subdirectories.
        '''
        paths = self.create_text_loading_paths_list(top_dir)
        print(
            len(paths),
            'The first file in the queue is:\n{}'.format(paths[0]),
            sep='### '
        )
        if not bc:
            return (self.load_text(path)
                    for path in paths)
        else:
            return (self._back_comp_load_text(path)
                    for path in paths)
    
    def write_text(self, text, path, custom_enc=True):
        with open(path, mode='w') as fle:
            if custom_enc:
                fle.write(text.encode(encoding=self.enc))
            else:
                fle.write(text)
    
    def create_text_writing_paths_list(self,
                                       fles_names_list,
                                       new_dir=None,
                                       date=False):
        if date:
            date = strftime('%Y-%m-%d')
            new_dir = (date+'_'+new_dir)
        paths_holder = []
        for fle in fles_names_list:
            dir_new = os.path.join(os.path.dirname(fle), new_dir)
            new_path = os.path.join(dir_new, os.path.basename(fle))
            if not os.path.exists(dir_new):
                os.mkdir(dir_new)
            paths_holder.append(new_path)
        return paths_holder

    def itreate_text_writing(self, paths, texts):
        pairs = zip(paths, texts)
        for item in pairs:
            path, text = item
            self.write_text(text, path)
    
    def write_text_to_csv(self,
                          file_name,
                          iter_txt_holder,
                          header=None,
                          zero_string=None):
        with open(file_name, mode='w', newline='', encoding=self.enc) as fle:
            writer = csv.writer(
                fle,
                delimiter='|',
                quotechar='#',
                quoting=csv.QUOTE_MINIMAL
            )
            if zero_string:
                zero_string = (
                    [zero_string] 
                    + ['' for i in range(len(iter_txt_holder[0]))-1]
                )
                assert len(zero_string) == len(iter_txt_holder[0])
                writer.writerow(zero_string)
            if header:
                writer.writerow(header)
            for row in iter_txt_holder:
                writer.writerow(row)

    def _write_pickle(self, py_obj, path):
        with open(path, mode='wb') as file_name:
            pickle.dump(py_obj,
                        file_name,
                        protocol=pickle.HIGHEST_PROTOCOL)
        
    def _load_pickle(self, path):
        with open(path, 'rb') as fle:
            data = pickle.load(fle)
        return data
    
    def _iterate_pickles_loading(self, top_dir):
        '''
        Return generator object iterating over all binary files
        in the top_dir subdirectories.
        '''
        paths = self.create_text_loading_paths_list(top_dir)#, txt=False)
        print(
            len(paths),
            'The first file in the queue is:\n{}'.format(paths[0]),
            sep='### '
        )
        return (self._load_pickle(path)
                for path in paths)
    

class CustomTextProcessor():
    '''
    Class provides a simple API for tokenization and lemmatization tasks.
    '''
    options = {
            'parser1': None,
            'parser2': (
                lambda x: MORPH.parse(x)[0][2]
            ),
            'parser3': (
                lambda x:
                MORPH.parse(x)[0].inflect({'sing', 'nomn'}).word
            )}

    def __init__(self, pars_type='parser2'):
        self.parser = CustomTextProcessor.options[pars_type]
        print('CTW class created')

    def court_decisions_cleaner(self, text):
        t0 = time()
        cleaned_text = re.subn(PATTERN_ACT_CLEAN, '', text, flags=re.DOTALL)[0]
        print('Acts were cleaned in {} seconds'.format(time()-t0))
        return cleaned_text
    
    def court_decisions_separator(self, text):
        t0 = time()
        separated_acts = re.split(PATTERN_ACT_SEP, text)
        print(
            'Acts were separated in {} seconds'.format(time()-t0),
            '{} acts were found'.format(len(separated_acts))
        )
        return separated_acts

    def text_separation(self, kwarg):
        options = {
            'option1':lambda : print('option1 - not implemented!'),
            'option2':lambda : print('option2 - not implemented!')
        }
        return options[kwarg]()
            
    def tokenize(self, text):
        text = text.lower().strip()
        return [token for token in re.split('\W', text) if len(token) > 1]
    
    def lemmatize(self, tokens_list):
        parser=self.parser
        if parser:
            return [parser(token) for token in tokens_list]
        else:
            raise BaseException('Parser wasn\'t defined during the class creation!')
    
    def change_parser(self, pars_type='parser2'):
        self.parser = CustomTextProcessor.options[pars_type]
    
    def full_process(self, text):
        tokens = self.tokenize(text)
        lemms = self.lemmatize(tokens)
        clear_text = ' '.join(lemms)
        return clear_text
    
    def iterate_full_processing(self, iter_obj):
        holder = []
        for text in iter_obj:
            text_holder=[]
            pars_list = text.split('\n')
            for par in pars_list:
                text_holder.append(self.full_process(par))
            holder.append(text.holder)
        return holder
    
    def words_count(self, acts_gen):
        t0 = time()
        vocab = Counter()
        for act in acts_gen:
            words = [
                w
                for par in act
                for w in par.split()
            ]
            vocab.update(words)
        t1 = time()
        print('Words were counted in {} seconds'.format(t1-t0))
        return vocab

class BCTextProcessor():
    '''
    API to process already stored acts. Implemented for bc reasons
    '''
    def __init__(self):
        print('BCTP class created')

    def process_act(self, act):
        act = re.subn('[\'\]\[]', '', act)[0]
        pars_list = act.split(', ')
        return pars_list
    
    def words_count(self, acts_gen):
        t0 = time()
        vocab = Counter()
        for act in acts_gen:
            pars_list = self.process_act(act)
            words = [
                w
                for par in pars_list
                for w in par.split()
            ]
            vocab.update(words)
        t1 = time()
        print('Words were counted in {} seconds'.format(t1-t0))
        return vocab
    
    def words_count_bin(self, acts_gen):
        t0 = time()
        vocab = Counter()
        for act in acts_gen:
            words = [
                w
                for par in act
                for w in par
            ]
            vocab.update(words)
        t1 = time()
        print('Words were counted in {} seconds'.format(t1-t0))
        return vocab
    
    def _rewrite(self, top_dir=PATH_TO_ACTS):
        rwt = ReadWriteTool()
        read_paths = rwt.create_text_loading_paths_list(top_dir)
        write_paths = rwt.create_text_writing_paths_list(read_paths, date=False)
        write_paths = deque(write_paths)
        acts = rwt.iterate_text_loading(top_dir, bc=True)
        for act in acts:
            pars_list = self.process_act(act)
            pars_list = [par.split() for par in pars_list]
            rwt._write_pickle(pars_list, write_paths.popleft())


class Vectorization():
    def __init__(self):
        self.vectorizer = CountVectorizer()
        print('Vct class created')
    
    def create_vocab(self, tokens_lst, threshold=1):
        toks_list = [w for par in tokens_lst for w in par if len(w) > threshold]
        self.vectorizer.fit(set(toks_list))
    
    def create_vectors(self, pars_lst):
        compressed_mtrx = self.vectorizer.transform(pars_lst)
        result_mtrx = compressed_mtrx.toarray()
        assert len(pars_lst) == result_mtrx.shape[0]
        return result_mtrx


class Constructor():
    def __init__(self,
                 path_to_acts=PATH_TO_ACTS,
                 path_to_conclusions=PATH_TO_CONCLUSIONS,
                 path_to_output=PATH_TO_OUTPUT,
                 enc='cp1251'):
        self.RWT = ReadWriteTool(enc=enc)
        self.CTP = CustomTextProcessor()
        self.BCTP = BCTextProcessor() #execute for bc reasons
        self.Vct = Vectorization()
        self.path1 = path_to_acts
        self.path2 = path_to_conclusions
        self.path3 = path_to_output
        self.holder = []
        self.tracker = []
        print('Constructor class created')
    
    def start_acts_iteration(self):
        '''
        Return generator object over paths to texts
        '''
        return self.RWT.iterate_text_loading(self.path1, bc=True)
    
    def all_words(self):
        return self.BCTP.words_count(
            self.start_acts_iteration()
        )
    
    def start_conclusions_iteration(self):
        '''
        Return generator object over paths to texts
        '''
        return self.RWT.iterate_text_loading(self.path2)
    
    def act_and_concl_to_mtrx(self, pars_list, concl, verbose=False):
        data = [concl] + pars_list
        self.Vct.vectorizer.fit(data)
        if verbose:
            print('Vectorized!')
        data_mtrx = self.Vct.create_vectors(data)
        return data_mtrx

    def eval_cos_dist(self, index_mtrx, output='best'):
        base = index_mtrx[0,:]
        holder = []
        for i in range(1,index_mtrx.shape[0],1):
            holder.append((i, sp_cosine(base, index_mtrx[i,:])))
        if output=='best':
            return sorted(holder, key = lambda x: x[1])[0]
        elif output=='all':
            return sorted(holder, key = lambda x: x[1])[0]
        else:
            raise TypeError('Wrong key argument for "output"!')

    def export_cd_eval_results(self):
        concls = self.start_conclusions_iteration()
        for concl in concls:
            concl = self.CTP.full_process(concl)
            acts = self.start_acts_iteration()
            print('\n', concl[:50], '\n', sep='')
            t0 = time()
            holder = []
            for act in acts:
                pars_list = self.BCTP.process_act(act) #execute for bc reasons
                data_mtrx = self.act_and_concl_to_mtrx(pars_list, concl)
                cos = self.eval_cos_dist(data_mtrx)[1]
                holder.append([pars_list[0], pars_list[2], cos])
            t1 = time()
            print(
                'Acts were processed!',
                'Time in seconds: {}'.format(t1-t0)
            )
            holder = sorted(holder, key=lambda x: x[2])
            t2 = time()
            print(
                'Results were sorted!',
                'Time in seconds: {}'.format(t2-t1)
            )
            name = concl[:40]+'.csv'
            self.RWT.write_text_to_csv(
                os.path.join(self.path3, name),
                holder,
                header=('Суд','Реквизиты','Косинус'),
                zero_string = concl
            )
            breaker = None
            while breaker != '1' and breaker != '0':
                breaker = input(
                ("Обработка вывода окончена. Обработать следующий вывод? "
                +"[1 for 'yes'/0 for 'no']")
                )
                if breaker != '1' and breaker != '0':
                    print('Вы ввели неподдерживаемое значение!')
            if breaker == '0':
                print('Programm was terminated')
                break
            elif breaker == '1':
                print('Continue execution')
        print('Execution complete')
    
    def export_vocab_to_csv(self, vocab=None, file_name='count_uniq'):
        file_name+='.csv'
        path = os.path.join(self.path3, file_name)
        if vocab:
            print('Sorting vocabulary')
            srt_vocab = vocab.most_common()
        else:
            print('Creating vocabulary, counting words...')
            vocab = self.all_words()
            srt_vocab = vocab.most_common()
        self.RWT.write_text_to_csv(
            path,
            srt_vocab,
            header=('Слова','Количество вхождений в корпус')
        )

