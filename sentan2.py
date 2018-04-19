import re
import numpy as np
import pymorphy2
import csv
import pickle
import pathlib as pthl
from time import time, strftime
from scipy.spatial.distance import cosine as sp_cosine
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter, deque
from nltk.corpus import stopwords

# Term 'bc' in comments below means 'backward compatibility'

# Each class method with leading single underscore in its title
# needs to be modified or removed or was provided for bc reasons.
# All such methods (except specified fo bc) are error prone
# and not recomended for usage

#acceptable data format:
#
#file_name.txt:
#----whole file
#    (string_obj)
#
#file_name.txt:
#----whole file wrapper:
#    (list_obj)
#------------whole paragraph wrapper:
#            (list_obj)
#--------------------splitted words in paragraph
#                    (string_objs)

#CONSTANTS:
PATTERN_ACT_CLEAN = (
    r'-{66}\nКонсультантПлюс.+?-{66}\n'
)
PATTERN_ACT_SEP = (
    r'\n\n\n-{66}\n\n\n'
)

#pymoprhy2 analyzer instance
MORPH = pymorphy2.MorphAnalyzer()


class ReadWriteTool():
    '''
    Class provides API to reading and writing options.
    '''
    def __init__(self, enc='cp1251'):
        self.enc=enc
        print('RWT class created')

    def create_dirs(self, dir_struct):
        paths = []
        for path in dir_struct.values():
            path.mkdir(parents=True, exist_ok=True)
            paths.append(str(path))
        print('Created directories:')
        for strg in sorted(paths):
            print('\t'+strg)
    
    def create_acts_subdirs(self, path, des_folder):
        p = pthl.Path(path)
        files = sorted([fle for fle in p.iterdir() if fle.is_file()])
        subdirs = [
            des_folder.joinpath(fle.stem)
            for fle in files
                if fle.is_file()
        ]
        for subdir in subdirs:
            subdir.mkdir(parents=True, exist_ok=True)
        return subdirs    

    def load_text(self, path):
        with open(path, mode='r', encoding=self.enc) as fle:
            text = fle.read()
        return text

    def collect_exist_file_paths(self, top_dir, suffix='.txt'):
        holder = []
        def inner_func(top_dir, suffix):
            p = pthl.Path(top_dir)
            nonlocal holder
            store = [path_obj for path_obj in p.iterdir()]
            for path_obj in store:
                if path_obj.is_dir():
                    inner_func(path_obj, suffix)
                elif path_obj.suffix == suffix:
                    holder.append(path_obj)
        inner_func(top_dir, suffix)
        return sorted(holder)
        
    def iterate_text_loading(self, top_dir):
        '''
        Return generator object iterating over all text files
        in the top_dir subdirectories.
        '''
        paths = self.collect_exist_file_paths(top_dir)
        print(
            len(paths),
            'The first file in the queue is:\n{}'.format(paths[0]),
            sep=' ### '
        )
        return (self.load_text(path) for path in paths)
    
    def write_text(self, text, path):
        with open(path, mode='w', encoding=self.enc) as fle:
                fle.write(text)
    
    def create_writing_paths(self, file_quant, path, suffix=''):
        p = pthl.Path(path)
        file_paths = [
            p.joinpath(str(i)).with_suffix(suffix)
            for i in range(file_quant)
        ]
        return file_paths

    def iterate_text_writing(self, texts, paths):
        pairs = zip(texts, paths)
        for item in pairs:
            text, path = item
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
                    [zero_string] + ['na' for i in range(len(header)-1)]
                )
                assert len(zero_string) == len(header)
                writer.writerow(zero_string)
            if header:
                writer.writerow(header)
            for row in iter_txt_holder:
                writer.writerow(row)

    def write_pickle(self, py_obj, path):
        with open(path, mode='wb') as file_name:
            pickle.dump(py_obj,
                        file_name,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def iterate_pickle_writing(self, py_objs, paths):
        pairs = zip(py_objs, paths)
        for item in pairs:
            py_obj, path = item
            self.write_pickle(py_obj, path)
        
    def load_pickle(self, path):
        with open(path, 'rb') as fle:
            data = pickle.load(fle)
        return data
    
    def iterate_pickle_loading(self, top_dir):
        '''
        Return generator object iterating over all binary files
        in the top_dir subdirectories.
        '''
        paths = self.collect_exist_file_paths(top_dir, suffix='')
        print(
            len(paths),
            'The first file in the queue is:\n{}'.format(paths[0]),
            sep='### '
        )
        return (self.load_pickle(path) for path in paths)
    

class CustomTextProcessor():
    '''
    Class provides a simple API for tokenization and lemmatization tasks.
    '''
    options = {
        'parser1': (
            lambda x: MORPH.parse(x)[0][2]
        ),
        'parser2': (
            lambda x:
            MORPH.parse(x)[0].inflect({'sing', 'nomn'}).word
        )
    }

    def __init__(self, par_type='parser1'):
        self.parser = CustomTextProcessor.options[par_type]
        print('CTW class created')
    
    def change_parser(self, par_type='parser1'):
        self.parser = CustomTextProcessor.options[par_type]

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
            
    def tokenize(self, text):
        text = text.lower().strip()
        return [token for token in re.split('\W', text) if len(token)>1]
    
    def iterate_tokenization(self, text_gen):
        holder = []
        for text in text_gen:
            text_holder=[]
            pars_list = text.split('\n')
            for par in pars_list:
                if par:
                    text_holder.append(self.tokenize(par))
            holder.append(text_holder)
        return holder
    
    def lemmatize(self, tokens_list, par_type='parser1'):
        self.change_parser(par_type=par_type)
        parser = self.parser
        if par_type == 'parser1':
            return [parser(token) for token in tokens_list]
        else:
            result = []
            for token in tokens_list:
                try:
                    norm = parser(token)
                except:
                    norm = token
                result.append(norm)
            return result
    
    def lemmatize_by_dict(self, lem_dict, tokens_list):
        return [lem_dict[token] for token in tokens_list]
    
    def iterate_lemmatize_by_dict(self, lem_dict, acts_gen):
        holder = []
        print('Start normalization!')
        t0 = time()
        for act in acts_gen:
            act = [
                self.lemmatize_by_dict(lem_dict, par)
                for par in act
            ]
            holder.append(act)
        print('Normalization complete in {} seconds'.format(time()-t0))
        return holder
    
    def full_process(self, text, par_type='parser1'):
        tokens = self.tokenize(text)
        lemms = self.lemmatize(tokens, par_type=par_type)
        return lemms
    
    def iterate_full_processing(self, iter_obj):
        holder = []
        for text in iter_obj:
            text_holder=[]
            pars_list = text.split('\n')
            for par in pars_list:
                text_holder.append(self.full_process(par))
            holder.append(text_holder)
        return holder
    
    def words_count(self, acts_gen):
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
                 enc='cp1251',
                 dir_structure=None):
        self.RWT = ReadWriteTool(enc=enc)
        self.CTP = CustomTextProcessor()
        self.Vct = Vectorization()
        if dir_structure:
            self.dir_struct = dir_structure
            print(
                "Dir structure dictionary must have the following "
                +"'{key : value} pairs:\n"
                +"'{'MainRoot' : dir_path,\n"
                +"'Raw_text' : dir_path,\n"
                +"'Divided_and_tokenized' : dir_path,\n"
                +"'Normalized_by_parser1' : dir_path,\n"
                +"'Normalized_by_parser2' : dir_path,\n"
                +"'Conclusions' : dir_path,\n"
                +"'Statistics_and_data' : dir_path,\n"
                +"'Results' : dir_path}\n"
                +"Othervise please use built-in presets!"
            )
        else:
            self.dir_struct = {
                'MainRoot': (
                    pthl.Path().home().joinpath('TextProcessing')
                ),
                'Raw_text': (
                    pthl.Path().home().joinpath('TextProcessing','RawText')
                ),
                'Divided_and_tokenized': (
                    pthl.Path().home().joinpath('TextProcessing','DivToks')
                ),
                'Normalized_by_parser1': (
                    pthl.Path().home().joinpath('TextProcessing','Norm1')
                ),
                'Normalized_by_parser2': (
                    pthl.Path().home().joinpath('TextProcessing','Norm2')
                ),
                'Conclusions': (
                    pthl.Path().home().joinpath('TextProcessing', 'Conclusions')
                ),
                'Statistics_and_data': (
                    pthl.Path().home().joinpath('TextProcessing', 'StatData')
                ),
                'Results': (
                    pthl.Path().home().joinpath('TextProcessing', 'Results')
                )
            }
        print('Constructor class created')
    
    def create_dir_struct(self):
        self.RWT.create_dirs(self.dir_struct)     

    def divide_and_tokenize_acts(self):
        raw_files = self.RWT.iterate_text_loading(self.dir_struct['Raw_text'])
        subdirs = deque(self.RWT.create_acts_subdirs(
            path = self.dir_struct['Raw_text'],
            des_folder = self.dir_struct['Divided_and_tokenized']
        ))
        for fle in raw_files:
            print('Starting new file processing!')
            cleaned = self.CTP.court_decisions_cleaner(fle)
            divided = self.CTP.court_decisions_separator(cleaned)
            print('\tStrating tokenization!')
            t0 = time()
            tokenized = self.CTP.iterate_tokenization(divided)
            print('\tTokenization complete in {} seconds!'.format(time()-t0))
            file_paths = self.RWT.create_writing_paths(
                len(divided),
                subdirs.popleft(),
                suffix=''
            )
            self.RWT.iterate_pickle_writing(tokenized, file_paths)
    
    def vocab_raw_words(self, path=None):
        top_dir = path if path else self.dir_struct['Divided_and_tokenized']
        all_files = self.RWT.iterate_pickle_loading(top_dir)
        vocab = self.CTP.words_count(all_files)
        return vocab
    
    def vocab_norm_words(self, par_type='parser1'):
        acts_path = self.dir_struct['Normalized_by_{}'.format(par_type)]
        all_files = self.RWT.iterate_pickle_loading(acts_path)
        vocab = self.CTP.words_count(all_files)
        return vocab
    
    def create_lem_dict(self,
                        vocab,
                        par_type='parser1',
                        save=False):
        all_words = list(vocab.keys())
        print('Strating normalization!')
        t0 = time()
        norm_words = self.CTP.lemmatize(all_words, par_type=par_type)
        print('Normalization complete in {} seconds'.format(time()-t0))
        lem_dict = {
            raw_word:norm_word
            for (raw_word, norm_word)
            in zip(all_words, norm_words)            
        }
        if save:
            self.save_lem_dict(lem_dict, par_type=par_type)
        else:
            return lem_dict
        
    def save_vocab(self, vocab, file_name):
        path = self.dir_struct['Statistics_and_data']
        path = path.joinpath(file_name)
        self.RWT.write_pickle(vocab, path)

    def load_vocab(self, file_name):
        path = self.dir_struct['Statistics_and_data']
        path = path.joinpath(file_name)
        return self.RWT.load_pickle(path)
    
    def save_lem_dict(self, lem_dict, par_type):
        path = self.dir_struct['Statistics_and_data']
        path = path.joinpath('lem_dict_normed_by_'+par_type)
        self.RWT.write_pickle(lem_dict, path)

    def load_lem_dict(self, par_type):
        path = self.dir_struct['Statistics_and_data']
        path = path.joinpath('lem_dict_normed_by_'+par_type)
        return self.RWT.load_pickle(path)
    
    def lemmatize_and_save_acts(self, lem_dict, par_type='parser1'):
        path = self.dir_struct['Divided_and_tokenized']
        all_acts = self.RWT.iterate_pickle_loading(path)
        all_lemmed_acts = self.CTP.iterate_lemmatize_by_dict(
            lem_dict,
            all_acts
        )
        subdirs = [
            path_item.name
            for path_item in path.iterdir()
                if path_item.is_dir()
        ]
        for subdir in subdirs:
            subpath = self.dir_struct['Normalized_by_{}'.format(par_type)]
            subpath = subpath.joinpath(subdir)
            subpath.mkdir(parents=True, exist_ok=True)
        all_files = self.RWT.collect_exist_file_paths(
            self.dir_struct['Divided_and_tokenized'],
            suffix=''
        )
        all_files = [
            file_path.parents[2].joinpath(
                self.dir_struct['Normalized_by_{}'.format(par_type)],
                *file_path.parts[-2:]
            )
            for file_path in all_files
        ]
        self.RWT.iterate_pickle_writing(
            all_lemmed_acts,
            all_files
        )
    
    def start_conclusions_iteration(self):
        '''
        Return generator object over text files' paths
        '''
        return self.RWT.iterate_text_loading(self.dir_struct['Conclusions'])
    
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
            concl = ' '.join(self.CTP.full_process(concl))
            acts = (
                self.RWT.iterate_pickle_loading\
                (self.dir_struct['Normalized_by_parser1'])
            )
            print('\n', concl[:50], '\n', sep='')
            t0 = time()
            holder = []
            for act in acts:
                act = [' '.join(par_lst) for par_lst in act]
                data_mtrx = self.act_and_concl_to_mtrx(act, concl)
                par_index, cos = self.eval_cos_dist(data_mtrx)
                holder.append([act[0], act[2], cos, act[par_index-1]])
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
            name = concl[:40]
            self.table_to_csv(
                holder,
                columns=4,
                header=('Суд','Реквизиты','Косинус', 'Абзац'),
                zero_string = concl,
                file_name=name
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
        print('Execution ended')
    
    def table_to_csv(self,
                     table,
                     columns=2,
                     path=None,
                     zero_string=None,
                     header=['Col1', 'Col2'],
                     file_name='py_table'):
        if path:
            path = pthl.Path(path)
            path.joinpath(file_name).with_suffix('.csv')
        else:
            path = self.dir_struct['Results']
            path = path.joinpath(file_name).with_suffix('.csv')
        assert columns == len(header)
        self.RWT.write_text_to_csv(
            path,
            table,
            zero_string=zero_string,
            header=header
        )

    def export_vocab_to_csv(self, vocab, file_name='count_uniq'):
        path = self.dir_struct['Results']
        path = path.joinpath(file_name).with_suffix('.csv')
        print('Sorting vocabulary')
        srt_vocab = vocab.most_common()
        self.table_to_csv(
            srt_vocab,
            columns=2,
            path=path,
            header=('Слова','Количество вхождений в корпус'),
            file_name=file_name
        )

