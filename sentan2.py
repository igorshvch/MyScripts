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
#from writer import writer

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
PATTERN_ACT_CLEAN1 = (
    '-{66}\nКонсультантПлюс.+?-{66}\n'
)
PATTERN_ACT_CLEAN2 = (
    'КонсультантПлюс.+?\n.+?\n'
)
PATTERN_ACT_CLEAN3 = (
    'Рубрикатор ФАС \(АСЗО\).*?Текст документа'
)
PATTERN_ACT_SEP1 = (
    '\n\n\n-{66}\n\n\n'
)
PATTERN_ACT_SEP2 = (
    'Документ предоставлен КонсультантПлюс'
)

FILE_NAMES = {
    'Как определить налоговую базу по НДС при': '541',
    'Необходимо ли для подтверждения ставки Н': '581',
    'Освобождается ли от НДС предоставление ж': '553',
    'Правомерно ли начисление штрафа, если эк': '590_1',
    'Увеличивается ли налоговая база по НДС н': '551',
    'Является ли непредставление поставщиком ': '525'
}

#pymoprhy2 analyzer instance
MORPH = pymorphy2.MorphAnalyzer()


class ReadWriteTool():
    '''
    Class provides API to reading and writing options.
    '''
    def __init__(self, enc='cp1251'):
        self.enc=enc
        print('RWT class created')

    def create_dirs(self, dir_struct, sub_dir=''):
        paths = []
        for key in dir_struct.keys():
            if key != 'MainRoot':
                path = dir_struct[key].joinpath(sub_dir)
                path.mkdir(parents=True, exist_ok=True)
                paths.append(str(path))
        print('Created directories:')
        for strg in sorted(paths):
            print('\t'+strg)
    
    def create_dir(self, dir_name, full_path_to_dir):
        path = pthl.Path().joinpath(full_path_to_dir)
        path = path.joinpath(dir_name)
        path.mkdir(parents=True, exist_ok=True)
        print('Created directory:')
        print('\t'+str(path))

    def load_text(self, path):
        with open(path, mode='r', encoding=self.enc) as fle:
            text = fle.read()
        return text

    def collect_exist_file_paths(self, top_dir, suffix=''):
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
        paths = self.collect_exist_file_paths(top_dir, suffix='.txt')
        print(
            len(paths),
            'The first file in the queue is:\n{}'.format(paths[0]),
            sep=' ### '
        )
        return (self.load_text(path) for path in paths)
    
    def write_text(self, text, path):
        with open(path, mode='w', encoding=self.enc) as fle:
                fle.write(text)
    
    def create_writing_paths(self, strt, stp, path, suffix=''):
        p = pthl.Path(path)
        file_paths = [
            p.joinpath(str(i)).with_suffix(suffix)
            for i in range(strt, stp, 1)
        ]
        return file_paths
    
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
        cleaned_text1 = re.subn(
            PATTERN_ACT_CLEAN1, '', text, flags=re.DOTALL
        )[0]
        cleaned_text2 = re.subn(PATTERN_ACT_CLEAN2, '', cleaned_text1)[0]
        cleaned_text3 = re.subn(
            PATTERN_ACT_CLEAN3, '', cleaned_text2, flags=re.DOTALL
        )[0]
        print('Acts were cleaned in {} seconds'.format(time()-t0))
        return cleaned_text3
    
    def court_decisions_separator(self, text, sep_type='sep1'):
        t0 = time()
        if sep_type=='sep1':
            separated_acts = re.split(PATTERN_ACT_SEP1, text)
        else:
            separated_acts = re.split(PATTERN_ACT_SEP2, text)
        print(
            'Acts were separated in {} seconds'.format(time()-t0),
            '{} acts were found'.format(len(separated_acts))
        )
        return separated_acts
            
    def tokenize(self, text, threshold=0):
        text = text.lower().strip()
        return [
            token
            for token in re.split('\W', text)
                if len(token) > threshold
        ]
    
    def iterate_tokenization(self, text_gen):
        '''
        Return generator object
        '''
        for text in text_gen:
            text_holder=[]
            pars_list = text.split('\n')
            for par in pars_list:
                if par:
                    text_holder.append(self.tokenize(par))
            yield text_holder
    
    def remove_stpw_from_list(self, list_obj, vocab):
        return [w for w in list_obj if w not in vocab]
    
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
    
    def lemmatize_by_dict(self, lem_dict, tokens_list, verifier):
        return [lem_dict[token] for token in tokens_list if token in verifier]
    
    def iterate_lemmatize_by_dict(self, lem_dict, acts_gen, verifier):
        '''
        Return generator object
        '''
        for act in acts_gen:
            act = [
                self.lemmatize_by_dict(lem_dict, par, verifier)
                for par in act
            ]
            yield act
    
    def full_process(self, text, par_type='parser1', vocab=None):
        tokens = self.tokenize(text)
        lemms = self.lemmatize(tokens, par_type=par_type)
        if vocab:
            lemms = [w for w in lemms if w not in vocab]
        return lemms
    
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
    
    def collect_all_words(self, act):
        return [
            w
            for par in act
            for w in par
        ]
    
    def create_2grams(self, par):
        holder=[]
        lp = len(par)
        if lp > 1:
            for i in range(1, lp, 1):
                bigram = par[i-1] + '_' + par[i]
                holder.append(bigram)
        return holder
    
    def create_2grams_with_voc(self, vocab):
        def inner_func(par):
            holder=[]
            par = [w for w in par if w not in vocab]
            la = len(par)
            if la > 1:
                for i in range(1, la, 1):
                    bigram = par[i-1] + '_' + par[i]
                    holder.append(bigram)
            return holder
        return inner_func
    
    def create_3grams(self, par):
        holder=[]
        la = len(par)
        if la > 2:
            for i in range(2, la, 1):
                trigram = (
                    par[i-2]
                    + '_' + par[i-1]
                    + '_' + par[i]
                )
                holder.append(trigram)       
        return holder

    def create_3grams_with_voc(self, vocab):
        def inner_func(par):
            holder=[]
            par = [w for w in par if w not in vocab]
            la = len(par)
            if la > 2:
                for i in range(2, la, 1):
                    trigram = (
                        par[i-2]
                        + '_' + par[i-1]
                        + '_' + par[i]
                    )
                    holder.append(trigram)       
            return holder
        return inner_func
    
    def extract_repetitive_ngrams(self, ngram_list, rep_num=2, verbose=True):
        holder = Counter()
        holder.update(ngram_list)
        holder_rep = [ngram for ngram,value in holder.items() if value >= rep_num]
        if verbose:
            print(holder_rep)
        return holder_rep
    
    def intersect_2gr(self, par, stpw, par_type='txt', verbose=True):
        if par_type == 'txt':
            lems = self.full_process(par)
            cleaned_lems = self.full_process(
                par,
                par_type='parser1',
                vocab=stpw
            )
        elif par_type == 'lst':
            lems = par
            cleaned_lems = [w for w in par if w not in stpw]
        bigr_lems = self.create_2grams(lems)
        bigr_cl_lems = self.create_2grams(cleaned_lems)
        common = set(bigr_lems) & set(bigr_cl_lems)
        result = [
            bigr
            for bigr in bigr_lems
                if
                bigr in common
        ]
        if verbose:
            print('Intersection result:\n{}'.format(result))
        return result


class Vectorization():
    def __init__(self, token_pattern='\w+'):
        self.vectorizer = CountVectorizer(token_pattern=token_pattern)
        print('Vct class created')
    
    def create_vocab(self, tokens_lst, threshold=0):
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
                 dir_structure=None,
                 token_pattern='\w+'):
        self.RWT = ReadWriteTool(enc=enc)
        self.CTP = CustomTextProcessor()
        self.Vct = Vectorization(token_pattern=token_pattern)
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
                '2grams':(
                    pthl.Path().home().joinpath('TextProcessing','Bigrams')
                ),
                '3grams':(
                    pthl.Path().home().joinpath('TextProcessing','Trigrams')
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

    def create_sub_dirs(self, dir_name):
        self.RWT.create_dirs(self.dir_struct, sub_dir=dir_name)

    def div_tok_acts(self,
                     dir_name='',
                     sep_type='sep1',
                     iden=''):
        path = self.dir_struct['Raw_text'].joinpath(dir_name)
        raw_files = self.RWT.iterate_text_loading(path)
        counter1 = 0
        counter2 = 0
        for fle in raw_files:
            print(iden+'Starting new file processing!')
            cleaned = self.CTP.court_decisions_cleaner(fle)
            divided = self.CTP.court_decisions_separator(
                cleaned,
                sep_type=sep_type
            )
            tokenized = self.CTP.iterate_tokenization(divided)
            counter2 += len(divided)
            t0=time()
            print(iden+'\tStarting tokenization and writing')
            file_paths = deque(self.RWT.create_writing_paths(
                counter1, counter2,
                self.dir_struct['Divided_and_tokenized'].joinpath(dir_name),
                suffix=''
            ))
            for tok_act in tokenized:
                self.RWT.write_pickle(
                    tok_act,
                    file_paths.popleft()
                )
            counter1 += len(divided)
            print(
                iden+'\tTokenization and writing '
                +'complete in {} seconds!'.format(time()-t0)
            )
    
    def create_vocab(self, dir_name='', spec='raw', iden=''):
        '''
        Accepted 'spec' args:
        raw, norm1
        '''
        options = {
            'raw' : (
                self.dir_struct['Divided_and_tokenized'].joinpath(dir_name)
            ),
            'norm1' : (
                self.dir_struct['Normalized_by_parser1'].joinpath(dir_name)
            ),
            'norm2' : (
                self.dir_struct['Normalized_by_parser2'].joinpath(dir_name)
            ),
        }
        t0=time()
        print(iden+'Starting vocab creation!')
        all_files = self.RWT.iterate_pickle_loading(options[spec])
        vocab = self.CTP.words_count(all_files)
        print(iden+'Vocab created in {} seconds!'.format(time()-t0))
        return vocab
    
    def create_lem_dict(self,
                        vocab,
                        par_type='parser1',
                        iden=''):
        all_words = list(vocab.keys())
        print(iden+'Strating normalization!')
        t0 = time()
        norm_words = self.CTP.lemmatize(all_words, par_type=par_type)
        print(iden+'Normalization complete in {} seconds'.format(time()-t0))
        lem_dict = {
            raw_word:norm_word
            for (raw_word, norm_word)
            in zip(all_words, norm_words)            
        }
        return lem_dict
        
    def save_vocab(self, vocab, spec='raw', dir_name=''):
        '''
        Accepted 'spec' args:
        raw, norm1, lem1
        '''
        path = self.dir_struct['Statistics_and_data']
        options = {
            'raw' : path.joinpath(dir_name, 'vocab_raw_words'),
            'norm1' : path.joinpath(dir_name, 'vocab_norm1_words'),
            'norm2' : path.joinpath(dir_name, 'vocab_norm2_words'),
            'lem1' : path.joinpath(dir_name, 'lem_dict_normed_by_parser1'),
            'lem2' : path.joinpath(dir_name, 'lem_dict_normed_by_parser2')
        }
        path = options[spec]
        self.RWT.write_pickle(vocab, path)

    def load_vocab(self, spec='raw', dir_name=''):
        '''
        Accepted 'spec' args:
        raw, norm1, lem
        '''
        path = self.dir_struct['Statistics_and_data']
        options = {
            'raw' : path.joinpath(dir_name, 'vocab_raw_words'),
            'norm1' : path.joinpath(dir_name, 'vocab_norm1_words'),
            'norm2' : path.joinpath(dir_name, 'vocab_norm2_words'),
            'lem1' : path.joinpath(dir_name, 'lem_dict_normed_by_parser1'),
            'lem2' : path.joinpath(dir_name, 'lem_dict_normed_by_parser2')
        }
        path = options[spec]
        return self.RWT.load_pickle(path)
    
    def lemmatize_and_save_acts(self,
                                lem_dict,
                                par_type='parser1',
                                load_dir_name='',
                                save_dir_name='',
                                iden=''):
        #load paths and lem gen
        load_path = (
            self.dir_struct['Divided_and_tokenized'].joinpath(load_dir_name)
        )
        all_acts_gen = self.RWT.iterate_pickle_loading(load_path)
        lemmed_acts_gen = self.CTP.iterate_lemmatize_by_dict(
            lem_dict,
            all_acts_gen,
            set(lem_dict)
        )
        #saves paths
        acts_quants = len(self.RWT.collect_exist_file_paths(load_path))
        save_dir = self.dir_struct['Normalized_by_{}'.format(par_type)]
        save_dir = save_dir.joinpath(save_dir_name)
        writing_paths = deque(sorted(self.RWT.create_writing_paths(
            0,
            acts_quants,
            save_dir
        )))
        #process
        t0 = time()
        print(iden+'Start normalization and writing')
        for lem_act in lemmed_acts_gen:
            self.RWT.write_pickle(
                lem_act,
                writing_paths.popleft()
            )
        print(
            iden+'Normalization and writing '
            +'complete in {} seconds'.format(time()-t0)
        )
    
    def act_and_concl_to_mtrx(self, vector_pop='concl'):
        '''
        Accepted 'vocab' args:
        'act', 'concl'
        '''
        def inner_func1(pars_list, concl):
            data = [concl] + pars_list
            self.Vct.vectorizer.fit(data)
            data_mtrx = self.Vct.create_vectors(data)
            update_mtrx = (
                np.append(data_mtrx, np.ones((len(data_mtrx),1)), 1)
            )
            return update_mtrx
        def inner_func2(pars_list, concl):
            data = [concl] + pars_list
            self.Vct.vectorizer.fit([concl])
            data_mtrx = self.Vct.create_vectors(data)
            update_mtrx = (
                np.append(data_mtrx, np.ones((len(data_mtrx),1)), 1)
            )
            return update_mtrx
        options = {
            'act' : inner_func1,
            'concl': inner_func2
        }
        return options[vector_pop]

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
    
    def summon_conclusions(self, dir_name):
        path = self.dir_struct['Conclusions'].joinpath(dir_name)
        paths = self.RWT.collect_exist_file_paths(path, suffix='.txt')
        holder = {}
        for p in paths:
            with open(p) as file:
                holder[p.stem] = file.read()
        return holder
    
    def export_cd_eval_results(self,
                               auto_mode=False,
                               concl_dir_name='',
                               load_dir_name='',
                               save_dir_name='',
                               vector_pop='concl',
                               par_type='parser1',
                               vocab=None,
                               bigram='join',
                               rep_ngram=False,
                               rep_ngram_act=False):
        #load concls and set mtrx creation finc
        concls_path = (
            self.dir_struct['Conclusions'].joinpath(concl_dir_name)
        )
        concls = self.RWT.iterate_text_loading(concls_path)
        mtrx_creator = self.act_and_concl_to_mtrx(vector_pop=vector_pop)
        for concl in concls:
            #load acts
            path_to_acts = self.dir_struct['Normalized_by_{}'.format(par_type)]
            ############################
            if bigram == 'intersection' and vocab:
                concl_prep = self.CTP.full_process(
                    concl,
                    par_type=par_type,
                    vocab=vocab
                )
                int_bigrs = self.CTP.intersect_2gr(concl, vocab)
                concl_cleaned = ' '.join(concl_prep+int_bigrs)
            ###########
            elif vocab:
                concl_prep = self.CTP.full_process(
                    concl,
                    par_type=par_type,
                    vocab=vocab
                )
                if bigram == 'join':
                    concl_gram = self.CTP.create_2grams(concl_prep)
                    if rep_ngram:
                        concl_rep_gram = (
                            self.CTP.extract_repetitive_ngrams(concl_gram)
                        )
                        concl_prep = concl_prep + concl_rep_gram
                        concl_cleaned = ' '.join(concl_prep)
                    else:
                        concl_prep = concl_prep + concl_gram
                        concl_cleaned = ' '.join(concl_prep)
                elif bigram == 'simple':
                    concl_gram = self.CTP.create_2grams(concl_prep)
                    if rep_ngram:
                        concl_rep_gram = (
                            self.CTP.extract_repetitive_ngrams(concl_gram)
                        )
                        concl_cleaned = ' '.join(concl_rep_gram)
                    else:
                        concl_cleaned = ' '.join(concl_gram)
                elif bigram == 'no':
                    concl_cleaned = ' '.join(concl_prep)          
            else:
                concl_prep = self.CTP.full_process(
                    concl,
                    par_type=par_type,
                )
                if bigram == 'join':
                    concl_gram = self.CTP.create_2grams(concl_prep)
                    if rep_ngram:
                        concl_rep_gram = (
                            self.CTP.extract_repetitive_ngrams(concl_gram)
                        )
                        concl_prep = concl_prep + concl_rep_gram
                        concl_cleaned = ' '.join(concl_prep)
                    else:
                        concl_prep = concl_prep + concl_gram
                        concl_cleaned = ' '.join(concl_prep)
                elif bigram == 'simple':
                    concl_gram = self.CTP.create_2grams(concl_prep)
                    if rep_ngram:
                        concl_rep_gram = (
                            self.CTP.extract_repetitive_ngrams(concl_gram)
                        )
                        concl_cleaned = ' '.join(concl_rep_gram)
                    else:
                        concl_cleaned = ' '.join(concl_gram)
                elif bigram == 'no':
                    concl_cleaned = ' '.join(concl_prep)
            #Uncleaned_acts
            uncl_acts = deque(self.RWT.collect_exist_file_paths(
                self.dir_struct['Divided_and_tokenized'].joinpath(load_dir_name)
            ))
            #load acts
            path_to_acts = path_to_acts.joinpath(load_dir_name)
            #print('this is it!', path_to_acts)
            acts = self.RWT.iterate_pickle_loading(path_to_acts)
            print('\n', concl[:50], '\n', sep='')
            print(concl_cleaned, '\n', sep='')
            t0 = time()
            holder = []
            #counter = 0
            for act in acts:
                uncl_act = self.RWT.load_pickle(uncl_acts.popleft())
                uncl_act = [' '.join(par_lst) for par_lst in uncl_act]
                ############################
                if bigram == 'intersection' and vocab:
                    act = [
                            self.CTP.remove_stpw_from_list(
                                par,
                                vocab
                            )
                            +
                            self.CTP.intersect_2gr(
                                par,
                                vocab,
                                verbose=False,
                                par_type='lst'
                            )
                            for par in act
                        ]
                    act = [' '.join(par) for par in act]
                    #writer(act, 'act{}'.format(counter), verbose=False)
                ######################        
                elif bigram == 'join':
                    if vocab:
                        act = [
                            self.CTP.remove_stpw_from_list(
                                par,
                                vocab
                            )
                            for par in act
                        ]
                    if rep_ngram_act:
                        act = [
                            ' '.join(
                                par_lst 
                                +
                                self.CTP.extract_repetitive_ngrams(
                                    self.CTP.create_2grams(par_lst),
                                    verbose=False
                                )
                            )
                            for par_lst in act
                        ]
                    else:
                        act = [
                            ' '.join(
                                par_lst + self.CTP.create_2grams(par_lst)
                            )
                            for par_lst in act
                        ]
                elif bigram == 'simple':
                    if vocab:
                        act = [
                            self.CTP.remove_stpw_from_list(
                                par,
                                vocab
                            )
                            for par in act
                        ]
                    if rep_ngram_act:
                        act = [
                            ' '.join(
                                self.CTP.extract_repetitive_ngrams(
                                    self.CTP.create_2grams(par),
                                    verbose=False
                                )
                            )
                            for par in act
                        ]
                    else:
                        act = [
                            ' '.join(self.CTP.create_2grams(par))
                            for par in act
                        ]
                elif bigram == 'no':
                    act = [
                            ' '.join(self.CTP.remove_stpw_from_list(
                                par,
                                vocab
                            ))
                            for par in act
                        ]
                #writer(act, 'act{}'.format(str(counter)), verbose=False)
                data_mtrx = mtrx_creator(act, concl_cleaned)
                par_index, cos = self.eval_cos_dist(data_mtrx)
                #counter+=1
                #if counter%500 == 0:
                    #print(counter)
                holder.append(
                    [uncl_act[0],
                    uncl_act[2],
                    cos,
                    uncl_act[par_index-1]] #act[par_index-1]]
                )
            t1 = time()
            print(
                'Acts were processed!',
                'Time in seconds: {}'.format(t1-t0)
            )
            holder = sorted(holder, key=lambda x: x[2])
            t2 = time()
            print(
                '\n',
                'Results were sorted!',
                'Time in seconds: {}'.format(t2-t1),
                sep=''
            )
            name = concl[:40]
            self.table_to_csv(
                holder,
                dir_name=save_dir_name,
                header=('Суд','Реквизиты','Косинус', 'Абзац'),
                zero_string = concl,
                file_name=FILE_NAMES[name]
            )
            if not auto_mode:
                breaker = None
                while breaker != '1' and breaker != '0':
                    breaker = input(
                    ("Обработка вывода окончена. Обработать следующий вывод? "
                    +"[1 for 'yes'/0 for 'no']\n")
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
                     file_name='py_table',
                     dir_name='',
                     zero_string=None,
                     header=['Col1', 'Col2']):
        path = self.dir_struct['Results']
        path = path.joinpath(dir_name, file_name).with_suffix('.csv')
        assert len(table[0]) == len(header)
        self.RWT.write_text_to_csv(
            path,
            table,
            zero_string=zero_string,
            header=header
        )

    def export_vocab_to_csv(self,
                            vocab,
                            file_name='count_uniq',
                            dir_name=''):
        print('Sorting vocabulary')
        srt_vocab = vocab.most_common()
        self.table_to_csv(
            srt_vocab,
            file_name=file_name,
            dir_name=dir_name,
            header=('Слова','Количество вхождений в корпус')            
        )
    
    def auto(self, dir_name='', sep_type='sep1'):
        t0=time()
        print('Starting division and tokenization!')
        self.div_tok_acts(
            dir_name=dir_name,
            sep_type=sep_type,
            iden='\t'
        )
        print('Acts are divided and tokenized')
        print('Creating raw words dictionary')
        vocab_rw = self.create_vocab(
            dir_name=dir_name,
            spec='raw',
            iden='\t'
        )
        print('Dictionary is created')
        print('Creating mapping')
        lem_dict = self.create_lem_dict(vocab_rw, iden='\t')
        print('Mapping is created')
        print('Starting lemmatization')
        self.lemmatize_and_save_acts(
            lem_dict,
            load_dir_name=dir_name,
            save_dir_name=dir_name,
            iden='\t')
        print('Creating norm words dictionary')
        vocab_nw = self.create_vocab(
            dir_name=dir_name,
            spec='norm1',
            iden='\t'
        )
        print('Dictionary is created')
        print('Saving all dictionaries')
        ###
        self.save_vocab(vocab_rw, spec='raw', dir_name=dir_name)
        self.save_vocab(vocab_nw, spec='norm1', dir_name=dir_name)
        self.save_vocab(lem_dict, spec='lem1', dir_name=dir_name)
        print('Dictionaries are saved')
        print('Total time costs: {} seconds'.format(time()-t0))

    def load_file(self, full_path):
        return self.RWT.load_pickle(full_path)
    
    def save_object(self, py_obj, file_name, full_path):
        path = pthl.Path()
        path = path.joinpath(full_path, file_name)
        self.RWT.write_pickle(py_obj, path)
    
    def file_to_text(self, full_path_to_file, output_file_name):
        path = pthl.Path(full_path_to_file)
        print('Entered path: {}'.format(path))
        path = path.joinpath(output_file_name)
        obj = self.RWT.load_pickle(path)
        self.RWT.write_text(obj, path)
    
    def obj_to_text(self, obj, file_name, dir_name=''):
        if type(obj) != str:
            obj = str(obj)
        path = self.dir_struct['Results']
        path = path.joinpath(dir_name, file_name)
        self.RWT.write_text(obj, path)
      
    def gram_to_csv(self, gram_obj, file_name='Ngram', dir_name=''):
        path = self.dir_struct['Results']
        path = path.joinpath(file_name).with_suffix('.csv')
        srt_gram = gram_obj.most_common()
        self.table_to_csv(
            srt_gram,
            file_name=file_name,
            dir_name=dir_name,
            header=('N-граммы','Количество вхождений в корпус')   
        )
    
    def collect_2grams(self,
                       gram=2,
                       load_dir_name='',
                       save_dir_name='',
                       vocab=None):
        path_to_acts = self.dir_struct['Normalized_by_parser1']
        path_to_acts = path_to_acts.joinpath(load_dir_name)
        
        acts = self.RWT.iterate_pickle_loading(path_to_acts)
        acts_quant = len(self.RWT.collect_exist_file_paths(path_to_acts))
        
        save_dir = self.dir_struct['{}grams'.format(gram)]
        save_dir = save_dir.joinpath(save_dir_name)
        writing_paths = deque(sorted(self.RWT.create_writing_paths(
            0,
            acts_quant,
            save_dir
        )))
        t0 = time()
        if vocab:
            func_2grams = self.CTP.create_2grams_with_voc(vocab)
        else:
            func_2grams = self.CTP.create_2grams
        print('Start collecting bigrmas and writing')
        for act in acts:
            if vocab:
                act = [
                    self.CTP.remove_stpw_from_list(par, vocab)
                    for par in act
                ]
                act = [par for par in act if par]
            bigram = [
                par + func_2grams(par)
                for par in act
            ]
            self.RWT.write_pickle(
                bigram,
                writing_paths.popleft()
            )
        print(
            'Collecting bigrams and writing '+
            'complete in {} seconds'.format(time()-t0)
        )
    
    def collect_3grams(self,
                       gram=3,
                       load_dir_name='',
                       save_dir_name='',
                       vocab=None):
        path_to_acts = self.dir_struct['Normalized_by_parser1']
        path_to_acts = path_to_acts.joinpath(load_dir_name)
        
        acts = self.RWT.iterate_pickle_loading(path_to_acts)
        acts_quant = len(self.RWT.collect_exist_file_paths(path_to_acts))
        
        save_dir = self.dir_struct['{}grams'.format(gram)]
        save_dir = save_dir.joinpath(save_dir_name)
        writing_paths = deque(sorted(self.RWT.create_writing_paths(
            0,
            acts_quant,
            save_dir
        )))
        t0 = time()
        if vocab:
            func_3grams = self.CTP.create_3grams_with_voc(vocab)
        else:
            func_3grams = self.CTP.create_3grams
        print('Start collecting bigrmas and writing')
        for act in acts:
            bigram = [
                par + func_3grams(par)
                for par in act
            ]
            self.RWT.write_pickle(
                bigram,
                writing_paths.popleft()
            )
        print(
            'Collecting bigrams and writing '+
            'complete in {} seconds'.format(time()-t0)
        )