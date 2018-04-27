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
PATTERN_ACT_CLEAN1 = (
    r'-{66}\nКонсультантПлюс.+?-{66}\n'
)
PATTERN_ACT_CLEAN2 = (
    r'\AКонсультантПлюс.+?\n.+?\n'
)
PATTERN_ACT_SEP1 = (
    r'\n\n\n-{66}\n\n\n'
)
PATTERN_ACT_SEP2 = (
    r'Документ предоставлен КонсультантПлюс'
)

#pymoprhy2 analyzer instance
MORPH = pymorphy2.MorphAnalyzer()


class ReadWriteTool():
    '''
    Class provides API to reading and writing options.
    '''
    def __init__(self, enc='cp1251'):
        self.enc=encasdasd
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
        cleaned_text1 = re.subn(PATTERN_ACT_CLEAN1, '', text, flags=re.DOTALL)[0]
        cleaned_text2 = re.subn(PATTERN_ACT_CLEAN2, '', cleaned_text1)[0]
        print('Acts were cleaned in {} seconds'.format(time()-t0))
        return cleaned_text2
    
    def court_decisions_separator(self, text, sep_type='sep1'):
        t0 = time()
        if sep_type=='sep1':
            print(PATTERN_ACT_SEP1)
            separated_acts = re.split(PATTERN_ACT_SEP1, text)
        else:
            print(PATTERN_ACT_SEP2)
            separated_acts = re.split(PATTERN_ACT_SEP2, text)
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
    
    def tokenize_wo_stpw(self, vocab):
        def inner_func(text):
            text = text.lower().strip()
            return [
                token
                for token in re.split('\W', text)
                    if len(token)>1 and token not in vocab
            ]
        return inner_func
    
    def remove_stpw_from_list(self, list_obj, vocab):
        return [w for w in list_obj if w not in vocab]
    
    def iterate_tokenization_wo_stpw(self, text_gen, vocab):
        holder = []
        tokenizer = self.tokenize_wo_stpw(vocab)
        for text in text_gen:
            text_holder=[]
            pars_list = text.split('\n')
            for par in pars_list:
                if par:
                    text_holder.append(tokenizer(par))
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
        if vocab:
            tokens = self.tokenize_wo_stpw(vocab)(text)
        else:
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
    
    def collect_all_words(self, act):
        return [
            w
            for par in act
            for w in par
        ]
    
    def create_2grams(self, par):
        holder=[]
        la = len(par)
        if la > 1:
            for i in range(1, la, 1):
                bigram = par[i-1] + ' ' + par[i]
                holder.append(bigram)
        return holder
    
    def create_2grams_with_voc(self, vocab):
        def inner_func(par):
            holder=[]
            par = [w for w in par if w not in vocab]
            la = len(par)
            if la > 1:
                for i in range(1, la, 1):
                    bigram = par[i-1] + ' ' + par[i]
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
                    + ' ' + par[i-1]
                    + ' ' + par[i]
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
                        + ' ' + par[i-1]
                        + ' ' + par[i]
                    )
                    holder.append(trigram)       
            return holder
        return inner_func
    
    def extract_repetative_ngrams(self, ngram_list, rep_num=2):
        holder = Counter()
        holder.update(ngram_list)
        holder_rep = [ngram for ngram,value in holder.items() if value >=2]
        print(holder_rep)
        return holder_rep



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

    def div_tok_acts(self, dir_name='', sep_type='sep1'):
        path = self.dir_struct['Raw_text'].joinpath(dir_name)
        raw_files = self.RWT.iterate_text_loading(path)
        counter1 = 0
        counter2 = 0
        for fle in raw_files:
            print('Starting new file processing!')
            cleaned = self.CTP.court_decisions_cleaner(fle)
            divided = self.CTP.court_decisions_separator(
                cleaned,
                sep_type=sep_type
            )
            print('\tStrating tokenization!')
            t0 = time()
            tokenized = self.CTP.iterate_tokenization(divided)
            print('\tTokenization complete in {} seconds!'.format(time()-t0))
            counter2 += len(divided)
            t0=time()
            print('\tStart writing')
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
            print('\tWriting complete in {} seconds!'.format(time()-t0))
    
    def vocab_raw_words(self, dir_name=''):
        top_dir = self.dir_struct['Divided_and_tokenized'].joinpath(dir_name)
        all_files = self.RWT.iterate_pickle_loading(top_dir)
        vocab = self.CTP.words_count(all_files)
        return vocab
    
    def vocab_norm_words(self, par_type='parser1', dir_name=''):
        acts_path = self.dir_struct['Normalized_by_{}'.format(par_type)]
        acts_path = acts_path.joinpath(dir_name)
        all_files = self.RWT.iterate_pickle_loading(acts_path)
        vocab = self.CTP.words_count(all_files)
        return vocab
    
    def create_lem_dict(self,
                        vocab,
                        par_type='parser1'):
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
        return lem_dict
        
    def save_vocab(self, vocab, file_name, dir_name=''):
        path = self.dir_struct['Statistics_and_data']
        path = path.joinpath(dir_name, file_name)
        self.RWT.write_pickle(vocab, path)

    def load_vocab(self, file_name, dir_name=''):
        path = self.dir_struct['Statistics_and_data']
        path = path.joinpath(dir_name, file_name)
        return self.RWT.load_pickle(path)
    
    def save_lem_dict(self, lem_dict, par_type, dir_name=''):
        path = self.dir_struct['Statistics_and_data']
        path = path.joinpath(dir_name, 'lem_dict_normed_by_'+par_type)
        self.RWT.write_pickle(lem_dict, path)

    def load_lem_dict(self, par_type, dir_name=''):
        path = self.dir_struct['Statistics_and_data']
        path = path.joinpath(dir_name, 'lem_dict_normed_by_'+par_type)
        return self.RWT.load_pickle(path)
    
    def lemmatize_and_save_acts(self,
                                lem_dict,
                                par_type='parser1',
                                load_dir_name='',
                                save_dir_name=''):
        path = self.dir_struct['Divided_and_tokenized'].joinpath(load_dir_name)
        all_acts_gen = self.RWT.iterate_pickle_loading(path)
        lemmed_acts_gen = self.CTP.iterate_lemmatize_by_dict(
            lem_dict,
            all_acts_gen,
            set(lem_dict)
        )
        acts_quant = len(self.RWT.collect_exist_file_paths(path))
        save_dir = self.dir_struct['Normalized_by_{}'.format(par_type)]
        save_dir = save_dir.joinpath(save_dir_name)
        writing_paths = deque(sorted(self.RWT.create_writing_paths(
            0,
            acts_quant,
            save_dir
        )))
        t0 = time()
        print('Start normalization and writing')
        for lem_act in lemmed_acts_gen:
            self.RWT.write_pickle(
                lem_act,
                writing_paths.popleft()
            )
        print('Normalization and writing complete in {} seconds'.format(time()-t0))
    
    def start_conclusions_iteration(self, dir_name=''):
        '''
        Return generator object over text files' paths
        '''
        path = self.dir_struct['Conclusions'].joinpath(dir_name)
        return self.RWT.iterate_text_loading(path)
    
    def act_and_concl_to_mtrx(self, pars_list, concl):
        data = [concl] + pars_list
        self.Vct.vectorizer.fit([concl])
        data_mtrx = self.Vct.create_vectors(data)
        update_mtrx = np.append(data_mtrx, np.ones((len(data_mtrx),1)), 1)
        return update_mtrx

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

    def export_cd_eval_results(self,
                               auto_mode=False,
                               concl_dir_name='',
                               load_dir_name='',
                               save_dir_name=''):
        concls = self.start_conclusions_iteration(dir_name=concl_dir_name)
        for concl in concls:
            concl_cleaned = ' '.join(self.CTP.full_process(concl))
            #uncl_acts = deque(self.RWT.collect_exist_file_paths(
            #    self.dir_struct['Divided_and_tokenized']
            #))
            path_to_acts = self.dir_struct['Normalized_by_parser1']
            path_to_acts = path_to_acts.joinpath(load_dir_name)
            acts = self.RWT.iterate_pickle_loading(path_to_acts)
            print('\n', concl[:50], '\n', sep='')
            t0 = time()
            holder = []
            for act in acts:
                #uncl_act = self.RWT.load_pickle(uncl_acts.popleft())
                act = [' '.join(par_lst) for par_lst in act]
                data_mtrx = self.act_and_concl_to_mtrx(act, concl_cleaned)
                par_index, cos = self.eval_cos_dist(data_mtrx)
                holder.append(
                    [act[0],
                    act[2],
                    cos,
                    act[par_index-1]] #uncl_act[par_index-1]]
                )
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
                dir_name=save_dir_name,
                header=('Суд','Реквизиты','Косинус', 'Абзац'),
                zero_string = concl,
                file_name=name
            )
            if not auto_mode:
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
        print('\t\tStarting division and tokenization!')
        self.div_tok_acts(dir_name=dir_name, sep_type=sep_type)
        print('\t\tActs are divided and tokenized')
        print('\t\tCreating raw words dictionary')
        vocab_rw = self.vocab_raw_words(dir_name=dir_name)
        print('\t\tDictionary is created')
        print('\t\tCreating mapping')
        self.lem_dict = self.create_lem_dict(vocab_rw)
        print('\t\tMapping is created')
        print('\t\tStarting lemmatization')
        self.lemmatize_and_save_acts(self.lem_dict,
                                     load_dir_name=dir_name,
                                     save_dir_name=dir_name)
        print('\t\tCreating norm words dictionary')
        self.vocab_nw = self.vocab_norm_words(dir_name=dir_name)
        print('\t\tDictionary is created')
        print('\t\tSaving all dictionaries')
        self.save_lem_dict(self.lem_dict,
                           par_type='parser1',
                           dir_name=dir_name)
        self.save_vocab(vocab_rw, 'vocab_raw_words', dir_name=dir_name)
        self.save_vocab(self.vocab_nw, 'vocab_norm1_words', dir_name=dir_name)
        print('\t\tDictionaries are saved')
        print('\t\tTotal time costs: {} seconds'.format(time()-t0))

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
    
    def _create_2_3_gram_doc(self,
                             doc_path=None,
                             gram=2):
        t0 = time()
        if doc_path.suffix == '.txt':
            doc = self.RWT.load_text(doc_path)
        else:
            doc = self.RWT.load_pickle(doc_path)
        vocab = Counter()
        all_words = self.CTP.collect_all_words(doc) 
        if gram == 2:
            holder = deque(maxlen=2)
            while all_words:
                try:
                    holder.appendleft(all_words.pop())
                    holder.appendleft(all_words.pop())
                except:
                    break
                vocab.update([' '.join(holder)])
        elif gram == 3:
            holder = deque(maxlen=3)
            while all_words:
                try:
                    holder.appendleft(all_words.pop())
                    holder.appendleft(all_words.pop())
                    holder.appendleft(all_words.pop())
                except:
                    break
                vocab.update([' '.join(holder)])
        print('Ngrams collected in {} seconds'.format(time()-t0))
        return vocab
    
    def _create_2_3_gram_corp(self,
                             path=None,
                             gram=2,
                             norm=True,
                             par_type='parser1',
                             dir_name=''):
        t0 = time()
        if path:
            acts_gen = self.RWT.iterate_pickle_loading(
                pthl.Path(path)
            )
        elif norm:
            load_path = self.dir_struct['Normalized_by_{}'.format(par_type)]
            load_path = load_path.joinpath(dir_name)        
            acts_gen = self.RWT.iterate_pickle_loading(load_path)
        else:
            load_path = self.dir_struct['Divided_and_tokenized']
            load_path = load_path.joinpath(dir_name)
            acts_gen = self.RWT.iterate_pickle_loading(load_path)
        vocab = Counter()
        for act in acts_gen:
            all_words = self.CTP.collect_all_words(act) 
            if gram == 2:
                holder = deque(maxlen=2)
                while all_words:
                    try:
                        holder.appendleft(all_words.pop())
                        holder.appendleft(all_words.pop())
                    except:
                        break
                    vocab.update([' '.join(holder)])
            elif gram == 3:
                holder = deque(maxlen=3)
                while all_words:
                    try:
                        holder.appendleft(all_words.pop())
                        holder.appendleft(all_words.pop())
                        holder.appendleft(all_words.pop())
                    except:
                        break
                    vocab.update([' '.join(holder)])
        print('Ngrams collected in {} seconds'.format(time()-t0))
        return vocab
    
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
    
    def export_cd_eval_results_bigram(self,
                                      gram=2,
                                      auto_mode=False,
                                      concl_dir_name='',
                                      load_dir_name='',
                                      save_dir_name='',
                                      vocab=None):
        concls = self.start_conclusions_iteration(dir_name=concl_dir_name)
        for concl in concls:
            if vocab:
                concl_prep = self.CTP.tokenize_wo_stpw(vocab)(concl)
            else:
                concl_prep = self.CTP.tokenize(concl)
            #print(concl_prep)
            concl_gram = self.CTP.create_2grams(concl_prep)
            concl_rep_gram = self.CTP.extract_repetative_ngrams(concl_gram)
            concl_prep = concl_prep + concl_rep_gram
            concl_cleaned = ' '.join(concl_prep)
            print(concl_cleaned)
            uncl_acts = deque(self.RWT.collect_exist_file_paths(
                self.dir_struct['Divided_and_tokenized'].joinpath('2018-04-24')
            ))
            path_to_acts = self.dir_struct['{}grams'.format(gram)]
            path_to_acts = path_to_acts.joinpath(load_dir_name)
            acts = self.RWT.iterate_pickle_loading(path_to_acts)
            print('\n', concl[:50], '\n', sep='')
            t0 = time()
            holder = []
            for act in acts:
                uncl_act = self.RWT.load_pickle(uncl_acts.popleft())
                uncl_act = [' '.join(par_lst) for par_lst in uncl_act]
                act = [' '.join(par_lst) for par_lst in act]
                data_mtrx = self.act_and_concl_to_mtrx(act, concl_cleaned)
                par_index, cos = self.eval_cos_dist(data_mtrx)
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
                'Results were sorted!',
                'Time in seconds: {}'.format(t2-t1)
            )
            name = concl[:40]
            self.table_to_csv(
                holder,
                dir_name=save_dir_name,
                header=('Суд','Реквизиты','Косинус', 'Абзац'),
                zero_string = concl,
                file_name=name
            )
            if not auto_mode:
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
    
    def export_cd_eval_results_trigram(self,
                                       gram=3,
                                       auto_mode=False,
                                       concl_dir_name='',
                                       load_dir_name='',
                                       save_dir_name=''):
        concls = self.start_conclusions_iteration(dir_name=concl_dir_name)
        for concl in concls:
            concl_prep = self.CTP.tokenize(concl)
            concl_prep = concl_prep + self.CTP.create_3grams(concl_prep)
            concl_cleaned = ' '.join(concl_prep)
            uncl_acts = deque(self.RWT.collect_exist_file_paths(
                self.dir_struct['Divided_and_tokenized'].joinpath(load_dir_name)
            ))
            path_to_acts = self.dir_struct['{}grams'.format(gram)]
            path_to_acts = path_to_acts.joinpath(load_dir_name)
            acts = self.RWT.iterate_pickle_loading(path_to_acts)
            print('\n', concl[:50], '\n', sep='')
            t0 = time()
            holder = []
            for act in acts:
                uncl_act = self.RWT.load_pickle(uncl_acts.popleft())
                uncl_act = [' '.join(par_lst) for par_lst in uncl_act]
                act = [' '.join(par_lst) for par_lst in act]
                data_mtrx = self.act_and_concl_to_mtrx(act, concl_cleaned)
                par_index, cos = self.eval_cos_dist(data_mtrx)
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
                'Results were sorted!',
                'Time in seconds: {}'.format(t2-t1)
            )
            name = concl[:40]
            self.table_to_csv(
                holder,
                dir_name=save_dir_name,
                header=('Суд','Реквизиты','Косинус', 'Абзац'),
                zero_string = concl,
                file_name=name
            )
            if not auto_mode:
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