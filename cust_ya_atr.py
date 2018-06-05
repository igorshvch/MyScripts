import re
import math
#import numpy as np
import pymorphy2
import csv
import pickle
import shelve
import math
import mysqlite
#import operator holder = sorted(holder, key=operator.itemgetter(0,1,2))
import pathlib as pthl
from time import time
#from scipy.spatial.distance import cosine as sp_cosine
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter, deque
#from nltk import sent_tokenize
#from nltk.corpus import stopwords
from writer import writer

# Term 'bc' in comments below means 'backward compatibility'

# Each class method with leading single underscore in its title
# needs to be modified or removed or was provided for bc reasons.
# All such methods (except specified fo bc) are error prone
# and not recomended for usage

#CONSTANTS:
PATTERN_ACT_CLEAN1 = (
    '-{66}\nКонсультантПлюс.+?-{66}\n'
)
PATTERN_ACT_CLEAN2 = (
    'КонсультантПлюс.+?\n.+?\n'
)
PATTERN_ACT_CLEAN3 = (
    'Рубрикатор ФАС \(АСЗО\).*?Текст документа\n'
)
PATTERN_ACT_SEP1 = (
    '\n\n\n-{66}\n\n\n'
)
PATTERN_ACT_SEP2 = (
    'Документ предоставлен КонсультантПлюс'
)
PATTERN_PASS = (
    'ОБЗОР\nСУДЕБНОЙ ПРАКТИКИ ПО ДЕЛАМ,'
    '+ РАССМОТРЕННЫМ\nАРБИТРАЖНЫМ СУДОМ УРАЛЬСКОГО ОКРУГА'
)

FILE_NAMES = {
    'Является ли непредставление поставщиком отчетности в налоговый орган признаком получения необоснованной налоговой выгоды по НДС (недобросовестности) (ст. ст. 54.1, 171 НК РФ) Непре': '525',
    'Как определить налоговую базу по НДС при безвозмездной реализации товаров (работ, услуг) (п. 2 ст. 154 НК РФ)? при определении налоговой базы по НДС при безвозмездной реализации то': '541',
    'Увеличивается ли налоговая база по НДС на суммы страхового возмещения, получаемые при наступлении страхового случая (пп. 2 п. 1 ст. 162 НК РФ) суммы страхового возмещения, получаем': '551',
    'Освобождается ли от НДС предоставление жилой площади в общежитиях (пп. 10 п. 2 ст. 149 НК РФ) Предоставление жилой площади в общежитиях облагается НДС что согласно ст. 16 ЖК РФ к ж': '553',
    'Необходимо ли для подтверждения ставки НДС 10 процентов представлять сертификаты соответствия (п. 2 ст. 164 НК РФ) Непредставление сертификатов не влияет на право применения ставки': '581',
    'Правомерно ли начисление штрафа, если экспорт подтвердили по истечении 180 календарных дней, но НДС при этом уплачен не был (п. 9 ст. 165 НК РФ) Если экспорт подтвердили по истечен': '590_1',
    'Правомерно ли применение вычета по НДС, если акт приемки-передачи отсутствует (ст. 54.1, п. 1 ст. 172 НК РФ) Отсутствие акта приема-передачи не влечет отказа в вычете НДС отсутстви': '43',
    'Должен ли налогоплательщик в целях применения освобождения от уплаты НДС включать в расчет выручки суммы, полученные от операций, не облагаемых НДС (освобождаемых от налогообложени': '44',
    'Являются ли плательщиками НДС физические лица, осуществляющие незаконную предпринимательскую деятельность без регистрации (п. 2 ст. 11, п. 1 ст. 143 НК РФ) физическое лицо сдавало ': '57',
    'Правомерно ли применение вычета по НДС при отсутствии товарно-транспортной накладной (п. 1 ст. 172 НК РФ) При отсутствии товарно-транспортной накладной вычет правомерен При отсутст': '6',
    'Имеет ли налогоплательщик право на вычет, если контрагент (субпоставщик) не перечислил НДС в бюджет (перечислил его не полностью) (ст. 54.1, п. 1 ст. 171 НК РФ) Право на вычет есть': '60',
    'Облагается ли НДС передача (в том числе безвозмездная) арендатором неотделимых улучшений арендодателю (пп. 1 п. 1 ст. 146 НК РФ) Передача неотделимых улучшений в рамках договора ар': '74',
    'Можно ли обязать контрагента выставить (исправить) счет-фактуру (п. 3 ст. 168 НК РФ) Контрагента можно обязать выставить (исправить) счет-фактуру (корректировочный счет-фактуру) На': '77',
    'Можно ли обязать контрагента выставить (исправить) счет-фактуру (п. 3 ст. 168 НК РФ) Контрагента можно обязать выставить (исправить) счет-фактуру (корректировочный счет-фактуру) пр': '77_2',
    'Имеет ли налогоплательщик право на вычет по НДС, если контрагент не представляет отчетность в налоговые органы (ст. 54.1, п. 1 ст. 171 НК РФ) Налогоплательщик имеет право на вычет,': '80'
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
    
    def map_concl_to_num(self, path_holder, lngth):
        dct = {}
        for p in path_holder:
            with open(str(p), mode='r') as file:
                text = file.read()
            dct[p.stem] = text[:lngth]
        return dct
    
    def map_num_to_concl(self, path_holder, lngth):
        dct = {}
        for p in path_holder:
            with open(str(p), mode='r') as file:
                text = file.read()
            dct[text[:lngth]] = p.stem
        return dct

    def load_text(self, path):
        with open(path, mode='r', encoding=self.enc) as fle:
            text = fle.read()
        return text
        
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
    
    def create_writing_paths(self, strt, stp, path, pref_len, suffix=''):
        p = pthl.Path(path)
        names = [
            '0'*(pref_len+1-len(str(i)))+str(i)
            for i in range(strt, stp, 1)
        ]
        file_paths = [
            p.joinpath(i).with_suffix(suffix)
            for i in names
        ]
        return sorted(file_paths)
    
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
    
    def iterate_pickle_loading(self, top_dir, suffix=''):
        '''
        Return generator object iterating over all binary files
        in the top_dir subdirectories.
        '''
        paths = self.collect_exist_file_paths(top_dir, suffix=suffix)
        print(
            len(paths),
            'The first file in the queue is:\n{}'.format(paths[0]),
            sep='### '
        )
        return (self.load_pickle(path) for path in paths)
    
    def iterate_shelve_reading(self, path):
        with shelve.open(str(path), flag='r') as db:
            keys = sorted(db.keys())
            for i in keys:
                yield db[i]
    
    
class DirManager(ReadWriteTool):
    def __init__(self, enc='cp1251'):
        ReadWriteTool.__init__(self, enc=enc)
        self.dir_struct = {
                'MainRoot': (
                    pthl.Path().home().joinpath('TextProcessing')
                ),
                'Raw_text': (
                    pthl.Path().home().joinpath('TextProcessing','RawText')
                ),
                'DivTok': (
                    pthl.Path().home().joinpath('TextProcessing','DivToks')
                ),
                'DivTokPars': (
                    pthl.Path().home().joinpath('TextProcessing','DivTokPars')
                ),
                'Norm1': (
                    pthl.Path().home().joinpath('TextProcessing','Norm1')
                ),
                'Norm1Pars': (
                    pthl.Path().home().joinpath('TextProcessing','Norm1Pars')
                ),
                'Concls': (
                    pthl.Path().home().joinpath('TextProcessing', 'Conclusions')
                ),
                'StatData': (
                    pthl.Path().home().joinpath('TextProcessing', 'StatData')
                ),
                'Results': (
                    pthl.Path().home().joinpath('TextProcessing', 'Results')
                ),
                'DivActs': (
                    pthl.Path().home().joinpath('TextProcessing', 'DivActs')
                ),
                'ActsInfo': (
                    pthl.Path().home().joinpath('TextProcessing', 'ActsInfo')
                ),
                'ParsInfo': (
                    pthl.Path().home().joinpath('TextProcessing', 'ParsInfo')
                )
            }
    
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

    def create_sub_dirs(self, dir_name):
        self.create_dirs(self.dir_struct, sub_dir=dir_name)
    

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
            if re.match(PATTERN_PASS, text):
                continue
            text_holder=[]
            pars_list = text.split('\n')
            for par in pars_list:
                if par:
                    text_holder.append(self.tokenize(par))
            yield text_holder
    
    def iterate_tokenization_by_par(self, text_gen):
        '''
        Return generator object
        '''
        for text in text_gen:
            if re.match(PATTERN_PASS, text):
                continue
            pars_list = text.split('\n')
            for par in pars_list[4:]:
                if par:
                    res = (
                        [' '.join(self.tokenize(pars_list[0]))] + 
                        [' '.join(self.tokenize(pars_list[3]))]
                    )
                    res += self.tokenize(par)
                    yield res
    
    def remove_stpw_from_list(self, list_obj, vocab):
        return [w for w in list_obj if w not in vocab]
    
    def lemmatize(self, tokens_list):
        parser = self.parser
        return [parser(token) for token in tokens_list]
    
    def lemmatize_by_dict(self, lem_dict, tokens_list):
        return [lem_dict[token] for token in tokens_list]
    
    def iterate_lemmatize_par_by_dict(self,
                                      lem_dict,
                                      pars_gen,
                                      stop_w=None,
                                      db=False):
        '''
        Return generator object
        '''
        if not db:
            for par in pars_gen:
                act_info, par = par[:2], par[2:]
                lems = [lem_dict[token] for token in par]
                if stop_w:
                    lems = [lem for lem in lems if lem not in stop_w]
                yield act_info + lems
        else:
            for par in pars_gen:
                par = par[1].split('#')
                act_info, par = par[:2], par[2:]
                lems = [lem_dict[token] for token in par]
                if stop_w:
                    lems = [lem for lem in lems if lem not in stop_w]
                yield '#'.join(act_info + lems)
    
    def iterate_lemmatize_by_dict(self, lem_dict, acts_gen):
        '''
        Return generator object
        '''
        for act in acts_gen:
            act = [
                self.lemmatize_by_dict(lem_dict, par)
                for par in act
            ]
            yield act
    
    def full_process(self, text, stop_w=None):
        tokens = self.tokenize(text)
        lemms = self.lemmatize(tokens)
        if stop_w:
            lemms = [w for w in lemms if w not in stop_w]
        return lemms
    
    def words_count(self, acts_gen):
        t0 = time()
        vocab = Counter()
        for act in acts_gen:
            for par in act:
                vocab.update(par)
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
                bigram = par[i-1], par[i]
                holder.append(bigram)
        return holder
    
    def position_search(self, words):
        d = {word:set() for word in words}
        counter = 0
        for word in words:
            d[word].add(counter)
            counter+=1
        d['total'] = len(words)
        for word in words:
            d['total#'+word]=len(d[word])
        return d
    
    def iterate_position_search(self, act_gen):
        for act in act_gen:
            act = [
                w
                for par in act
                for w in par
            ]
            yield self.position_search(act)
    
    def position_search_pars(self, words):
        words = words[2:]
        d = {word:set() for word in words}
        counter = 0
        for word in words:
            d[word].add(counter)
            counter+=1
        d['total'] = len(words)
        for word in words:
            d['total#'+word]=len(d[word])
        return d
    
    def position_search_pars_db(self, words):
        d = {word:set() for word in words}
        counter = 0
        for word in words:
            d[word].add(counter)
            counter+=1
        d['total'] = len(words)
        for word in words:
            d['total#'+word]=len(d[word])
        return d
    
    
class DivTokLem():
    def __init__(self):
        self.DM = DirManager()
        self.CTP = CustomTextProcessor()
        print('DivTokLem class created')
    
    def div_tok_acts(self,
                     dir_name='',
                     sep_type='sep1',
                     inden=''):
        load_path = (
            self.DM.dir_struct['Raw_text'].joinpath(dir_name)
        )
        raw_files_gen = self.DM.iterate_text_loading(load_path)
        counter = 0
        for fle in raw_files_gen:
            print(inden+'Starting new file processing!')
            cleaned = self.CTP.court_decisions_cleaner(fle)
            divided = self.CTP.court_decisions_separator(
                cleaned,
                sep_type=sep_type
            )
            tokenized = self.CTP.iterate_tokenization(divided)
            t0=time()
            print(inden+'\tStarting tokenization and writing')
            for tok_act in tokenized:
                self.DM.write_pickle(
                    tok_act,
                    self.DM.dir_struct['DivTok'].joinpath(
                        dir_name,
                        '0'*(4+1-len(str(counter)))+str(counter)
                        )
                )
                counter +=1
            print(
                inden+'\tTokenization and writing '
                +'complete in {} seconds!'.format(time()-t0)
            )
        
    def div_tok_pars(self,
                     load_dir_name='',
                     save_dir_name='',
                     sep_type='sep1',
                     inden='',
                     write_to_db=False):
        load_path = (
            self.DM.dir_struct['Raw_text']\
            .joinpath(load_dir_name)
        )
        save_path = (
                self.DM.dir_struct['DivTokPars'].joinpath(save_dir_name)
            )
        if write_to_db:
            DB = mysqlite.DataBase(raw_path=save_path, base_name='DivDB')
            DB.create_tabel(
                'DivTokPars', 
                (('id', 'TEXT', 'PRIMARY KEY'), ('par', 'TEXT'))
            )
        else:
            save_path = save_path.joinpath('DivDB')
        raw_files_gen = self.DM.iterate_text_loading(load_path)
        counter = 0
        t_1 = time()
        for fle in raw_files_gen:
            print(inden+'Starting new file processing!')
            cleaned = self.CTP.court_decisions_cleaner(fle)
            divided = self.CTP.court_decisions_separator(
                cleaned,
                sep_type=sep_type
            )
            tokenized = self.CTP.iterate_tokenization_by_par(divided)
            t0=time()
            print(inden+'\tStarting tokenization and writing')
            if write_to_db:
                holder = []
                for tok_par in tokenized:
                    name = ('0'*(6+1-len(str(counter)))+str(counter))
                    #holder.append((name, tok_par))
                    #data = [(name, '_'.join(tok_par))]
                    holder.append((name, '#'.join(tok_par)))
                    #print('Data length:', len(data))
                    #print(data)
                    #sqldb.insert_data('DivTokPars', data)
                    counter += 1
                    #if counter % 50000 == 0:
                DB.insert_data(holder)
                holder=[]
                print(
                    inden+'\tTokenization and writing '
                    +'complete in {} seconds!'.format(time()-t0)
                )
            else: 
                db = shelve.open(str(save_path), flag='c', writeback=False)
                for tok_par in tokenized:
                    name = ('0'*(6+1-len(str(counter)))+str(counter))
                    db[name] = tok_par
                    counter += 1
                print(inden+'\t', counter)
                db.close()
                print(
                    inden+'\tTokenization and writing '
                    +'complete in {} seconds!'.format(time()-t0)
                )
        print(
            inden+('Total time costs: {}'.format(time()-t_1))
            )
        DB.close()
        
    def load_file(self, full_path):
        return self.DM.load_pickle(full_path)
    
    def save_object(self, py_obj, file_name, full_path):
        path = pthl.Path()
        path = path.joinpath(full_path, file_name)
        self.DM.write_pickle(py_obj, path)
    
    def table_to_csv(self,
                     table,
                     file_name='py_table',
                     dir_name='',
                     zero_string=None,
                     header=['Col1', 'Col2']):
        path = self.DM.dir_struct['Results']
        path = path.joinpath(dir_name, file_name).with_suffix('.csv')
        assert len(table[0]) == len(header)
        self.DM.write_text_to_csv(
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
    
    def create_vocab(self, dir_name='', spec='raw', inden=''):
        '''
        Accepted 'spec' args:
        raw, norm1
        '''
        options = {
            'raw' : (
                self.DM.dir_struct['DivTok'].joinpath(dir_name)
            ),
            'norm1' : (
                self.DM.dir_struct['Norm1'].joinpath(dir_name)
            )
        }
        t0=time()
        print(inden+'Starting vocab creation!')
        all_files = self.DM.iterate_pickle_loading(options[spec])
        vocab = self.CTP.words_count(all_files)
        print(inden+'Vocab created in {} seconds!'.format(time()-t0))
        return vocab
    
    def count_all_words(self, dir_name='', spec='norm1', inden=''):
        '''
        Accepted 'spec' args:
        raw, norm1
        '''
        options = {
            'raw' : (
                self.DM.dir_struct['StatData'].joinpath(
                    dir_name,
                    'vocab_raw_words'
                )
            ),
            'norm1' : (
                self.DM.dir_struct['StatData'].joinpath(
                    dir_name,
                    'vocab_norm1_words'
                )
            )
        }
        vocab =self.DM.load_pickle(options[spec])
        total = sum(vocab.values())
        vocab['all_words'] = total
        print(inden+'All words were counted! Total: {}'.format(total))
        self.save_vocab(vocab, spec=spec, dir_name=dir_name)
    
    def create_lem_dict(self,
                        vocab,
                        inden=''):
        all_words = list(vocab.keys())
        print(inden+'Strating normalization!')
        t0 = time()
        norm_words = self.CTP.lemmatize(all_words)
        print(inden+'Normalization complete in {} seconds'.format(time()-t0))
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
        path = self.DM.dir_struct['StatData']
        options = {
            'raw' : path.joinpath(dir_name, 'vocab_raw_words'),
            'norm1' : path.joinpath(dir_name, 'vocab_norm1_words'),
            'lem1' : path.joinpath(dir_name, 'lem_dict_normed_by_parser1')
        }
        path = options[spec]
        self.DM.write_pickle(vocab, path)

    def load_vocab(self, spec='raw', dir_name=''):
        '''
        Accepted 'spec' args:
        raw, norm1, lem
        '''
        path = self.DM.dir_struct['StatData']
        options = {
            'raw' : path.joinpath(dir_name, 'vocab_raw_words'),
            'norm1' : path.joinpath(dir_name, 'vocab_norm1_words'),
            'lem1' : path.joinpath(dir_name, 'lem_dict_normed_by_parser1')
        }
        path = options[spec]
        return self.DM.load_pickle(path)
    
    def lemmatize_and_save_acts(self,
                                lem_dict,
                                load_dir_name='',
                                save_dir_name='',
                                inden=''):
        #load paths and lem gen
        load_path = (
            self.DM.dir_struct['DivTok'].joinpath(load_dir_name)
        )
        all_acts_gen = self.DM.iterate_pickle_loading(load_path)
        lemmed_acts_gen = self.CTP.iterate_lemmatize_by_dict(
            lem_dict,
            all_acts_gen
        )
        #saves paths
        acts_quants = len(self.DM.collect_exist_file_paths(load_path))
        writing_paths = deque(
            self.DM.create_writing_paths(
                0,
                acts_quants,
                self.DM.dir_struct['Norm1'].joinpath(save_dir_name),
                4,
                suffix=''
            )
        )
        #process
        t0 = time()
        print(inden+'Start normalization and writing')
        for lem_act in lemmed_acts_gen:
            self.DM.write_pickle(
                lem_act,
                writing_paths.popleft()
            )
        print(
            inden+'Normalization and writing '
            +'complete in {} seconds'.format(time()-t0)
        )
    
    def lemmatize_and_save_pars(self,
                                lem_dict,
                                stop_w,
                                load_dir_name='',
                                save_dir_name='',
                                inden='',
                                use_db=False):
        #load paths and lem gen
        if use_db:
            DB_load = mysqlite.DataBase(
                dir_name='TextProcessing/DivTokPars/'+load_dir_name,
                base_name='DivDB',
                tb_name=True
            )
            DB_save = mysqlite.DataBase(
                dir_name='TextProcessing/Norm1Pars/'+save_dir_name,
                base_name='Norm1DB'
            )
            DB_save.create_tabel(
                'Norm1Pars',
                (('id', 'TEXT', 'PRIMARY KEY'), ('par', 'TEXT'))
            )
            pars_gen = DB_load.iterate_row_retr()
        else:
            load_path = (
                self.DM.dir_struct['DivTokPars'].joinpath(load_dir_name, 'DivDB')
            )
            save_path =(
                self.DM.dir_struct['Norm1Pars'].joinpath(save_dir_name, 'NormDB')
            )
            print('This is load path:', load_path)
            pars_gen = self.DM.iterate_shelve_reading(load_path)
            lemmed_pars_gen = self.CTP.iterate_lemmatize_par_by_dict(
                lem_dict,
                pars_gen,
                stop_w
            )
        counter=0
        t0 = time()
        t1 = time()
        print(inden+'Start normalization and writing')
        if use_db:
            for bath in pars_gen:
                holder = []
                lemmed_pars_gen = self.CTP.iterate_lemmatize_par_by_dict(
                    lem_dict,
                    bath,
                    stop_w,
                    db=True
                )
                for lem_par in lemmed_pars_gen:
                    name = ('0'*(6+1-len(str(counter)))+str(counter))
                    holder.append((name, lem_par))
                    counter += 1
                DB_save.insert_data(holder)
                if counter % 50000 == 0:
                    print(
                        '\tAt this moment '
                        +'{} pars were normalized. {:8.4f}'.format(
                            counter, (time()-t1)
                        )
                    )
                    t1=time()
            DB_load.close()
            DB_save.close()
        else:
            db = shelve.open(str(save_path), flag='c', writeback=True)
            for lem_par in lemmed_pars_gen:
                name = ('0'*(6+1-len(str(counter)))+str(counter))
                db[name] = lem_par
                counter += 1
                if counter % 50000 == 0:
                    print(
                        '\tAt this moment '
                        +'{} pars were normalized. {:8.4f}'.format(
                            counter, (time()-t1)
                        )
                    )
                    t1=time()
                    db.sync()
            print(counter)
            db.close()
        print(
            inden+'Normalization and writing '
            +'complete in {} seconds'.format(time()-t0)
        )
        
    def auto(self, dir_name='', sep_type='sep1'):
        t0=time()
        print('Starting division and tokenization!')
        self.div_tok_acts(
            dir_name=dir_name,
            sep_type=sep_type,
            inden='\t'
        )
        print('Acts are divided and tokenized')
        print('Creating raw words dictionary')
        vocab_rw = self.create_vocab(
            dir_name=dir_name,
            spec='raw',
            inden='\t'
        )
        print('Dictionary was created')
        print('Creating mapping')
        lem_dict = self.create_lem_dict(vocab_rw, inden='\t')
        print('Mapping was created')
        print('Starting lemmatization')
        self.lemmatize_and_save_acts(
            lem_dict,
            load_dir_name=dir_name,
            save_dir_name=dir_name,
            inden='\t')
        print('Creating norm words dictionary')
        vocab_nw = self.create_vocab(
            dir_name=dir_name,
            spec='norm1',
            inden='\t'
        )
        print('Dictionary was created')
        print('Saving all dictionaries')
        ###
        self.save_vocab(vocab_rw, spec='raw', dir_name=dir_name)
        self.save_vocab(vocab_nw, spec='norm1', dir_name=dir_name)
        self.save_vocab(lem_dict, spec='lem1', dir_name=dir_name)
        print('Dictionaries are saved')
        print('Add words count info to the dictionaries!')
        self.count_all_words(
            dir_name=dir_name,
            spec='raw',
            inden='\t')
        self.count_all_words(
            dir_name=dir_name,
            spec='norm1',
            inden='\t')
        print('Total time costs: {} seconds'.format(time()-t0))
    
    def summon_conclusions(self, dir_name):
        path = self.DM.dir_struct['Concls'].joinpath(dir_name)
        paths = self.DM.collect_exist_file_paths(path, suffix='.txt')
        holder = {}
        for p in paths:
            with open(p) as file:
                holder[p.stem] = file.read()
        return holder
    
    def create_index_tables(self,
                            load_dir_name='',
                            save_dir_name='',
                            inden=''):
        #load paths and lem gen
        load_path = (
            self.DM.dir_struct['Norm1'].joinpath(load_dir_name)
        )
        all_acts_gen = self.DM.iterate_pickle_loading(load_path)
        index_acts_gen = self.CTP.iterate_position_search(
            all_acts_gen
        )
        #saves paths
        acts_quants = len(self.DM.collect_exist_file_paths(load_path))
        writing_paths = deque(
            self.DM.create_writing_paths(
                0,
                acts_quants,
                self.DM.dir_struct['ActsInfo'].joinpath(save_dir_name),
                4,
                suffix='.info'
            )
        )
        #process
        t0 = time()
        print(inden+'Start indexing and writing')
        for index_table in index_acts_gen:
            self.DM.write_pickle(
                index_table,
                writing_paths.popleft()
            )
        print(
            inden+'Indexing and writing '
            +'complete in {} seconds'.format(time()-t0)
        )
    
    def create_index_tables_pars(self,
                                 load_dir_name='',
                                 save_dir_name='',
                                 inden=''):
        #load paths and lem gen
        load_path = (
            self.DM.dir_struct['Norm1Pars'].joinpath(load_dir_name, 'NormDB')
        )
        save_path =(
            self.DM.dir_struct['ParsInfo'].joinpath(save_dir_name, 'IndexDB')
        )
        print('This is load path:', load_path)
        pars_gen = self.DM.iterate_shelve_reading(load_path)
        counter=0
        t0 = time()
        t1 = time()
        print(inden+'Start indexing and writing')
        db = shelve.open(str(save_path), flag='c', writeback=True)
        for par in pars_gen:
            index = self.CTP.position_search_pars(par)
            name = ('0'*(6+1-len(str(counter)))+str(counter))
            db[name] = index
            if counter % 50000 == 0:
                print(
                    '\tAt this moment '
                    +'{} pars were indexed. {:8.4f}'.format(
                        counter, (time()-t1)
                    )
                )
                t1=time()
                db.sync()
            counter+=1
        print(counter)
        db.close()
        print(
            inden+'Indexing and writing '
            +'complete in {} seconds'.format(time()-t0)
        )

    def create_index_tables_pars_db(self,
                                    load_dir_name='',
                                    save_dir_name='',
                                    inden=''):
        DB_load = mysqlite.DataBase(
                dir_name='TextProcessing/Norm1Pars/'+load_dir_name,
                base_name='Norm1DB',
                tb_name=True
        )
        DB_save = mysqlite.DataBase(
                dir_name='TextProcessing/ParsInfo/'+save_dir_name,
                base_name='IndexDB'
        )
        DB_save.create_tabel(
                'ParsInfo',
                (('id', 'TEXT', 'PRIMARY KEY'), ('par', 'TEXT'))
        )
        pars_gen = DB_load.iterate_row_retr()
        counter=0
        t0 = time()
        t1 = time()
        print(inden+'Start indexing and writing')
        for batch in pars_gen:
            holder = []
            for par in batch:
                par = par[1].split('#')
                act_info, par = par[:2], par[2:]
                name = ('0'*(6+1-len(str(counter)))+str(counter))
                index = self.CTP.position_search_pars_db(par)
                for item in index.items():
                    key, info = item
                    holder.append((name+'#'+key, str(info)))
                try:
                    holder.append((name+'#'+'req', act_info[1]))
                    holder.append((name+'#'+'court', act_info[0]))
                except:
                    print(act_info)
                    print(name)
                if counter % 50000 == 0:
                    print(
                        '\tAt this moment '
                        +'{} pars were indexed. {:8.4f}'.format(
                            counter, (time()-t1)
                        )
                    )
                    t1=time()
                counter+=1
            DB_save.insert_data(holder)
        DB_load.close()
        DB_save.close()
        print(
            inden+'Indexing and writing '
            +'complete in {} seconds'.format(time()-t0)
        )


class Scorer():
    def __init__(self, math_log = math.log10):
        self.DM = DirManager()
        self.CTP = CustomTextProcessor()
        self.DTL = DivTokLem()
        self.D = None
        self.math_log = math_log
        print('Scorer class created')
        
    def count_acts(self, acts_dir):
        path_to_acts = (
            self.DM.dir_struct['Norm1'].joinpath(acts_dir)
        )
        self.D = len(self.DM.collect_exist_file_paths(path_to_acts))
        print('\nActs in total: {}\n'.format(self.D))

    def change_lob_base(self, log_func):
        self.math_log = log_func
    
    def process_concl_pars(self,
                           auto_mode=False,
                           concl_dir_name='',
                           dpars_dir_name='',
                           npars_dir_name='',
                           info_dir_name='',
                           save_dir_name='',
                           stop_w=None,
                           doc_len=None,
                           add_file_name=''):
        self.D=doc_len
        path_to_concl = (
            self.DM.dir_struct['Concls'].joinpath(concl_dir_name)
        )
        dpath_to_pars = (
            self.DM.dir_struct['DivTokPars'].joinpath(npars_dir_name, 'DivDB')
        )
        npath_to_pars = (
                self.DM.dir_struct['Norm1Pars'].joinpath(npars_dir_name, 'NormDB')
        )
        path_to_info = (
                self.DM.dir_struct['ParsInfo'].joinpath(info_dir_name, 'IndexDB')
        )
        concls = self.DM.iterate_text_loading(path_to_concl)
        vocab_nw = self.DTL.load_vocab(spec='norm1', dir_name='2018-05-22')
        dpar_db = shelve.open(str(dpath_to_pars), flag='r')
        print('\nConnection with DivDB established!')
        npar_db = shelve.open(str(npath_to_pars), flag='r')
        print('Connection with NormDB established!')
        info_db = shelve.open(str(path_to_info), flag='r')
        print('Connection with InfoDB established!\n')
        dct_cya={}
        for concl in concls:
            print(concl[:40])
            concl_prep = self.CTP.full_process(concl, stop_w=stop_w)
            print(' '.join(concl_prep)+'\n')
            holder =[] #holder={}
            ###processing
            t0=time()
            t1=time()
            print('Starting corpus scoring!')
            for counter in range(self.D):
                key = ('0'*(6+1-len(str(counter)))+str(counter))
                npar = npar_db[key]
                npar_court, npar_req, npar = *npar[:2], npar[2:]
                dpar = ' '.join(dpar_db[key][2:])
                info = info_db[key]
                sc = self.score(concl_prep, npar, info, vocab_nw)
                holder.append((npar_court, npar_req, sc, dpar))
                #if npar_court in holder:
                #    holder[npar_court].append(npar_req, sc, dpar)
                #else:
                #    holder[npar_court] = []
                #    holder[npar_court].append(npar_req, sc, dpar)
                if counter % 50000 == 0:
                    print(
                        '\tAt this moment '
                        +'{} pars were scored. {:8.4f}'.format(
                            counter, (time()-t1)
                        )
                    )
                    t1=time()
            print('Corpus was scored in {} seconds.'.format(time()-t0))
            print('Starting pars sorting!')
            t2=time()
            hl = {}
            for i in holder:
                c, r, s, p = i
                key = (c, r)
                if key in hl:
                    if hl[key][0] < s:
                        hl[key] = (s, p)
                else:
                    hl[key] = (s, p)
            print('Totla acts1:', len(hl))
            del holder
            hl2 = []
            for i in hl.items():
                hl2.append((i[0][0], i[0][1], i[1][0], i[1][1]))
            del hl
            print('Totla acts2:', len(hl2))
            hl2 = sorted(hl2, key=lambda x: x[2], reverse=True)
            print('Sorting complete in {} seconds'.format(time()-t2))
            print('Starting writing!')                
            #self.DTL.table_to_csv(
            #    hl2,
            #    dir_name=save_dir_name,
            #    header=('Суд','Реквизиты','Оценка', 'Абзац'),
            #    zero_string = concl,
            #    file_name=FILE_NAMES[concl[:40]]+add_file_name
            #)
            dct_cya[FILE_NAMES[concl[:180]]] = hl2
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
                    return dct_cya
                elif breaker == '1':
                    print('Continue execution')
        dpar_db.close()
        npar_db.close()
        info_db.close()
        print('\nDB connections were terminated!\n')
        print('Execution ended')
        return dct_cya
    
    def process_concl_pars_db(self,
                              auto_mode=False,
                              concl_dir_name='',
                              dpars_dir_name='',
                              info_dir_name='',
                              save_dir_name='',
                              stop_w=None,
                              doc_len=None,
                              add_file_name=''):
        self.D=doc_len
        ###Paths
        path_to_concl = (
            self.DM.dir_struct['Concls'].joinpath(concl_dir_name)
        )
        path_to_info = (
                self.DM.dir_struct['ParsInfo'].joinpath(info_dir_name)
        )
        path_to_dpars = (
            self.DM.dir_struct['DivTokPars'].joinpath(dpars_dir_name)
        )
        #DB connection and concls loading
        DB_load_info = mysqlite.DataBase(
            raw_path = path_to_info,
            base_name = 'IndexDB',
            tb_name = True
        )
        DB_load_dpars = mysqlite.DataBase(
            raw_path = path_to_dpars,
            base_name = 'DivDB',
            tb_name = True
        )
        concls = self.DM.iterate_text_loading(path_to_concl)
        vocab_nw = self.DTL.load_vocab(spec='norm1', dir_name='2018-05-22')
        dct_cya={}
        for concl in concls:
            print(concl[:40])
            concl_prep = self.CTP.full_process(concl, stop_w=stop_w)
            print(' '.join(concl_prep)+'\n')
            holder =[] #holder={}
            ###processing
            t0=time()
            t1=time()
            gen_info = DB_load_info.iterate_row_retr(output=20000)
            print('Starting corpus scoring!')
            for counter in range(self.D):
                key = ('0'*(6+1-len(str(counter)))+str(counter))
                pass


    
    def process_one_phrase_acts(self,
                                phrase,
                                acts_dir,
                                info_dir,
                                save_dir_name=None,
                                stop_w=None,
                                add_file_name=''):
        ###loading
        self.count_acts(acts_dir)
        path_to_acts = (
            self.DM.dir_struct['Norm1'].joinpath(acts_dir)
        )
        path_to_ifno = (
            self.DM.dir_struct['ActsInfo'].joinpath(info_dir)
        )
        path_to_uncl_acts = (
            self.DM.dir_struct['DivTok'].joinpath(acts_dir)
        )
        acts_gen = self.DM.iterate_pickle_loading(path_to_acts)
        info_gen = self.DM.iterate_pickle_loading(
            path_to_ifno,
            suffix='.info'
        )
        uncl_acts_gen = self.DM.iterate_pickle_loading(path_to_uncl_acts)
        vocab_nw = self.DTL.load_vocab(spec='norm1', dir_name='2018-05-22')
        phrase = self.CTP.full_process(phrase, stop_w)
        print(' '.join(phrase)+'\n')
        holder=[]
        ###processing
        t0=time()
        print('Starting corpus scoring!')
        counter = 0
        for act, info, uncl_act in zip(acts_gen, info_gen, uncl_acts_gen):
            sc = self.score(phrase, act, info, vocab_nw)
            holder.append((' '.join(uncl_act[0]), ' '.join(uncl_act[2]), sc))
            counter += 1
            if counter % 500 == 0:
                print(counter)
        print('Corpus was scored in {} seconds.'.format(time()-t0))
        name = FILE_NAMES[phrase[:180]]
        self.DTL.table_to_csv(
                     sorted(holder, key=lambda x: x[2]),
                     file_name=name+add_file_name,
                     dir_name=save_dir_name,
                     zero_string=None,
                     header=['Суд', 'Реквизиты', 'Оценка']
        )
    
    def extract_pairs_term_freq(self, word1, word2, info):
        counter = 0
        if word1 in info and word2 in info:
            order1 = info[word1]
            order2 = info[word2]
            if info['total_'+word1]<info['total_'+word2]:
                min_order = order1
                max_order = order2
            else:
                min_order = order2
                max_order = order1
            for place in min_order:
                if (place+1) in max_order:
                    counter+=1
                elif (place+2) in max_order:
                    counter+=0.5
                elif (place-1) in max_order:
                    counter+=0.5
            return counter
        else:
            return 0
    
    def extract_p_ontopic(self, word, vocab):
        CF = vocab.get(word)
        if CF:
            return 1-math.exp(-1.5*CF/self.D)
        else:
            return 0

    def w_single(self, word, act, info, vocab):
        DL = info['total']
        TF = info.get('total_'+word)
        if not TF:
            return 0
        else:
            p_ontopic = self.extract_p_ontopic(word, vocab)
            return self.math_log(p_ontopic)*(TF/(TF+1+(1/350)+DL))
    
    def w_pair(self, word1, word2, info, vocab):
        pTF = self.extract_pairs_term_freq(word1, word2, info)
        if not pTF:
            return 0
        p_ontopic1 = self.extract_p_ontopic(word1, vocab)
        p_ontopic2 = self.extract_p_ontopic(word2, vocab)
        return (
            0.3*(self.math_log(p_ontopic1)+self.math_log(p_ontopic2))*pTF/(1+pTF)
        )
    
    def w_allwords(self, phrase_words, act, vocab):
        act_words = set([w for par in act for w in par])
        mis_count = 0
        for w in phrase_words:
            if w not in act_words:
                mis_count += 1
        p_ontopics = [
            self.extract_p_ontopic(w, vocab)
            for w in phrase_words
            if w in act_words
        ]
        log_ps = [self.math_log(p) for p in p_ontopics]
        return 0.2*sum(log_ps)*0.03**mis_count
    
    def score(self, phrase_words, act, info, vocab):
        w_singles = [self.w_single(w, act, info, vocab) for w in phrase_words]
        #print(w_singles)
        pairs = self.CTP.create_2grams(phrase_words)
        #print(pairs)
        w_pairs = [
            self.w_pair(*pair, info, vocab)
            for pair in pairs
        ]
        #print(w_pairs)
        w_allwords = self.w_allwords(phrase_words, act, vocab)
        #print(w_allwords)
        return abs(sum(w_singles)+sum(w_pairs)+w_allwords)

def open_norm(index):
    index = str(index)
    key = '0'*(6+1-len(index))+index
    path = r'C:\Users\EA-ShevchenkoIS\TextProcessing\Norm1Pars\2018-05-23\NormDB'
    with shelve.open(path, flag='r') as db:
        item = db[key]
    return item

def open_index(index):
    index = str(index)
    key = '0'*(6+1-len(index))+index
    path = r'C:\Users\EA-ShevchenkoIS\TextProcessing\ParsInfo\2018-05-24\IndexDB'
    with shelve.open(path, flag='r') as db:
        item = db[key]
    return item




#######################################
#######################################
#######################################
#######################################

class Debugger(DirManager, CustomTextProcessor):
    def __init__(self, enc='cp1251', par_type='parser1'):
        DirManager.__init__(self, enc=enc)
        CustomTextProcessor.__init__(self, par_type=par_type)
    
    def get_acts_names(self, full_path, suffix=''):
        acts_gen = self.iterate_pickle_loading(full_path, suffix=suffix)
        holder = []
        counter = 0
        for act in acts_gen:
            court = ' '.join(act[0])
            req = ' '.join(act[2])
            holder.append(court+' '+req)
            counter+=1
            if counter % 5000 == 0:
                print(counter)
        print(len(holder))
        return holder
    
    def find_errors(self, names_holder):
        verifier = set()
        errors = []
        counter = 0
        for i in names_holder:
            if i in verifier:
                errors.append(counter)
            else:
                verifier.add(i)
            counter +=1
        print(len(errors))
        return errors
    
    def conv_er_index_to_names(self, errors, names_holder):
        er_names = set()
        for i in errors:
            er_names.add(names_holder[i])
        return er_names
    
    def find_all_er_pos(self, er_names, names_holder):
        in_hold = []
        for i in er_names:
            counter = 0
            in2_hold = []
            for j in names_holder:
                if i == j:
                    in2_hold.append(counter)
                counter+=1
            in_hold.append((i, in2_hold))
        return in_hold
    
    def debug_acts_reps(self, full_path, suffix=''):
        names_holder = self.get_acts_names(full_path, suffix='')
        err_ind = self.find_errors(names_holder)
        er_names = self.conv_er_index_to_names(err_ind, names_holder)
        total = self.find_all_er_pos(er_names, names_holder)
        return total
    
    def extr_dt_act(self, index, dir_name='2018-05-22'):
        ind = str(index)
        name = '0'*(4+1-len(str(ind)))+str(ind)
        print(name)
        fle = self.load_pickle(
            r'C:\Users\EA-ShevchenkoIS'
            +r'\TextProcessing\DivToks\{}\{}'.format(dir_name, name)
        )
        return fle
    
    def find_missed_names(self, names_holder, lngth):
        names = [
        '0'*(4+1-len(str(i)))+str(i)
        for i in range(lngth)
        ]
        print(len(names))
        holder2 = []
        for i in names:
            if i not in names_holder:
                print(i)
                holder2.append(i)
        return holder2
    
    def timer(self, rep, func, *args, **kwargs):
        t0=time()
        for i in range(rep):
            x = func(*args, **kwargs)
        total = time()-t0
        aver = total/rep
        print(
            'Totla: {} reps in '.format(rep)
            +'{:8.5f} seconds. '.format(total)
            +'Aver: {:2.5}'.format(aver)
        )

        
class AverScorer(Debugger):
    def __init__(self, enc='cp1251', par_type='parser1', stop_w=None):
        Debugger.__init__(self, enc=enc, par_type=par_type)
        self.stpw = stop_w
    
    def table_to_csv(self,
                     table,
                     file_name='py_table',
                     dir_name='',
                     zero_string=None,
                     header=['Col1', 'Col2']):
        path = self.dir_struct['Results']
        path = path.joinpath(dir_name, file_name).with_suffix('.csv')
        assert len(table[0]) == len(header)
        self.write_text_to_csv(
            path,
            table,
            zero_string=zero_string,
            header=header
        )
    
    def tahn(self, x):
        return (1-math.exp(-2*x))/(1+math.exp(-2*x))
    
    def find_reps_index(self, holder:list):
        names = [i[0]+'#'+i[1] for i in holder]
        verifier = set()
        holder = []
        counter = 0
        for name in names:
            if name in verifier:
                holder.append(counter)
            else:
                verifier.add(name)
            counter+=1
        print(len(holder))
        return names, holder
    
    def aver_reps_score(self, holder:list, add_file_name=''):
        names, rep_ind = self.find_reps_index(holder)
        rep_names = self.conv_er_index_to_names(rep_ind, names)
        total_reps = self.find_all_er_pos(rep_names, names)
        writer(total_reps, 'total_reps'+add_file_name)
        for i in total_reps:
            score_box=[]
            for j in i[1]:
                score_box.append(holder.pop(j)[2])
            print(score_box, end=' === ')
            print(sum(score_box), end=' === ')
            print(sum(score_box)/len(score_box))
            aver_score = sum(score_box)/len(score_box)
            holder.append([*i[0].split('#'), aver_score, '==Замещающий текст=='])
        return sorted(holder, key=lambda x: x[2])
    
    def map_to_one(self, holder):
        maped_holder = []
        while len(holder) != 0:
            item = holder.pop()
            item = [*item]
            item[2] = 1 - self.tahn(item[2])
            maped_holder.append(item)
        return sorted(maped_holder, key=lambda x: x[2])
    
    def addition(self,
                 holder1,
                 holder2,
                 save_dir_name,
                 zero_string=None,
                 add_file_name='Проба'):
        #assert len(holder1) == len(holder2)
        dct_hl1 = {}
        dct_hl2 = {}
        total = []
        for i in holder1:
            court, req, score, par = i
            #court = ' '.join(self.full_process(court, self.stpw))
            #req = ' '.join(self.full_process(req, self.stpw))
            key = court+'#'+req
            if key in dct_hl1:
                if  dct_hl1[key][0] < score:
                    dct_hl1[key] = (score, par)
            else:
                dct_hl1[key] = (score, par)
        for j in holder2:
            court, req, score, par = j
            court = ' '.join(self.tokenize(court))
            req = ' '.join(self.tokenize(req))
            key = court+'#'+req
            dct_hl2[key] = (score, par)
        dct_hl1.pop('арбитражный суд северо западного округа#от 7 февраля 2017 г по делу n а56 72948 2015')
        dct_hl2.pop('#постановление')
        if not dct_hl1.keys() == dct_hl2.keys():
            hl8 = []
            for kk in dct_hl1:
                if kk not in dct_hl2:
                    hl8.append(kk)
            hl9 = []
            for kk in dct_hl2:
                if kk not in dct_hl1:
                    hl9.append(kk)
            writer(sorted(hl8), 'hl8'+add_file_name)
            writer(sorted(hl9), 'hl9'+add_file_name)
            return 'Bad keys'
        for k in dct_hl1.keys():
            sc1, p1 = dct_hl1[k]
            sc2, p2 = dct_hl2[k]
            total.append((*k.split('#'), sc1+sc2, p1, p2, 'да' if p1==p2 else 'НЕСОВП'))
        total = sorted(total, key=lambda x: x[2])
        self.table_to_csv(
                total,
                dir_name=save_dir_name,
                header=(
                    'Суд',
                    'Реквизиты',
                    'Общая оценка',
                    'Абзац (векторы)',
                    'Абзац (Яндекс)',
                    'абзацы совпадают?'
                ),
                zero_string = zero_string,
                file_name=add_file_name
            )
    
    def cut_top(self, db, alg, top=5):
        holder = []
        #print(db.keys())
        for key in db:
            cut = db[key][:top]
            holder_cut = []
            for row in cut:
                #print('This is row: ', row[:3])
                #print(type(row[0]), type(row[1]), type(row))
                row = ['_'.join((row[0], row[1])), row[2], row[3], alg, key]
                holder_cut.append(row)
            holder.extend(holder_cut)
        return holder
    
    def summon_reps(self,
                    top=5,
                    write=False,
                    file_name=None,
                    save_dir_name=None,
                    **algs):
        holder = []
        #print(algs.keys())
        for key in algs:
            holder.extend(self.cut_top(algs[key], key, top=top))
        if not write:
            return holder
        elif write:
            self.table_to_csv(
               holder,
               file_name=file_name,
               dir_name=save_dir_name,
               zero_string=None,
               header=['Акт', 'Оценка', 'Абзац', 'Алгоритм', 'Вывод']
            )
    
    def count_oqur(self, holder:list):
        algs = set([i[3] for i in holder])
        print(algs)
        concls = set([i[4] for i in holder])
        print(concls)
        names = set([i[0] for i in holder])
        print(names)
        dct_nm = {concl:{name:{alg:0 for alg in algs} for name in names} for concl in concls}
        for concl in concls:
            for name in names:
                for alg in algs:
                    for row in holder:
                        if (
                            concl in row and
                            name in row and
                            alg in row
                        ):
                            dct_nm[concl][name][alg]+=1
        return dct_nm
    
    def extr(self, dct:dict, write=False, file_name='', save_dir_name=''):
        alg_keys = None
        holder = []
        for k in dct:
            for l in dct[k]:
                if not alg_keys:
                    alg_keys = sorted(dct[k][l].keys())
                    print(alg_keys)
                vals = [dct[k][l][m] for m in alg_keys]
                row = [k, *l.split('_'), *vals]
                row.append(sum(row[3:]))
                holder.append(row)
        if not write:
            return holder
        elif write:
            self.table_to_csv(
               holder,
               file_name=file_name,
               dir_name=save_dir_name,
               zero_string=None,
               header=['Вывод', 'Суд', 'Реквизиты', *alg_keys, 'Сумма']
            )



st =(
'''
                npar = npar_db[key]
                npar_court, npar_req, npar = *npar[:2], npar[2:]
                dpar = ' '.join(dpar_db[key][2:])
                info = info_db[key]
                sc = self.score(concl_prep, npar, info, vocab_nw)
                holder.append((npar_court, npar_req, sc, dpar))
                #if npar_court in holder:
                #    holder[npar_court].append(npar_req, sc, dpar)
                #else:
                #    holder[npar_court] = []
                #    holder[npar_court].append(npar_req, sc, dpar)
                if counter % 50000 == 0:
                    print(
                        '\tAt this moment '
                        +'{} pars were scored. {:8.4f}'.format(
                            counter, (time()-t1)
                        )
                    )
                    t1=time()
            print('Corpus was scored in {} seconds.'.format(time()-t0))
            print('Starting pars sorting!')
            t2=time()
            hl = {}
            for i in holder:
                c, r, s, p = i
                key = (c, r)
                if key in hl:
                    if hl[key][0] < s:
                        hl[key] = (s, p)
                else:
                    hl[key] = (s, p)
            print('Totla acts1:', len(hl))
            del holder
            hl2 = []
            for i in hl.items():
                hl2.append((i[0][0], i[0][1], i[1][0], i[1][1]))
            del hl
            print('Totla acts2:', len(hl2))
            hl2 = sorted(hl2, key=lambda x: x[2], reverse=True)
            print('Sorting complete in {} seconds'.format(time()-t2))
            print('Starting writing!')                
            #self.DTL.table_to_csv(
            #    hl2,
            #    dir_name=save_dir_name,
            #    header=('Суд','Реквизиты','Оценка', 'Абзац'),
            #    zero_string = concl,
            #    file_name=FILE_NAMES[concl[:40]]+add_file_name
            #)
            dct_cya[FILE_NAMES[concl[:180]]] = hl2
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
                    return dct_cya
                elif breaker == '1':
                    print('Continue execution')
        dpar_db.close()
        npar_db.close()
        info_db.close()
        print('\nDB connections were terminated!\n')
        print('Execution ended')
        return dct_cya
'''
)