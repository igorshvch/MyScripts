import re
import math
import numpy as np
import pymorphy2
import csv
import pickle
import json
import mysqlite
import pathlib as pthl
from time import time
from scipy.spatial.distance import cosine as sp_cosine
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter, deque
from nltk import sent_tokenize
#from nltk.corpus import stopwords
from writer import writer

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
#file_name:
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

SHORT_FILE_NAMES = {
    'Является ли непредставление поставщиком отчетности в налоговый орган признаком получения необоснованной налоговой выгоды по НДС (недобросовестности) (ст. ст. 54.1, 171 НК РФ) Непре': '525',
    'Как определить налоговую базу по НДС при безвозмездной реализации товаров (работ, услуг) (п. 2 ст. 154 НК РФ)? при определении налоговой базы по НДС при безвозмездной реализации то': '541',
    'Увеличивается ли налоговая база по НДС на суммы страхового возмещения, получаемые при наступлении страхового случая (пп. 2 п. 1 ст. 162 НК РФ) суммы страхового возмещения, получаем': '551',
    'Освобождается ли от НДС предоставление жилой площади в общежитиях (пп. 10 п. 2 ст. 149 НК РФ) Предоставление жилой площади в общежитиях облагается НДС что согласно ст. 16 ЖК РФ к ж': '553',
    'Необходимо ли для подтверждения ставки НДС 10 процентов представлять сертификаты соответствия (п. 2 ст. 164 НК РФ) Непредставление сертификатов не влияет на право применения ставки': '581',
    'Правомерно ли начисление штрафа, если экспорт подтвердили по истечении 180 календарных дней, но НДС при этом уплачен не был (п. 9 ст. 165 НК РФ) Если экспорт подтвердили по истечен': '590_1',
    'Правомерно ли применение вычета по НДС, если акт приемки-передачи отсутствует (ст. 54.1, п. 1 ст. 172 НК РФ)': '43',
    'Должен ли налогоплательщик в целях применения освобождения от уплаты НДС включать в расчет выручки суммы, полученные от операций, не облагаемых НДС (освобождаемых от налогообложения) (п. 1 ст. 145, ст. 149 НК РФ)': '44',
    'Являются ли плательщиками НДС физические лица, осуществляющие незаконную предпринимательскую деятельность без регистрации (п. 2 ст. 11, п. 1 ст. 143 НК РФ)': '57',
    'Правомерно ли применение вычета по НДС при отсутствии товарно-транспортной накладной (п. 1 ст. 172 НК РФ)': '6',
    'Имеет ли налогоплательщик право на вычет, если контрагент (субпоставщик) не перечислил НДС в бюджет (перечислил его не полностью) (ст. 54.1, п. 1 ст. 171 НК РФ)': '60',
    'Облагается ли НДС передача (в том числе безвозмездная) арендатором неотделимых улучшений арендодателю (пп. 1 п. 1 ст. 146 НК РФ)': '74',
    'Можно ли обязать контрагента выставить (исправить) счет-фактуру (п. 3 ст. 168 НК РФ)': '77',
    'Можно ли обязать контрагента выставить (исправить) счет-фактуру (п. 3 ст. 168 НК РФ)...': '77_2',
    'Имеет ли налогоплательщик право на вычет по НДС, если контрагент не представляет отчетность в налоговые органы (ст. 54.1, п. 1 ст. 171 НК РФ)': '80'
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
    
    def map_num_to_concl(self, path_holder, lngth=None):
        dct = {}
        for p in path_holder:
            with open(str(p), mode='r') as file:
                text = file.read()
            if lngth:
                dct[p.stem] = text[:lngth]
            else:
                dct[p.stem] = text
        return dct
    
    def map_concl_to_num(self, path_holder, lngth=None):
        dct = {}
        for p in path_holder:
            with open(str(p), mode='r') as file:
                text = file.read()
            if lngth:
                dct[text[:lngth]] = p.stem
            else:
                dct[text] = p.stem
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
            if re.match(PATTERN_PASS, text):
                continue
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
    
    def full_process(self, text, par_type='parser1', stop_w=None):
        tokens = self.tokenize(text)
        lemms = self.lemmatize(tokens, par_type=par_type)
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
                stop_w=stpw
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
        self.C_vectorizer = CountVectorizer(token_pattern=token_pattern)
        self.T_vectorizer = TfidfVectorizer(token_pattern=token_pattern)
        print('Vct class created')
    
    def create_vectors(self, pars_lst, vectorizer):
        compressed_mtrx = vectorizer.transform(pars_lst)
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
        print('Constructor class created')
    
    def create_dir_struct(self):
        self.RWT.create_dirs(self.dir_struct)

    def create_sub_dirs(self, dir_name):
        self.RWT.create_dirs(self.dir_struct, sub_dir=dir_name)

    def div_tok_acts(self,
                     dir_name='',
                     sep_type='sep1',
                     inden=''):
        path = self.dir_struct['Raw_text'].joinpath(dir_name)
        raw_files = self.RWT.iterate_text_loading(path)
        counter1 = 0
        counter2 = 0
        t_0 = time()
        for fle in raw_files:
            print(inden+'Starting new file processing!')
            cleaned = self.CTP.court_decisions_cleaner(fle)
            divided = self.CTP.court_decisions_separator(
                cleaned,
                sep_type=sep_type
            )
            tokenized = self.CTP.iterate_tokenization(divided)
            counter2 += len(divided)
            t0=time()
            print(inden+'\tStarting tokenization and writing')
            file_paths = deque(self.RWT.create_writing_paths(
                counter1, counter2,
                self.dir_struct['DivTok'].joinpath(dir_name),
                suffix=''
            ))
            for tok_act in tokenized:
                self.RWT.write_pickle(
                    tok_act,
                    file_paths.popleft()
                )
            counter1 += len(divided)
            print(
                inden+'\tTokenization and writing '
                +'complete in {} seconds!'.format(time()-t0)
            )
        print('Total time costs: {}'.format(time()-t_0))
    
    def div_tok_acts_db(self,
                        load_dir_name='',
                        save_dir_name='',
                        sep_type='sep1',
                        inden=''):
        t0=time()
        #Initiate concls iterator:
        load_path = self.dir_struct['Raw_text'].joinpath(load_dir_name)
        raw_files = self.RWT.iterate_text_loading(load_path)
        #Initiate DB:
        DB_save = mysqlite.DataBase(
            dir_name='TextProcessing/DivToks/'+save_dir_name,
            base_name='BigDivDB'
        )
        DB_save.create_tabel(
            'BigDivToks',
            (('id', 'TEXT', 'PRIMARY KEY'), ('par1', 'TEXT'))
        )
        counter = 0
        for fle in raw_files:
            t1=time()
            holder=[]
            print(inden+'Starting new file processing!')
            cleaned = self.CTP.court_decisions_cleaner(fle)
            divided = self.CTP.court_decisions_separator(
                cleaned,
                sep_type=sep_type
            )
            tokenized = self.CTP.iterate_tokenization(divided)
            print(inden+'\tStarting tokenization and writing')
            for tok_act in tokenized:
                name = ('0'*(4+1-len(str(counter)))+str(counter))
                enc = json.dumps(tok_act)
                holder.append((name, enc))
                counter+=1
            DB_save.insert_data(holder, col_num=2)
            print(
                inden+'\tTokenization and writing '
                +'complete in {:4.5f} seconds!'.format(time()-t1)
            )
        print('Total time costs: {}'.format(time()-t0))
    
    def create_vocab(self, dir_name='', spec='raw', inden=''):
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
        print(inden+'Starting vocab creation!')
        all_files = self.RWT.iterate_pickle_loading(options[spec])
        vocab = self.CTP.words_count(all_files)
        print(inden+'Vocab created in {} seconds!'.format(time()-t0))
        return vocab
    
    def create_lem_dict(self,
                        vocab,
                        par_type='parser1',
                        inden=''):
        all_words = list(vocab.keys())
        print(inden+'Strating normalization!')
        t0 = time()
        norm_words = self.CTP.lemmatize(all_words, par_type=par_type)
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
        path = self.dir_struct['StatData']
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
        path = self.dir_struct['StatData']
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
                                inden=''):
        #load paths and lem gen
        load_path = (
            self.dir_struct['DivTok'].joinpath(load_dir_name)
        )
        all_acts_gen = self.RWT.iterate_pickle_loading(load_path)
        lemmed_acts_gen = self.CTP.iterate_lemmatize_by_dict(
            lem_dict,
            all_acts_gen,
            set(lem_dict)
        )
        #saves paths
        acts_quants = len(self.RWT.collect_exist_file_paths(load_path))
        save_dir = self.dir_struct['Norm1']
        save_dir = save_dir.joinpath(save_dir_name)
        writing_paths = deque(sorted(self.RWT.create_writing_paths(
            0,
            acts_quants,
            save_dir
        )))
        #process
        t0 = time()
        print(inden+'Start normalization and writing')
        for lem_act in lemmed_acts_gen:
            self.RWT.write_pickle(
                lem_act,
                writing_paths.popleft()
            )
        print(
            inden+'Normalization and writing '
            +'complete in {} seconds'.format(time()-t0)
        )
    
    def lemmatize_and_save_acts_db(self,
                                   load_dir_name='',
                                   save_dir_name='',
                                   inden=''):
        t0 = time()
        #Load vocab
        lem_dict = self.load_vocab(spec='lem1', dir_name='2018-05-22')
        lem_keys = set(lem_dict)
        #Initiat DBs
        DB_load = mysqlite.DataBase(
            dir_name='TextProcessing/DivToks/'+load_dir_name,
            base_name='BigDivDB',
            tb_name=True
        )
        DB_save = mysqlite.DataBase(
            dir_name='TextProcessing/Norm1/'+save_dir_name,
            base_name='BigNorm1DB'
        )
        DB_save.create_tabel(
            'BigNorm1DB',
            (
                (('id', 'TEXT', 'PRIMARY KEY'), ('par1', 'TEXT'), ('par2', 'TEXT'))
            )
        )
        acts_gen = DB_load.iterate_row_retr(length=8588, output=1000)
        for batch in acts_gen:
            t1 = time()
            print('Starting new batch!')
            holder = []
            for row in batch:
                divtoks = row[1]
                #print('Starting new row!')
                #t_1 = time()
                dec = json.loads(divtoks)
                #print('\tTime: {:4.4f}'.format(time()-t_1))
                #print('Starting new lem!')
                #t_2 = time()
                act = [
                    [lem_dict[token] for token in par if token in lem_keys]
                    for par in dec
                ]
                #print('\tTime: {:4.4f}'.format(time()-t_2))
                #print('Starting new enc!')
                #t_3 = time()
                enc = json.dumps(act)
                #print('\tTime: {:4.4f}'.format(time()-t_3))
                holder.append((row[0], enc, divtoks))
            DB_save.insert_data(holder, col_num=3)
            print('Batch was proceed in {:4.5f} seconds'.format(time()-t1))
        print('Total time costs: {:4.5f}'.format(time()-t0))
    
    def act_and_concl_to_mtrx(self,
                              vector_pop='concl',
                              vector_model=None,
                              addition=True,
                              fill_val=1):
        '''
        Accepted 'vector_pop' args:
        'act', 'concl', 'mixed'
        Accepted 'vect_model' args:
        'count', 'tfidf'
        '''
        if vector_model == 'count':
            vectorizer = self.Vct.C_vectorizer
        elif vector_model == 'tfidf':
            vectorizer = self.Vct.T_vectorizer
        def inner_func1(pars_list, concl):
            data = [concl] + pars_list
            vectorizer.fit(data)
            data_mtrx = self.Vct.create_vectors(data, vectorizer)
            if addition:
                update_mtrx = (
                    np.append(
                        data_mtrx,
                        np.full(
                            (len(data_mtrx),1), fill_val
                            ),
                        1
                    )
                )
                return update_mtrx
            else:
                return data_mtrx
        def inner_func2(pars_list, concl, bigrs=None):
            data = [concl] + pars_list
            vectorizer.fit([concl])
            data_mtrx = self.Vct.create_vectors(data, vectorizer)
            if addition:
                update_mtrx = (
                    np.append(
                        data_mtrx,
                        np.full(
                            (len(data_mtrx),1), fill_val
                            ),
                        1
                    )
                )
                return update_mtrx
            else:
                return data_mtrx 
        def inner_func3(pars_list, concl, bigrs):
            data = [concl] + pars_list
            vectorizer.fit(data+[bigrs])
            data_mtrx = self.Vct.create_vectors(data, vectorizer)
            if addition:
                update_mtrx = (
                    np.append(
                        data_mtrx,
                        np.full(
                            (len(data_mtrx),1), fill_val
                            ),
                        1
                    )
                )
                return update_mtrx
            else:
                return data_mtrx   
        options = {
            'act' : inner_func1,
            'concl': inner_func2,
            'mixed': inner_func3
        }
        return options[vector_pop]

    def eval_cos_dist(self, index_mtrx, output='best'):
        base = index_mtrx[0,:]
        holder = []
        for i in range(1,index_mtrx.shape[0],1):
            cos = sp_cosine(base, index_mtrx[i,:])
            cos = cos if not np.isnan(cos) else 1.0
            holder.append((i, cos))
        if output=='best':
            return sorted(holder, key = lambda x: x[1])[0]
        elif output=='all':
            return sorted(holder, key = lambda x: x[1])
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
    
    def concl_2gram(self, concl, stop_w=None, reps=False, result=None):
        concl_prep = self.CTP.full_process(
            concl,
            par_type='parser1',
            stop_w=stop_w
        )
        bigrms = self.CTP.create_2grams(concl_prep)
        if reps:
            bigrms = self.CTP.extract_repetitive_ngrams(bigrms)
        if result =='join':
            return ' '.join(concl_prep+bigrms)
        elif result =='simple':
            return  ' '.join(bigrms)
    
    def concl_uniq_words(self, concl, stop_w=None):
        concl_prep = self.CTP.full_process(
            concl,
            par_type='parser1',
            stop_w=stop_w
        )
        return ' '.join(set(concl_prep))
    
    def export_cd_eval_results(self,
                               auto_mode=False,
                               concl_dir_name='',
                               load_dir_name='',
                               save_dir_name='',
                               vector_pop='concl',
                               vector_model='count',
                               addition=True,
                               fill_val=1,
                               par_type='parser1',
                               stop_w=None,
                               rep_ngram_act=False,
                               add_file_name='',
                               intersection=False,
                               params=None,
                               old_uniq=False,
                               old_bigram='join',
                               par_len=None,
                               act_borders=None
                               ):
        #load concls and set mtrx creation finc
        path_to_concl = (
            self.dir_struct['Concls'].joinpath(concl_dir_name, 'concls.dct')
        )
        concls = self.RWT.load_pickle(str(path_to_concl))
        mtrx_creator = self.act_and_concl_to_mtrx(
            vector_pop=vector_pop,
            vector_model=vector_model,
            addition=addition,
            fill_val=fill_val
        )
        dct_holder = {}
        for concl in concls:
            ############################
            if intersection and stop_w:
                concl_prep = self.CTP.full_process(
                    concl,
                    par_type=par_type,
                    stop_w=stop_w
                )
                concl_int_bigrs = self.CTP.intersect_2gr(concl, stop_w)
                if old_uniq:
                    concl_prep = set(concl_prep)
                    concl_int_bigrs = set(concl_int_bigrs)
                    concl_cleaned = ' '.join(concl_prep | concl_int_bigrs)
                else:
                    concl_cleaned = ' '.join(concl_prep+concl_int_bigrs)
            ###########
            elif len(params) == 4:
                options = {
                    ('W_stpw', 'reps', 'bigrs_join', 'NOuniq'): (lambda:
                        self.concl_2gram(concl, stop_w=None, reps=True, result='join')
                    ),
                    ('WO_stpw', 'reps', 'bigrs_join', 'NOuniq'): (lambda:
                        self.concl_2gram(concl, stop_w=stop_w, reps=True, result='join')
                    ),
                    ('W_stpw', 'NOreps', 'bigrs_join', 'NOuniq'): (lambda:
                        self.concl_2gram(concl, stop_w=None, reps=False, result='join')    
                    ),
                    ('WO_stpw', 'NOreps', 'bigrs_join', 'NOuniq'): (lambda:
                        self.concl_2gram(concl, stop_w=stop_w, reps=False, result='join')    
                    ),
                    ('W_stpw', 'reps', 'bigrs_simple', 'NOuniq'): (lambda:
                        self.concl_2gram(concl, stop_w=None, reps=True, result='simple')
                    ),
                    ('WO_stpw', 'reps', 'bigrs_simple', 'NOuniq'): (lambda:
                        self.concl_2gram(concl, stop_w=stop_w, reps=True, result='simple')
                    ),
                    ('W_stpw', 'NOreps', 'bigrs_simple', 'NOuniq'): (lambda:
                        self.concl_2gram(concl, stop_w=None, reps=False, result='simple')    
                    ),
                    ('WO_stpw', 'NOreps', 'bigrs_simple', 'NOuniq'): (lambda:
                        self.concl_2gram(concl, stop_w=stop_w, reps=False, result='simple')    
                    ),
                    #########
                    ('W_stpw', 'NOreps', 'NObigrs', 'NOuniq'): (lambda:
                        ' '.join(
                            self.CTP.full_process(
                                concl,
                                par_type=par_type,
                                stop_w=None
                            )
                        )
                    ),
                    ('WO_stpw', 'NOreps', 'NObigrs', 'NOuniq'): (lambda:
                        ' '.join(
                            self.CTP.full_process(
                                concl,
                                par_type=par_type,
                                stop_w=stop_w
                            )
                        )
                    ),
                    #########
                    ('W_stpw', 'NOreps', 'NObigrs', 'uniq'): (lambda:
                        self.concl_uniq_words(concl, stop_w=None)
                    ),
                    ('WO_stpw', 'NOreps', 'NObigrs', 'uniq'): (lambda:
                        self.concl_uniq_words(concl, stop_w=stop_w)
                    )
                }
                concl_cleaned = options[params]()
            else:
                raise TypeError('Wrong params args!')
            #Uncleaned_acts
            #uncl_acts = deque(self.RWT.collect_exist_file_paths(
            #    self.dir_struct['Divided_and_tokenized'].joinpath(load_dir_name)
            #))
            #load acts
            #path_to_acts = path_to_acts.joinpath(load_dir_name)
            #load acts
            #path_to_acts = self.dir_struct['Normalized_by_{}'.format(par_type)]
            DB_load = mysqlite.DataBase(
                dir_name='TextProcessing/Norm1/'+load_dir_name,
                base_name='BigNorm1DB',
                tb_name=True
            )
            #print('this is it!', path_to_acts)
            #acts = self.RWT.iterate_pickle_loading(path_to_acts)
            acts_gen = DB_load.iterate_row_retr(length=8588, output=2000)
            print('\n', concl[:50], '\n', sep='')
            print(concl_cleaned, '\n', sep='')
            t0 = time()
            holder = []
            #counter = 0
            for batch in acts_gen:
                print('Start Batch!')
                for row in batch:
                    uncl_act = [
                        ' '.join(par_lst)
                        for par_lst in json.loads(row[2])
                    ]
                    act = json.loads(row[1])
                    #print('Acts_extracted')
                    if intersection and stop_w:
                        if par_len:
                            act2 = []
                            for par in act:
                                if len(''.join(par)) > par_len:
                                    act2.append(par)
                                else:
                                    act2.append('')
                            act = act2
                        if old_uniq:
                            act = [
                                set(
                                    [w for w in par if w not in stop_w]
                                )
                                |
                                set(self.CTP.intersect_2gr(
                                    par,
                                    stop_w,
                                    verbose=False,
                                    par_type='lst'
                                ))
                                for par in act
                            ]
                            act = [' '.join(par) for par in act]
                        else:
                            act = [
                                    [w for w in par if w not in stop_w]
                                    +
                                    self.CTP.intersect_2gr(
                                        par,
                                        stop_w,
                                        verbose=False,
                                        par_type='lst'
                                    )
                                    for par in act
                                ]
                            act = [' '.join(par) for par in act]
                    ######################        
                    elif old_bigram == 'join':
                        if par_len:
                            act2 = []
                            for par in act:
                                if len(''.join(par)) > par_len:
                                    act2.append(par)
                                else:
                                    act2.append('')
                            act = act2
                        if stop_w:
                            act = [
                                [w for w in par if w not in stop_w]
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
                    elif old_bigram == 'simple':
                        if par_len:
                            act2 = []
                            for par in act:
                                if len(''.join(par)) > par_len:
                                    act2.append(par)
                                else:
                                    act2.append('')
                            act = act2
                        if stop_w:
                            act = [
                                [w for w in par if w not in stop_w]
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
                    elif old_bigram == 'no':
                        if par_len:
                            act2 = []
                            for par in act:
                                if len(''.join(par)) > par_len:
                                    act2.append(par)
                                else:
                                    act2.append('')
                            act = act2
                        if old_uniq:
                            act = [
                                    ' '.join(set(
                                        [w for w in par if w not in stop_w]
                                    ))
                                    for par in act
                                ]
                        else:
                            act = [
                                    ' '.join(
                                        [w for w in par if w not in stop_w]
                                    )
                                    for par in act
                                ]
                    if vector_pop=='mixed':
                        bigrs = ' '.join(concl_int_bigrs)
                        data_mtrx = mtrx_creator(act, concl_cleaned, bigrs)
                    else:
                        data_mtrx = mtrx_creator(act, concl_cleaned)
                    par_index, cos = self.eval_cos_dist(data_mtrx)
                    #if act_borders:
                    #    holder.append(
                    #        [act_court,
                    #        act_name,
                    #        cos,
                    #        uncl_act[par_index-1]] #act[par_index-1]]
                    #    )
                    #else:
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
            if old_uniq:
                dct_holder[concls[concl]] = holder
            else:
                dct_holder[concls[concl]] = holder
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
                    return dct_holder
                    #break
                elif breaker == '1':
                    print('Continue execution')
        print('Execution ended')
        return dct_holder




            #for act in acts:
            #    uncl_act = self.RWT.load_pickle(uncl_acts.popleft())
            #    if act_borders:
            #        act_court = ' '.join(uncl_act[0])
            #        act_name = ' '.join(uncl_act[2])
            #        uncl_act = uncl_act[act_borders[0]:act_borders[1]]
            #        act = act[act_borders[0]:act_borders[1]]
                    #writer([len(uncl_act)], 'unclean_acts_len', mode='a', verbose=False)
            #    uncl_act = [' '.join(par_lst) for par_lst in uncl_act]
                ############################
                #writer(act, 'act{}'.format(counter), verbose=False)
                #writer(data_mtrx, 'act_mtrx{}'.format(counter), verbose=False)
                #counter+=1
                #if counter%500 == 0:
                    #print(counter)
            #name = concl[:40]
            #self.table_to_csv(
            #    holder,
            #    dir_name=save_dir_name,
            #    header=('Суд','Реквизиты','Косинус', 'Абзац'),
            #    zero_string = concl,
            #    file_name=FILE_NAMES[name]+add_file_name
            #)
    
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
            inden='\t'
        )
        print('Acts are divided and tokenized')
        print('Creating raw words dictionary')
        vocab_rw = self.create_vocab(
            dir_name=dir_name,
            spec='raw',
            inden='\t'
        )
        print('Dictionary is created')
        print('Creating mapping')
        lem_dict = self.create_lem_dict(vocab_rw, inden='\t')
        print('Mapping is created')
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
    
    def special_sentence_count(self, load_dir_name):
        path = self.dir_struct['Raw_text'].joinpath(load_dir_name)
        raw_files = self.RWT.iterate_text_loading(path)
        holder=[]
        for fle in raw_files:
            print('Starting new file processing!')
            cleaned = self.CTP.court_decisions_cleaner(fle)
            divided = self.CTP.court_decisions_separator(
                cleaned,
                sep_type='sep1'
            )
            t0=time()
            for act in divided:
                act = act.split('\n')
                act = [p for p in act if p]
                for par in act:
                    sentences = sent_tokenize(par)
                    holder.extend(sentences)
            print(len(holder))
            print('Acts were divided in {} seconds'.format(time()-t0))
        print('Start counting!')
        holder2=[]
        t0=time()
        while holder:
            sent = holder.pop()
            lngth = len(sent)
            holder2.append((sent, lngth))
        print(len(holder2))
        print('Lengths were counted in {} seconds'.format(time()-t0))
        return holder2
    
    def special_raw_pars_count(self, load_dir_name):
        path = self.dir_struct['Raw_text'].joinpath(load_dir_name)
        raw_files = self.RWT.iterate_text_loading(path)
        holder=[]
        for fle in raw_files:
            print('Starting new file processing!')
            cleaned = self.CTP.court_decisions_cleaner(fle)
            divided = self.CTP.court_decisions_separator(
                cleaned,
                sep_type='sep1'
            )
            t0=time()
            for act in divided:
                act = act.split('\n')
                act = [p for p in act if p]
                holder.extend(act)
            print(len(holder))
            print('Acts were divided in {} seconds'.format(time()-t0))
        print('Start counting!')
        holder2=[]
        t0=time()
        while holder:
            par = holder.pop()
            lngth = len(par)
            holder2.append((par, lngth))
        print(len(holder2))
        print('Lengths were counted in {} seconds'.format(time()-t0))
        return holder2
    
    def special_cleaned_pars_count(self, load_dir_name):
        path = self.dir_struct['Normalized_by_parser1'].joinpath(load_dir_name)
        files = self.RWT.iterate_pickle_loading(path)
        t0=time()
        holder=[]
        for fle in files:
            holder.extend(fle)
        print(len(holder))
        print('Pars extracted in {} seconds'.format(time()-t0))
        t0=time()
        holder2 = []
        while holder:
            par = holder.pop()
            words_count = len(par)
            total_len = len(''.join(par))
            holder2.append((par,total_len, words_count))
        print(len(holder2))
        print('Estimation ended in {} seconds'.format(time()-t0))
        writer(sorted(holder2, key=lambda x: x[0]), 'pras_len_count_ALPH1')
        writer(sorted(holder2, key=lambda x: x[1], reverse=True), 'pras_LEN_count_alph2')
        writer(sorted(holder2, key=lambda x: x[2], reverse=True), 'pras_len_COUNT_alph3')
        return holder2
    
    def special_get_ind(self, x, word, mode='e', order='forward'):
        if order == 'forward':
            for i in range(len(x)):
                if mode == 'i':
                    if word in x[i]:
                        return i
                elif mode == 'e':
                    if word == x[i][0]:
                        return i
        elif order == 'backward':
            for i in range(1, len(x)+1):
                if mode == 'i':
                    if word in x[-i]:
                        return i
                elif mode == 'e':
                    if word == x[-i][0]:
                        return i
    
    def special_locate_borders(self, load_dir_name, top, bottom):
        holder = []
        cn = Counter()
        path_to_acts = (
            self.dir_struct['Divided_and_tokenized'].joinpath(load_dir_name)
        )
        acts = self.RWT.iterate_pickle_loading(path_to_acts)
        counter = 0
        for act in acts:
            try:
                t = self.special_get_ind(
                    act,
                    top,
                    mode='e',
                    order='forward')
            except:
                print ('No top in {}'.format(counter))
            try:
                b = self.special_get_ind(
                    act,
                    bottom,
                    mode='e',
                    order='backward')
            except:
                print ('No bottom in {}'.format(counter))
            if t:
                t = str(t)
            else:    
                t = 'No top'
            if b:
                b = '-'+str(b)
            else:
                b = 'No bottom'
            holder.append((t, b))
            cn.update([t, b])
            counter+=1
        return holder, cn

    def simple_divide(self,
                      dir_name='',
                      sep_type='sep1',
                      inden=''):
        path = self.dir_struct['Raw_text'].joinpath(dir_name)
        raw_files = self.RWT.iterate_text_loading(path)
        counter1 = 0
        counter2 = 0
        for fle in raw_files:
            print(inden+'Starting new file processing!')
            cleaned = self.CTP.court_decisions_cleaner(fle)
            divided = self.CTP.court_decisions_separator(
                cleaned,
                sep_type=sep_type
            )
            #tokenized = self.CTP.iterate_tokenization(divided)
            counter2 += len(divided)
            t0=time()
            print(inden+'\tStarting writing')
            file_paths = deque(self.RWT.create_writing_paths(
                counter1, counter2,
                self.dir_struct['Divided_acts'].joinpath(dir_name),
                suffix=''
            ))
            for div_act in divided:
                self.RWT.write_pickle(
                    div_act,
                    file_paths.popleft()
                )
            counter1 += len(divided)
            print(
                inden+'\tWriting '
                +'complete in {} seconds!'.format(time()-t0)
            )