import pymorphy2
import re
from collections import Counter
from time import time
#from sentan.lowlevl import *


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

#pymoprhy2 analyzer instance
MORPH = pymorphy2.MorphAnalyzer()


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
        d = {word:[] for word in words}
        counter = 0
        for word in words:
            d[word].append(counter)
            counter+=1
        d['total'] = len(words)
        for word in words:
            d['total#'+word]=len(d[word])
        return d