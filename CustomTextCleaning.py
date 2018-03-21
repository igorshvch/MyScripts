import pymorphy2 as pmr
import re
import textextrconst as tec
from nltk import sent_tokenize
from nltk.corpus import stopwords

#pattern_clean = r'-{66}\nКонсультантПлюс.+?-{66}\n'
#pattern_sep = r'\n\n\n-{66}\n\n\n'

pattern = tec.RU_word_strip_pattern
usefull_stops = set([word for word in stopwords.words('russian') if len(word)<=3 and word != 'нет'])

morph = pmr.MorphAnalyzer()
parser = morph.parse

class Cleaner():
    def __init__(self, text=None, path=None, auto_mode=True):
        if text:
            self.text = text
        if path:
            with open(path) as file:
                self.text = file.read()
        self.spl_pars = None
        self.lc = False
        if auto_mode == True:
            self.split_paragraphs()
            self.remove_num_punct()
            self.lowcase()
            self.remove_stopwords()
            #self.normalizer()

    
    def split_paragraphs(self):
        '''
        Function splits document into paragraphs and saves them as list object into .spl_pars attribute
        '''
        self.spl_pars = self.text.split('\n')
        print('Text was separated by the "\\n" symbol')
    
    def remove_num_punct(self):
        '''
        Function removes all numerical and punctuation characters. Hiphens are saved.
        '''
        if self.spl_pars:
            container = []
            for par in self.spl_pars:
                par_tokens = re.findall(pattern, par)
                par_text = ' '.join(par_tokens)
                container.append(par_text)
            self.spl_pars = container
        else:
            verifier = input('При выполнении данной операции текст невозможно будет разделить на абзацы. Продолжить[y/n]?\n')
            if verifier == 'y':
                tokens = re.findall(pattern, self.text)
                self.text = ' '.join(tokens)
            elif verifier == 'n':
                print('Operation was aborted')
            else:
                print('Unknown command. Operation was aborted')
    
    def remove_stopwords(self):
        '''
        Function removes stopwords.
        '''
        if not self.lc:
            print('При выполнении данной операции до операции .lowcase существует вероятность, \
            что не все стоп-слова будут удалены из тектса. Операция прекращена')
        else:
            if self.spl_pars:
                container = []
                for par in self.spl_pars:
                        par_tokens = par.split(' ')
                        par_words = [word for word in par_tokens if word not in usefull_stops]
                        par_text = ' '.join(par_words)
                        container.append(par_text)
                self.spl_pars = container    
            else:
                tokens = self.text.split(' ')
                words = [word for word in tokens if word not in usefull_stops]
                self.text = ' '.join(words)
    
    def lowcase(self):
        '''
        Function lowers the case in words
        '''
        if self.spl_pars:
            container = []
            for par in self.spl_pars:
                par = par.lower()
                container.append(par)
            self.spl_pars = container
            self.lc = True
        else:
            verifier = input('При выполнении данной операции текст невозможно будет разделить на абзацы. Продолжить[y/n]?\n')
            if verifier == 'y':
                self.text = self.text.lower()
                self.lc = True
            elif verifier == 'n':
                print('Operation was aborted')
            else:
                print('Unknown command. Operation was aborted')
    
    def normalizer(self):
        '''
        Function normalises word forms in the text
        '''
        if self.spl_pars:
            container = []
            for par in self.spl_pars:
                par_tokens = par.split(' ')
                norm_tokens = [parser(word)[0].normal_form for word in par_tokens]
                text_par = ' '.join(norm_tokens)
                container.append(text_par)
            self.spl_pars = container
        else:
            verifier = input('При выполнении данной операции текст невозможно будет разделить на абзацы. Продолжить[y/n]?\n')
            if verifier == 'y':
                tokens = self.text.split(' ')
                norm_tokens = [parser(word)[0].normal_form for word in tokens]
                self.text = ' '.join(norm_tokens)
            elif verifier == 'n':
                print('Operation was aborted')
            else:
                print('Unknown command. Operation was aborted')



