import pymorphy2 as pmr
import re
import textextrconst as tec
from nltk import sent_tokenize
from nltk.corpus import stopwords

#pattern_clean = r'-{66}\nКонсультантПлюс.+?-{66}\n'
#pattern_sep = r'\n\n\n-{66}\n\n\n'

pattern = tec.RU_word_strip_pattern

morph = pmr.MorphAnalyzer()
parser = morph.parse

class Cleaner():
    def __init__(self, text=None, path=None, auto_mode=True, custom_stopwords=False):
        if text:
            self.text = text
        if path:
            with open(path) as file:
                self.text = file.read()
        if custom_stopwords:
            self.usefull_stops = set([word for word in stopwords.words('russian') if len(word)<=3 and word != 'нет'])
        else:
            self.usefull_stops = stopwords.words('russian')
        self.splitted = None
        self.lc = False
        if auto_mode:
            if auto_mode == 'sent_split':
                self.splitter(mode='sent')
            else:
                self.splitter()
            self.remove_num_punct()
            self.lowcase()
            self.remove_stopwords()
            #self.normalizer()

    
    def splitter(self, mode='par'):
        '''
        Function splits document into paragraphs or sentences and saves them as list object into .splitted attribute
        '''
        if mode == 'par':
            self.splitted = self.text.split('\n')
            print('Text was separated by the "\\n" symbol')
        elif mode == 'sent':
            self.splitted = sent_tokenize(self.text)
            print('Text was splitted into sentences')
    
    def remove_num_punct(self):
        '''
        Function removes all numerical and punctuation characters. Hiphens are saved.
        '''
        if self.splitted:
            container = []
            for par in self.splitted:
                par_tokens = re.findall(pattern, par)
                par_text = ' '.join(par_tokens)
                container.append(par_text)
            self.splitted = container
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
            if self.splitted:
                container = []
                for par in self.splitted:
                        par_tokens = par.split(' ')
                        par_words = [word for word in par_tokens if word not in self.usefull_stops]
                        par_text = ' '.join(par_words)
                        container.append(par_text)
                self.splitted = container    
            else:
                tokens = self.text.split(' ')
                words = [word for word in tokens if word not in self.usefull_stops]
                self.text = ' '.join(words)
    
    def lowcase(self):
        '''
        Function lowers the case in words
        '''
        if self.splitted:
            container = []
            for par in self.splitted:
                par = par.lower()
                container.append(par)
            self.splitted = container
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
        if self.splitted:
            container = []
            for par in self.splitted:
                par_tokens = par.split(' ')
                norm_tokens = [parser(word)[0].normal_form for word in par_tokens]
                text_par = ' '.join(norm_tokens)
                container.append(text_par)
            self.splitted = container
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



