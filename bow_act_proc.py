'import act_sep as acts'
import re
import shelve
from string import punctuation
from nltk.corpus import stopwords
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense

BASE_PATH = r'C:\Users\EA-ShevchenkoIS\Python36\AP\SimpleBase'

def srch_lims(start, stop, inclusion=True):
    '''
    The function returns reg.exp object with setted search borders
    '''
    if inclusion:
        pattern = r'(?={0}).+(?<={1})'.format(start, stop)
    else:
        pattern = r'(?<={0}).+(?={1})'.format(start, stop)
    return re.compile(pattern, flags=re.DOTALL)

def extract_databse_info(data_key):
    '''
    The function exctracts info from the data base and loads it
    to memory
    '''
    data_base = shelve.open(BASE_PATH)
    instances = {
        'tags': sorted(data_base['tags'].keys()),
        'marked_acts': data_base['marked_acts'].values(),
        'opened_acts': 0,
        'acts_keys': 0,
        'all_acts': 0,
        'processed_acts': 0
    }
    data = instances[data_key]
    data_base.close()
    return data

def find_act_part(tag, acts):
    re_obj = srch_lims(tag, ('/'+tag), inclusion=False)
    acts_parts = [re_obj.search(act).group(0).lower() for act in acts]
    return acts_parts

def clean_text(txt):
    #preparations
    splt = txt.split()
    re_punc = re.compile('[{}]'.format(re.escape(punctuation)))
    stp_w = set(stopwords.words('russian'))
    #cleaning block
    tokens = [re_punc.sub('', w) for w in splt]
    tokens = [
        w for w in tokens
        if w.isalpha()
        and w not in stp_w
        and len(w)>1]
    #return
    return tokens

def add_txt_to_vocab(tag='02_DEM'):
    vocab = Counter()
    acts = extract_databse_info('marked_acts')
    acts_parts = find_act_part(tag, acts)
    for txt in acts_parts:
        txt = clean_text(txt)
        vocab.update(txt)
    common_dict = [k for k,v in vocab.items() if v >= 2]
    return set(common_dict)

def docs_to_lines(tag_pos='02_DEM', tag_neg='07_REASON'):
    common_dict = add_txt_to_vocab(tag='02_DEM')
    lines_pos = []
    lines_neg = []
    acts = extract_databse_info('marked_acts')
    acts_parts_pos = find_act_part(tag_pos, acts)
    acts_parts_neg = find_act_part(tag_neg, acts)
    for act in acts_parts_pos:
        tokens = clean_text(act)
        tokens = [w for w in tokens if w in common_dict]
        line = ' '.join(tokens)
        lines_pos.append(line)
    for act in acts_parts_neg:
        tokens = clean_text(act)
        tokens = [w for w in tokens if w in common_dict]
        line = ' '.join(tokens)
        lines_neg.append(line)
    all_lines = lines_pos + lines_neg
    return all_lines

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer




'''
stop_words = set(stopwords.words('russian'))
Acts = acts.ActSep(string_ending='usual')

div_acts = Acts.store
Acts.act_clean()
cln_act_words = Acts.store

def clean_tokens():
    holder = []
    for act in cln_act_words:
        holder.extend(act)
    tokens = [w for w in holder if w not in stop_words]
    tokens = [w for w in tokens if len(w) > 1]
    return tokens


tokens = clean_tokens()
vocab = Counter()
vocab.update(tokens)
print(len(vocab))
tokens = [k for k,i in vocab.items() if i >= 4]
print(len(tokens))

def clean_acts():
    new_store = []
    for act in cln_act_words:
        new_act = [w for w in act if w in tokens]
        new_store.append(new_act)
    return new_store

cleaned_acts = clean_acts()
tokenizer = Tokenize()
tokenizer.fit_on_texts(cleaned_acts)
'''