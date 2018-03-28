import act_sep as acts
import re
from nltk.corpus import stopwords
from collections import Counter
from keras.preprocessing.text import Tokenize
from keras.models import Sequential
from keras.layers import Dense


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
    new_act = []
    new_store = []
    for act in cln_act_words:
        new_act = [w for w in act if w in tokens]
        new_store.append(act)
    return new_store

cleaned_acts = clean_acts()
tokenizer = Tokenize()
tokenizer.fit_on_texts(cleaned_acts)
