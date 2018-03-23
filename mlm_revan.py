import string
import re
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense

def load_doc(filename):
    with open(filename, mode='r') as file:
        text = file.read()
    return text

def clean_doc(doc):
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

def add_doc_to_vocab(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)

def doc_to_line(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

def process_docs_to_vocab(directory, vocab):
    for filename in listdir(directory):
        if filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        add_doc_to_vocab(path, vocab)

def save_list(lines, filename):
    data = '\n'.join(lines)
    with open(filename, mode='w') as file:
        file.write(data)

def process_docs(directory, vocab, is_train):
    lines = list()
    for filename in listdir(directory):
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        line = doc_to_line(path, vocab)
        lines.append(line)
    return lines

def load_clean_dataset(vocab, is_train):
    neg = process_docs((r'C:\Users\EA-ShevchenkoIS\Continuum'+
    r'\anaconda3\MyScripts\Data\txt_sentoken\neg'), vocab, is_train)
    pos = process_docs((r'C:\Users\EA-ShevchenkoIS\Continuum'+
    r'\anaconda3\MyScripts\Data\txt_sentoken\pos'), vocab, is_train)
    docs = neg+pos
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def define_model(n_words):
    model = Sequential()
    model.add(Dense(50, input_shape=(n_words,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
    model.summary()
    return model

def vocab_setting():
    vocab = Counter()
    print('Start word collecting')
    process_docs_to_vocab((r'C:\Users\EA-ShevchenkoIS\Continuum'+
    r'\anaconda3\MyScripts\Data\txt_sentoken\pos'), vocab)
    process_docs_to_vocab((r'C:\Users\EA-ShevchenkoIS\Continuum'+
    r'\anaconda3\MyScripts\Data\txt_sentoken\neg'), vocab)
    print('Vocab len: {}'.format(len(vocab)))
    tokens = [k for k,i in vocab.items() if i >= 2]
    print('Cleaned vocab len: {}'.format(len(tokens)))
    save_list(tokens, (r'C:\Users\EA-ShevchenkoIS\Continuum'+
    r'\anaconda3\MyScripts\Data\vocab.txt'))

def data_preparation():
    vocab_filename = (r'C:\Users\EA-ShevchenkoIS\Continuum'+
    r'\anaconda3\MyScripts\Data\vocab.txt')
    print('Start data preparation')
    vocab = load_doc(vocab_filename)
    vocab = set(vocab.split())
    print('Vocab len: {}'.format(len(vocab)))
    train_docs, ytrain = load_clean_dataset(vocab, True)
    test_docs, ytest = load_clean_dataset(vocab, False)
    tokenizer = create_tokenizer(train_docs)
    Xtrain = tokenizer.texts_to_matrix(train_docs, mode='freq')
    Xtest = tokenizer.texts_to_matrix(test_docs, mode='freq')
    print(Xtrain.shape, Xtest.shape)
    return [(Xtrain, ytrain), (Xtest, ytest)]

