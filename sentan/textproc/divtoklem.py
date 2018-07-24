import json
import re
from time import time
from sentan import mysqlite
from sentan.dirman import DIR_STRUCT
from sentan.lowlevel.rwtool import (
    collect_exist_file_paths, load_text, save_object, load_pickle
)
from sentan.lowlevel.mypars import DataStore, tokenize, lemmatize
from sentan.lowlevel import textsep
from writer import writer

__version__ = 0.1

###Content=====================================================================
DataStore = DataStore()

def raw_files_to_db(load_dir_name='',
                    sep_type=textsep.SEP_TYPE,
                    inden='',
                    DS=DataStore):
    t0=time()
    PATTERN_PASS = textsep.PATTERN_PASS
    #Initiate concls loading:
    path = DIR_STRUCT['RawText'].joinpath(load_dir_name)
    raw_files = (
        load_text(p) for p
        in collect_exist_file_paths(top_dir=path, suffix='.txt')
    )   
    #Initiate DB connection:
    DB_save = mysqlite.DataBase(
        dir_name='TextProcessing/DivActs/',
        base_name='DivActs'
    )
    DB_save.create_tabel(
        'DivActs',
        (
            ('id', 'INTEGER', 'PRIMARY KEY'),
            ('COURT', 'TEXT'),
            ('REQ', 'TEXT'),
            ('ACT', 'TEXT')
        )
    )
    #Start acts separation
    counter = 1
    for fle in raw_files:
        t1=time()
        holder=[]
        print(inden+'Starting new file processing!')
        cleaned = textsep.court_decisions_cleaner(fle, inden='\t')
        divided = textsep.court_decisions_separator(
            cleaned,
            sep_type=sep_type,
            inden='\t'
        )
        print(inden+'\tStarting blanklines deletion')
        for act in divided:
            if re.match(PATTERN_PASS, act):
                continue
            splitted = act.split('\n')
            splitted2 = [tokenize(row) for row in splitted if row] #re.split('\W', row)
            DS.words_count(splitted2)
            holder.append(
                (
                    counter,
                    splitted[0],
                    splitted[2],
                    '\n'.join(splitted)
                )
            )
            counter+=1
        DB_save.insert_data(holder, col_num=4)
        print(
            inden+'\tRaw text processing '
            +'complete in {:4.5f} seconds!'.format(time()-t1)
        )
    print(inden+'Total time costs: {}'.format(time()-t0))
    DS.create_lem_map()
    save_object(DS.lem_map,
                'lem_map',
                r'C:\Users\EA-ShevchenkoIS\TextProcessing\StatData')

def 


def div_tok_acts_db(load_dir_name='',
                    save_dir_name='',
                    sep_type=textsep.SEP_TYPE,
                    inden=''):
    t0=time()
    #Initiate concls loading:
    path = DIR_STRUCT['RawText'].joinpath(load_dir_name)
    raw_files = (
        load_text(p) for p
        in collect_exist_file_paths(top_dir=path, suffix='.txt')
    )
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
        cleaned = textsep.court_decisions_cleaner(fle)
        divided = textsep.court_decisions_separator(
            cleaned,
            sep_type=sep_type
        )
        tokenized = [1,2,3]#self.CTP.iterate_tokenization(divided)
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

###Testing=====================================================================
if __name__ == '__main__':
    import sys
    try:
        sys.argv[1]
        if sys.argv[1] == '-v':
            print('Module name: {}'.format(sys.argv[0]))
            print('Version info:', __version__)
        elif sys.argv[1] == '-t':
            print('Testing mode!')
            print('Not implemented!')
        else:
            print('Not implemented!')
    except IndexError:
        print('Mode var wasn\'t passed!')