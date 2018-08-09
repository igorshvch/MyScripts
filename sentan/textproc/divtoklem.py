#built-in modules
import json
import re
from datetime import datetime as dt
from time import time
#my modules
import sentan.lowlevel.mypars as mypars
from sentan import mysqlite
from sentan.dirman import DIR_STRUCT
from sentan.stringbreakers import RAWPAR_B, TOKLEM_B, BGRSEP_B, DCTITM_B
from sentan.lowlevel import textsep
from sentan.lowlevel.texttools import (
    create_indexdct_from_tokens_list, indexdct_to_string #, create_bigrams
)
from sentan.lowlevel.rwtool import (
    collect_exist_file_paths, load_text, save_object, load_pickle
)

__version__ = 0.3

###Content=====================================================================
DataStore1 = mypars.ParsDataStore()
DataStore2 = mypars.ParsDataStore()
TOTAL_ACTS = 183

def raw_files_to_db(load_dir_name='',
                    sep_type=textsep.SEP_TYPE,
                    inden='',
                    DS=DataStore1):
    #Initialise local vars
    t0=time()
    PATTERN_PASS = textsep.PATTERN_PASS
    tokenize = mypars.tokenize
    separator = RAWPAR_B
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
    counter = 0
    for fle in raw_files:
        t1=time()
        holder=[]
        print(inden+'\tStarting new file processing!')
        cleaned = textsep.court_decisions_cleaner(fle, inden='\t\t')
        divided = textsep.court_decisions_separator(
            cleaned,
            sep_type=sep_type,
            inden='\t\t'
        )
        print(inden+'\t\tStarting blanklines deletion')
        for act in divided:
            if re.match(PATTERN_PASS, act):
                continue
            counter+=1
            splitted = act.split('\n')
            splitted = [par for par in splitted if par]
            splitted2 = [tokenize(row) for row in splitted if row] #re.split('\W', row)
            DS.words_count(splitted2)
            holder.append(
                (
                    counter,
                    splitted[0],
                    splitted[2],
                    separator.join(splitted)
                )
            )
        DB_save.insert_data(holder, col_num=4)
        print(
            inden+'\t\tRaw text processing '
            +'complete in {:4.5f} seconds!'.format(time()-t1)
        )
    global TOTAL_ACTS
    TOTAL_ACTS = counter
    print(inden+'Total time costs: {}'.format(time()-t0))
    DS.create_lem_map()
    save_object(
        DS.lem_map,
        ('lem_map_' + str(dt.date(dt.now()))),
        r'C:\Users\EA-ShevchenkoIS\TextProcessing\StatData'
    )

def make_tok_lem_bigr_indx(lem_dict_name='', inden='', DS = DataStore2):
    #Initialise local vars
    t0 = time()
    TA = TOTAL_ACTS
    OUTPUT = TA//10 if TA > 10 else TA//2
    lem_dict = load_pickle(str(DIR_STRUCT['StatData'].joinpath(lem_dict_name)))
    sep_par = RAWPAR_B
    sep_toklem = TOKLEM_B
    sep_bigr = BGRSEP_B
    sep_dctitm = DCTITM_B
    lemmed_pars_counter = 0
    #Initialise local funcs
    tokenize = mypars.tokenize
    #cr2gr = create_bigrams
    create_indexdct = create_indexdct_from_tokens_list
    indexdct_tostr = indexdct_to_string
    #Initiate DB connection:
    DB_load = mysqlite.DataBase(
        dir_name='TextProcessing/DivActs/',
        base_name='DivActs',
        tb_name=True
    )
    DB_save = mysqlite.DataBase(
        dir_name='TextProcessing/TNBI/',
        base_name='TNBI'
    )
    DB_save.create_tabel(
        'TNBI',
        (
            ('id', 'INTEGER', 'PRIMARY KEY'),
            ('COURT', 'TEXT'),
            ('REQ', 'TEXT'),
            ('RAWPARS', 'TEXT'),
            ('DIV', 'TEXT'),
            ('LEM', 'TEXT'),
            #('BIGRAMS', 'TEXT'),
            ('INDXACT', 'TEXT'),
            ('INDXPAR', 'TEXT')
        )
    )
    #Start division and lemmatization
    acts_gen = DB_load.iterate_row_retr(length=TA, output=OUTPUT)
    for batch in acts_gen:
        t1 = time()
        holder = []
        print(inden+'\tStarting new batch! {:4.5f}'.format(time()-t0))
        for row in batch:
            idn, court, req, act = row
            rawpars = [
                par for par in act.split(sep_par) if len(tokenize(par))>1
            ]
            tokens_by_par = [tokenize(par) for par in rawpars]
            lems_by_par = [
                [lem_dict[word] for word in par] for par in tokens_by_par
            ]
            lemmed_pars_counter += len(lems_by_par)
            DS.words_count(lems_by_par)
            lems_by_act = [word for par in lems_by_par for word in par]
            #bigrams = [['',''], ['',''], ['','']] #[cr2gr(par) for par in lems_by_par]
            index_act = indexdct_tostr(create_indexdct(lems_by_act))
            index_pars = [
                indexdct_tostr(create_indexdct(par)) for par in lems_by_par
            ]
            holder.append(
                (
                    idn,
                    court,
                    req,
                    sep_par.join(rawpars),
                    sep_par.join(
                        [sep_toklem.join(par) for par in tokens_by_par]
                    ),
                    sep_par.join(
                        [sep_toklem.join(par) for par in lems_by_par]
                    ),
                    #sep_par.join(
                    #    [sep_bigr.join(par) for par in bigrams]
                    #),
                    sep_dctitm.join(index_act),
                    sep_par.join(
                        [sep_dctitm.join(par) for par in index_pars]
                    )
                )
            )
        DB_save.insert_data(holder, col_num=8)
        print(inden+'\t\tBatch was proceed in {:4.5f} seconds'.format(time()-t1))
    print(inden+'Total time costs: {}'.format(time()-t0))
    save_object(
        DS.vocab,
        'vocab_nw',
        r'C:\Users\EA-ShevchenkoIS\TextProcessing\StatData'
    )
    save_object(
        lemmed_pars_counter,
        'total_lem_pars',
        r'C:\Users\EA-ShevchenkoIS\TextProcessing\StatData'
    )


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