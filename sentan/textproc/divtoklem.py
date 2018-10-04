#built-in modules
import json
import re
import itertools
from datetime import datetime as dt
from time import time
from collections import Counter
#my modules
from sentan.lowlevel import mypars
from sentan.lowlevel import textsep
from sentan.lowlevel.texttools import (
    create_indexdct_from_tokens_list, indexdct_to_string
)
from sentan.lowlevel.rwtool import (
    collect_exist_files, read_text, save_obj, load_pickle
)
from sentan import mysqlite
from sentan.stringbreakers import (
    RAWPAR_B, TOKLEM_B, BGRSEP_B, DCTITM_B
)

__version__ = '0.6.1'

###Content=====================================================================
def raw_files_to_db(paths,
                    load_dir_name,
                    inden='',
                    sep_type='sep1'):
    #Initialise local vars
    t0=time()
    PATTERN_PASS1 = textsep.PATTERN_PASS1
    PATTERN_PASS2 = textsep.PATTERN_PASS2
    tokenize = mypars.tokenize
    separator = RAWPAR_B
    words_store = set()
    lemz = mypars.PARSER
    #Initiate concls loading:
    path = paths['root_struct']['RawText'].joinpath(load_dir_name)
    raw_files = (
        read_text(p) for p
        in collect_exist_files(top_dir=path, suffix='.txt')
    )   
    #Initiate DB connection:
    DB_save = mysqlite.DataBase(
        paths['proj_struct']['ActsBase'],
        'DivActs'
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
            if re.match(PATTERN_PASS1, act) or re.match(PATTERN_PASS2, act):
                continue
            counter+=1
            splitted = act.split('\n')
            splitted = [par for par in splitted if par]
            splitted2 = [tokenize(row) for row in splitted if row] #re.split('\W', row)
            words_store.update(itertools.chain(*splitted2))
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
    TOTAL_ACTS = counter
    print(inden+'Acts in total: {}'.format(TOTAL_ACTS))
    print(inden+'Total time costs: {}'.format(time()-t0))
    t2 = time()
    lem_dict = {key:lemz(key) for key in words_store}
    print('Lem_map was created in {:4.5f} seconds'.format(time()-t2))
    save_obj(
        lem_dict,
        'lem_dict',
        paths['proj_struct']['StatData']
    )
    save_obj(
        TOTAL_ACTS,
        'total_acts',
        paths['proj_struct']['StatData']
    )

def make_tok_lem_bigr_indx(paths, inden=''):
    #Initialise local vars
    t0 = time()
    counter = 1
    TA = load_pickle(
        str(paths['proj_struct']['StatData'].joinpath('total_acts'))
    )
    OUTPUT = TA//10 if TA > 10 else TA//2
    lem_dict = load_pickle(
        str(paths['proj_struct']['StatData'].joinpath('lem_dict'))
    )
    sep_par = RAWPAR_B
    sep_toklem = TOKLEM_B
    sep_dctitm = DCTITM_B
    lemmed_pars_counter = 0
    words_store = Counter()
    #Initialise local funcs
    tokenize = mypars.tokenize
    create_indexdct = create_indexdct_from_tokens_list
    indexdct_tostr = indexdct_to_string
    #Initiate DB connection:
    DB_load = mysqlite.DataBase(
        paths['proj_struct']['ActsBase'],
        'DivActs'
    )
    DB_save = mysqlite.DataBase(
        paths['proj_struct']['ActsBase'],
        'TLI'
    )
    DB_save.create_tabel(
        'TLI',
        (
            ('id', 'INTEGER', 'PRIMARY KEY'),
            ('COURT', 'TEXT'),
            ('REQ', 'TEXT'),
            ('RAWPARS', 'TEXT'),
            ('DIV', 'TEXT'),
            ('LEM', 'TEXT'),
            ('INDXACT', 'TEXT'),
            ('INDXPAR', 'TEXT')
        )
    )
    #Start division and lemmatization
    acts_gen = DB_load.iterate_row_retr(length=TA, output=OUTPUT)
    for batch in acts_gen:
        t1 = time()
        holder = []
        print(
            inden+'\tStarting batch # {:2d}! {:4.5f}'.format(counter, time()-t0)
        )
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
            words_store.update(itertools.chain(*lems_by_par))
            lems_by_act = [word for par in lems_by_par for word in par]
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
                    sep_dctitm.join(index_act),
                    sep_par.join(
                        [sep_dctitm.join(par) for par in index_pars]
                    )
                )
            )
        DB_save.insert_data(holder, col_num=8)
        print(
            inden+'\t\tBatch was proceed in {:4.5f} seconds'.format(time()-t1)
        )
        counter+=1
    print(inden+'Total time costs: {}'.format(time()-t0))
    save_obj(
        words_store,
        'vocab_nw',
        paths['proj_struct']['StatData']
    )
    save_obj(
        lemmed_pars_counter,
        'total_lem_pars',
        paths['proj_struct']['StatData']
    )

def main(paths, load_dir_name, inden=''):
    print('='*80)
    print('='*80)
    print(inden*2+'Starting acts separation!')
    raw_files_to_db(paths, load_dir_name)
    print('='*80)
    print(inden*2+'Starting text preprocessing!')
    make_tok_lem_bigr_indx(paths)


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