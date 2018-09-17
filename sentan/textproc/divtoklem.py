#built-in modules
import json
import re
from datetime import datetime as dt
from time import time
#my modules
from sentan import shared
import sentan.lowlevel.mypars as mypars
from sentan import dirman
from sentan.stringbreakers import RAWPAR_B, TOKLEM_B, BGRSEP_B, DCTITM_B
from sentan.lowlevel import textsep
from sentan.lowlevel.texttools import (
    create_indexdct_from_tokens_list, indexdct_to_string #, create_bigrams
)
from sentan.lowlevel.rwtool import (
    collect_exist_files, read_text, save, load_pickle
)

__version__ = '0.5.1'

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
    PATTERN_PASS1 = textsep.PATTERN_PASS1
    PATTERN_PASS2 = textsep.PATTERN_PASS2
    tokenize = mypars.tokenize
    separator = RAWPAR_B
    #Initiate concls loading:
    path = shared.GLOBS['root_struct']['RawText'].joinpath(load_dir_name)
    raw_files = (
        read_text(p) for p
        in collect_exist_files(top_dir=path, suffix='.txt')
    )   
    #Initiate DB connection:
    DB_save = shared.DB['DivActs']
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
    print(inden+'Acts in total: {}'.format(TOTAL_ACTS))
    print(inden+'Total time costs: {}'.format(time()-t0))
    DS.create_lem_map()
    save(
        DS.lem_map,
        'lem_map',
        to='ProjStatData'
    )
    save(
        TOTAL_ACTS,
        'total_acts',
        to='ProjStatData'
    )

def make_tok_lem_bigr_indx(inden='', DS = DataStore2):
    #Initialise local vars
    t0 = time()
    TA = load_pickle(
        str(shared.GLOBS['proj_struct']['StatData'].joinpath('total_acts'))
    )
    OUTPUT = TA//10 if TA > 10 else TA//2
    lem_dict = load_pickle(
        str(shared.GLOBS['proj_struct']['StatData'].joinpath('lem_map'))
    )
    sep_par = RAWPAR_B
    sep_toklem = TOKLEM_B
    sep_dctitm = DCTITM_B
    lemmed_pars_counter = 0
    #Initialise local funcs
    tokenize = mypars.tokenize
    create_indexdct = create_indexdct_from_tokens_list
    indexdct_tostr = indexdct_to_string
    #Initiate DB connection:
    DB_load = shared.DB['DivActs']
    DB_save = shared.DB['TLI']
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
        print(
            inden+'\t\tBatch was proceed in {:4.5f} seconds'.format(time()-t1)
        )
    print(inden+'Total time costs: {}'.format(time()-t0))
    save(
        DS.vocab,
        'vocab_nw',
        to='ProjStatData'
    )
    save(
        lemmed_pars_counter,
        'total_lem_pars',
        to='ProjStatData'
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