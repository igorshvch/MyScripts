from writer import writer
from time import time
from sentan import mysqlite
from sentan.textproc import myvect as mv
from sentan.lowlevel.rwtool import load_pickle
from sentan.lowlevel.texttools import (
    clean_txt_and_remove_stpw as ctrs,
    #clean_txt_and_remove_stpw_add_bigrams as ctrsab,
    clean_txt_and_remove_stpw_add_bigrams_splitted as ctrsabs,
    clean_txt_and_remove_stpw_add_intersect_bigrams as ctrsaib,
    bigrams_intersection as bgrint
)
from sentan.stringbreakers import (
    RAWPAR_B, TOKLEM_B
)

__version__ = 0.1

###Content=====================================================================
STPW = load_pickle(
    r'C:\Users\EA-ShevchenkoIS\TextProcessing\StatData\custom_stpw'
)

def model_1_count_concl_stopw(concl_lemmed,
                              addition=True,
                              fill_val=1):
    #Initialise local vars
    t0 = time()
    sep_par = RAWPAR_B
    sep_lems = TOKLEM_B
    TA = None # num of rows in DB table
    OUTPUT = None # size of batch
    stpw = STPW
    concl = ' '.join(word for word in concl_lemmed if word not in stpw)
    print(concl)
    #Initialise local funcs
    vectorizer = mv.act_and_concl_to_mtrx(
        vector_pop='concl',
        vector_model='count',
        addition=addition,
        fill_val=fill_val
    )
    estimator = mv.eval_cos_dist
    local_cleaner = ctrs
    #Initiate DB connection:
    DB_load = mysqlite.DataBase(
        raw_path = r'C:\Users\EA-ShevchenkoIS\TextProcessing\TNBI',
        base_name='TNBI',
        tb_name=True
    )
    TA = DB_load.total_rows()
    print('Total acts num: {}'.format(TA))
    OUTPUT = TA//10 if TA > 10 else TA//2
    acts_gen = DB_load.iterate_row_retr(length=TA, output=OUTPUT)
    holder = []
    for batch in acts_gen:
        t1 = time()
        print('\tStarting new batch! {:4.5f}'.format(time()-t0))
        for row in batch:
            _, court, req, rawpars, _, lems, _, _ = row
            pars = [
                local_cleaner(par, sep_lems, stpw)
                for par in lems.split(sep_par)
            ]
            par_index, cos = estimator(vectorizer(pars, concl))
            holder.append(
                [court, req, cos, rawpars.split(sep_par)[par_index-1]]
            )
        print('\t\tBatch processed! Time: {:4.5f}'.format(time()-t1))
    print(
            '\tActs were processed!'
            +' Time in seconds: {}'.format(time()-t0)
        )
    holder = sorted(holder, key=lambda x: x[2])
    return holder

def model_2_count_concl_bigrint_stopw(concl_lemmed,
                                      addition=True,
                                      fill_val=1):
    #Initialise local vars
    t0 = time()
    sep_par = RAWPAR_B
    sep_lems = TOKLEM_B
    TA = None # num of rows in DB table
    OUTPUT = None # size of batch
    stpw = STPW
    concl = None # prepared concl
    #Initialise local funcs
    vectorizer = mv.act_and_concl_to_mtrx(
        vector_pop='concl',
        vector_model='count',
        addition=addition,
        fill_val=fill_val
    )
    estimator = mv.eval_cos_dist
    local_cleaner = ctrsaib # ctrsab
    #Initiate DB connection:
    DB_load = mysqlite.DataBase(
        raw_path = r'C:\Users\EA-ShevchenkoIS\TextProcessing\TNBI',
        base_name='TNBI',
        tb_name=True
    )
    TA = DB_load.total_rows()
    print('Total acts num: {}'.format(TA))
    OUTPUT = TA//10 if TA > 10 else TA//2
    acts_gen = DB_load.iterate_row_retr(length=TA, output=OUTPUT)
    concl = [word for word in concl_lemmed if word not in stpw]
    concl = ' '.join(concl + bgrint(concl_lemmed, stpw))
    print(concl)
    holder = []
    for batch in acts_gen:
        t1 = time()
        print('\tStarting new batch! {:4.5f}'.format(time()-t0))
        for row in batch:
            _, court, req, rawpars, _, lems, _, _ = row
            pars = [
                local_cleaner(par, sep_lems, stpw)
                for par in lems.split(sep_par)
            ]
            par_index, cos = estimator(vectorizer(pars, concl))
            holder.append(
                [court, req, cos, rawpars.split(sep_par)[par_index-1]]
            )
        print('\t\tBatch processed! Time: {:4.5f}'.format(time()-t1))
    print(
            '\tActs were processed!'
            +' Time in seconds: {}'.format(time()-t0)
        )
    holder = sorted(holder, key=lambda x: x[2])
    return holder

def model_3_tfidf_concl_bigrint_stopw_parlen(concl_lemmed,
                                             addition=True,
                                             fill_val=0.001,
                                             par_len=140):
    #Initialise local vars
    t0 = time()
    sep_par = RAWPAR_B
    sep_lems = TOKLEM_B
    TA = None # num of rows in DB table
    OUTPUT = None # size of batch
    stpw = STPW
    concl = None # prepared concl
    #Initialise local funcs
    vectorizer = mv.act_and_concl_to_mtrx(
        vector_pop='concl',
        vector_model='tfidf',
        addition=addition,
        fill_val=fill_val
    )
    estimator = mv.eval_cos_dist
    local_cleaner = ctrsaib # ctrsab
    #Initiate DB connection:
    DB_load = mysqlite.DataBase(
        raw_path = r'C:\Users\EA-ShevchenkoIS\TextProcessing\TNBI',
        base_name='TNBI',
        tb_name=True
    )
    TA = DB_load.total_rows()
    print('Total acts num: {}'.format(TA))
    OUTPUT = TA//10 if TA > 10 else TA//2
    acts_gen = DB_load.iterate_row_retr(length=TA, output=OUTPUT)
    concl = [word for word in concl_lemmed if word not in stpw]
    concl = ' '.join(concl + bgrint(concl_lemmed, stpw))
    print(concl)
    holder = []
    for batch in acts_gen:
        t1 = time()
        print('\tStarting new batch! {:4.5f}'.format(time()-t0))
        for row in batch:
            _, court, req, rawpars, _, lems, _, _ = row
            pars = [
                local_cleaner(par, sep_lems, stpw) if len(par)>par_len else ''
                for par in lems.split(sep_par)
            ]
            par_index, cos = estimator(vectorizer(pars, concl))
            holder.append(
                [court, req, cos, rawpars.split(sep_par)[par_index-1]]
            )
        print('\t\tBatch processed! Time: {:4.5f}'.format(time()-t1))
    print(
            '\tActs were processed!'
            +' Time in seconds: {}'.format(time()-t0)
        )
    holder = sorted(holder, key=lambda x: x[2])
    return holder

def model_4_tfidf_act_stopw_parlen(concl_lemmed,  # model N 5
                                   addition=True,
                                   fill_val=0.001,
                                   par_len=140):
    #Initialise local vars
    t0 = time()
    sep_par = RAWPAR_B
    sep_lems = TOKLEM_B
    TA = None # num of rows in DB table
    OUTPUT = None # size of batch
    stpw = STPW
    concl = None # prepared concl
    #Initialise local funcs
    vectorizer = mv.act_and_concl_to_mtrx(
        vector_pop='act',
        vector_model='tfidf',
        addition=addition,
        fill_val=fill_val
    )
    estimator = mv.eval_cos_dist
    local_cleaner = ctrs
    #Initiate DB connection:
    DB_load = mysqlite.DataBase(
        raw_path = r'C:\Users\EA-ShevchenkoIS\TextProcessing\TNBI',
        base_name='TNBI',
        tb_name=True
    )
    TA = DB_load.total_rows()
    print('Total acts num: {}'.format(TA))
    OUTPUT = TA//10 if TA > 10 else TA//2
    acts_gen = DB_load.iterate_row_retr(length=TA, output=OUTPUT)
    concl = [word for word in concl_lemmed if word not in stpw]
    concl = ' '.join(concl)
    print(concl)
    holder = []
    for batch in acts_gen:
        t1 = time()
        print('\tStarting new batch! {:4.5f}'.format(time()-t0))
        for row in batch:
            _, court, req, rawpars, _, lems, _, _ = row
            pars = [
                local_cleaner(par, sep_lems, stpw) if len(par)>par_len else ''
                for par in lems.split(sep_par)
            ]
            par_index, cos = estimator(vectorizer(pars, concl))
            holder.append(
                [court, req, cos, rawpars.split(sep_par)[par_index-1]]
            )
        print('\t\tBatch processed! Time: {:4.5f}'.format(time()-t1))
    print(
            '\tActs were processed!'
            +' Time in seconds: {}'.format(time()-t0)
        )
    holder = sorted(holder, key=lambda x: x[2])
    return holder

def model_5_tfidf_act_bigrintMIX_stopw_parlen(concl_lemmed, # model N 6
                                              addition=True,
                                              fill_val=0.001,
                                              par_len=140):
    #Initialise local vars
    t0 = time()
    sep_par = RAWPAR_B
    sep_lems = TOKLEM_B
    TA = None # num of rows in DB table
    OUTPUT = None # size of batch
    stpw = STPW
    concl = None # prepared concl
    #Initialise local funcs
    vectorizer = mv.act_and_concl_to_mtrx(
        vector_pop='act',
        vector_model='tfidf',
        addition=addition,
        fill_val=fill_val
    )
    estimator = mv.eval_cos_dist
    local_cleaner = ctrsaib
    #Initiate DB connection:
    DB_load = mysqlite.DataBase(
        raw_path = r'C:\Users\EA-ShevchenkoIS\TextProcessing\TNBI',
        base_name='TNBI',
        tb_name=True
    )
    TA = DB_load.total_rows()
    print('Total acts num: {}'.format(TA))
    OUTPUT = TA//10 if TA > 10 else TA//2
    acts_gen = DB_load.iterate_row_retr(length=TA, output=OUTPUT)
    concl = [word for word in concl_lemmed if word not in stpw]
    concl = ' '.join(concl + bgrint(concl_lemmed, stpw))
    print(concl)
    holder = []
    for batch in acts_gen:
        t1 = time()
        print('\tStarting new batch! {:4.5f}'.format(time()-t0))
        for row in batch:
            _, court, req, rawpars, _, lems, _, _ = row
            pars = [
                local_cleaner(par, sep_lems, stpw) if len(par)>par_len else ''
                for par in lems.split(sep_par)
            ]
            par_index, cos = estimator(vectorizer(pars, concl))
            holder.append(
                [court, req, cos, rawpars.split(sep_par)[par_index-1]]
            )
        print('\t\tBatch processed! Time: {:4.5f}'.format(time()-t1))
    print(
            '\tActs were processed!'
            +' Time in seconds: {}'.format(time()-t0)
        )
    holder = sorted(holder, key=lambda x: x[2])
    return holder

def model_6_tfidf_act_bigrint_stopw_parlen(concl_lemmed, # model N 7
                                           addition=True,
                                           fill_val=0.001,
                                           par_len=140):
    #Initialise local vars
    t0 = time()
    sep_par = RAWPAR_B
    sep_lems = TOKLEM_B
    TA = None # num of rows in DB table
    OUTPUT = None # size of batch
    stpw = STPW
    concl = None # prepared concl
    #Initialise local funcs
    vectorizer = mv.act_and_concl_to_mtrx(
        vector_pop='mixed',
        vector_model='tfidf',
        addition=addition,
        fill_val=fill_val
    )
    estimator = mv.eval_cos_dist
    local_cleaner = ctrsabs
    #Initiate DB connection:
    DB_load = mysqlite.DataBase(
        raw_path = r'C:\Users\EA-ShevchenkoIS\TextProcessing\TNBI',
        base_name='TNBI',
        tb_name=True
    )
    TA = DB_load.total_rows()
    print('Total acts num: {}'.format(TA))
    OUTPUT = TA//10 if TA > 10 else TA//2
    acts_gen = DB_load.iterate_row_retr(length=TA, output=OUTPUT)
    concl = [word for word in concl_lemmed if word not in stpw]
    concl = ' '.join(concl + bgrint(concl_lemmed, stpw))
    print(concl)
    holder = []
    for batch in acts_gen:
        t1 = time()
        print('\tStarting new batch! {:4.5f}'.format(time()-t0))
        for row in batch:
            _, court, req, rawpars, _, lems, _, _ = row
            pars, pars_and_bigrs = local_cleaner(
                lems, par_len, sep_par, sep_lems, stpw
            )
            par_index, cos = estimator(
                vectorizer(pars, concl, pars_with_bigrs=pars_and_bigrs)
            )
            holder.append(
                [court, req, cos, rawpars.split(sep_par)[par_index-1]]
            )
        print('\t\tBatch processed! Time: {:4.5f}'.format(time()-t1))
    print(
            '\tActs were processed!'
            +' Time in seconds: {}'.format(time()-t0)
        )
    holder = sorted(holder, key=lambda x: x[2])
    return holder


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