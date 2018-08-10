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

__version__ = 0.5

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

def aggregate_model_csd(concl_lemmed,
                        par_len=140):
    #Initialise local vars=======================
    t0 = time()
    sep_par = RAWPAR_B
    sep_lems = TOKLEM_B
    TA = None # num of rows in DB table
    OUTPUT = None # size of batch
    stpw = STPW
    ##Concl holders for different models=====================
    concl_m1 = concl_m4 = ' '.join(
        word for word in concl_lemmed if word not in stpw
    )
    concl_m2 = concl_m3 = concl_m5 = concl_m6 = ' '.join(
        [word for word in concl_lemmed if word not in stpw]
        + bgrint(concl_lemmed, stpw)
    )
    #Initialise local funcs======================
    estimator = mv.eval_cos_dist
    ##Initialise vectorizers for different models============
    vectorizer_m1 = vectorizer_m2 = mv.act_and_concl_to_mtrx(
        vector_pop='concl',
        vector_model='count',
        addition=True,
        fill_val=1
    )
    vectorizer_m3 = mv.act_and_concl_to_mtrx(
        vector_pop='concl',
        vector_model='tfidf',
        addition=True,
        fill_val=0.001
    )
    vectorizer_m4 = vectorizer_m5 = mv.act_and_concl_to_mtrx(
        vector_pop='act',
        vector_model='tfidf',
        addition=True,
        fill_val=0.001
    )
    vectorizer_m6 = mv.act_and_concl_to_mtrx(
        vector_pop='mixed',
        vector_model='tfidf',
        addition=True,
        fill_val=0.001
    )
    ##Initialise cleanres for different models===============
    local_cleaner_m1 = local_cleaner_m4 = ctrs
    local_cleaner_m2 = local_cleaner_m3 = local_cleaner_m5 = ctrsaib
    local_cleaner_m6 = ctrsabs
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
    holder_m1 = []
    holder_m2 = []
    holder_m3 = []
    holder_m4 = []
    holder_m5 = []
    holder_m6 = []
    counter = 1
    for batch in acts_gen:
        t1 = time()
        print(
            '\tStarting new batch! Batch # {}. {:4.5f}'.format(
                counter, time()-t0
            )
        )
        for row in batch:
            _, court, req, rawpars, _, lems, _, _ = row
            #Split strings
            rawpars_splitted = rawpars.split(sep_par)
            lems_splitted = lems.split(sep_par)
            #Process pars
            pars_m1 = [
                local_cleaner_m1(par, sep_lems, stpw)
                for par in lems_splitted
            ]
            pars_m2 = [
                local_cleaner_m2(par, sep_lems, stpw)
                for par in lems_splitted
            ]
            pars_m3 = [
                local_cleaner_m3(par, sep_lems, stpw) if len(par)>par_len else ''
                for par in lems_splitted
            ]
            pars_m4 = [
                local_cleaner_m4(par, sep_lems, stpw) if len(par)>par_len else ''
                for par in lems_splitted
            ]
            pars_m5 = [
                local_cleaner_m5(par, sep_lems, stpw) if len(par)>par_len else ''
                for par in lems_splitted
            ]
            pars_m6, pars_and_bigrs_m6 = local_cleaner_m6(
                lems, par_len, sep_par, sep_lems, stpw
            )
            #Eval cosdist
            par_index_m1, cos_m1 = estimator(vectorizer_m1(pars_m1, concl_m1))
            holder_m1.append(
                [court, req, cos_m1, rawpars_splitted[par_index_m1-1]]
            )
            par_index_m2, cos_m2 = estimator(vectorizer_m2(pars_m2, concl_m2))
            holder_m2.append(
                [court, req, cos_m2, rawpars_splitted[par_index_m2-1]]
            )
            par_index_m3, cos_m3 = estimator(vectorizer_m3(pars_m3, concl_m3))
            holder_m3.append(
                [court, req, cos_m3, rawpars_splitted[par_index_m3-1]]
            )
            par_index_m4, cos_m4 = estimator(vectorizer_m4(pars_m4, concl_m4))
            holder_m4.append(
                [court, req, cos_m4, rawpars_splitted[par_index_m4-1]]
            )
            par_index_m5, cos_m5 = estimator(vectorizer_m5(pars_m5, concl_m5))
            holder_m5.append(
                [court, req, cos_m5, rawpars_splitted[par_index_m5-1]]
            )
            par_index_m6, cos_m6 = estimator(
                vectorizer_m6(pars_m6, concl_m6, pars_with_bigrs=pars_and_bigrs_m6)
            )
            holder_m6.append(
                [court, req, cos_m6, rawpars_splitted[par_index_m6-1]]
            )
            ################################################
            ################################################
            ################################################
            counter += 1
        print('\t\tBatch processed! Time: {:4.5f}'.format(time()-t1))
    print(
            '\tActs were processed!'
            +' Time in seconds: {}'.format(time()-t0)
        )
    holder_m1 = sorted(holder_m1, key=lambda x: x[2])
    holder_m2 = sorted(holder_m2, key=lambda x: x[2])
    holder_m3 = sorted(holder_m3, key=lambda x: x[2])
    holder_m4 = sorted(holder_m4, key=lambda x: x[2])
    holder_m5 = sorted(holder_m5, key=lambda x: x[2])
    holder_m6 = sorted(holder_m6, key=lambda x: x[2])
    holders = {
        'm1':holder_m1,
        'm2':holder_m2,
        'm3':holder_m3,
        'm4':holder_m4,
        'm5':holder_m5,
        'm6':holder_m6
    }
    return holders


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