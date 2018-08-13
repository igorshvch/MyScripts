from time import time
from math import (
    log10 as math_log,
    exp as math_exp
)
from sentan import mysqlite
from sentan.textproc import myvect as mv
from sentan.lowlevel import rwtool
from sentan.lowlevel.texttools import (
    bigrams_intersection as bgrint,
    clean_txt_and_remove_stpw as ctrs,
    clean_txt_and_remove_stpw_add_bigrams_splitted as ctrsabs,
    clean_txt_and_remove_stpw_add_intersect_bigrams as ctrsaib,
    create_bigrams as crtbgr,
    string_to_indexdct as str_to_indct
)
from sentan.stringbreakers import (
    DCTKEY_B, DCTITM_B, TOKLEM_B, RAWPAR_B
)
from sentan.textproc.scorer import score

__version__ = 0.2

###Content=====================================================================
VOCAB_NW = rwtool.load_pickle(
    r'C:\Users\EA-ShevchenkoIS\TextProcessing\StatData\vocab_nw'
)
TOTAL_PARS = rwtool.load_pickle(
    r'C:\Users\EA-ShevchenkoIS\TextProcessing\StatData\total_lem_pars'
)
STPW = rwtool.load_pickle(
    r'C:\Users\EA-ShevchenkoIS\TextProcessing\StatData\custom_stpw'
)
DB_CONNECTION = mysqlite.DataBase(
            raw_path = r'C:\Users\EA-ShevchenkoIS\TextProcessing\TNBI',
            base_name='TNBI',
            tb_name=True
)
TOTAL_ACTS = DB_CONNECTION.total_rows()
    

def aggregate_model(concl_lemmed,
                    par_len=140):
    #Initialise local vars=======================
    t0 = time()
    sep_par = RAWPAR_B
    sep_lems = TOKLEM_B
    sep_dctitm = DCTITM_B
    vocab_nw = VOCAB_NW
    TA = None # num of rows in DB table
    TA_pars = TOTAL_PARS
    OUTPUT = None # size of batch
    stpw = STPW
    ##Concl holders for different models=====================
    concl_YA = [word for word in concl_lemmed if word not in stpw]
    concl_m1 = concl_m4 = ' '.join(concl_YA)
    concl_m2 = concl_m3 = concl_m5 = concl_m6 = ' '.join(
        concl_YA + bgrint(concl_lemmed, stpw)
    )
    #Initialise local funcs======================
    estimator = mv.eval_cos_dist
    local_scorer = score
    local_str_to_indct = str_to_indct
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
    DB_load = DB_CONNECTION
    TA = DB_load.total_rows()
    print('Total acts num: {}'.format(TA))
    print('Total pars num: {}'.format(TA_pars))
    OUTPUT = TA//10 if TA > 10 else TA//2
    acts_gen = DB_load.iterate_row_retr(length=TA, output=OUTPUT)
    holder_m1 = []
    holder_m2 = []
    holder_m3 = []
    holder_m4 = []
    holder_m5 = []
    holder_m6 = []
    holder_YA = []
    counter = 1
    for batch in acts_gen:
        t1 = time()
        print(
            '\tStarting new batch! Batch # {}. {:4.5f}'.format(
                counter, time()-t0
            )
        )
        for row in batch:
            _, court, req, rawpars, _, lems, _, index_pars = row
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
            #Eval par score:
            raw_pars_for_scr_par = rawpars.split(sep_par)
            lems_by_par = [par for par in lems.split(sep_par)]
            scr_par_holder = []
            #Find par with the best score through the current act in the row
            for ind, index_par in enumerate(index_pars.split(sep_par)):
                sc_par = local_scorer(
                concl_YA,
                lems_by_par[ind].split(sep_lems),
                local_str_to_indct(index_par.split(sep_dctitm)),
                vocab=vocab_nw,
                total_parts=TA_pars
                )
                scr_par_holder.append((sc_par, raw_pars_for_scr_par[ind]))
            best_par_scr, best_par = sorted(scr_par_holder)[-1]
            holder_YA.append([court, req, best_par_scr, best_par])
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
    holder_YA = sorted(holder_YA, key=lambda x:x[2], reverse=True)
    holders = {
        'm1':holder_m1,
        'm2':holder_m2,
        'm3':holder_m3,
        'm4':holder_m4,
        'm5':holder_m5,
        'm6':holder_m6,
        'YA':holder_YA
    }
    return holders

def count_result_scores(res_dict, top=5):
    holder_acts_set = set()
    holder_acts = []
    for key in res_dict:
        val = res_dict[key]
        reqs = [val[i][0]+' '+val[i][1] for i in range(top)]
        for req in reqs:
            holder_acts_set.add(req)
        holder_acts.extend(reqs)
    acts_score = {}
    for act_req in holder_acts_set:
        acts_score[act_req] = holder_acts.count(act_req)
    return sorted(
        [[key_dct, value] for key_dct, value in acts_score.items()],
        key=lambda x: x[1],
        reverse=True
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