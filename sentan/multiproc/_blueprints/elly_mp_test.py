# coding=cp1251

from multiprocessing import Pool, Manager, Lock, current_process, cpu_count
import os
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
from sentan.lowlevel.mypars import (
    tokenize as my_tok,
    lemmatize as my_lem
)
from sentan.stringbreakers import (
    DCTKEY_B, DCTITM_B, TOKLEM_B, RAWPAR_B
)
from sentan.textproc.scorer import score

__version__ = 0.3

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
CPUS = cpu_count()-1

def print_cust(message):
    print(23*'=')
    print(message)
    print(23*'=')

def producer(store1, store2, lock):
    local_worker = worker
    with lock:
        print('Starting', current_process().name)
    pid = os.getpid()
    while True:
        try:
            item = store1.get()
        except:
            with lock:
                print('\t\tPID: {}. Exception!'.format(pid))
            continue
        if item == None:
            with lock:
                print('\t\tPID: {}. End loop, bye!'.format(pid))
            break
        else:
            with lock:
                print('\tPID: {}, starting new batch!'.format(pid))
                print('\t\tPID: {}, Batch size: {}'.format(pid, len(item[1])))
            result = local_worker(item)
            with lock:
                print('\tPID: {}, LEN: {}'.format(pid, len(result['YA'])))
            store2.put(
                result,
                block=False
            )

def worker(args):
    #Initialise local vars=======================
    concl_lemmed, batch = args
    par_len = 140
    TA_pars = TOTAL_PARS
    stpw = STPW
    #t0 = time()
    sep_par = RAWPAR_B
    sep_lems = TOKLEM_B
    sep_dctitm = DCTITM_B
    vocab_nw = VOCAB_NW
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
    ##Initialize local storages for processed data===========
    holder_m1 = []
    holder_m2 = []
    holder_m3 = []
    holder_m4 = []
    holder_m5 = []
    holder_m6 = []
    holder_YA = []
    results = {
        'm1':0, 'm2':0, 'm3':0, 'm4':0, 'm5':0, 'm6':0, 'YA':0
    }
    ##Start iteration onver rows of passed data==============
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
            local_cleaner_m3(par, sep_lems, stpw) \
            if len(par)>par_len else ''
            for par in lems_splitted
        ]
        pars_m4 = [
            local_cleaner_m4(par, sep_lems, stpw) \
            if len(par)>par_len else ''
            for par in lems_splitted
        ]
        pars_m5 = [
            local_cleaner_m5(par, sep_lems, stpw) \
            if len(par)>par_len else ''
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
            vectorizer_m6(
                pars_m6, concl_m6, pars_with_bigrs=pars_and_bigrs_m6
            )
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
    results['m1'] = holder_m1
    results['m2'] = holder_m2
    results['m3'] = holder_m3
    results['m4'] = holder_m4
    results['m5'] = holder_m5
    results['m6'] = holder_m6
    results['YA'] = holder_YA
    return results

def main(raw_concl):
    #Initialise local vars=======================
    lock = Lock()
    t0 = time()
    TA = TOTAL_ACTS
    TA_pars = TOTAL_PARS
    OUTPUT = TA//10 if TA > 10 else TA//2
    concl_lemmed = my_lem(my_tok(raw_concl))
    DB_load = DB_CONNECTION
    acts_gen = DB_load.iterate_row_retr(length=TA, output=OUTPUT)
    #PROC_UNITS = 11 if TA > 10 else 3
    store1 = Manager().Queue(maxsize=CPUS)
    store2 = Manager().Queue()
    gen = ((concl_lemmed, batch) for batch in acts_gen)
    results = {
        'm1':[], 'm2':[], 'm3':[], 'm4':[], 'm5':[], 'm6':[], 'YA':[]
    }
    #Initialise local funcs======================
    local_worker = producer
    #Info========================================
    print('\nTotal acts num: {}'.format(TA))
    print('Total pars num: {}'.format(TA_pars))
    #Start data processing=======================
    t1 = time()
    end_time0 = time()-t0
    print_cust(
        'Start data processnig! '
        +'TIME: min: {:3.5f}, '.format(end_time0/60)
        +'sec: {:3.5f}'.format(end_time0)
    )
    with Pool(CPUS*3+2, local_worker, initargs=(store1, store2, lock)) as pool:
        for new_batch in gen:
            store1.put(new_batch)
        #for batch in acts_gen:
        for _ in range(CPUS):
            store1.put(None)
    end_time1 = time()-t1
    print_cust(
        'Data processed! '
        +'TIME: min: {:3.5f}, '.format(end_time1/60)
        +'sec: {:3.5f}'.format(end_time1)
    )
    print('Results extracting')
    while not store2.empty():
        res_next = store2.get()
        for key in res_next:
            results[key].extend(res_next[key])
    end_time2 = time()-t0
    print_cust(
        'Operation ended. '
        +'TIME_TOTAL: min: {:3.5f}, '.format(end_time2/60)
        +'sec: {:3.5f}'.format(end_time2)
    )
    end_res = {}
    for key in results:
        if key == 'YA':
            end_res[key] = sorted(results[key], key=lambda x:x[2], reverse=True)
        else:
            end_res[key] = sorted(results[key], key=lambda x:x[2])
    rwtool.save_object(
        end_res, 'TEST_RES', r'C:\Users\EA-ShevchenkoIS\TextProcessing'
    )
    return 0


###Testing=====================================================================
if __name__ == '__main__':
    print(DB_CONNECTION)
    print(TOTAL_ACTS)
    main(
                '1 .1. Являются ли плательщиками НДС государственные (муниципальные) органы, имеющие статус юридического лица (государственные и муниципальные учреждения) (  п. 1 ст. 143   НК РФ)?  В   п. 1    данного Постановления указано, что государственные (муниципальные) органы, имеющие статус юридического лица (государственные и муниципальные учреждения), в силу   п. 1 ст. 143   НК РФ могут являться плательщиками НДС по совершаемым ими финансово-хозяйственным операциям, если они действуют в собственных интересах в качестве самостоятельных хозяйствующих субъектов, а не реализуют публично-правовые функции соответствующего публично-правового образования и не выступают от его имени в гражданских правоотношениях в порядке, предусмотренном   ст. 125   ГК РФ.'
            )
#    import sys
#    try:
#        sys.argv[1]
#        if sys.argv[1] == '-v':
#            print('Module name: {}'.format(sys.argv[0]))
#            print('Version info:', __version__)
#        elif sys.argv[1] == '-t':
#            print('Testing mode!')
#            print('Not implemented!')
#        elif sys.argv[1] == '-start_mp_test':
#            main(
#                '1 .1. Являются ли плательщиками НДС государственные (муниципальные) органы, имеющие статус юридического лица (государственные и муниципальные учреждения) (  п. 1 ст. 143   НК РФ)?  В   п. 1    данного Постановления указано, что государственные (муниципальные) органы, имеющие статус юридического лица (государственные и муниципальные учреждения), в силу   п. 1 ст. 143   НК РФ могут являться плательщиками НДС по совершаемым ими финансово-хозяйственным операциям, если они действуют в собственных интересах в качестве самостоятельных хозяйствующих субъектов, а не реализуют публично-правовые функции соответствующего публично-правового образования и не выступают от его имени в гражданских правоотношениях в порядке, предусмотренном   ст. 125   ГК РФ.'
#            )
#        else:
#            print('Not implemented!')
#    except IndexError:
#        print('Mode var wasn\'t passed!')