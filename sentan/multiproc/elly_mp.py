# coding=cp1251

from multiprocessing import (
    Process, Queue, Lock, current_process, cpu_count
)
import queue
import os
from time import time, sleep
from math import (
    log10 as math_log,
    exp as math_exp
)
from sentan import mysqlite, dirman, shared
from sentan.textproc import myvect as mv
from sentan.lowlevel import rwtool
from sentan.lowlevel.texttools import (
    bigrams_intersection as bgrint,
    clean_txt_and_remove_stpw as ctrs,
    clean_txt_and_remove_stpw_add_bigrams_splitted as ctrsabs,
    clean_txt_and_remove_stpw_add_intersect_bigrams as ctrsaib,
    create_bigrams as crtbgr,
   string_to_indexdct as str_to_indct,
   form_string_numeration
)
from sentan.lowlevel.mypars import (
    tokenize as my_tok,
    lemmatize as my_lem
)
from sentan.stringbreakers import (
   DCTKEY_B, DCTITM_B, TOKLEM_B, RAWPAR_B
)
from sentan.textproc.scorer import score
from sentan.gui.dialogs import (
    ffp, fdp, pmb, giv
)

__version__ = '0.4.1'

###Content=====================================================================
VOCAB_NW = rwtool.load_pickle(
    str(shared.GLOBS['proj_struct']['StatData'].joinpath('vocab_nw'))
)
TOTAL_PARS = rwtool.load_pickle(
    str(shared.GLOBS['proj_struct']['StatData'].joinpath('total_lem_pars'))
)
TOTAL_ACTS = rwtool.load_pickle(
    str(shared.GLOBS['proj_struct']['StatData'].joinpath('total_acts'))
)
STPW = rwtool.load_pickle(
    str(shared.GLOBS['root_struct']['Common'].joinpath('custom_stpw'))
)
CPUS = cpu_count()
LOCK = Lock()

def processor(item):
    #Initialise local vars=======================
    concl_lemmed, batch = item
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

def mp_processor(store1, store2, lock):
    local_worker = processor
    pid = os.getpid()
    with lock:
        print(
            'Starting', current_process().name,
            'PID: {:>7}, {:>28s}'.format(pid, mp_processor.__name__)
        )
    while True:
        item = store1.get()
        if item == None:
            with lock:
                print(
                    'PID: {:>7}. CONSUMER HAS ENDED THE LOOP, bye!'.format(pid)
                )
            break
        else:
            with lock:
                print(
                    '\t'
                    +'PID: {:>7}, starting new batch! '.format(pid)
                    +'Batch size: {:>7}'.format(len(item[1]))
                )
            result = local_worker(item)
            with lock:
                print('\tPID: {:>7}, result: DONE!'.format(pid)+26*'=')
            store2.put(
                result,
                block=False
            )
    store2.put(None)

def mp_queue_fill(concl, inner_queue, diapason, lock):
    pid = os.getpid()
    with lock:
        print(
            'Starting', current_process().name,
            'PID: {:>7}, {:>28s}'.format(pid, mp_queue_fill.__name__)
        )
        DB_load = shared.DB['TLI']
    TA = TOTAL_ACTS
    OUTPUT = TA//10 if TA > 10 else TA//2
    acts_gen = DB_load.iterate_row_retr(length=TA, output=OUTPUT)
    for new_batch in acts_gen:
        inner_queue.put((concl, new_batch))
    for _ in range(diapason):
        inner_queue.put(None)
    with lock:
        print('PID: {:>7}, QUEUE FILLER HAS ENDED THE LOOP, bye!'.format(pid))

def mp_writer(inner_queue, lock, diapason, indx, save_path):
    pid = os.getpid()
    results = {
        'm1':[], 'm2':[], 'm3':[], 'm4':[], 'm5':[], 'm6':[], 'YA':[]
    }
    stopper = diapason
    counter = 0
    with lock:
        print(
            'Starting', current_process().name,
            'PID: {:>7}, {:>28s}'.format(pid, mp_writer.__name__)
        )
    while True:
        try:
            item = inner_queue.get()
        except queue.Empty:
            continue
        if item == None:
            stopper-=1
            if stopper == 0:
                with lock:
                    print(
                        '\t\t\tPID: {:>7}. '.format(pid)
                        +'Writer has ended the loop, start writing!'
                    )
                break
        else:
            counter += 1
            with lock:
                print(
                    '\tPID: {:>7}, storing # {:>5} batch of data'.format(
                        pid, counter
                    )
                    +' RES_NEXT INFO: {:>5}'.format(len(item))
                )
                for key in item:
                    results[key].extend(item[key])
    end_res = {}
    for key in results:
        if key == 'YA':
            end_res[key] = sorted(results[key], key=lambda x:x[2], reverse=True)
        else:
            end_res[key] = sorted(results[key], key=lambda x:x[2])
    rwtool.save(
        end_res,
        indx + '_TEST_RES',
        to='ProjRes'
    )
    print('\t\t\tPID: {:>7}. Results are written to file'.format(pid)) 
        
def print_cust(message):
    print(92*'=')
    print(message)
    print(92*'=')

def main(raw_concl, indx, save_path, cpus, local_lock=LOCK):
    #Initialise local vars=======================
    pid = os.getpid()
    concl_lemmed = my_lem(my_tok(raw_concl))
    #rwtool.save_object(
    #    concl_lemmed, 'CL', r'C:\Users\EA-ShevchenkoIS\TextProcessing'
    #)
    lock = local_lock
    t0 = time()
    TA_pars = TOTAL_PARS
    #===========================================
    PROC_UNITS = cpus
    #Tests showed that 5 processing units compute data with optimal speed
    #acts_gen = DB_load.iterate_row_retr(length=TA, output=OUTPUT)
    #gen = ((concl_lemmed, batch) for batch in acts_gen)
    store1 = Queue(maxsize=PROC_UNITS)
    store2 = Queue(maxsize=PROC_UNITS)
    #Info========================================
    print('\nTotal acts num: {:>7}'.format(TOTAL_ACTS))
    print('Total pars num: {:>7}'.format(TA_pars))
    #Start data processing=======================
    end_time0 = time()-t0
    print_cust(
        'Start data processnig! PID: {:>7}, '.format(pid)
        +'TIME: min: {:>9.5f}, '.format(end_time0/60)
        +'sec: {:>9.5f}'.format(end_time0)
    )
    #Multiprocessing starts======================
    QUEUE_FILLER = Process(
        target=mp_queue_fill,
        args=(concl_lemmed, store1, PROC_UNITS, lock)
    )
    WORKERS_HOLDER = [
        Process(target=mp_processor, args=(store1, store2, lock))
        for i in range(PROC_UNITS)
    ]
    RESULTS_CONSUMER = Process(
        target=mp_writer, args=(store2, lock, PROC_UNITS, indx, save_path)
    )
    QUEUE_FILLER.start()
    for WP in WORKERS_HOLDER: WP.start()
    RESULTS_CONSUMER.start()
    QUEUE_FILLER.join()
    for WP in WORKERS_HOLDER: WP.join()
    RESULTS_CONSUMER.join()
    #Multiprocessing ends. Info==================
    end_time2 = time()-t0
    print_cust(
        'Operation ended. '
        +'TIME_TOTAL: min: {:>9.5f}, '.format(end_time2/60)
        +'sec: {:>9.5f}'.format(end_time2)
    )

def nextiter(path_to_file=None, local_lock=LOCK, CP_UNITS=5):
    message1 = (
        'Chose concls FILE and DIRECTORY to save results'
    )
    pmb(message1)
    if not path_to_file:
        path = ffp()
    else:
        path = path_to_file
    lock = local_lock
    save_path = fdp()
    #message2 = (
    #    'Number of CPUS: {:>2}.'.format(CPUS)
    #    +'\nSelect number of worker processes:'
    #)
    cpus = int(CP_UNITS)
    with lock:
        print(92*'=')
        print(92*'=')
        print(92*'=')
        print('ITERATION BEGINS!')
        print('PATH:', path)
        print('SAVE PATH:', save_path)
        print('Worker processes total: {}'.format(cpus))
    t0 = time()
    if path[-4:] == '.txt':
        with open(path, mode='r') as fle:
            text = fle.read().strip('\n')
        concls = text.split('\n')
    else:
        concls = rwtool.load_pickle(path)    
    digits_num = len(str(len(concls)))
    formatter = form_string_numeration(digits_num)
    for ind, concl in enumerate(concls):
        with lock:
            print(
                '\n\n'
                +35*'!'
                +'NEW CONCLUSION! # {:>3d}'.format(ind)
                +36*'!'
                +'\n\n'
            )
        main(
            concl, formatter.format(ind), save_path, cpus, local_lock=lock
        )
    end_time = time()-t0
    with lock:
        print(
            '\n\nITERATION ENDS! TOTAL TIME: '
            +'min: {:>9.5f}, sec: {:>9.5f}'.format(end_time/60, end_time)
        )
        print(92*'=')
        print(92*'=')
        print(92*'=')
    

###Testing=====================================================================
if __name__ == '__main__':
    import sys
    print(38*'=' +'PROGRAM STARTS!' + 39*'=')
    try:
        sys.argv[1]
        if sys.argv[1] == '-v':
            print('Module name: {}'.format(sys.argv[0]))
            print('Version info:', __version__)
        elif sys.argv[1] == '-t':
            print('Testing mode!')
            print('Not implemented!')
        elif sys.argv[1] == '-r':
            nextiter(CP_UNITS=sys.argv[2])
        else:
            print('Not implemented!')
    except IndexError:
        print('Mode var wasn\'t passed!')