# coding=cp1251

from concurrent.futures import (
    ProcessPoolExecutor as PPE,
    ThreadPoolExecutor as TPE
)
from time import time
from math import (
    log10 as math_log,
    exp as math_exp
)
from sentan import mysqlite
from sentan.lowlevel import rwtool
from sentan.lowlevel.mypars import (
    tokenize as my_tok,
    lemmatize as my_lem
)
from sentan.textproc import cosdalg as csd

__version__ = 0.1

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

def aggregate_model_processes(raw_concl, workers):
    print('='*23)
    print('Start multiprocessing elly evaluation!')
    t0 = time()
    concl_lemmed = my_lem(my_tok(raw_concl))
    with PPE(int(workers)) as ex:
        r1 = ex.submit(
            csd.model_1_count_concl_stopw, concl_lemmed
        )
        r2 = ex.submit(
            csd.model_2_count_concl_bigrint_stopw, concl_lemmed
        )
        r3 = ex.submit(
            csd.model_3_tfidf_concl_bigrint_stopw_parlen,
            concl_lemmed
        )
        r4 = ex.submit(
            csd.model_4_tfidf_act_stopw_parlen,
            concl_lemmed
        )
        r5 = ex.submit(
            csd.model_5_tfidf_act_bigrintMIX_stopw_parlen,
            concl_lemmed
        )
        r6 = ex.submit(
            csd.model_6_tfidf_act_bigrint_stopw_parlen,
            concl_lemmed
        )
    holders = {
        'm1':r1.result(),
        'm2':r2.result(),
        'm3':r3.result(),
        'm4':r4.result(),
        'm5':r5.result(),
        'm6':r6.result()
    }
    t1 = time()
    print('='*23)
    print('Ended multiprocessing elly evaluation!')
    print(
        'Time: sec: {:3.5f}, mins: {:3.5f}'.format(
            t1-t0, (t1-t0)/60
        )
    )

def aggregate_model_threads(raw_concl):
    print('='*23)
    print('Start multithreading elly evaluation!')
    t0 = time()
    concl_lemmed = my_lem(my_tok(raw_concl))
    with TPE(4) as ex:
        r1 = ex.submit(
            csd.model_1_count_concl_stopw, concl_lemmed
        )
        r2 = ex.submit(
            csd.model_2_count_concl_bigrint_stopw, concl_lemmed
        )
        r3 = ex.submit(
            csd.model_3_tfidf_concl_bigrint_stopw_parlen,
            concl_lemmed
        )
        r4 = ex.submit(
            csd.model_4_tfidf_act_stopw_parlen,
            concl_lemmed
        )
        r5 = ex.submit(
            csd.model_5_tfidf_act_bigrintMIX_stopw_parlen,
            concl_lemmed
        )
        r6 = ex.submit(
            csd.model_6_tfidf_act_bigrint_stopw_parlen,
            concl_lemmed
        )
    holders = {
        'm1':r1.result(),
        'm2':r2.result(),
        'm3':r3.result(),
        'm4':r4.result(),
        'm5':r5.result(),
        'm6':r6.result()
    }
    t1 = time()
    print('='*23)
    print('Ended multithreading elly evaluation!')
    print(
        'Time: sec: {:3.5f}, mins: {:3.5f}'.format(
            t1-t0, (t1-t0)/60
        )
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
        elif sys.argv[1] == '-start_mp_test':
            aggregate_model_processes(
                '1 .1. Являются ли плательщиками НДС государственные (муниципальные) органы, имеющие статус юридического лица (государственные и муниципальные учреждения) (  п. 1 ст. 143   НК РФ)?  В   п. 1    данного Постановления указано, что государственные (муниципальные) органы, имеющие статус юридического лица (государственные и муниципальные учреждения), в силу   п. 1 ст. 143   НК РФ могут являться плательщиками НДС по совершаемым ими финансово-хозяйственным операциям, если они действуют в собственных интересах в качестве самостоятельных хозяйствующих субъектов, а не реализуют публично-правовые функции соответствующего публично-правового образования и не выступают от его имени в гражданских правоотношениях в порядке, предусмотренном   ст. 125   ГК РФ.',
                sys.argv[2]
            )
        elif sys.argv[1] == '-start_mt_test':
            aggregate_model_threads(
                '1 .1. Являются ли плательщиками НДС государственные (муниципальные) органы, имеющие статус юридического лица (государственные и муниципальные учреждения) (  п. 1 ст. 143   НК РФ)?  В   п. 1    данного Постановления указано, что государственные (муниципальные) органы, имеющие статус юридического лица (государственные и муниципальные учреждения), в силу   п. 1 ст. 143   НК РФ могут являться плательщиками НДС по совершаемым ими финансово-хозяйственным операциям, если они действуют в собственных интересах в качестве самостоятельных хозяйствующих субъектов, а не реализуют публично-правовые функции соответствующего публично-правового образования и не выступают от его имени в гражданских правоотношениях в порядке, предусмотренном   ст. 125   ГК РФ.'
            )
        else:
            print('Not implemented!')
    except IndexError:
        print('Mode var wasn\'t passed!')