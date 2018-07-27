from time import time
from sentan import mysqlite
from sentan.textproc import myvect as mv
from sentan.lowlevel.rwtool import load_pickle
from sentan.stringbreakers import (
    RAWPAR_B, TOKLEM_B
)

__version__ = 0.1

###Content=====================================================================
TOTAL_ACTS = 183
STPW = load_pickle(
    r'C:\Users\EA-ShevchenkoIS\TextProcessing\StatData\custom_stpw'
)

def model_1_count_vect_count_concl_stopw(concl_lemmed,
                                         output_filename='',
                                         total_acts=None,
                                         addition=True,
                                         fill_val=1):
    #Initialise local vars
    t0 = time()
    sep_par = RAWPAR_B
    sep_lems = TOKLEM_B
    TA = total_acts if total_acts else TOTAL_ACTS
    print(TA, TOTAL_ACTS, total_acts)
    OUTPUT = TA//10 if TA > 10 else TA//2
    #Initialise local funcs
    vectorizer = mv.act_and_concl_to_mtrx(
        vector_pop='concl',
        vector_model='count',
        addition=addition,
        fill_val=fill_val
    )
    estimator = mv.eval_cos_dist
    #Initiate DB connection:
    DB_load = mysqlite.DataBase(
        raw_path = r'C:\Users\EA-ShevchenkoIS\TextProcessing\TNBI',
        base_name='TNBI',
        tb_name=True
    )
    acts_gen = DB_load.iterate_row_retr(length=TA, output=OUTPUT)
    for batch in acts_gen:
        t1 = time()
        holder = []
        print('\tStarting new batch! {:4.5f}'.format(time()-t0))
        for row in batch:
            _, court, req, rawpars, _, lems, _, _ = row
            pars = [par.replace(sep_lems, ' ') for par in lems.split(sep_par)]
            par_index, cos = estimator(vectorizer(pars, ' '.join(concl_lemmed)))
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