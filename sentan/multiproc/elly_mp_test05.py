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
#My modules
from sentan.lowlevel import rwtool
from sentan import shared, dirman
#==============================================================================

def initializer(lock):
    pid = os.getpid()
    with lock:
        print(
            'Starting', current_process().name,
            'PID: {:>6}, {:>20s}'.format(pid, initializer.__name__)
        )
        shared.init_globs()
        shared.init_db()
        dirman.init_project()
        shared.GLOBS.update(
            rwtool.load_pickle(
                str(dirman.DIR_STRUCT_ROOT['TEMP'].joinpath('GLOBS'))
            )
        )
        dirman.
        print('Shared GLOBS were loaded!')


    
    
    
    
    



from sentan import shared

shared.init_globs()
shared.init_db()

from sentan import dirman

dirman