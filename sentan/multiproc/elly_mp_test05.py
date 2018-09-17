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
#==============================================================================

def initializer(lock):
    with lock:
        print(
            'Starting', current_process().name,
            'PID: {:>6}, {:>20s}'.format(pid, mp_processor.__name__)
        )
    



from sentan import shared

shared.init_globs()
shared.init_db()

from sentan import dirman

dirman