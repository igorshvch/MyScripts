from multiprocessing import Pool, Manager, Lock, current_process
import os
import random as rd
from time import time, sleep

alph = [chr(i) for i in range(1040, 1040+32, 1)]
alph2 = [chr(i) for i in range(1072, 1072+32, 1)]

cpus = os.cpu_count() - 1

def generate_name():
    return ''.join(rd.sample(alph, 4))

def estimate(data):
    return ''.join([str(i)*2 for i in data])


def worker(args):
    store, data, proc_num = args
    sleep(1)
    name = generate_name()
    est_data = estimate(data)
    pid = os.getpid()
    print(
        'PROCESS: {}, PROC_NUM: {}, PID: {}, DATA: {:4s}'.format(
            name, proc_num, pid, est_data
        )
    )
    store.put(
        {'n':name, 'p':proc_num, 'pid':pid, 'ed':est_data},
        block=False
    )

def start_process():
    print('Starting', current_process().name)

if __name__ == '__main__':
    global_pid = os.getpid()
    holder = []
    t0 = time()
    store = Manager().Queue()
    print(
        'Parent started, PID: {}, TIME: {:2.3f}'.format(global_pid, time()-t0)
    )
    with Pool(cpus*2, start_process) as pool:
        for num in range(cpus*2):
            data = ''.join(rd.sample(alph2, 7))
            holder.append(pool.apply_async(worker, [[store, data, num]]))
        for r in holder: r.get()
    print('Parent ended, PID: {}, TIME: {:2.3f}'.format(global_pid, time()-t0))
    while not store.empty():
        print(store.get())