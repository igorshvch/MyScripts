from multiprocessing import Pool, Queue, Lock, current_process
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
    data, proc_num = args
    sleep(4)
    name = generate_name()
    est_data = estimate(data)
    pid = os.getpid()
    print(
        'PROCESS: {}, PROC_NUM: {}, PID: {}, DATA: {:4s}'.format(
            name, proc_num, pid, est_data
        )
    )

def start_process():
    print('Starting', current_process().name)

if __name__ == '__main__':
    global_pid = os.getpid()
    holder = []
    t0 = time()
    print('Parent started, PID: {}, TIME: {:2.3f}'.format(global_pid, time()-t0))
    with Pool(cpus*2, start_process) as pool:
        for num in range(cpus*2):
            data = ''.join(rd.sample(alph2, 7))
            holder.append(pool.map_async(worker, iterable=[[data, num]]))
        for r in holder: r.get()
    print('Parent ended, PID: {}, TIME: {:2.3f}'.format(global_pid, time()-t0))