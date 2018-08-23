from multiprocessing import Pool, Manager, Lock, current_process, cpu_count
import queue
import os
import random as rd
from time import time, sleep

alph = [chr(i) for i in range(1040, 1040+32, 1)]
alph2 = [chr(i) for i in range(1072, 1072+32, 1)]

cpus =  cpu_count() - 1

def generate_name():
    return ''.join(rd.sample(alph, 4))

def estimate(data):
    return ''.join([str(i)*3 for i in data])

def worker(q1, q2, lock):
    #sleep(1)
    with lock:
        print('Starting', current_process().name)
    pid = os.getpid()
    while True:
        item = q1.get(timeout=2)
        if item == None:
            with lock:
                print('\t\t\tPID: {}. End loop, bye!'.format(pid))
            break
        else:
            data, proc_num = item
            name = generate_name()
            est_data = estimate(data)
            #with lock:
            #    print(
            #        'PROCESS: {}, PROC_NUM: {}, PID: {}, DATA: {:4s}'.format(
            #            name, proc_num, pid, est_data
            #        )
            #    )
            q2.put(
                {'n':name, 'p':proc_num, 'pid':pid, 'ed':est_data},
                #None,
                block=False
            )

if __name__ == '__main__':
    global_pid = os.getpid()
    holder = []
    t0 = time()
    store1 = Manager().Queue(maxsize=cpus)
    store2 = Manager().Queue()
    lock = Lock()
    print(
        'Parent started, PID: {}, TIME: {:2.3f}'.format(global_pid, time()-t0)
    )
    gen = ((''.join(rd.sample(alph2, 7)), i) for i in range(0, cpus*3))
    local_worker = worker
    with Pool(cpus, worker, initargs=(store1, store2, lock)) as pool:
        for new in gen:
            store1.put(new)
            #with lock:
            #    print('\tPID: {}, stuffed!'.format(global_pid))
        for _ in range(cpus):
            store1.put(None)
    print('Parent ended, PID: {}, TIME: {:2.3f}'.format(global_pid, time()-t0))
    while not store2.empty():
        print(store2.get())