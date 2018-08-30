import os
from threading import Thread, Lock
from multiprocessing import (
    Process, Queue as mp_Queue, Lock, current_process, cpu_count
)
from queue import Queue
from time import time, sleep
import random as rd

CPUS = cpu_count()

alph = [chr(i) for i in range(1040, 1040+32, 1)]
alph2 = [chr(i) for i in range(1072, 1072+32, 1)]

def producer(store, diapason, lock, tid, data=None, multpr=False):
    if multpr:
        data = (''.join(rd.sample(alph2, 7)) for i in range(0, diapason*20000))
    t0 = time()
    for i in data:
        store.put(i)
    with lock:
        print(
            'PRODUCER # {:>3d}:'.format(tid),
            'data stored in {:0>5.5f} sec'.format(time()-t0)
        )
    for i in range(int(diapason)):
        store.put(None)

def consumer(worker, store, lock, tid):
    #sleep(1)
    while True:
        item = store.get()
        if item == None:
            with lock:
                print(
                    'CONSUMER # {:>3d}:'.format(tid),
                    'end of loop, bye!'
                )
            break
        else:
            worker(item, tid, lock)

def worker(item, tid, lock):
    res = ''.join([str(i)*3 for i in item])
    #with lock:
        #print(
        #    '='*10+'WORKER # {:>3d}'.format(tid),
        #    'RESULT: {:->25s}'.format(res)
        #)

def main_multhr(diapason=8):
    t0=time()
    store = Queue(maxsize=int(diapason))
    lock = Lock()
    with lock:
        #print('='*92)
        #print('='*92)
        #print('='*92)
        print('MAIN_MULTHR # {:>3d}: PROGRAM STARTS!'.format(diapason+2))
    gen = (''.join(rd.sample(alph2, 7)) for i in range(0, diapason*10000))
    t_prod = Thread(
        target=producer, args=(store, diapason, lock, diapason+1, gen, False)
    )
    holder = [
        Thread(target=consumer, args=(worker, store, lock, i))
        for i in range(int(diapason))
    ]
    t_prod.start()
    for th in holder: th.start()
    t_prod.join()
    for th in holder: th.join()
    with lock:
        print('MAIN # {:>3d}: PROGRAM ENDED!'.format(diapason+2))
        print('TOTAL TIME: {:0>5.5f} sec'.format(time()-t0))
        #print('='*92)
        #print('='*92)
        #print('='*92)

def main_multproc(diapason=4):
    t0=time()
    store = mp_Queue(maxsize=int(diapason))
    lock = Lock()
    with lock:
        #print('='*92)
        #print('='*92)
        #print('='*92)
        print('MAIN_MULTPROC: PROGRAM STARTS!')
    
    p_prod = Process(
        target=producer, args=(store, diapason, lock, diapason+1, None, True)
    )
    holder = [
        Process(target=consumer, args=(worker, store, lock, i))
        for i in range(CPUS)
    ]
    p_prod.start()
    for pr in holder: pr.start()
    p_prod.join()
    for pr in holder: pr.join()
    with lock:
        print('MAIN_MULTPROC: PROGRAM ENDED!')
        print('TOTAL TIME: {:0>5.5f} sec'.format(time()-t0))
        #print('='*92)
        #print('='*92)
        #print('='*92)

def main_sng_prc(diapason=8):
    t0 = time()
    #print('='*92)
    #print('='*92)
    #print('='*92)
    print('MAIN_SNG_PRC: PROGRAM STARTS!')
    store = Queue()
    gen = (''.join(rd.sample(alph2, 7)) for i in range(0, diapason*10000))
    for i in gen:
        store.put(i)
    while not store.empty():
        item = store.get()
        res = ''.join([str(i)*3 for i in item])
    print('MAIN_SNG_PRC: PROGRAM ENDED!')
    print('TOTAL TIME: {:0>5.5f} sec'.format(time()-t0))
    #print('='*92)
    #print('='*92)
    #print('='*92)

if __name__ == '__main__':
    main_multhr()
    print('\n'+92*'*'+'\n'+92*'*'+'\n'+92*'*'+'\n')
    main_multproc()
    print('\n'+92*'*'+'\n'+92*'*'+'\n'+92*'*'+'\n')
    main_sng_prc()
    


