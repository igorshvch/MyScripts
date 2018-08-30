import os
from threading import Thread, Lock
from queue import Queue
from time import time, sleep
import random as rd

alph = [chr(i) for i in range(1040, 1040+32, 1)]
alph2 = [chr(i) for i in range(1072, 1072+32, 1)]

def producer(data, store, diapason, lock, tid):
    t0 = time()
    for i in data:
        store.put(i)
    with lock:
        print(
            'PRODUCER # {:>3d}:'.format(tid),
            'data stored in {:0>5.5f} sec'.format(time()-t0)
        )
    for i in range(int(diapason/2)):
        store.put(None)

def consumer(worker, store, lock, tid):
    while True:
        item = store.get()
        if item == None:
            with lock:
                print(
                    'CONSUMER # {:>3d}:'.format(tid),
                    'end of loop, bye!'
                )
        else:
            worker(item, tid, lock)

def worker(item, tid, lock):
    sleep(10)
    res = ''.join([str(i)*3 for i in item])
    with lock:
        print(
            '='*10+'WORKER # {:>3d}'.format(tid),
            'RESULT: {:->25s}'.format(res)
        )

def main(diapason=8):
    t0=time()
    store = Queue(maxsize=int(diapason/2))
    lock = Lock()
    with lock:
        print('='*92)
        print('='*92)
        print('='*92)
        print('MAIN # {:>3d}: PROGRAM STARTS!'.format(diapason+2))
    gen = (''.join(rd.sample(alph2, 7)) for i in range(0, diapason))
    t_prod = Thread(
        target=producer, args=(gen, store, diapason, lock, diapason+1)
    )
    holder = [
        Thread(target=consumer, args=(worker, store, lock, i))
        for i in range(diapason)
    ]
    t_prod.start()
    for th in holder: th.start()
    t_prod.join()
    for th in holder: th.join()
    with lock:
        print('MAIN # {:>3d}: PROGRAM ENDED!'.format(diapason+2))
        print('TOTAL TIME: {:0>5.5f} sec'.format(time()-t0))

if __name__ == '__main__':
    main()
    


