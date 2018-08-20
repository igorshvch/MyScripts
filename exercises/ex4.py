from concurrent.futures import (
    ProcessPoolExecutor as PPE,
    wait as cf_wait
)
import os
from time import time, sleep

def f1(x):
    t0 = time()
    sleep(5)
    pid = str(os.getpid())
    print(pid+'\ntime end: {:3.2f}'.format(time()-t0))
    return str(x**2)
def f2(x):
    t0 = time()
    sleep(2)
    pid = str(os.getpid())
    print(pid+'\ntime end: {:3.2f}'.format(time()-t0))
    return str(x**3)
def f3(x):
    t0 = time()
    sleep(10)
    pid = str(os.getpid())
    print(pid+'\ntime end: {:3.2f}'.format(time()-t0))
    return str(x**4)
def f4(x):
    t0 = time()
    sleep(1)
    pid = str(os.getpid())
    print(pid+'\ntime end: {:3.2f}'.format(time()-t0))
    return str(x**5)

def mpr(args):
    t0 = time()
    results = []
    with PPE() as ex:
        r1 = ex.submit(f1, args)
        r2 = ex.submit(f2, args)
        r3 = ex.submit(f3, args)
        r4 = ex.submit(f4, args)
    print('time end: {:3.2f}'.format(time()-t0))
    print(r1.result(), r2.result(), r3.result(), r4.result())

if __name__ == '__main__':
    mpr(2)