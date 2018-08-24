import os
from multiprocessing import Process, Lock
from time import sleep

def whoami(label, lock):
    sleep(3)
    msg = '%s: name:%s, pid:%s'
    with lock:
        print(msg % (label, __name__, os.getpid()))

if __name__ == '__main__':
    lock = Lock()
    whoami('function call', lock)
    p = Process(target=whoami, args=('spawned child', lock))
    p.start()
    p.join()
    #for i in range(5):
        #Process(target=whoami, args=(('run process %s' % i), lock)).start()
    holder = [
        Process(target=whoami, args=(('run process %s' % i), lock))
        for i in range(5)
    ]
    for pr in holder: pr.start()
    for pr in holder: pr.join()
    for pr in holder: pr.terminate()
    with lock:
        print('Main process exit.')