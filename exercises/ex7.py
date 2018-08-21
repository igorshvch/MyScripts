from multiprocessing import Process, Queue, Lock
import os
import random as rd

alph = [chr(i) for i in range(1040, 1040+32, 1)]
alph2 = [chr(i) for i in range(1072, 1072+32, 1)]

def generate_name():
    return ''.join(rd.sample(alph, 4))

def estimate(data):
    return ''.join([str(i)*2 for i in data])


def worker(data, lock, store):
    name = generate_name()
    est_data = estimate(data)
    pid = os.getpid()
    store.put({name: est_data}, block=False)
    with lock:
        print(
            'PROCESS: {}, NAME: {}, DATA: {:4s}'.format(
                name, pid, est_data
            )
        )

if __name__ == '__main__':
    global_pid = os.getpid()
    global_store = Queue()
    lock = Lock()
    holder = []
    print('Parent started, PID: {}'.format(global_pid))
    for i in range(3):
        data = ''.join(rd.sample(alph2, 7))
        p = Process(target=worker, args=(data, lock, global_store))
        p.start()
        holder.append(p)
    for i in holder: i.join()
    with lock:
        print('Parent ended, PID: {}'.format(global_pid))
        while not global_store.empty():
            print(global_store.get())



