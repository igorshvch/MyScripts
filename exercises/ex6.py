from multiprocessing import Process, Pipe, Lock
import os

def show_pipe_end(pipe, lock):
    pid = os.getpid()
    data = pipe.recv()
    pipe.send(['VERTICAL+{}'.format(pid)])
    pipe.close()
    with lock:
        print('process {}, data: {}'.format(pid, data))

if __name__ == '__main__':
    print('Parent starts')
    lock = Lock()
    holder=[]
    (parent_E1, child_E1) = Pipe()
    (parent_E2, child_E2) = Pipe()
    (parent_E3, child_E3) = Pipe()
    (parent_E4, child_E4) = Pipe()
    data_store = [
        'INFORMATION1', 'INFORMATION2', 'INFORMATION3', 'INFORMATION4'
    ]
    while data_store:
        holder = []
        chunk = data_store.pop()
        parent_E1.send(chunk)
        parent_E2.send(chunk)
        parent_E3.send(chunk)
        parent_E4.send(chunk)
        for i in child_E1, child_E2, child_E3, child_E4:
            p = Process(target=show_pipe_end, args=[i, lock])
            p.start()
            holder.append(p)
    for p in holder: p.join()
    d1, d2, d3, d4 = (
        parent_E1.recv(), parent_E2.recv(), parent_E3.recv(), parent_E4.recv()
    )
    parent_E1.close()
    parent_E2.close()
    parent_E3.close()
    parent_E4.close()
    with lock:
        for i in (d1,d2,d3,d4):
            print('Stored data: {}'.format(i))
        print('Parent ends')
