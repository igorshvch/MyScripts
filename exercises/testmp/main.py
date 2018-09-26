from multiprocessing import (
    Manager, Process, Queue, Lock, freeze_support, current_process
)
import importlib

lock = Lock()

def loader(module, *args, **kwargs):
    importlib.invalidate_caches()
    mod = importlib.import_module(module)
    p = Process(target=mod.worker, args=args, kwargs=kwargs)
    p.start()
    p.join()

def main(message):
    with lock:
        print('\n')
        print('='*96)
        print(__name__)
        print('Run: ', current_process())
    loader('exercises.testmp.one', lock, message=message['p1'])
    loader('exercises.testmp.two', lock, message=message['p2'])

if __name__ == '__main__':
    GLOBS = Manager().dict()
    GLOBS['p1'] = 'path # 1'
    GLOBS['p2'] = 'path # 2'
    with lock:
        print('='*96)
        print('START')
        print('='*96)
        print('GLOBS', type(GLOBS))
        print(GLOBS)
    main(GLOBS)
    print('END!')
    val = input('Type here: ')
    print(val)


