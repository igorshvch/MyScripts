from multiprocessing import (
    Manager, Process, Queue, Lock, freeze_support, current_process
)
import importlib
import pathlib as pthl

from sentan import dirman

__version__='0.0.1'

#CONTENT=======================================================================
LOCK = Lock()

ROOT_STRUCT = {
    'Root': (pthl.Path().home().joinpath('TextProcessing')),
    'RawText': (pthl.Path().home().joinpath('TextProcessing','RawText')),
    'Projects': (pthl.Path().home().joinpath('TextProcessing','Projects')),
    'TEMP':(pthl.Path().home().joinpath('TextProcessing', '_TEMP')),
    'Common':(pthl.Path().home().joinpath('TextProcessing', 'CommonData'))
}
COMMANDS = {
    '0': 'print GLOBS',
    '1': 'create new project',
    '2': 'switch to another project',
    '3': 'close project',
    '4': 'make databases',
    '5': 'start elly',
    '6': '',
    '7': 'list all exist projects',
    '8': 'print commands',
    '9': 'end program',
}

def init_register():
    global GLOBS
    GLOBS = Manager().dict()
    GLOBS['root_struct'] = ROOT_STRUCT
    #GLOBS = {
    #    'root_struct': {
    #        'Root':None,
    #        'RawText':None,
    #        'Projects':None,
    #        'TEMP':None,
    #        'Common':None
    #    },
    #    'proj_struct': {
    #        'ActsBase':None,
    #        'TEMP':None,
    #        'Conclusions':None,
    #        'StatData':None,
    #        'Results':None
    #    },
    #    'proj_path':None,
    #    'proj_name':None,
    #    'old':None,
    #}

def make_dtl(load_dir_name):
    if 'proj_name' not in GLOBS:
        return 'ERROR! There is no current project! Plese, chose one'
    dtl = importlib.import_module('sentan.textproc.divtoklem')
    pr = Process(target = dtl.main, args=(GLOBS, load_dir_name))
    pr.start()
    pr.join()

def start_elly(cpus, start):
    if 'proj_name' not in GLOBS:
        return 'ERROR! There is no current project! Plese, chose one'
    elly = importlib.import_module('sentan.multiproc.elly_mp_n')
    pr = Process(target=elly.main, args=(GLOBS, LOCK, cpus, start))
    pr.start()
    pr.join()

def print_commands(inden=''):
    print('Commands:')
    for key in sorted(COMMANDS.keys()):
        if COMMANDS[key]:
            print(inden+'{0} - {1}'.format(key, COMMANDS[key]))

def interface():
    print('Init GLOBS path register')
    init_register()
    if not GLOBS['root_struct']['Root'].exists():
        dirman.create_and_register_root_struct(GLOBS['root_struct'])
    reg = dirman.Registrator(GLOBS, GLOBS['root_struct'])
    print_commands()
    while True:
        breaker = input('\n==>Type command here: ')
        if breaker == '0':
            print('==Execute: \'{}\'==\n'.format(COMMANDS[breaker]))
            for item in GLOBS.items():
                print(item[0], item[1])
        elif breaker == '1':
            print('==Execute: \'{}\'==\n'.format(COMMANDS[breaker]))
            reg.create_project()
        elif breaker == '2':
            print('==Execute: \'{}\'==\n'.format(COMMANDS[breaker]))
            reg.swith_to_project()
        elif breaker == '3':
            print('==Execute: \'{}\'==\n'.format(COMMANDS[breaker]))
            reg.close_current_project()
        elif breaker == '4':
            print('==Execute: \'{}\'==\n'.format(COMMANDS[breaker]))
            load_dir_name = input('Type raw acts folder: ')
            make_dtl(load_dir_name)
            print_commands('\t')
        elif breaker == '5':
            print('==Execute: \'{}\'==\n'.format(COMMANDS[breaker]))
            cpus, start = input(
                'Type number of workerks and start value here: '
            ).split(' ')
            start_elly(cpus, start)
            print_commands('\t')
        elif breaker == '7':
            print('==Execute: \'{}\'==\n'.format(COMMANDS[breaker]))
            print('All exist projects:')
            if GLOBS['proj_name']:
                print('\t\t'+str(GLOBS['proj_name']))
            for pj in GLOBS['old'].keys():
                print('\t\t'+pj)
        elif breaker == '8':
            print('==Execute: \'{}\'==\n'.format(COMMANDS[breaker]))
            print_commands('\t')
        elif breaker == '9':
            print('==Execute: \'{}\'==\n'.format(COMMANDS[breaker]))
            print('Program ends here!')
            break
        else:
            print('Incorrect value!')


#TESTS=========================================================================
if __name__ == '__main__':
    interface()