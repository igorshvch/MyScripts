from multiprocessing import (
    Manager, Process, Queue, Lock, freeze_support, current_process
)
import importlib

__version__='0.0.1'

#CONTENT=======================================================================

def init_register():
    global GLOBS
    GLOBS = Manager().dict()
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
    #    'DB':
    #        'DivActs':None,
    #        'TLI':None
    #}

def init_save_paths(register):
    global SAVE_OPTIONS
    SAVE_OPTIONS = {
        'RootCommon':register['root_struct']['Common'],
        'RootTemp':register['root_struct']['TEMP'],
        'ProjStatData':register['proj_struct']['StatData'],
        'ProjRes':register['proj_struct']['Results'],
        'ProjConcls':register['proj_struct']['Conclusions'],
        'ProjTemp':register['proj_struct']['TEMP']
    }

def interface():
    while True:
        print('Init GLOBS path register')
        init_register()

        breaker = False

        if breaker:
            break


#TESTS=========================================================================
if __name__ == '__main__':
    interface()