import pathlib as pthl
import time
from . import shared

__version__ = '0.3'

###Content=====================================================================
DIR_STRUCT_ROOT = {
    'Root': (pthl.Path().home().joinpath('TextProcessing')),
    'RawText': (pthl.Path().home().joinpath('TextProcessing','RawText')),
    'Projects': (pthl.Path().home().joinpath('TextProcessing','Projects')),
    'TEMP':(pthl.Path().home().joinpath('TextProcessing', '_TEMP')),
    'Common':(pthl.Path().home().joinpath('TextProcessing', 'CommonData'))
}

TODAY = time.strftime(r'%Y-%m-%d')

def create_project_dirs(project_name, dir_struct=DIR_STRUCT_ROOT):
    inner_dirs = ['ActsBase', 'StatData', 'Conclusions', 'Results', '_TEMP']
    project_path = dir_struct['Projects'].joinpath(TODAY+'_'+project_name)
    shared.GLOBS['proj_path'] = project_path
    paths_to_inner_dirs = [project_path.joinpath(dr) for dr in inner_dirs]
    for p in paths_to_inner_dirs:
        p.mkdir(parents=True, exist_ok=True)
        shared.GLOBS['proj_struct'][p.name] = (p)

def create_and_register_root_dirs():
    shared.GLOBS['root_struct']={}
    for key in DIR_STRUCT_ROOT:
        path = DIR_STRUCT_ROOT[key]
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        shared.GLOBS['root_struct'][key] = path

def create_and_register_new_project():
    shared.GLOBS['proj_struct']={}
    if 'proj_name' not in shared.GLOBS:
        proj_name = input('type project name======>')
        shared.GLOBS['proj_name'] = proj_name
        create_project_dirs(project_name=proj_name)
    else:
        print('There is an opened project already: {: >40s}'.format(shared.GLOBS['proj_name']))
        print('You need to end current session to create new project!')

create_and_register_root_dirs()
create_and_register_new_project()


###Testing=====================================================================
if __name__ == "__main__":
    import sys
    try:
        sys.argv[1]
        if sys.argv[1] == '-v':
            print('Module name: {}'.format(sys.argv[0]))
            print('Version info:', __version__)
        elif sys.argv[1] == '-t':
            print('Testing mode!')
            print('Not implemented!')
        else:
            print('Not implemented!')
    except IndexError:
        print('Mode var wasn\'t passed!')