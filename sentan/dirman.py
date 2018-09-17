import pathlib as pthl
import time
from . import shared
from . import mysqlite
from .lowlevel import rwtool

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

def create_and_register_root_struct():
    if not pthl.Path().home().joinpath('TextProcessing').exists():
        shared.GLOBS['root_struct']={}
        for key in DIR_STRUCT_ROOT:
            path = DIR_STRUCT_ROOT[key]
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            shared.GLOBS['root_struct'][key] = path
    else:
        print('root_struct already exists!')

def create_project_struct(project_name, dir_struct=DIR_STRUCT_ROOT):
    inner_dirs = ['ActsBase', 'StatData', 'Conclusions', 'Results', '_TEMP']
    project_path = dir_struct['Projects'].joinpath(TODAY+'_'+project_name)
    shared.GLOBS['proj_path'] = project_path
    paths_to_inner_dirs = [project_path.joinpath(dr) for dr in inner_dirs]
    for p in paths_to_inner_dirs:
        p.mkdir(parents=True, exist_ok=True)
        shared.GLOBS['proj_struct'][p.name.strip('_')] = (p)
    register_db_connection(shared.GLOBS['proj_struct']['ActsBase'])

def register_db_connection(path, tb_name=False):
    shared.DB['DivActs'] = mysqlite.DataBase(
        dir_name= path,
        base_name='DivActs',
        tb_name=tb_name
    )
    if not tb_name:
        shared.DB['DivActs'].create_tabel(
            'DivActs',
            (
                ('id', 'INTEGER', 'PRIMARY KEY'),
                ('COURT', 'TEXT'),
                ('REQ', 'TEXT'),
                ('ACT', 'TEXT')
            )
        )
    shared.DB['TLI'] = mysqlite.DataBase(
        dir_name=path,
        base_name='TLI',
        tb_name=tb_name
    )
    if not tb_name:
        shared.DB['TLI'].create_tabel(
            'TLI',
            (
                ('id', 'INTEGER', 'PRIMARY KEY'),
                ('COURT', 'TEXT'),
                ('REQ', 'TEXT'),
                ('RAWPARS', 'TEXT'),
                ('DIV', 'TEXT'),
                ('LEM', 'TEXT'),
                ('INDXACT', 'TEXT'),
                ('INDXPAR', 'TEXT')
            )
        )

def register_old_projects():
    old_pjs = [p for p in DIR_STRUCT_ROOT['Projects'].iterdir() if p.is_dir()]
    shared.GLOBS['old'] = {p.name:p for p in old_pjs}
        
def register_project(new_pj=True):
    shared.GLOBS['proj_name'] = None
    shared.GLOBS['proj_path'] = None
    shared.GLOBS['proj_struct'] = {}
    if new_pj:
        proj_name = input('type project name======>')
        if not proj_name:
            raise TypeError('No name was typed!')
        shared.GLOBS['proj_name'] = proj_name
        create_project_struct(project_name=proj_name)
    else:
        print('None new project was created and registered!')
    register_old_projects()

def close_current_project():
    pn = shared.GLOBS['proj_name']
    for name in 'proj_path', 'proj_struct', 'proj_name':
        shared.GLOBS[name] = None
    if shared.DB:
        shared.DB = {key:None for key in shared.DB}
    if pn:
        print('Current project {} was succeffuly closed!'.format(pn))

def swith_to_another_project(another_project_name):
    olds = shared.GLOBS['old']
    close_current_project()
    if another_project_name in olds:
        proj_path = olds[another_project_name]
        shared.GLOBS['proj_name'] = another_project_name
        shared.GLOBS['proj_path'] = proj_path
        shared.GLOBS['proj_struct'] = {
            p.name.strip('_'):p for p in proj_path.iterdir() if p.is_dir()
        }
    else:
        raise KeyError('No "{}" key'.format(another_project_name))
    register_old_projects()
    register_db_connection(shared.GLOBS['proj_struct']['ActsBase'], tb_name=True)
    print('Current project is: {}'.format(another_project_name))

def store_global_data_for_subproc_access():
    rwtool.save(shared.GLOBS, 'GLOBS', to='RootTemp')


create_and_register_root_struct()
register_project(new_pj=False)
register_old_projects()


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