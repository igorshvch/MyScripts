import pathlib as pthl
import time
from sentan import mysqlite
from sentan.lowlevel import rwtool
from sentan.gui.dialogs import ffp, fdp

__version__ = '0.5'

###Content=====================================================================
DIR_STRUCT_ROOT = {
    'Root': (pthl.Path().home().joinpath('TextProcessing')),
    'RawText': (pthl.Path().home().joinpath('TextProcessing','RawText')),
    'Projects': (pthl.Path().home().joinpath('TextProcessing','Projects')),
    'TEMP':(pthl.Path().home().joinpath('TextProcessing', '_TEMP')),
    'Common':(pthl.Path().home().joinpath('TextProcessing', 'CommonData'))
}

TODAY = time.strftime(r'%Y-%m-%d')

def create_and_register_root_struct(register):
    if not pthl.Path().home().joinpath('TextProcessing').exists():
        register['root_struct']={}
        for key in DIR_STRUCT_ROOT:
            path = DIR_STRUCT_ROOT[key]
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            register['root_struct'][key] = path
    else:
        print('root_struct already exists!')

def create_project_struct(register, project_name, dir_struct=DIR_STRUCT_ROOT):
    inner_dirs = ['ActsBase', 'StatData', 'Conclusions', 'Results', '_TEMP']
    project_path = dir_struct['Projects'].joinpath(project_name)
    register['proj_path'] = project_path
    paths_to_inner_dirs = [project_path.joinpath(dr) for dr in inner_dirs]
    for p in paths_to_inner_dirs:
        p.mkdir(parents=True, exist_ok=True)
        register['proj_struct'][p.name.strip('_')] = (p)
    register_db_paths(register, register['proj_struct']['ActsBase'])

def register_db_paths(register, path):
    register['DB'] = {}
    register['DB']['DivActs'] = path.joinpath('DivActs')
    register['DB']['TLI'] = path.joinpath('TLI')

    #= mysqlite.DataBase(
    #    dir_name= path,
    #    base_name='DivActs',
    #    tb_name=tb_name
    #)
    #if not tb_name:
    #    shared.DB['DivActs'].create_tabel(
    #        'DivActs',
    #        (
    #            ('id', 'INTEGER', 'PRIMARY KEY'),
    #           ('COURT', 'TEXT'),
    #           ('REQ', 'TEXT'),
    #           ('ACT', 'TEXT')
    #       )
    #   )
    #shared.DB['TLI'] = mysqlite.DataBase(
    #   dir_name=path,
    #   base_name='TLI',
    #    tb_name=tb_name
    #)
    #if not tb_name:
    #    shared.DB['TLI'].create_tabel(
    #        'TLI',
    #        (
    #            ('id', 'INTEGER', 'PRIMARY KEY'),
    #            ('COURT', 'TEXT'),
    #            ('REQ', 'TEXT'),
    #            ('RAWPARS', 'TEXT'),
    #            ('DIV', 'TEXT'),
    #            ('LEM', 'TEXT'),
    #            ('INDXACT', 'TEXT'),
    #            ('INDXPAR', 'TEXT')
    #        )
    #    )

def register_old_projects(register):
    old_pjs = [p for p in DIR_STRUCT_ROOT['Projects'].iterdir() if p.is_dir()]
    register['old'] = {p.name:p for p in old_pjs}
        
def register_project(register):
    register['proj_name'] = None
    register['proj_path'] = None
    register['proj_struct'] = {}
    proj_name = input('type project name======>')
    if not proj_name:
        raise TypeError('No project name was typed!')
    register['proj_name'] = TODAY+'_'+proj_name
    create_project_struct(register, project_name=proj_name)
    register_old_projects(register)

def close_current_project(register):
    pn = register['proj_name']
    for name in 'proj_path', 'proj_struct', 'proj_name':
        register[name] = None
    if register['DB']:
        register['DB'] = {key:None for key in register['DB']}
    if pn:
        print('Current project {} was succeffuly closed!'.format(pn))

def swith_to_project(register, another_project_name):
    olds = register['old']
    close_current_project(register)
    if another_project_name in olds:
        proj_path = olds[another_project_name]
        register['proj_name'] = another_project_name
        register['proj_path'] = proj_path
        register['proj_struct'] = {
            p.name.strip('_'):p for p in proj_path.iterdir() if p.is_dir()
        }
    else:
        raise KeyError('No "{}" key'.format(another_project_name))
    register_old_projects(register)
    register_db_paths(register, register['proj_struct']['ActsBase'])
    print('Current project is: {}'.format(another_project_name))

def init_project(register):
    create_and_register_root_struct(register)
    register_old_projects(register)


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