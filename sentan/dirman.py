import time

__version__ = '0.5.1'

###Content=====================================================================
TODAY = time.strftime(r'%Y-%m-%d')

def create_and_register_root_struct(file_struct):
    for key in file_struct:
        file_struct[key].mkdir(parents=True, exist_ok=True)
    print('Root file structure created:')
    for val in file_struct.values():
        print('\t'+str(val))

def load_to_register(func):
    def wrapper(*args):
        func(*args)
        args[0].register.update(args[0].ps)
    return wrapper

class Registrator():
    project_inner_dirs = [
        '_TEMP',
        '01_Conclusions',
        '02_RawText',
        '03_ActsBase',
        '04_StatData',
        '05_Results',
    ]

    def __init__(self, register):
        self.fs = register['root_struct']
        self.ps ={
            'proj_struct' : {},
            'proj_path' : None,
            'proj_name' : None,
            'old' : {}
        }
        self.register = register
        self.register_old_projs()
    
    @load_to_register
    def create_project(self):
        if self.ps['proj_name']:
            print(
                'ERROR: YOU NEED TO CLOSE CURRENT'
                +' PROJECT BEFORE CREATING NEW ONE'
            )
            return 0
        new_pj = input('Type name of the project to create: ')
        self.ps['proj_name'] = TODAY+'_'+new_pj
        print('Current project full name is:\n\t' + self.ps['proj_name'])
        self.ps['proj_path'] = (
            self.fs['Projects'].joinpath(self.ps['proj_name'])
        )
        self.ps['proj_path'].mkdir()
        for folder in Registrator.project_inner_dirs:
            self.ps['proj_struct'][folder.strip('0123456789_')] = (
                self.ps['proj_path'].joinpath(folder)
            )
            self.ps['proj_struct'][folder.strip('0123456789_')].mkdir()
    
    @load_to_register
    def register_old_projs(self):
        self.ps['old'] = {
            p.name:p
            for p in self.fs['Projects'].iterdir()
            if p.is_dir()
        }
        if self.ps['proj_name']:
            self.ps['old'].pop(self.ps['proj_name'])
    
    @load_to_register
    def swith_to_project(self, old_pj_name=None):
        if not self.ps['old']:
            self.register_old_projs()
        if not old_pj_name:
            print('Exist projects:\n')
            for pj in sorted(self.ps['old'].keys(), reverse=True):
                print('\t'+pj)
            old_pj = input('Type full name of the project to swith: ')
        else:
            old_pj = old_pj_name
        self.ps['proj_name'] = old_pj
        self.ps['proj_path'] = (
            self.fs['Projects'].joinpath(self.ps['proj_name'])
        )
        self.ps['proj_struct'] = {
            p.name.strip('0123456789_'):p for p in
            self.ps['proj_path'].iterdir()
            if p.is_dir()
        }
        self.register_old_projs()
    
    @load_to_register
    def close_current_project(self):
        if not self.ps['proj_name']:
            print('ERORR: NO CURRENT PROJECT TO CLOSE!')
            return 0
        pj_name = self.ps['proj_name']
        self.ps.update(
            {
                'proj_struct' : {},
                'proj_path' : None,
                'proj_name' : None,
            }
        )
        self.register_old_projs()
        print('Current project \'{}\' was succeffuly closed!'.format(pj_name))


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