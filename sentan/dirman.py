import pathlib as pthl
import time

__version__ = '0.2'

###Content=====================================================================
DIR_STRUCT = {
    'Root': (pthl.Path().home().joinpath('TextProcessing')),
    'RawText': (pthl.Path().home().joinpath('TextProcessing','RawText')),
    'Concls': (pthl.Path().home().joinpath('TextProcessing', 'Conclusions')),
    'StatData': (pthl.Path().home().joinpath('TextProcessing', 'StatData')),
    'Res': (pthl.Path().home().joinpath('TextProcessing', 'Results')),
    'DivActs': (pthl.Path().home().joinpath('TextProcessing', 'DivActs')),
    'TLI':(pthl.Path().home().joinpath('TextProcessing', 'TLI')),
    'TEMP':(pthl.Path().home().joinpath('TextProcessing', '_TEMP'))
}
TODAY = time.strftime(r'%Y-%m-%d')

def create_dirs(dir_struct, sub_dir=''):
    paths = []
    for key in dir_struct.keys():
        if key != 'MainRoot':
            path = dir_struct[key].joinpath(sub_dir)
            path.mkdir(parents=True, exist_ok=True)
            paths.append(str(path))
    print('Created directories:')
    for strg in sorted(paths):
        print('\t'+strg)
    
def create_one_dir(dir_name, date=False, main_dir=None):
    if main_dir in DIR_STRUCT and date:
        path = DIR_STRUCT[main_dir].joinpath(TODAY, dir_name)
    elif main_dir in DIR_STRUCT and not date:
        path = DIR_STRUCT[main_dir].joinpath(dir_name)
    elif main_dir not in DIR_STRUCT and date:
        path = DIR_STRUCT['Root'].joinpath(TODAY, dir_name)
    elif main_dir not in DIR_STRUCT and not date:
        path = DIR_STRUCT['Root'].joinpath(dir_name)
    else:
        raise TypeError(
            'Fail to process arguments!'
            +'\ndir_name={}, date={}, main_dir{}'\
            .format(dir_name, date, main_dir)
        )
    path.mkdir(parents=True, exist_ok=True)
    print('Created directory:')
    print('\t'+str(path))


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
        elif sys.argv[1] == '-create':
            create_dirs(DIR_STRUCT)
        elif sys.argv[1] == '-create_sd':
            create_dirs(DIR_STRUCT, sub_dir=sys.argv[2])
        else:
            print('Not implemented!')
    except IndexError:
        print('Mode var wasn\'t passed!')