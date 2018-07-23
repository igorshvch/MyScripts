import pathlib as pthl

GLOB_ENC = 'cp1251'
__version__ = 0.1

DIR_STRUCT = {
    'MainRoot': (pthl.Path().home().joinpath('TextProcessing')),
    'RawText': (pthl.Path().home().joinpath('TextProcessing','RawText')),
    'DivToks': (pthl.Path().home().joinpath('TextProcessing','DivToks')),
    'DivTokPars': (pthl.Path().home().joinpath('TextProcessing','DivTokPars')),
    'Norm1': (pthl.Path().home().joinpath('TextProcessing','Norm1')),
    'Norm1Pars': (pthl.Path().home().joinpath('TextProcessing','Norm1Pars')),
    'Concls': (pthl.Path().home().joinpath('TextProcessing', 'Conclusions')),
    'StatData': (pthl.Path().home().joinpath('TextProcessing', 'StatData')),
    'Results': (pthl.Path().home().joinpath('TextProcessing', 'Results')),
    'DivActs': (pthl.Path().home().joinpath('TextProcessing', 'DivActs')),
    'ActsInfo': (pthl.Path().home().joinpath('TextProcessing', 'ActsInfo')),
    'ParsInfo': (pthl.Path().home().joinpath('TextProcessing', 'ParsInfo'))
}

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
    
def create_dir(dir_name, full_path_to_dir):
    path = pthl.Path().joinpath(full_path_to_dir)
    path = path.joinpath(dir_name)
    path.mkdir(parents=True, exist_ok=True)
    print('Created directory:')
    print('\t'+str(path))

###Testing
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
        else:
            print('Not implemented!')
    except IndexError:
        print('Mode var wasn\'t passed!')