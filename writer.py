import pathlib as pthl

path = pthl.Path().home().joinpath('MYWRITE')
path.mkdir(parents=True, exist_ok=True)

def writer(iterable_object,
           file_name_string,
           prefix='custome',
           mode='a',
           verbose=True):
    '''
    Accepts iterable objects containing strings
    or a string object itselfe
    
    '''
    import datetime
    today = datetime.date.today

    file_name = (
        str(today())+'_'+file_name_string
        if prefix=='custome'
        else prefix+'_'+file_name_string
    )
    inner_path = path.joinpath(file_name)
    inner_path = inner_path.with_suffix('.txt')
    
    if type(iterable_object) != str:
        with open(inner_path, mode=mode) as file:
            for i in iterable_object:
                i = str(i) + '\n'
                file.write(i)
    else:
        with open(inner_path, mode=mode) as file:
            file.write(iterable_object)
    if verbose:
        print('OK')

def recode(root_path, paths, encodings, verbose=True):
    root_path = pthl.Path(root_path)
    for path in paths:
        with open(
            root_path.joinpath(path), mode='r', encoding=encodings[0]
        ) as file:
            text = file.read()
        with open(
            root_path.joinpath(path), mode='w', encoding=encodings[1]
        ) as file:
            file.write(text)
        if verbose:
            print("File '{}' recoded!".format(path))