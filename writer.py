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
        str(today())+'__'+file_name_string
        if prefix=='custome'
        else (
            str(today())+'__'
            +prefix+'_'
            +file_name_string
        )
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
        print(
            'File \'{}\' was written in mode \'{}\''.format(
                file_name_string, mode
            )
        )

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

def find_files(top_dir, suffix=''):
        holder = []
        def inner_func(top_dir, suffix):
            p = pthl.Path(top_dir)
            nonlocal holder
            store = [path_obj for path_obj in p.iterdir()]
            for path_obj in store:
                if path_obj.is_dir():
                    inner_func(path_obj, suffix)
                elif path_obj.suffix == suffix:
                    holder.append(path_obj)
        inner_func(top_dir, suffix)
        return sorted(holder)
