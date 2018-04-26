import pathlib as pthl

path = pthl.Path().home().joinpath('MYWRITE')
path.mkdir(parents=True, exist_ok=True)

def writer(iterable_object, file_name_string, prefix='custome', mode='a'):
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
    print('OK')
