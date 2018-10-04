import csv
import pathlib as pthl
from time import time


__version__ = '0.2.8'

###Content=====================================================================
GLOB_ENC = 'cp1251'

def collect_exist_files(top_dir, suffix=''):
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

def collect_exist_dirs(top_dir):
    holder = []
    def inner_func(top_dir):
        p = pthl.Path(top_dir)
        nonlocal holder
        store = [path_obj for path_obj in p.iterdir() if path_obj.is_dir()]
        for path_obj in store:
            holder.append(path_obj)
            inner_func(path_obj)
    inner_func(top_dir)
    return sorted(holder)

def collect_exist_files_and_dirs(top_dir, suffix=''):
    holder = []
    def inner_func(top_dir, suffix):
        p = pthl.Path(top_dir)
        nonlocal holder
        store = [path_obj for path_obj in p.iterdir()]
        for path_obj in store:
            if path_obj.is_dir():
                holder.append(path_obj)
                inner_func(path_obj, suffix)
            elif path_obj.suffix == suffix:
                holder.append(path_obj)
    inner_func(top_dir, suffix)
    return sorted(holder)

def read_text(path):
    with open(path, mode='r', encoding=GLOB_ENC) as fle:
        text = fle.read()
    return text

def write_text(text, path):
    if path[-4:] != '.txt':
        path += '.txt'
    with open(path, mode='w', encoding=GLOB_ENC) as fle:
        fle.write(text)

def write_iterable_to_csv(full_path,
                          iter_txt_holder,
                          header=None,
                          zero_string=None):
    with open(full_path, mode='w', newline='', encoding=GLOB_ENC) as fle:
        writer = csv.writer(
            fle,
            delimiter='|',
            quotechar='#',
            quoting=csv.QUOTE_MINIMAL
        )
        if zero_string:
            zero_string = (
                [zero_string] + ['na' for i in range(len(header)-1)]
            )
            assert len(zero_string) == len(header)
            writer.writerow(zero_string)
        if header:
            writer.writerow(header)
        for row in iter_txt_holder:
            writer.writerow(row)

def save_pickle(py_obj, path):
    import pickle
    with open(path, mode='wb') as file_name:
        pickle.dump(py_obj,
                    file_name,
                    protocol=pickle.HIGHEST_PROTOCOL)
        
def load_pickle(path):
    import pickle
    with open(path, 'rb') as fle:
        data = pickle.load(fle)
    return data            

def save_obj(py_obj, name, save_path):
    path = save_path.joinpath(name)
    save_pickle(py_obj, str(path))


###Testing=====================================================================
if __name__ == '__main__':
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
