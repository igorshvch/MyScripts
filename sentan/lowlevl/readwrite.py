import csv
import json
import pathlib as pthl
import pickle
import shelve
from time import time


class ReadWriteTool():
    '''
    Class provides API to reading and writing options.
    '''
    def __init__(self, enc='cp1251'):
        self.enc=enc
        print('RWT class created')

    def collect_exist_file_paths(self, top_dir, suffix=''):
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
    
    def map_num_to_concl(self, path_holder, lngth=None):
        dct = {}
        for p in path_holder:
            with open(str(p), mode='r') as file:
                text = file.read()
            if lngth:
                dct[p.stem] = text[:lngth]
            else:
                dct[p.stem] = text
        return dct
    
    def map_concl_to_num(self, path_holder, lngth=None):
        dct = {}
        for p in path_holder:
            with open(str(p), mode='r') as file:
                text = file.read()
            if lngth:
                dct[text[:lngth]] = p.stem
            else:
                dct[text] = p.stem
        return dct

    def load_text(self, path):
        with open(path, mode='r', encoding=self.enc) as fle:
            text = fle.read()
        return text
        
    def iterate_text_loading(self, top_dir):
        '''
        Return generator object iterating over all text files
        in the top_dir subdirectories.
        '''
        paths = self.collect_exist_file_paths(top_dir, suffix='.txt')
        print(
            len(paths),
            'The first file in the queue is:\n{}'.format(paths[0]),
            sep=' ### '
        )
        return (self.load_text(path) for path in paths)
    
    def write_text(self, text, path):
        with open(path, mode='w', encoding=self.enc) as fle:
                fle.write(text)
    
    def create_writing_paths(self, strt, stp, path, pref_len, suffix=''):
        p = pthl.Path(path)
        names = [
            '0'*(pref_len+1-len(str(i)))+str(i)
            for i in range(strt, stp, 1)
        ]
        file_paths = [
            p.joinpath(i).with_suffix(suffix)
            for i in names
        ]
        return sorted(file_paths)
    
    def write_text_to_csv(self,
                          file_name,
                          iter_txt_holder,
                          header=None,
                          zero_string=None):
        with open(file_name, mode='w', newline='', encoding=self.enc) as fle:
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

    def write_pickle(self, py_obj, path):
        with open(path, mode='wb') as file_name:
            pickle.dump(py_obj,
                        file_name,
                        protocol=pickle.HIGHEST_PROTOCOL)
        
    def load_pickle(self, path):
        with open(path, 'rb') as fle:
            data = pickle.load(fle)
        return data
    
    def iterate_pickle_loading(self, top_dir, suffix=''):
        '''
        Return generator object iterating over all binary files
        in the top_dir subdirectories.
        '''
        paths = self.collect_exist_file_paths(top_dir, suffix=suffix)
        print(
            len(paths),
            'The first file in the queue is:\n{}'.format(paths[0]),
            sep='### '
        )
        return (self.load_pickle(path) for path in paths)
    
    def iterate_shelve_reading(self, path):
        with shelve.open(str(path), flag='r') as db:
            keys = sorted(db.keys())
            for i in keys:
                yield db[i]


class DirManager(ReadWriteTool):
    def __init__(self, enc='cp1251'):
        ReadWriteTool.__init__(self, enc=enc)
        self.dir_struct = {
                'MainRoot': (
                    pthl.Path().home().joinpath('TextProcessing')
                ),
                'Raw_text': (
                    pthl.Path().home().joinpath('TextProcessing','RawText')
                ),
                'DivTok': (
                    pthl.Path().home().joinpath('TextProcessing','DivToks')
                ),
                'DivTokPars': (
                    pthl.Path().home().joinpath('TextProcessing','DivTokPars')
                ),
                'Norm1': (
                    pthl.Path().home().joinpath('TextProcessing','Norm1')
                ),
                'Norm1Pars': (
                    pthl.Path().home().joinpath('TextProcessing','Norm1Pars')
                ),
                'Concls': (
                    pthl.Path().home().joinpath('TextProcessing', 'Conclusions')
                ),
                'StatData': (
                    pthl.Path().home().joinpath('TextProcessing', 'StatData')
                ),
                'Results': (
                    pthl.Path().home().joinpath('TextProcessing', 'Results')
                ),
                'DivActs': (
                    pthl.Path().home().joinpath('TextProcessing', 'DivActs')
                ),
                'ActsInfo': (
                    pthl.Path().home().joinpath('TextProcessing', 'ActsInfo')
                ),
                'ParsInfo': (
                    pthl.Path().home().joinpath('TextProcessing', 'ParsInfo')
                )
            }
    
    def create_dirs(self, dir_struct, sub_dir=''):
        paths = []
        for key in dir_struct.keys():
            if key != 'MainRoot':
                path = dir_struct[key].joinpath(sub_dir)
                path.mkdir(parents=True, exist_ok=True)
                paths.append(str(path))
        print('Created directories:')
        for strg in sorted(paths):
            print('\t'+strg)
    
    def create_dir(self, dir_name, full_path_to_dir):
        path = pthl.Path().joinpath(full_path_to_dir)
        path = path.joinpath(dir_name)
        path.mkdir(parents=True, exist_ok=True)
        print('Created directory:')
        print('\t'+str(path))