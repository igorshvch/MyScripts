import pathlib as pthl
import random as rd
from writer import find_files

class FileRand():
    def __init__(self):
        self.files_dict = None
        self.load_dir_name = None
        self.keys = None
        self.prefs = None
    
    def constr_dict(self, full_path):
        self.load_dir_name = full_path
        files = find_files(full_path, suffix='.mp3')
        self.files_dict = {key:key.name for key in files}
        self.keys = set(self.files_dict.keys())
        return len(self.keys)
    
    def pref_cr(self, dict_len):
        pref_len = len(str(dict_len))
        holder = []
        for i in range(dict_len):
            stc = str(i)
            pref = '0'*(pref_len+1-len(stc))+stc+'_'
            holder.append(pref)
        self.prefs = holder
    
    def rand_prefs(self):
        rd.shuffle(self.prefs)
    
    def add_prefs(self):
        if not self.prefs:
            return 'No prefs were found! Create prefs!'
        for key in self.keys:
            name = self.files_dict[key]
            name = self.prefs.pop()+name
            self.files_dict[key] = name
        
    def rename_files(self):
        for i in self.keys:
            i.rename(i.parent.joinpath(self.files_dict[i]))
    
    def replace_files(self, save_dir_name):
        for i in self.keys:
            i.replace(i.parents[1].joinpath(save_dir_name, self.files_dict[i]))

    def names_clean(self, chars_quant):
        self.constr_dict(self.load_dir_name)
        for i in self.keys:
            i.rename(i.parent.joinpath(self.files_dict[i][chars_quant:]))
    
    def process(self, full_load_dir_name, verbose=False):
        self.constr_dict(full_load_dir_name)
        self.pref_cr(len(self.keys))
        self.rand_prefs()
        self.add_prefs()
        if verbose:
            print(self.files_dict)
        self.rename_files()
    
    def re_rename(self, chars_quant):
        self.names_clean(chars_quant)
        self.process(self.load_dir_name)
    



    




            
        

