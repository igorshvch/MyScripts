#built-in modules
from collections import Counter
from time import time
#my modules
from sentan.stringbreakers import (
    BGRINR_B, DCTKEY_B, KEYVAL_B, INDEXP_B
)

__version__ = '0.2.1'

###Content=====================================================================
def bigrams_intersection(tokens_list, stpw):
    #Initialize local funcs:
    crbg = create_bigrams
    tokens_wostpw = [token for token in tokens_list if token not in stpw]
    bigrams = crbg(tokens_list)
    cleaned_bigrams = crbg(tokens_wostpw)
    return list(set(bigrams).intersection(cleaned_bigrams))

def clean_txt_and_remove_stpw(par, sep, stpw):
    cleaned = [word for word in par.split(sep) if word not in stpw]
    return ' '.join(cleaned)

def clean_txt_and_remove_stpw_add_bigrams_splitted(lemmed_pars,
                                                   par_len,
                                                   sep_par,
                                                   sep_lems,
                                                   stpw):
    crbg = create_bigrams
    holder_pars = []
    holder_pars_and_bigrs = []
    spl = lemmed_pars.split(sep_par)
    for par in spl:
        if len(par) > par_len:
            par_spl = par.split(sep_lems)
            cleaned = [word for word in par_spl if word not in stpw]
            bigrams = crbg(par_spl)
            cleaned_bigrams = crbg(cleaned)
            ints_bigrams = list(set(bigrams).intersection(cleaned_bigrams))
            holder_pars.append(' '.join(cleaned))
            holder_pars_and_bigrs.append(
                ' '.join(cleaned + ints_bigrams)
            )
        else:
            holder_pars.append('')
            holder_pars_and_bigrs.append('')
    return holder_pars, holder_pars_and_bigrs

def clean_txt_and_remove_stpw_add_intersect_bigrams(par, sep, stpw):
    bgrint = bigrams_intersection
    tokens_list = par.split(sep)
    cleaned = [word for word in tokens_list if word not in stpw]
    cleaned += bgrint(tokens_list, stpw)
    return ' '.join(cleaned)
    
def create_bigrams(tokens_list):
    separator = BGRINR_B
    holder=[]
    holder = [
        separator.join((tokens_list[i-1], tokens_list[i]))
        for i in range(1, len(tokens_list), 1)
    ]
    return holder
    
def create_indexdct_from_tokens_list(tokens_list):
    separator = DCTKEY_B
    index_dct = {word:set() for word in tokens_list}
    counter = 0
    for word in tokens_list:
        index_dct[word].add(counter)
        counter+=1
    index_dct['total'] = [len(tokens_list)]
    for word in tokens_list:
        index_dct['total'+separator+word]=[len(index_dct[word])]
    return index_dct

def indexdct_to_string(index_dct):
    sep_keyval = KEYVAL_B
    sep_index = INDEXP_B
    holder = [
        key+sep_keyval+sep_index.join([str(item) for item in val])
        for key,val in index_dct.items()
    ]
    return holder

def string_to_indexdct(list_of_strings):
    sep_keyval = KEYVAL_B
    sep_index = INDEXP_B
    par_dict = {}
    for string in list_of_strings:
        key, vals = string.split(sep_keyval)
        if 'total' in key:
            par_dict[key] = int(vals)
        else:
            par_dict[key] = set([int(i) for i in vals.split(sep_index)])
    return par_dict

def form_string_numeration(digits_num):
    st = ['{:0>', 'd}']
    form = str(digits_num).join(st)
    return form

def form_string_pattern(char, typ, fields):
    st = ['{:%s>' % char, '%s}' % typ]
    form = str(fields).join(st)
    return form


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