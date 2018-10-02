import re
from sentan.gui.dialogs import ffp, fdp, pmb
from sentan.lowlevel import rwtool
from collections import deque

__version__ = '0.1.2'

###Content=====================================================================
ARTICLE = 'Статья [0-9]{3}\. .+'
PARAGRAPH = '[0-9]{1,2}\. .+'
DESICION_INDEX = '[0-9]{1,2}\.[0-9]{1,2}\. .+'
DESICION_PURE = '(?<=[0-9]\.[0-9]\. Вывод из судебной практики: ).*'
DESICION_ALT = (
    '(?<=Вывод из судебной практики: По вопросу о).*(?=у судов нет единой позиции|существует две позиции судов|существует три позиции судов|существует четыре позиции судов|существует пять позиций судов)'
)
DESICION_CLEANED = (
    '(?<=По вопросу о).*(?=у судов нет единой позиции|существует две позиции судов|существует три позиции судов|существует четыре позиции судов|существует пять позиций судов)'
)
POSITION_DIRTY = 'Позиция [1-9]\. .+'
POSITION_PURE = '(?<=Позиция [1-9]\.).*'

PATTERNS = {
    'art': ARTICLE,
    'par' : PARAGRAPH,
    'di' : DESICION_INDEX, 
    'dp': DESICION_PURE,
    'da': DESICION_ALT,
    'dc': DESICION_CLEANED,
    'pd': POSITION_DIRTY,
    'p': POSITION_PURE
}

def process_concls_in_index():
    pmb('Chose file with conclusions!')
    path_to_file = ffp()
    with open(path_to_file, mode='r') as fle:
        text = fle.read()
    spl = [
        line for line in text.split('\n')
        if line and not re.match(PATTERNS['art'], line)
    ]
    spl = deque(spl)
    counter = -1
    flag_del = False
    holder_current_des = None
    holder_p = []
    holder = []
    while spl:
        cursor = spl.popleft()
        counter += 1
        if re.match(PATTERNS['par'], cursor):
            flag_del = False
            holder_p.append(
                re.match(PATTERNS['par'], cursor).group()
            )
        elif re.match(PATTERNS['di'], cursor):
            flag_del = False
            holder.append(
                holder_p[-1] + '#' + re.match(PATTERNS['di'], cursor).group()
            )
        elif re.match(PATTERNS['pd'], cursor):
            if not flag_del:
                flag_del = True
                holder_current_des = holder.pop()
            holder.append(
                holder_current_des + '#' + re.match(PATTERNS['pd'], cursor).group()
            )
        else:
            raise ValueError(
                'String didn\'t match to any mpattern:'
                +'\n{}, ind: {}'.format(cursor, counter)
                +'Flags: Del: {}'\
                .format(flag_del)
            )
    return holder

def func(cleaned_dec, stored_dec):
    from writer import writer
    holder_errors = []
    holder_res = []
    cleaned_dec = [line.replace('Вывод из судебной практики: ', '') for line in cleaned_dec]
    cleaned_dec = [line.replace('\n', '') for line in cleaned_dec]
    spl_cl = [line.split('#')+['',''] for line in cleaned_dec]
    print('spl_cl', len(spl_cl))
    spl_st = [line.split('#')+['',''] for line in stored_dec]
    print('spl_st', len(spl_st))
    dct_com =  {(item[0]+item[1].strip('.')).replace(' ', ''):item[0]+'#'+item[1].strip('.') for item in spl_cl}
    print('dct_com', len(dct_com))
    spl_cl_keys = [(item[0]+item[1].strip('.')).replace(' ', '') for item in spl_cl]
    spl_st_keys = [(item[0]+item[1]).replace(' ', '') for item in spl_st]
    spl_cl = [item[2:] for item in spl_cl]
    spl_st = [item[2:] for item in spl_st]
    writer(spl_st_keys, 'spl_st_keys', mode='w')
    writer(list(dct_com.keys()), 'dct_com_keys', mode='w')
    flag_fisrt = True
    for ind, key in enumerate(dct_com.keys()):
        print(ind, end=' :: ')
        try:
            pos_in_st = spl_st_keys.index(key)
            print ('pos_in_st', pos_in_st,)
        except:
            print('error!', end=' == ')
            holder_errors.append(key)
            continue
        pos_in_cl = spl_cl_keys.index(key)
        if flag_fisrt:
            pos_in_st+=1
        while pos_in_st:
            if flag_fisrt:
                flag_fisrt = False
                pos_in_st-=1
            st = '#'.join([dct_com[key], *spl_st[pos_in_st], *spl_cl[pos_in_cl]])
            holder_res.append(st)
            pos_in_st+=1
            pos_in_cl+=1
            try:
                val = spl_st_keys[pos_in_st]
            except:
                pos_in_st = None
            if val != key:
                break
    holder_res = [re.subn('#{1,4}', '#', line)[0].rstrip('#') for line in holder_res]
    return holder_res, holder_errors

def cleaner(string):
    holder = []
    spl = [line.lstrip('0123456789. ') for line in string.split('#')]
    for line in spl:
        if re.search(PATTERNS['dc'], line):
            holder.append(re.search(PATTERNS['dc'], line).group())
        elif re.search(PATTERNS['p'], line):
            holder.append(re.search(PATTERNS['p'], line).group())
        else:
            holder.append(line)
    return ' '.join(holder)
        

def clean_string_from_pattern(lst):
    holder = []
    for item in lst:
        res = cleaner(item)
        holder.append(res)
    return holder



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