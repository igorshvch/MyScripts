import re
from sentan.gui.dialogs import ffp, fdp, pmb
from sentan.lowlevel import rwtool

__version__ = '0.1'

###Content=====================================================================
DESICION_PURE = '(?<=[0-9]\.[0-9]\. Вывод из судебной практики: ).*'
DESICION_ALT = (
    '(?<=Вывод из судебной практики: По вопросу о).*(?=у судов нет единой позиции|существует две позиции судов|существует три позиции судов|существует четыре позиции судов|существует пять позиций судов)'
)
POSITION = '(?<=Позиция [1-9]\.).*'

PATTERNS = {
    'dp': DESICION_PURE,
    'da': DESICION_ALT,
    'p': POSITION
}

def clean_stored_concls():
    pmb('Chose file with conclusions!')
    holder = []
    path_to_file = ffp()
    with open(path_to_file, mode='r') as fle:
        text = fle.read()
    spl = [line.split('#') for line in text.split('\n') if line]
    for _, concl in enumerate(spl):
        if len(concl) == 2:
            item1, item2 = concl
            item1 = item1[3:]
            try:
                item2 = re.search(PATTERNS['dp'], item2).group().strip(' ')
            except:
                print(item2)
                break
            holder.append(' '.join([item1, item2]))
        elif len(concl) == 3:
            item1, item2, item3 = concl
            item1 = item1[3:]
            try:
                item2 = re.search(PATTERNS['da'], item2).group().strip(' ')
            except:
                print('item 2', _, 'part', item2)
                break
            try:
                item3 = re.search(PATTERNS['p'], item3).group().strip(' ')
            except:
                print('item 3', _, 'part', item3)
                break
            holder.append(' '.join([item1, item2, item3]))
        else:
            print(_, len(concl), concl)
            raise TypeError('Something has gone wrong!')
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