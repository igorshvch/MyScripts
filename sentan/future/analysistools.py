import re
from sentan.lowlevel import rwtool
from sentan.gui.dialogs import (
    ffp, fdp, pmb
)

__version__ = '0.1'

###Content=====================================================================

def concord(text, pars=None):
    char, char_num_min, char_num_max, dist = pars
    pattern = (
        r'(?=(.{{{0:d}}} {1:s}{{{2:d},{3:d}}}[ ,\.].{{{4:d}}}))'\
        .format(dist, char, char_num_min, char_num_max, dist)
    )
    print(pattern)
    res = re.findall(pattern, text, flags=re.DOTALL)
    formatter = rwtool.form_string_pattern(' ', 's', char_num_max)
    return [
        ' :: '.join(
            [line[:dist],
            formatter.format(line[dist+1:len(line)-dist-1]),
            line[-dist:]]
         )
         for line in res]

def observe_many_acts(pars=None):
    pmb('Chose directory to load text files!')
    path_to_text_files_dir = fdp()
    paths = rwtool.collect_exist_files(path_to_text_files_dir, suffix='.txt')
    holder = []
    for p in paths:
        with open(p, mode='r') as fle:
            text = fle.read()
        text = text.replace('\n', ' ')
        if pars:
            res = concord(text, pars=pars)
        else:
            res = concord(text, pars=(r'\d', 1, 4, 40))
        holder.extend(res)
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
