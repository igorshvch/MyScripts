import re
from time import time

__version__ = '0.2.1'

###Content=====================================================================
#Patterns
PATTERN_ACT_CLEAN1 = '-{66}\nКонсультантПлюс.+?-{66}\n'
PATTERN_ACT_CLEAN2 = 'КонсультантПлюс.+?\n.+?\n'
PATTERN_ACT_CLEAN3 = 'Рубрикатор ФАС \(АСЗО\).*?Текст документа\n'
PATTERN_ACT_SEP1 = '\n\n\n-{66}\n\n\n'
PATTERN_ACT_SEP2 = 'Документ предоставлен КонсультантПлюс'
PATTERN_PASS1 = (
    'ОБЗОР\nСУДЕБНОЙ ПРАКТИКИ ПО ДЕЛАМ,'
    '+ РАССМОТРЕННЫМ\nАРБИТРАЖНЫМ СУДОМ УРАЛЬСКОГО ОКРУГА'
)
PATTERN_PASS2 = (
    'Утвержден\nпрезидиумом Арбитражного суда\nСеверо-Кавказского округа'
)

#Funcs
def court_decisions_cleaner(text, inden=''):
    t0 = time()
    cleaned_text1 = re.subn(PATTERN_ACT_CLEAN1, '', text, flags=re.DOTALL)[0]
    cleaned_text2 = re.subn(PATTERN_ACT_CLEAN2, '', cleaned_text1)[0]
    cleaned_text3 = re.subn(
        PATTERN_ACT_CLEAN3, '', cleaned_text2, flags=re.DOTALL
        )[0]
    print('{}Acts were cleaned in {} seconds'.format(inden, time()-t0))
    return cleaned_text3[1:-68]

def court_decisions_separator(text, sep_type='sep1', inden=''):
    t0 = time()
    if sep_type=='sep1':
        separated_acts = re.split(PATTERN_ACT_SEP1, text)
    else:
        separated_acts = re.split(PATTERN_ACT_SEP2, text)
    print(
        '{}Acts were separated in {} seconds,'.format(inden, time()-t0),
        '{} acts were found'.format(len(separated_acts)),
    )
    return separated_acts
    

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