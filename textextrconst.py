'''
Module provides string patterns
for regexp processing
'''

alph = 'абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯёЁ'
alph_up = 'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯЁ'
alph_low = 'абвгдежзийклмнопрстуфхцчшщъыьэюяё'

demand_find_pattern = r'(?<=Установил:\n\n|УСТАНОВИЛ:\n\n|установил:\n\n).*(?=\n)'

note_find_pattern = r'-{66}\nКонсультантПлюс.*?-{66}\n'

result_find_pattern = (r'.*[Рр]уководствуясь статьями.*?Арбитражного процессуального кодекса.*?(?=\n)|'
       r'.*[Рр]уководствуясь статьями.*?АПК.*?(?=\n)|'
       r'.*[Рр]уководствуясь ст.*?Арбитражного процессуального кодекса.*?(?=\n)|'
       r'.*[Рр]уководствуясь ст.*?АПК.*?(?=\n)')

acts_separator_pattern = r'АРБИТРАЖНЫЙ СУД.*?(?=-{66})'

RU_word_strip_pattern = (
    r'[{0}]+-?[{1}]*-?[{2}]*-?[{3}]*'.format(alph,
                                             alph,
                                             alph,
                                             alph))

RU_word_strip_pattern_with_sentmarks = (
    r'[{0}]+-?[{1}]*-?[{2}]*-?[{3}]*|<S>|</S>'.format(alph,
                                                      alph,
                                                      alph,
                                                      alph))

RU_word_strip_pattern_with_sentmarks_n_articles = (
    r'[{0}]+-?[{1}]*-?[{2}]*-?[{3}]*|<S>|</S>|[0-9]+[.]*[0-9]*'.format(
        alph,
        alph,
        alph,
        alph))

word_strip_pattern = (
    r'[{0}]+-?[{1}]*-?[{2}]*-?[{3}]*|'
    r'[A-Za-z]+-?[A-Za-z]*-?[A-Za-z]*-?[A-Za-z]*|'
    r'[0-9]+[-.,]*[0-9]*[-.,]*[0-9]*[-.,]*[0-9]*[-.,]*[{4}]*'.format(alph,
                                                                     alph,
                                                                     alph,
                                                                     alph,
                                                                     alph))
sentence_breaker_pattern = (
    r'(?<=[\w][\w\)])\.\s+(?=[%s]\s|[%s][%s])'
                            % (alph_up, alph_up, alph))
