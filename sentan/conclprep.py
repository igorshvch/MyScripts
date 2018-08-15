import re
from collections import deque

__version__ = 0.1

###Content=====================================================================
QUEST = '[0-9] [.0-9]+ ?[.0-9]*? .+'
POSITION = '(Позиция|Способ) [.0-9]+ .+'
COURT = (
    '(Постановление +?(((Арбитражного суда|ФАС) (Волго-Вятского|Восточно-Сибирского|Дальневосточного|Западно-Сибирского|Московского|Поволжского|Северо-Западного|Северо-Кавказского|Уральского|Центрального) округа)|(Верховного|Конституционного) Суда РФ|(Пленума|Президиума) ВАС РФ)|Определение (Верховного|Конституционного) Суда РФ|Решение ВАС РФ) от.+'
)
NONJUDJ = '(Консультация эксперта|Информационное сообщение +?ФНС|Письм[оа]|Приказ|Статья +?:)'

PATTERNS = {
    'Q' : QUEST,
    'P' : POSITION,
    'C' : COURT,
    'N' : NONJUDJ
}

def cleaner(raw_text, pat_dict = PATTERNS):
    #Remove internal 'K+' marks
    marks_removed = re.subn('\{.+?\}', '', raw_text)[0]
    #Change endline sequences
    endline_normilized = marks_removed.replace(' \n ', '\n')
    #Split into situations
    situations_list = re.split('\n-{69}\n', endline_normilized)
    #Split situations into paragraphs
    situations_list_spl = [
        situation.split('\n') for situation in situations_list
    ]
    print('Total situations num: {}'.format(len(situations_list)))
    #Find all questions
    questions = [
        re.match(pat_dict['Q'], situation).group()
        for situation in situations_list if re.match(pat_dict['Q'], situation)
    ]
    print('Total questions num: {}'.format(len(questions)))
    #Format text in the situations
    format_dct = {}
    for_strings = []
    sep = '#'
    trigger_pos = False
    trigger_act = False
    inden = '\t'
    #for spl_situation in situations_list_spl:
    for idn, spl_situation in enumerate(situations_list_spl):
        dct_holer = {}
        list_holder = []
        spl_situation = deque(spl_situation)
        new_par = None
        pos_count = 1
        court_count = 1
        #print('Iteration # {}'.format(idn), end=' === ')
        while spl_situation:
            par = new_par if new_par else spl_situation.popleft() 
            if not trigger_pos and re.match(pat_dict['Q'], par):
                dct_holer['sit'] = par
                list_holder.append(par)
                trigger_pos = True
            elif re.match(pat_dict['P'], par):
                dct_holer['pos'+sep+str(pos_count)] = par
                list_holder.append(inden+par)
                #trigger_inner_pos = True
                trigger_act = False
                court_count = pos_count
                pos_count+=1
            elif not trigger_act and re.match(pat_dict['C'], par):
                dct_holer['court'+sep+str(court_count)] = par
                list_holder.append(2*inden+par)
                trigger_act = True
                try:
                    new_par = spl_situation.popleft()
                except:
                    new_par=None
                    continue
                if (not re.match(pat_dict['Q'], new_par)
                    and not re.match(pat_dict['P'], new_par)
                    and not re.match(pat_dict['C'], new_par)
                    and not re.match(pat_dict['N'], new_par)):
                    inner_holder = new_par
                    new_par = None
                    try:
                        new_par = spl_situation.popleft()
                        if (not re.match(pat_dict['Q'], new_par)
                            and not re.match(pat_dict['P'], new_par)
                            and not re.match(pat_dict['C'], new_par)
                            and not re.match(pat_dict['N'], new_par)):
                            dct_holer['ann'+sep+str(court_count)] = (
                            inner_holder + '\n' + new_par
                                )
                            list_holder.append(
                                3*inden+inner_holder
                                +'\n'+3*inden
                                + new_par
                            )
                            new_par = None
                        else:
                            dct_holer['ann'+sep+str(court_count)] = inner_holder
                            list_holder.append(3*inden+inner_holder)
                            new_par = None              
                    except:
                        dct_holer['ann'+sep+str(court_count)] = inner_holder
                        list_holder.append(3*inden+inner_holder)
                        new_par = None

                else:
                    new_par = None
        dct_holer['total'] = pos_count
        format_dct[idn]=dct_holer
        for_strings.extend(list_holder)
        trigger_pos = False
        trigger_act = False
    #Return results
    return {
        'spl_pars' : situations_list_spl,
        'poses' : questions,
        'format' : for_strings,
        'dct' : format_dct
    }

def dct_to_list_of_concls(main_dct):
    holder = []
    outter_dct = main_dct['dct']
    for j in range(len(main_dct['poses'])):
        inner_dct = outter_dct[j]
        lngth = inner_dct['total'] if inner_dct['total'] > 1 else 2
        for i in range(1, lngth):
            inner_holder = []
            inner_holder.append(inner_dct['sit'])
            inner_holder.append(
                inner_dct['pos#'+str(i)][11:] if 'pos#'+str(i) in inner_dct.keys() else ''
                )
            inner_holder.append(inner_dct.get('ann#'+str(i), ''))
            holder.append(' '.join(inner_holder))
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