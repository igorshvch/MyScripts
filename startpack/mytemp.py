import re
from collections import deque

court = (
    '(Постановление +?(((Арбитражного суда|ФАС) (Волго-Вятского|Восточно-Сибирского|Дальневосточного|Западно-Сибирского|Московского|Поволжского|Северо-Западного|Северо-Кавказского|Уральского|Центрального) округа)|(Верховного|Конституционного) Суда РФ|(Пленума|Президиума) ВАС РФ)|Определение (Верховного|Конституционного) Суда РФ|Решение ВАС РФ) от.+'
)

quest = '[0-9][0-9]*\.[0-9][0-9]*\..+[А-я)?](?=\n)'
pos = '(Позиция|Способ) [.0-9]+ .+'

pattern_pos_last_ver = '[0-9] [.0-9]+ ?[.0-9]*? .+'
pattern_nonj = pat = '(Консультация эксперта|Информационное сообщение +?ФНС|Письм[оа]|Приказ|Статья +?:)'

def cleaner(raw_text, pattern=pattern_pos_last_ver):
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
    #Find all positions
    positions = [
        re.match(pattern, situation).group()
        for situation in situations_list if re.match(pattern, situation)
    ]
    print('Total positions num: {}'.format(len(positions)))
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
            if not trigger_pos and re.match(pattern, par):
                dct_holer['sit'] = par
                list_holder.append(par)
                trigger_pos = True
            elif re.match(pos, par):
                dct_holer['pos'+sep+str(pos_count)] = par
                list_holder.append(inden+par)
                #trigger_inner_pos = True
                trigger_act = False
                court_count = pos_count
                pos_count+=1
            elif not trigger_act and re.match(court, par):
                dct_holer['court'+sep+str(court_count)] = par
                list_holder.append(2*inden+par)
                trigger_act = True
                try:
                    new_par = spl_situation.popleft()
                except:
                    new_par=None
                    continue
                if (not re.match(pattern, new_par)
                    and not re.match(pos, new_par)
                    and not re.match(court, new_par)
                    and not re.match(pattern_nonj, new_par)):
                    inner_holder = new_par
                    new_par = None
                    try:
                        new_par = spl_situation.popleft()
                        if (not re.match(pattern, new_par)
                            and not re.match(pos, new_par)
                            and not re.match(court, new_par)
                            and not re.match(pattern_nonj, new_par)):
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
        'spl_pars':situations_list_spl,
        'poses':positions,
        'format':for_strings,
        'dct':format_dct
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

def start_up():
    outter_holder = []
    t0 = elly.time()
    for idn, concl in enumerate(concls12):
        t1 = elly.time()
        print('CONCLUSION # {} started.\nTime:\ntotal: {:3.5f}\nsub: {:3.5f}'.format(idn, elly.time()-t0, elly.time()-t1), end='\n'+23*'='+'\n')
        outter_holder.append((elly.count_result_scores(elly.aggregate_model(concl), top=5)))
    return outter_holder

class Collector():
    def __init__(self):
        self.dct = {'quest':0, 'pos': 0, 'court':0}
        self.holder = []
        self.info = []

    
    def find(self, text):
        holder = []
        info = []
        text = re.subn('\n\n', '\n', text)[0]
        d_chunks = deque(re.split('^(%s)' % quest, text, flags=re.MULTILINE))
        while d_chunks:
            chunk = d_chunks.popleft()
            if re.match(quest[:-5], chunk):
                q = re.match(quest[:-5], chunk).group(0)
                holder.append(q)
                self.dct['quest'] += 1
            else:
                d_subchunks = deque(re.split('(%s)' % pos, chunk))
                while d_subchunks:
                    subchunk = d_subchunks.popleft()
                    if re.match(pos[:-5], subchunk):
                        p = re.match(pos[:-5], subchunk).group(0)
                        holder.append('\t'+p)
                        self.dct['pos']+=1
                    else:
                        d_lines = deque(subchunk.split('\n'))
                        while d_lines:
                            d_line = d_lines.popleft()
                            if re.search(court, d_line) and not self.dct['court']:
                                c = re.search('(?=%s).+' % court, d_line).group(0)
                                if '\n' in c:
                                    print('This is it!', end='==')
                                st_c = ('\t\t'+c) #('\t'+c) if not self.dct['pos'] else ('\t\t'+c)
                                holder.append(st_c)
                                #holder.append(d_lines.popleft())
                                self.dct['court'] +=1
                                break
                        self.dct['court'] = 0
                self.dct['pos'] = 0
            info.append(len(holder))
        #return holder    
        self.holder = holder
        self.info = info






p1 = '[0-9]\.[0-9]\..+[А-я)?](?=\n)'
p2 = 'Позиция [0-9].+\n.*\n.+\n.+\n'
p3 = 'Позиция [0-9].+\n.*\nОбратите внимание.+\n.*\n.+\n.+\n'
p4 = 'Подробнее см\. документы\n'
p5 = 'Подробнее см\. документы\n.*\n.+\n.+\n'
p6 = 'Подробнее см\. документы\n.*\nПозиция [0-9].+\n.*\n.+\n.+\n'
p7 = (
    'Подробнее см\. документы\n.*\nОбратите внимание.+\n.*\nПозиция [0-9].+\n.*\n.+\n.+\n'
)

joined_p = '(' + p1 + '|' + p2 + '|' + p3 + '|' + p4 + ')'
j_p = '(' + p1 + '|' + p6 + '|' + p5 + '|' + p3 + '|' +p2 + ')'
j_p2 =  '(' + p1 + '|' +  p7 + '|' + p6 + '|' + p5 + '|' + p3 + '|' +p2 + ')'

pattern_quest = '[0-9]\.[0-9]\..+[А-я)?]\n'
pattern_pos = 'Позиция [0-9].+\n'
pattern_docs = 'Подробнее см\. документы\n'
pattern_alert = 'Обратите внимание.+\n'

def parser(d_lines:deque):
    #cash = []
    holder = []
    while d_lines:
        cursor = d_lines.popleft()
        if re.match(pattern_quest, cursor):
            match_obj = re.match(pattern_quest, cursor)
            holder.append(match_obj.group(0)[:-1])
        elif re.match(pattern_docs, cursor):
            match_obj = re.match(pattern_docs, cursor)
            holder.append('\t'+match_obj.group(0)[:-1])
            d_lines.popleft()
            cursor = d_lines.popleft()
            if re.match(pattern_pos, cursor):
                match_obj = re.match(pattern_pos, cursor)
                holder.append('\t'+match_obj.group(0)[:-1])
                d_lines.popleft()
                holder.append('\t\t\t'+d_lines.popleft()[:-1])
                #holder.append('\t\t\t'+d_lines.popleft()[:-1])
            elif re.match(pattern_alert, cursor):
                match_obj = re.match(pattern_alert, cursor)
                holder.append('\t'+match_obj.group(0)[:-1])
                holder.append('\t\t\t\t\t'+d_lines.popleft()[:-1])
                holder.append('\t\t\t\t\t'+d_lines.popleft()[:-1])
                holder.append('\t\t\t\t\t'+d_lines.popleft()[:-1])
            else:
                holder.append('\t\t\t'+cursor[:-1])
                holder.append('\t'+d_lines.popleft()[:-1])
                holder.append('\t'+d_lines.popleft()[:-1])
    return holder

stt='''     par = spl_situation.popleft() if not new_par else new_par
            if not trigger_pos and re.match(pattern, par):
                #print('Pos # {}'.format(idn))
                par_holder.append(par)
                trigger_pos = True
                continue
            elif re.match(pos, par):
                print('InnerPos # {}'.format(idn))
                par_holder.append('\t'+par)
                print('Appended', end=', ')
                trigger_inner_pos = True
                #One level down
                new_par = spl_situation.popleft()
                #print('New el\n{}'.format(new_par), end=', ')
                if re.match(court, new_par):
                    print('matched', end=', ')
                    par_holder.append('\t\t'+new_par)
                    par_holder.append('\t\t\t'+spl_situation.popleft())
                    try:
                        print('trying to load', end=', ')
                        new_par = spl_situation.popleft()
                        print('Lower level: new el', end=', ')
                        if (
                            not re.match(pattern, new_par)
                            and not re.match(pos, new_par)
                            and not re.match(court, new_par)
                        ):
                            print('Lowlevel if passed')
                            par_holder.append('\t\t\t'+new_par)
                            new_par = None
                        else:
                            print('Lowlevel if failed')
                    except:
                        pass
            elif (
                not trigger_inner_pos
                and not trigger_act
                and re.match(court, par)               
            ):
                #print('Court # {}'.format(idn))
                par_holder.append('\t'+par)
                trigger_act = True
                #One level down
                try:
                    new_par = spl_situation.popleft()
                    if (
                        not re.match(pattern, new_par)
                        and not re.match(pos, new_par)
                        and not re.match(court, new_par)
                    ):
                        par_holder.append('\t\t\t'+new_par)
                        new_par = None
                    else:
                        pass
                except:
                    pass
                break
        trigger_pos = False
        trigger_inner_pos = False
        trigger_act = False'''