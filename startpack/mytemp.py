import re
from collections import deque

court = (
    '(Постановление (((Арбитражного суда|ФАС) (Волго-Вятского|Восточно-Сибирского|Дальневосточного|Западно-Сибирского|Московского|Поволжского|Северо-Западного|Северо-Кавказского|Уральского|Центрального) округа)|(Верховного|Конституционного) Суда РФ|(Пленума|Президиума) ВАС РФ)|Определение (Верховного|Конституционного) Суда РФ|Решение ВАС РФ) от'
)

quest = '[0-9][0-9]*\.[0-9][0-9]*\..+[А-я)?](?=\n)'
pos = '(Позиция|Способ) [0-9].+(?=\n)'

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
