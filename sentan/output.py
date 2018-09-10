from .lowlevel import rwtool
from . import elgranderoyal as elly
from .lowlevel import rwtool
from time import time
from .gui.dialogs import (
    find_file_path as ffp,
    find_directory_path as fdp
)
from . import mysqlite, dirman
from .gui.dialogs import (
    ffp, fdp, pmb, giv
)

__version__ = '0.3.1'

###Content=====================================================================
def count_result_scores(res_dict, top=5):
    holder_acts_set = set()
    holder_acts = []
    for key in res_dict:
        val = res_dict[key]
        reqs = [val[i][0]+' '+val[i][1] for i in range(top)]
        for req in reqs:
            holder_acts_set.add(req)
        holder_acts.extend(reqs)
    acts_score = {}
    for act_req in holder_acts_set:
        acts_score[act_req] = holder_acts.count(act_req)
    return sorted(
        [[key_dct, value] for key_dct, value in acts_score.items()],
        key=lambda x: x[1],
        reverse=True
    )

def start_up(concls):
    '''Starts iterations on data'''
    outter_holder = []
    t0 = time()
    t1 = time()
    for idn, concl in enumerate(concls):
        print(
            (
                23*'='
                +'\n'
                +'CONCLUSION # {} started.'.format(idn)
                +'\nTime:\ntotal: {:3.5f}'.format(time()-t0)
                +'\nsub: {:3.5f}'.format(time()-t1)
            ),
            end='\n'+23*'='+'\n'
        )
        t1 = time()
        outter_holder.append(
            (count_result_scores(elly.aggregate_model(concl), top=5))
        )
    print(
        (
            '\n'
            +23*'='
            +'\n'
            +'Total time costs in mins: {:3.5f}'.format((time()-t0)/60)
        )
    )
    return outter_holder

def write_res(concls, results, dir_name=None):
    '''Write results to txt files'''
    assert dir_name
    path = (
        r'C:\Users\EA-ShevchenkoIS\TextProcessing\Results\{}'.format(dir_name)
    )
    for idn, item in enumerate(concls):
        rwtool.write_text_to_csv(
            path+'/res'+str(idn+1)+'.txt',
            results[idn],
            header=['act', 'score'],
            zero_string=item
        )

def collect_results_from_csv(path_to_folder):
    paths = rwtool.collect_exist_files(path_to_folder, suffix='.txt')
    global_trash_holder = {}
    global_valid_holder = {}
    for p in paths:
        with open(p, mode='r') as file:
            text = file.read()
        spl = [line.split('|') for line in text.split('\n')]
        spl = spl[:-1]
        if int(spl[2][1]) > 5:
           global_valid_holder[spl[0][0]] = []
           for i in spl[2:]:
               if int(i[1]) > 5:
                   global_valid_holder[spl[0][0]].append(i[0])
        else:
            global_trash_holder[p.name] = spl[0][0]
    return global_valid_holder, global_trash_holder

def write_cleaned_results_from_csv(valid_holder, common_file_name='res'):
    from writer import writer
    count = 1
    for key in valid_holder:
        writer(key+'\n', common_file_name+str(count), mode='a', verbose=False)
        for i in valid_holder[key]:
            writer(
                i+'\n', common_file_name+str(count), mode='a', verbose=False
            )
        count+=1
    print('All results are written to files!')

def print_output_to_console(path):
    import re
    months = {
        'января':'01','февраля':'02',
        'марта':'03','апреля':'04','мая':'05',
        'июня':'06','июля':'07','августа':'08',
        'сентября':'09','октября':'10','ноября':'11',
        'декабря':'12'
    }
    dct = {key:0 for key in range(1,8)}
    res_tot = rwtool.load_pickle(path)
    counted = count_result_scores(res_tot)
    for i in counted:
        if len(re.split(' от ', i[0])) < 2:
            print(i[0], i[1])
        else:
            court, req = re.split(' от ', i[0])
            _, date, case = re.split(r'([0-9]{1,2} [А-я]+? [0-9]{4} г.)', req)
            spl_date = date.split(' ')
            spl_date[1] = months[spl_date[1]]
            if len(spl_date[0]) == 1:
                spl_date[0] = '0'+spl_date[0]
            date2 = '.'.join(spl_date[:-1])
            dct[i[1]] +=1
            print('{:-<43s} :: {:<10s} :: {:-<35} :: {:>2d}'.format(court, date2, case[1:],i[1]))
    for i in range(7, 0, -1):
        print('with rank {} :: total :: {}'.format(i, dct[i]))
    total_greater_3 = 0
    for key in dct:
        if key >= 3:
            total_greater_3 += dct[key]
    print('with rank 3 or greater :: {}'.format(total_greater_3))

def print_output_to_console_2(file_name1, file_name2, file_name3=None):
    import re
    months = {
        'января':'01','февраля':'02',
        'марта':'03','апреля':'04','мая':'05',
        'июня':'06','июля':'07','августа':'08',
        'сентября':'09','октября':'10','ноября':'11',
        'декабря':'12'
    }
    if file_name3:
        dir_path = fdp()
        path1 = dir_path+'/{}'.format(file_name1)
        path2 = dir_path+'/{}'.format(file_name2)
        path3 = dir_path+'/{}'.format(file_name3)
        holder1 = count_result_scores(rwtool.load_pickle(path1))
        holder2 = count_result_scores(rwtool.load_pickle(path2))
        holder3 = count_result_scores(rwtool.load_pickle(path3))
        holder_all = holder1+holder2+holder3
    else:
        dir_path = fdp()
        path1 = dir_path+'/{}'.format(file_name1)
        path2 = dir_path+'/{}'.format(file_name2)
        holder1 = count_result_scores(rwtool.load_pickle(path1))
        holder2 = count_result_scores(rwtool.load_pickle(path2))
        holder_all = holder1+holder2
    all_acts = [line[0] for line in holder_all]
    dct_all_acts = {act_name:[] for act_name in all_acts}
    for ind, act in enumerate(all_acts):
        dct_all_acts[act].append(holder_all[ind][1])
    counted = sorted(
        [(act, max(ranks)) for act,ranks in dct_all_acts.items()],
        key=lambda x: x[1],
        reverse=True
    )
    dct_rank = {key:0 for key in range(1,8)}
    for i in counted:
        if len(re.split(' от ', i[0])) < 2:
            print(i[0], i[1])
        else:
            court, req = re.split(' от ', i[0])
            _, date, case = re.split(r'([0-9]{1,2} [А-я]+? [0-9]{4} г.)', req)
            spl_date = date.split(' ')
            spl_date[1] = months[spl_date[1]]
            if len(spl_date[0]) == 1:
                spl_date[0] = '0'+spl_date[0]
            date2 = '.'.join(spl_date[:-1])
            dct_rank[i[1]] +=1
            print('{:-<43s} :: {:<10s} :: {:-<35} :: {:>2d}'.format(court, date2, case[1:],i[1]))
    for i in range(7, 0, -1):
        print('with rank {} :: total :: {}'.format(i, dct_rank[i]))
    total_greater_3 = 0
    for key in dct_rank:
        if key >= 3:
            total_greater_3 += dct_rank[key]
    print('with rank 3 or greater :: {}'.format(total_greater_3))

def export_court_reqs(file_name):
    DB_load = mysqlite.DataBase(
        raw_path = r'C:\Users\EA-ShevchenkoIS\TextProcessing\TNBI',
        base_name='TNBI',
        tb_name=True
    )
    TA = DB_load.total_rows()
    OUTPUT = TA//10 if TA > 10 else TA//2
    acts_gen = DB_load.iterate_row_retr(length=TA, output=OUTPUT)
    holder = []
    for batch in acts_gen:
        for row in batch:
            ind, court, req, _, _, _, _, _ = row
            holder.append([ind, court, req])
    rwtool.write_text_to_csv(
        'C:/Users/EA-ShevchenkoIS/TextProcessing/Results/{}.txt'.format(file_name),
        holder
    )

def write_output_to_file():
    import re
    pmb('Select file with concls data')
    path_to_concls = ffp()
    if path_to_concls[-4:] == '.txt':
        with open(path_to_concls, mode='r') as fle:
            text = fle.read().strip('\n')
        concls = text.split('\n')
        concls = [[item[0], int(item[1])] for item in concls]
    else:
        concls = rwtool.load_pickle(path_to_concls)
    pmb('Select load and save directories')
    path_to_load = fdp()
    path_to_save = fdp()
    paths = rwtool.collect_exist_files(path_to_load)
    total_input_files = len(paths)
    digits_num = len(str(total_input_files))
    formatter = rwtool.form_string_numeration(digits_num)
    res_gen = ((ind, rwtool.load_pickle(p)) for ind, p in enumerate(paths))
    for ind, res in res_gen:
        counted = count_result_scores(res)
        holder = []
        holder.append(concls[ind])
        for item in counted:
            if len(re.split(' от ', item[0])) < 2:
                print(item[0], item[1])
                continue
            else:
                if item[1] >= 3:
                    holder.append(
                        '{:<100s} :: {:>2d}'.format(item[0], item[1])
                    )
        try:
            rwtool.write_text(
                '\n'.join(holder),
                path_to_save+'/'+formatter.format(ind)+'_RESULT'
            )
        except:
            print(holder)


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