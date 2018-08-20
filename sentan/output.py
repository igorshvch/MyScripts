from sentan.lowlevel import rwtool
from sentan import elgranderoyal as elly
from time import time

__version__ = 0.3

###Content=====================================================================
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
            (elly.count_result_scores(elly.aggregate_model(concl), top=5))
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
    paths = rwtool.collect_exist_file_paths(path_to_folder, suffix='.txt')
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