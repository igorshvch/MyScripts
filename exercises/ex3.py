import random as rd
import mysqlite
import multiprocessing as mp
from time import time

ALPH = [chr(i) for i in range(1040, 1072)]

def str_cre(alph_quant, rep_num):
    st = ''
    while rep_num:
        word = ''.join(rd.choices(ALPH, k=alph_quant))
        st+=word+' '
        rep_num-=1
    return st[:-1]

def create_data_list(lst_len, bacth_size=20, alph_quant=4, rep_num=10, mode='g'):
    if mode == 'g':
        while bacth_size:
            yield [
                (str_cre(alph_quant, rep_num),)
                for j in range(lst_len//bacth_size)
            ]
            bacth_size-=1
    elif mode == 'f':
        return [str_cre(alph_quant, rep_num) for i in range(lst_len)]

def populate_base(lst_len, base_dir='TextProcessing', base_name='TestMP'):
    db = mysqlite.DataBase(dir_name=base_dir, base_name=base_name)
    db.create_tabel(
        'TestText',
        (('txt', 'TEXT', 'PRIMARY KEY'),)
    )
    data_gen = create_data_list(lst_len=lst_len)
    t0=time()
    for batch in data_gen:
        t1=time()
        print('Start new batch! {:4.4f}'.format(time()-t1))
        db.insert_data(batch, col_num=1)
    print('Data writing ended in {:4.4f} seconds!'.format(time()-t0))
    db.close()

def collect_info_mp(base_dir='TextProcessing', base_name='TestMP'):
    db = mysqlite.DataBase(dir_name=base_dir, base_name=base_name, tb_name=True)
    batch_gen = db.iterate_row_retr(length=1000000, output=50000)
    voc=set()
    t0 = time()
    for batch in batch_gen:
        t1 = time()
        print('Start new batch! {:4.4f}'.format(time()-t1))
        with mp.Pool() as p:
            for row in batch:
                for word in row[0].split(' '):
                    p.imap_unordered(voc.add, (word,))
    print('Execution ended in {:4.4f}'.format(time()-t0))
    db.close()
    return voc

def collect_info(base_dir='TextProcessing', base_name='TestMP'):
    db = mysqlite.DataBase(dir_name=base_dir, base_name=base_name, tb_name=True)
    batch_gen = db.iterate_row_retr(length=1000000, output=50000)
    voc=set()
    t0 = time()
    for batch in batch_gen:
        t1 = time()
        print('Start new batch! {:4.4f}'.format(time()-t1))
        for row in batch:
            for word in row[0].split(' '):
                voc.add(word)
    print('Execution ended in {:4.4f}'.format(time()-t0))
    db.close()
    return voc


    
