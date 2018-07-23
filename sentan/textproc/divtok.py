import json
from time import time
from sentan import mysqlite

def div_tok_acts_db(load_dir_name='',
                    save_dir_name='',
                    sep_type='sep1',
                    inden=''):
        t0=time()
        #Initiate concls iterator:
        DB_load = mysqlite.DataBase(
            dir_name='TextProcessing/RawText/'+load_dir_name,
            base_name='RawText',
            tb_name=True
        )
        #Initiate DB:
        DB_save = mysqlite.DataBase(
            dir_name='TextProcessing/DivToks/'+save_dir_name,
            base_name='BigDivDB'
        )
        DB_save.create_tabel(
            'BigDivToks',
            (('id', 'TEXT', 'PRIMARY KEY'), ('par1', 'TEXT'))
        )
        counter = 0
        for fle in raw_files:
            t1=time()
            holder=[]
            print(inden+'Starting new file processing!')
            cleaned = self.CTP.court_decisions_cleaner(fle)
            divided = self.CTP.court_decisions_separator(
                cleaned,
                sep_type=sep_type
            )
            tokenized = self.CTP.iterate_tokenization(divided)
            print(inden+'\tStarting tokenization and writing')
            for tok_act in tokenized:
                name = ('0'*(4+1-len(str(counter)))+str(counter))
                enc = json.dumps(tok_act)
                holder.append((name, enc))
                counter+=1
            DB_save.insert_data(holder, col_num=2)
            print(
                inden+'\tTokenization and writing '
                +'complete in {:4.5f} seconds!'.format(time()-t1)
            )
        print('Total time costs: {}'.format(time()-t0))