import sqlite3
import pathlib as pthl

PK = 'PRIMARY KEY'

class DataBase():
    options = {
            (1,1,1) : (
                lambda x,y,z: pthl.Path(x).joinpath\
                (y, z).with_suffix('.db')
            ),
            (1,1,0) : (
                lambda x,y,z: pthl.Path(x).joinpath\
                (y, 'Test').with_suffix('.db')
            ),
            (1,0,1) : (
                lambda x,y,z: pthl.Path(x).joinpath\
                (z).with_suffix('.db')
            ),
            (1,0,0): (
                lambda x,y,z: pthl.Path(x).joinpath\
                ('Test').with_suffix('.db')
            ),
            (0,1,1): (
                lambda x,y,z: pthl.Path().home().joinpath\
                (y, z).with_suffix('.db')
            ),
            (0,1,0): (
                lambda x,y,z: pthl.Path().home().joinpath\
                (y, 'Test').with_suffix('.db')
            ),
            (0,0,1): (
                lambda x,y,z: pthl.Path().home().joinpath\
                (z).with_suffix('.db')
            ),
            (0,0,0): (
                lambda x,y,z: pthl.Path().home().joinpath\
                ('Test').with_suffix('.db')
            )
        }

    def __init__(self,
                 raw_path=None,
                 dir_name=None,
                 base_name=None,
                 tb_name=False,
                 mode='rowid'):
        self.conn = None 
        self.cur = None
        self.tables = set()
        self.open(
            raw_path=raw_path, dir_name=dir_name, base_name=base_name
        )
        self.path = {
            'raw_path':raw_path,
            'dir_name':dir_name,
            'base_name':base_name
        }
        if tb_name:
            self.table_name = self.retrive_tabel_name()
            self.tables.add(self.table_name)
        self.mode = mode
        #print('CUR: {}'.format(self.cur))
        #print('CONN: {}'.format(self.conn))
    
    def __call__(self,
                 raw_path=None,
                 dir_name=None,
                 base_name=None,
                 print_path=True):
        if not raw_path and not dir_name and not base_name:
            self.open(**self.path)
        else:
            key = (bool(raw_path), bool(dir_name), bool(base_name))
            p = DataBase.options[key](raw_path, dir_name, base_name)
            if print_path:
                print(p)
            self.conn = sqlite3.connect(str(p))
            self.cur = self.conn.cursor()
            print('CALL: DB connection is established!')
    
    def __getitem__(self, key):
        if not self.cur:
            self.open(**self.path)
        if self.mode == 'id':
            self.cur.execute(
                'SELECT * FROM {tb} WHERE id=="{txt}"'\
                .format(tb = self.table_name, txt = key)
            )
            val = self.cur.fetchone()
        elif self.mode == 'rowid':
            self.cur.execute(
                'SELECT * FROM {tb} WHERE rowid=="{txt}"'\
                .format(tb = self.table_name, txt = key)
            )
            val = self.cur.fetchone()
        return val
    
    def open(self,
             raw_path=None,
             dir_name=None,
             base_name=None,
             print_path=True):
        key = (bool(raw_path), bool(dir_name), bool(base_name))
        p = DataBase.options[key](raw_path, dir_name, base_name)
        if print_path:
            print(p)
        self.conn = sqlite3.connect(str(p))
        self.cur = self.conn.cursor()
        print('DB connection is established!')
    
    def close(self, save=True):
        if save:
            self.conn.commit()
            print('Changes are saved!')
        self.conn.close()
        self.conn = None
        self.cur = None
        print('DB is closed!')
    
    def retrive_rows(self, num_of_rows, first_row=0, only_par_col=False):
        if not self.cur:
            self.open(**self.path)
        if only_par_col:
            self.cur.execute(
                'SELECT par FROM {tn} LIMIT {nr} OFFSET {fr}'\
                .format(tn=self.table_name,nr=num_of_rows,fr=first_row)
            )
        else:
            self.cur.execute(
                'SELECT * FROM {tn} LIMIT {nr} OFFSET {fr}'\
                .format(tn=self.table_name,nr=num_of_rows,fr=first_row)
            )
        rows = self.cur.fetchall()
        return rows
    
    def iterate_row_retr(self,
                         length=543434,
                         output=50000,
                         first_row=0,
                         only_par_col=False
                         ):
        def inner_func(num_of_rows, inner_fr, only_par_col=only_par_col):
            if not self.cur:
                self.open(**self.path)
            if only_par_col:
                self.cur.execute(
                    'SELECT par FROM {tn} LIMIT {nr} OFFSET {fr}'\
                    .format(tn=self.table_name,nr=num_of_rows,fr=inner_fr)
                )
            elif only_par_col==False:
                self.cur.execute(
                    'SELECT * FROM {tn} LIMIT {nr} OFFSET {fr}'\
                    .format(tn=self.table_name,nr=num_of_rows,fr=inner_fr)
                )
            rows = self.cur.fetchall()
            return rows
        counter = first_row
        base = (length-first_row)//output + 1
        while base:
            batch = inner_func(
                num_of_rows = output,
                inner_fr = counter)
            counter += output
            base-=1
            yield batch
    
    def retrive_tabel_name(self):
        if not self.cur:
            self.open(**self.path)
        tb_name = self.cur.execute(
            '''
            SELECT name FROM sqlite_master
            WHERE type="table"
            '''
        )
        for name in tb_name:
            return name[0]
    
    def create_tabel(self, table_name, columns, verbose=False):
        col_struct=''
        for i in columns:
            if len(i) == 3:
                col_struct += '{} {} {},'.format(*i)
            else:
                col_struct += '{} {},'.format(*i)
        col_struct = col_struct[:-1]
        if not self.cur:
            self.open(**self.path)
        self.cur.execute(
            'CREATE TABLE IF NOT EXISTS {tn} ({cs})'\
            .format(tn=table_name, cs=col_struct)
        )
        if verbose:
            print(
                'Tabel \'{}\' with columns:\n\'{}\'\nwas created!'\
                .format(table_name, col_struct)
            )
        self.table_name = table_name
        self.tables.add(table_name)
        self.conn.commit()
    
    def insert_data(self, data, col_num=2):
        if not self.cur:
            self.open(**self.path)
        if len(data) == 1:
            self.cur.execute(
                'INSERT INTO {tn} VALUES (?,?)'\
                .format(tn=self.table_name), (data[0][0], data[0][1])
            )
            print('Data was inserted!')
        else:
            q = '?,'*col_num
            q='('+q[:-1]+')'
            self.cur.executemany(
                'INSERT INTO {tn} VALUES {cols}'\
                .format(tn=self.table_name, cols=q), data
            )
        self.conn.commit()
    
    def total_rows(self):
        if not self.cur:
            self.open(**self.path)
        return self.cur.execute("SELECT Count(*) FROM TNBI").fetchone()[0]
    
    

    

    
