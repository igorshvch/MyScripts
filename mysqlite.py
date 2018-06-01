import sqlite3
import pathlib as pthl

PK = 'PRIMARY KEY'

class DataBase():
    options = {
            (1,1,1) : (
                lambda x,y,z: pthl.Path(x).joinpath\
                (y, y).with_suffix('.db')
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
                 base_name=None):
        self.conn = None 
        self.cur = None
        self.open(
            raw_path=raw_path, dir_name=dir_name, base_name=base_name
        )
        self.path = {
            'raw_path':raw_path,
            'dir_name':dir_name,
            'base_name':base_name
        }
        self.tables = set()
    
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
            print('DB connection is established!')
    
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
        print('DB connection is established!')
    
    def close(self, save=True):
        if save:
            self.conn.commit()
            print('Changes are saved!')
        self.conn.close()
        self.conn = None
        self.cur = None
        print('DB is closed!')
    
    def create_conn(self, dir_name, base_name):
        p = pthl.Path(r'C:\Users\EA-ShevchenkoIS\TextProcessing')
        p = p.joinpath(dir_name, base_name).with_suffix('.db')
        print(p)
        self.conn = sqlite3.connect(str(p))

    def create_cur(self, base_name=None, dir_name=None):
        if base_name and not self.conn:
            p = pthl.Path(r'C:\Users\EA-ShevchenkoIS\TextProcessing')
            p = p.joinpath(dir_name, base_name).with_suffix('.db')
            print(p)
            self.conn = sqlite3.connect(str(p))
            self.cur = self.conn.cursor()
        elif not base_name and self.conn:
            self.cur = self.conn.cursor()
        else:
            print(
                'Error! Try to set or path either self.conn'
            )
    
    def create_tabel(self, table_name, columns):
        col_struct=''
        for i in columns:
            if len(i) == 3:
                col_struct += '{} {} {},'.format(*i)
            else:
                col_struct += '{} {},'.format(*i)
        #print(col_struct)
        #print('CREATE TABLE {} ({})'.format(table_name, col_struct))
        col_struct = col_struct[:-1]
        if not self.cur:
            self.create_cur()
        self.cur.execute(
            'CREATE TABLE {} ({})'.format(table_name, col_struct)
        )
        print(
            'Tabel \'{}\' with columns:\n\'{}\'\nwas created!'\
            .format(table_name, col_struct)
        )
        self.tables.add(table_name)
        self.conn.commit()
    
    def insert_data(self, table_name, data):
        if not self.cur:
            self.create_cur()
        #print('Data len:', len(data))
        if len(data) == 1:
            self.cur.execute(
                'INSERT INTO {} VALUES (?,?)'\
                .format(table_name), (data[0][0], data[0][1])
            )
        else:
            self.cur.executemany(
                'INSERT INTO {} VALUES (?,?)'\
                .format(table_name), data
            )
        #print('Data was inserted!')
        self.conn.commit()

    

    
