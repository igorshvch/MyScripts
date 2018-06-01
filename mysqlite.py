import sqlite3
import pathlib as pthl

PK = 'PRIMARY KEY'

class DataBase():
    def __init__(self):
        self.conn = None
        self.cur = None
        self.tables = set()
    
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

    

    
