import sqlite3
import pathlib as pthl

__version__ = '0.4.1'

###Content=====================================================================
class DataBase():
    def __init__(self,
                 path,
                 base_name):
        self.conn = None 
        self.cur = None
        self.path = path.joinpath(base_name)
        self.open(
            self.path
        )
        tb_name = self.retrive_tabel_name()
        if tb_name:
            self.table_name = tb_name
    
    def open(self,
             path,
             print_path=True):
        if print_path:
            print(path)
        self.conn = sqlite3.connect(str(path))
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
            self.open(self.path)
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
                self.open(self.path)
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
            self.open(self.path)
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
            self.open(self.path)
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
        self.conn.commit()
    
    def insert_data(self, data, col_num=2):
        if not self.cur:
            self.open(self.path)
        if len(data) == 1:
            q = '?,'*col_num
            q='('+q[:-1]+')'
            self.cur.execute(
                'INSERT INTO {tn} VALUES {cols}'\
                .format(tn=self.table_name, cols=q), *data
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
            self.open(self.path)
        return self.cur.execute("SELECT Count(*) FROM TNBI").fetchone()[0]
    
    
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