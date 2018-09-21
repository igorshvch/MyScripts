import re
from collections import deque
from sentan.gui.dialogs import ffp, fdp, pmb
from openpyxl import load_workbook

__version__ = '0.1.1'

###Content=====================================================================

COLS = ('E', 'F', 'AP', 'AR', 'AS', 'AT', 'AU',)
KLASS = {
    'Может быть разворот' : True,
    'Устоялась есть динамика' : True,
    'Устоялась нет динамики' : True,
    'К удалению' : False
}
VALS = {'ЦИТАТА АКТА', 'ДОБОР НЕ НУЖЕН', 'ЦИТАТА НОРМЫ'}


class RowsCollectror():
    def __init__(self, rows, cols=COLS, klass=KLASS, break_vals=VALS):
        self.rows = range(*rows)
        self.cols = cols[0],
        self.klass = klass,
        self.break_vals = break_vals
        self.wb=None
    
    def open_excel(self):
        pmb('Выберите книгу Excel')
        self.wb = load_workbook(ffp())
    
    def select_sheet(self):
        print(self.wb.sheetnames)
        self.wsh = input('Введите название рабочего листа ======>> ')
    
    def iterate_over_rows(self):
        holder = []
        wsh = self.wb[self.wsh]
        rows = list(self.rows)
        cols = self.cols
        keys = ['par', 'concl', 'klass', 'pos1', 'pos2', 'pos3', 'pos4']
        for i in rows:
            cells = [col+str(i) for col in cols]
            vals = {key:wsh[cell].value for key,cell in zip(keys, cells)}
            if vals['klass'] == 'К удалению':
                continue
            else:
                par_concl = vals['par']+'#'+vals['concl']+'#'
                holder.append(par_concl+vals['pos1'])
                holder.append(par_concl+vals['pos2'])
                holder.append(par_concl+vals['pos3'])
                holder.append(par_concl+vals['pos4'])
        return holder



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