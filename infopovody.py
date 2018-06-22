import tkinter as tk
import datetime as dt

class GUI():
    def __init__(self, parent):
        self.root = tk.Frame(parent, relief='sunken', bd=1)
        #self.root.pack(side='top')
        self.make_widgets()

    def make_widgets(self):
        buffer = tk.Frame(self.root)
        self.ent = tk.Entry(buffer, width=40)
        button = tk.Button(buffer, text='Записать инфоповоды', command=self.wrt)
        buffer.pack()
        self.ent.pack(side='left')
        button.pack(side='right', expand='no', fill='x')

    def wrt(self):
        string = self.ent.get()
        if not string:
            new_win = tk.Toplevel(self.root)
            tk.Label(new_win, text='Данные не введены!', bg='red').pack()
        else:
            time = str(dt.datetime.now())[:-7]
            path = (r'C:\Users\EA-ShevchenkoIS'
                    +'\Documents\Рабочая\{}'.format(time[:7])
                    +'\Инфоповоды.txt')
            with open(path, mode='a') as file:
                string = time  + ' ::...  ' + string +'\n'
                file.write(string)
            self.ent.delete(first='0', last='end')
            new_win = tk.Toplevel(self.root)
            tk.Label(new_win, text='Данные сохранены!', bg='yellow').pack()

#GUI().root.mainloop()
        

    
