import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import messagebox as msgb

__version__ = '0.2'

###Content=====================================================================
def find_file_path():
    tk.Tk().withdraw()
    return fd.askopenfilename()

def find_directory_path():
    tk.Tk().withdraw()
    return fd.askdirectory()

def popup_message_box(message):
    tk.Tk().withdraw()
    msgb.showinfo(title='Информация', message=message)

def entry():
    root = tk.Tk()
    username = tk.StringVar()
    ent = ttk.Entry(master=root, textvariable=username)
    but = ttk.Button(master=root, text='Ok', command=lambda:root.destroy())
    ent.pack()
    but.pack()
    root.mainloop()
    print(username.get())

###Short names=================================================================
ffp = find_file_path
fdp = find_directory_path
pmb = popup_message_box

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