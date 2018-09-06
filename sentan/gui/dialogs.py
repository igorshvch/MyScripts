import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import messagebox as msgb
from time import sleep

__version__ = '0.3'

###Content=====================================================================
def find_file_path():
    tk.Tk().withdraw()
    return fd.askopenfilename()

def find_directory_path():
    tk.Tk().withdraw()
    return fd.askdirectory()

def popup_message_box(message='Spam!'):
    tk.Tk().withdraw()
    msgb.showinfo(title='Информация', message=message)

def get_integer_value(message='Spam!'):
    root = tk.Tk()
    userval = tk.IntVar()
    lab = ttk.Label(master=root, text=message)
    ent = ttk.Entry(master=root, textvariable=userval)
    but = ttk.Button(master=root, text='Ok', command=root.quit())
    lab.pack()
    ent.pack()
    but.pack()
    root.mainloop()
    return userval.get()

class SampleApp(tk.Tk):
#Sample get at
#https://stackoverflow.com/questions/7310511/how-to-create-downloading-progress-bar-in-ttk
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.button = ttk.Button(text="start", command=self.start)
        self.button.pack()
        self.progress = ttk.Progressbar(self, orient="horizontal",
                                        length=200, mode="determinate")
        self.progress.pack()

        self.bytes = 0
        self.maxbytes = 0

    def start(self):
        self.progress["value"] = 0
        self.maxbytes = 50000
        self.progress["maximum"] = 50000
        self.read_bytes()

    def read_bytes(self):
        '''simulate reading 500 bytes; update progress bar'''
        self.bytes += 500
        self.progress["value"] = self.bytes
        if self.bytes < self.maxbytes:
            # read more bytes after 100 ms
            self.after(100, self.read_bytes)

class SampleApp2(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.button = ttk.Button(text="start", command=self.start)
        self.button.pack()
        self.progress = ttk.Progressbar(self, orient="horizontal",
                                        length=200, mode="determinate")
        self.progress.pack()

        self.bytes = 0
        self.maxbytes = 0

    def start(self):
        self.progress["value"] = 0
        self.maxbytes = 50000
        self.progress["maximum"] = 50000
        self.read_bytes()

    def read_bytes(self):
        '''simulate reading 500 bytes; update progress bar'''
        self.bytes += 500
        self.progress["value"] = self.bytes
        self.progress.update()
        while self.bytes < self.maxbytes:
            sleep(0.1)
            self.bytes += 500
            self.progress["value"] = self.bytes
            self.progress.update()


###Short names=================================================================
ffp = find_file_path
fdp = find_directory_path
pmb = popup_message_box
giv = get_integer_value

###Testing=====================================================================
if __name__ == '__main__':
    options = {
        '-f':ffp,
        '-d':fdp,
        '-m':pmb,
        '-v':giv
    }
    import sys
    try:
        sys.argv[1]
        if sys.argv[1] == '-v':
            print('Module name: {}'.format(sys.argv[0]))
            print('Version info:', __version__)
        elif sys.argv[1] == '-t':
            print('Testing mode!')
            print(sys.argv[2])
            res = options[sys.argv[2]]()
            print('='*(len('result: ') + len(str(res))))
            print('result:', res)
            print('='*(len('result: ') + len(str(res))))
        else:
            print('Not implemented!')
    except IndexError:
        print('Mode var wasn\'t passed!')