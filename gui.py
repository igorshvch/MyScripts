https://stackoverflow.com/questions/46403292/how-should-a-scrollable-spreadsheet-be-displayed-within-tkinter
from tkinter import *

class App:
    def __init__(self, root):
        self.entry = []
        self.sv = []
        self.root = root
        self.canvas = Canvas(self.root, background="#ffffff", borderwidth=0)
        self.frame = Frame(self.canvas, background="#ffffff")
        self.scrolly = Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollx = Scrollbar(self.root, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.scrolly.set)#, xscrollcommand=self.scrollx.set)
        self.canvas.create_window((4,4), window=self.frame, anchor="nw", tags="self.frame")
        self.scrolly.pack(side="left", fill="y")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.scrollx.pack(side="bottom", fill="x")
        self.frame.bind("<Configure>", self.onFrameConfigure)
        for i in range(15):
            self.entry.append([])
            self.sv.append([])
            for c in range(30):
                self.sv[i].append(StringVar())
                self.sv[i][c].trace("w", lambda name, index, mode, sv=self.sv[i][c], i=i, c=c: self.callback(sv, i, c))
                self.entry[i].append(Entry(self.frame, textvariable=self.sv[i][c]).grid(row=c, column=i))
    def onFrameConfigure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    def callback(self, sv, column, row):
        print("Column: "+str(column)+", Row: "+str(row)+" = "+sv.get())

root = Tk()
App(root)
root.mainloop()