import tkinter as tk
from tkinter import filedialog as fd

def find_file_path():
    tk.Tk().withdraw()
    return fd.askopenfilename()

def find_directory_path():
    tk.Tk().withdraw()
    return fd.askdirectory()