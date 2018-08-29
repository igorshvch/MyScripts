import tkinter as tk
from tkinter.filedialog import askopenfilename as ask_fn

def find_file_path():
    tk.Tk().withdraw()
    return ask_fn()