import tkinter as tk

root = tk.Tk()

for i in range(100):
    for j in range(10):
        tk.Entry(root).grid(row=i, column=j)

root.mainloop()