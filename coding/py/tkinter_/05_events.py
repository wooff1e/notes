import tkinter as tk
from tkinter import ttk


root = tk.Tk()
root.resizable(False, False)
root.title("Events")


my_input = ttk.Entry(root, width=30)
my_input.insert(tk.END, 'something')
my_input.grid(row=0, column=0)

def edit_input(arg=None):
    print(f"Edit input: {arg}")
    my_input.delete(0, tk.END)
    my_input.insert(tk.END, 'something else')
    my_input.event_generate('<<EditedInput>>')

def report(event):
    print('input changed')

    
my_input.bind('<Return>', edit_input)
my_input.bind('<Return>', lambda event: edit_input('some_argument'))

my_input.bind('<<EditedInput>>', report)


# -----------------------------------
root.mainloop()
