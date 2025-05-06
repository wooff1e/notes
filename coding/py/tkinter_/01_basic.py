import tkinter as tk
from tkinter import ttk # newer "themed widgets"

# high dpi for windows 10
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass


# main window
root = tk.Tk()
root.title("Greeter")
root.geometry('600x400')

name_label = ttk.Label(root, text="Name: ")
name_label.pack(side="left", padx=(0, 10))


# ----- user input field -----
user_name = tk.StringVar()
name_entry = ttk.Entry(root, width=15, textvariable=user_name)
name_entry.pack(side="left")
name_entry.focus()

def greet():
    print(f"Hello, {user_name.get() or 'World'}!")

greet_button = ttk.Button(root, text="Greet", command=greet)
greet_button.pack(side="left", fill="x", expand=True)
#-----------------------------

quit_button = ttk.Button(root, text="Quit", command=root.destroy)
quit_button.pack(side="right", fill="x", expand=True)


root.mainloop()