import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.geometry('600x400')

style = ttk.Style(root)
print(style.theme_names())
print(style.theme_use()) # current theme
#style.theme_use("clam")

name_label = ttk.Label(root, text="Name: ")
name_label.pack(side="left", padx=(0, 10))
print(name_label['style']) # nothing means default

print(name_label.winfo_class()) # TLabel
print(style.layout('TLabel'))

print(style.element_options('Label.border'))
print(style.lookup('TLabel', 'font'))

# without relief='solid' the changes won't be visible
style.configure('TLabel', font=('Segoe UI', 20), 
                relief='solid', borderwidth=4, 
                bordercolor='#f00') # not available in this theme

root.mainloop()



root = tk.Tk()

style = ttk.Style(root)
style.configure('CustomStyle.TLabel', padding=20)

name_label = ttk.Label(root, text="Name: ")
name_label.pack(side="left")
name_label['style'] = 'CustomStyle.TLabel'

another_lbl = ttk.Label(root, text="one more", style='CustomStyle.TLabel')
another_lbl.pack(side="left")
root.mainloop()