import tkinter as tk
'''
by default element goes to the top center of the parent and keeps its dimensions. 
Several elements will form a verticle stack in the middle


.pack(side="left") will create a horizontal stack in the middle

fill='x' --> take up all horizontal space reserved
(if side="left", then the reserved space = width of the element)
fill='y' --> take up all vertical space reserved
fill='both'

expand=True --> take up all available space (not occupied by other widgets)
and grow with the window (but only space with specified with 'fill' will be colored,
creating a margin)
'''

root = tk.Tk()
root.geometry('600x400')
tk.Label(root, text="Label 1", bg="green").pack(side="left", fill="y", expand=True)
tk.Label(root, text="Label 2", bg="red").pack(side="left")
root.mainloop()


root = tk.Tk()
root.geometry('600x400')
tk.Label(root, text="Label 1", bg="green").pack(side="left", fill="y")
tk.Label(root, text="Label 2", bg="red").pack(side="top", fill="x")
root.mainloop()


root = tk.Tk()
root.geometry('600x400')
tk.Label(root, text="Label 1", bg="green").pack(side="left", fill="both")
tk.Label(root, text="Label 2", bg="red").pack(side="top", fill="both")
root.mainloop()


# Even if either label doesn't fill
root = tk.Tk()
root.geometry('600x400')
tk.Label(root, text="Label 1", bg="green").pack(side="left")
tk.Label(root, text="Label 2", bg="red").pack(side="top", fill="both")
root.mainloop()


root = tk.Tk()
root.geometry('600x400')
tk.Label(root, text="Label 1", bg="green").pack(side="left", fill="both")
tk.Label(root, text="Label 2", bg="red").pack(side="top")
root.mainloop()


# expand can make it grow as much as possible. 
# It won't hide other widgets, but other widgets will be compressed
root = tk.Tk()
root.geometry('600x400')
tk.Label(root, text="Label 1", bg="green").pack(side="left", expand=True, fill="both")
tk.Label(root, text="Label 2", bg="red").pack(side="top")
root.mainloop()


# expanding two widgets means they share the available space evenly
root = tk.Tk()
root.geometry('600x400')
tk.Label(root, text="Label 2", bg="red").pack(side="top", expand=True, fill="both")
tk.Label(root, text="Label 2", bg="red").pack(side="top", expand=True, fill="both")
root.mainloop()


# whichever side comes first gets expansion priority
root = tk.Tk()
root.geometry('600x400')
tk.Label(root, text="Label left", bg="green").pack(side="left", expand=True, fill="both")
tk.Label(root, text="Label top", bg="red").pack(side="top", expand=True, fill="both")
tk.Label(root, text="Label top", bg="red").pack(side="top", expand=True, fill="both")
root.mainloop()


root = tk.Tk()
root.geometry('600x400')
tk.Label(root, text="Label top", bg="red").pack(side="top", expand=True, fill="both")
tk.Label(root, text="Label top", bg="red").pack(side="top", expand=True, fill="both")
tk.Label(root, text="Label left", bg="green").pack(side="left", expand=True, fill="both")
root.mainloop()


root = tk.Tk()
root.geometry("600x400")
rectangle_1 = tk.Label(root, text="Rectangle 1", bg="green", fg="white") 
rectangle_1.pack(side="left", ipadx=10, ipady=10, fill="both", expand=True)
rectangle_2 = tk.Label(root, text="Rectangle 2", bg="red", fg="white")
rectangle_2.pack(side="top", ipadx=10, ipady=10, fill="both", expand=True)
rectangle_3 = tk.Label(root, text="Rectangle 3", bg="black", fg="white")
rectangle_3.pack(side="left", ipadx=10, ipady=10, fill="both", expand=True)
rectangle_4 = tk.Label(root, text="Rectangle 4", bg="black", fg="white")
rectangle_4.pack(side="top", ipadx=10, ipady=10, fill="both", expand=True)
root.mainloop()