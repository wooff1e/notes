import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


root = tk.Tk()
root.resizable(False, False)
root.title("Widget Examples")


# -- Label as text --
label = ttk.Label(root, text="Hello, World!", padding=20)
label.config(font=("Segoe UI", 20))  # Could be in the constructor instead.
label.pack()

# -- Labels with images --
image = Image.open("test_image.png").resize((64, 64))
photo = ImageTk.PhotoImage(image)
ttk.Label(root, image=photo, padding=5).pack()

# This is how you change an image associated with a label, if necessary:
# label["image"] = photo


# SEPARATOR
main_sep = ttk.Separator(root, orient="horizontal")
main_sep.pack(fill="x")  # without it not visible


# -- Changing the text of a label dynamically --
greeting = tk.StringVar()
label = ttk.Label(root, padding=10)
label["textvariable"] = greeting
greeting.set("Hello, John!")  # This can change during your program and the label will update.
label.pack()

# -- Combining text and images --
text_image = Image.open("test_image.png")
text_photo = ImageTk.PhotoImage(text_image)
ttk.Label(root, text="Image with text.", image=text_photo, padding=5, compound="right").pack()


# CHECK BUTTONS
check_button = ttk.Checkbutton(root, text="Check me!")
check_button.pack()
check_button["state"] = "disabled"  # "normal" is the counterpart

# -- All options --
selected_option = tk.StringVar()

def print_current_option():
    print(selected_option.get())

check = tk.Checkbutton(
    root,
    text="Check Example",
    variable=selected_option,
    command=print_current_option,
    onvalue="On",
    offvalue="Off"
)
check.pack()


# RADIO BUTTONS
storage_variable = tk.StringVar()

option_one = ttk.Radiobutton(
	root,
	text="Option 1",
	variable=storage_variable,
	value="First option"
)
option_two = ttk.Radiobutton(
	root,
	text="Option 2",
	variable=storage_variable,
	value="Second option"
)
option_three = ttk.Radiobutton(
	root,
	text="Option 3",
	variable=storage_variable,
	value="Third option"
)
option_one.pack()
option_two.pack()
option_three.pack()


# COMBO BOXES
selected_weekday = tk.StringVar()
weekday = ttk.Combobox(root, textvariable=selected_weekday)
weekday["values"] = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
weekday["state"] = "readonly"  # "normal is the counterpart"
weekday.pack()

def handle_selection(event):
    print("Today is", weekday.get())
    print("But we're gonna change it to Friday.")
    weekday.set("Friday")
    print(weekday.current())  # This can return -1 if the user types their own value.

weekday.bind("<<ComboboxSelected>>", handle_selection)


# LIST BOXES
programming_languages = ("C", "Go", "JavaScript", "Perl", "Python", "Rust")

pl = tk.StringVar(value=programming_languages)
pl_select = tk.Listbox(root, listvariable=pl, height=6)
pl_select.pack(padx=10, pady=10)
# Allows multiple selection, "browse" is the counterpart (yeah, I know it's a bad name!)
pl_select["selectmode"] = "extended"  

def handle_selection_change(event):
    selected_indices = pl_select.curselection()
    for i in selected_indices:
        print(pl_select.get(i))

pl_select.bind("<<ListboxSelect>>", handle_selection_change)


# SPIN BOXES
initial_value = tk.StringVar(value=20)
spin_box = tk.Spinbox(
    root,
    from_=0,
    to=30,
    textvariable=initial_value,
    wrap=False)
# spin_box = tk.Spinbox(root, values=(5, 10, 15, 20, 25, 30), textvariable=initial_value, wrap=False)
# The alternative uses values instead of a range.

spin_box.pack()

print(spin_box.get())  # You'd usually use this when clicking a button (e.g. to submit)
# Can't call `.get()` after the mainloop finishes, of course.


# SCALES
def handle_scale_change(event):
    print(scale.get())  # `.set()` can be used to change the value dynamically.

scale = ttk.Scale(root, orient="horizontal", from_=0, to=10, command=handle_scale_change)
scale.pack(fill="x")

# scale["state"] = "disabled"  # "normal" is the counterpart


# SCROLL BAR
text = tk.Text(root, height=4)
text.pack(side='left')
text.insert("1.0", "Enter multiline comment...")
# The position is given as two numbers, separated by a `.`.
# First number is the line number, starting at 1.
# Second number is character number within the line, starting at 0.
# So 1.0 is the first line, first character.

text_scroll = ttk.Scrollbar(root, orient="vertical", command=text.yview)
text_scroll.pack( side = 'right', fill = 'y') # north-south
text['yscrollcommand'] = text_scroll.set


# -- Disable text widget --
#text["state"] = "disabled"  # "normal" is the counterpart

# -- Get text content --
text_content = text.get("1.0", "end")
print(text_content)


# -----------------------------------
root.mainloop()
