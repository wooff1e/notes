# import tkinter as tk
# import tkinter.font as font

# window = tk.Tk()
# families = font.families()

# for family in families:
#     if "Liberation" in family:
#         print(family)
# window.mainloop()

'''
When using Conda's Python for a Tkinter program, fonts may not display correctly due to the lack of Freetype support in Conda's build of the Tk library. This issue can be addressed by building the Tcl/Tk libraries with Freetype support yourself and then using them in your Conda environment
'''
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as font

class MyFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.m_title = tk.StringVar(value="My Title")

        title_font = font.Font(family='Liberation Serif', size=25)
        self.m_title_lbl = ttk.Label(self, textvariable=self.m_title, font=title_font)
        self.m_title_lbl.grid(row=0, column=0) # Simple grid placement

if __name__ == "__main__":
    window = tk.Tk()
    frame = MyFrame(window)
    frame.grid(row=0, column=0) # Simple grid for the frame as well
    window.mainloop()