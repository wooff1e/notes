
from kivy.app import App
from kivy.uix.button import Button
# if you want your .kv file to be called something else
from kivy.lang import Builder
#Builder.load_file('path/to/your/file.kv')
from kivy.core.window import Window
# preferred way to change size
Window.size = (800, 600)
'''
Using Config.write() will set your changes to the default settings. 
If you accidentally run this and want to revert the changes, 
you can either run Config.write() again with the default values (therefore reverting your changes) 
or you can open up $HOME/.kivy/config.ini with a text editor and edit it.
'''
from kivy.config import Config
Config.set('graphics', 'width', '200')
Config.set('graphics', 'height', '200')
Config.write()

# custom widget
def callback(instance):
    print('The button <%s> is being pressed' % instance.text)

class FunkyButton(Button):
    def __init__(self, **kwargs):
        super(FunkyButton, self).__init__(**kwargs)
        self.text='Hello world'
        self.pos=(50, 50)
        self.size = (100, 100)   
        self.size_hint = (None, None)

'''
name the .kv file as you App class but in lowercase and without the "App" suffix 
(note that the name doesn't have to end with App but if it does, kivy will ignore it 
when searching for .kv file)
'''
class ExampleApp(App):
    def build(self):
        # return your Root Widget
        return Button(
            text='Hello world', 
            pos=(50, 50),               # absolute position: x, y (px)
            size = (100, 100),          # absolute size: w, h
            # needed for size parameter to work
            size_hint = (None, None)    # fraction of the parent size in range [0.0, 1.0], default=(1, 1)
            #pos_hint=(0, 0),           # relative to the parent, default=(0, 0) <-- BOTTOM LEFT !            
        )    
    
# ExampleApp().run()

class MyApp(App):
    def build(self):
        Window.clearcolor = (1,1,1,1)
        btn = FunkyButton()
        btn.bind(on_press=callback)
        return btn

MyApp().run()

'''
UX widgets:
    Label, 
    Button, 
    CheckBox, 
    Image, 
    Slider, 
    Progress Bar, 
    Text Input, 
    Toggle button, 
    Switch, 
    Video

Layouts: A layout widget does no rendering but just acts as a trigger that arranges its children in a specific way.

AnchorLayout:   widgets can be anchored to the 'top', 'bottom', 'left', 'right' or 'center'.
BoxLayout:      sequential arrangement (either 'vertical' or 'horizontal' orientation)
FloatLayout:    unrestricted.
RelativeLayout: child widgets are positioned relative to the layout.
GridLayout:     a grid defined by the rows and cols properties.
PageLayout:     simple multi-page layouts, in a way that allows easy flipping from one page to another using borders.
ScatterLayout:  similar to a RelativeLayout, but widgets can be translated, rotated and scaled.
StackLayout:    widgets are stacked in a lr-tb (left to right then top to bottom) or tb-lr order.


Complex UX widgets: Non-atomic widgets that are the result of combining multiple classic widgets. 
We call them complex because their assembly and usage are not as generic as the classical widgets.
    Bubble, 
    Drop-Down List, 
    FileChooser, 
    Popup, 
    Spinner, 
    RecycleView, 
    TabbedPanel, 
    Video player, 
    VKeyboard,

Behaviors widgets: These widgets do no rendering but act on the graphics instructions or interaction (touch) behavior of their children.
    Scatter, 
    Stencil View

Screen manager: Manages screens and transitions when switching from one to another.
    Screen Manager
'''