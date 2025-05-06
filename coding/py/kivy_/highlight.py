
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.stacklayout import StackLayout
from kivy.properties import ListProperty
from kivy.core.window import Window

class ArrowButton(Button):
    original_color = ListProperty([0, 0, 0, 1])
    highlight_color = ListProperty([.9, .9, .9, 1])

    def __init__(self, *args, **kwargs):
        super(ArrowButton, self).__init__(*args, **kwargs)
        self.color = self.original_color
        Window.bind(mouse_pos=self.on_mouseover)
        
    def on_mouseover(self, window, pos):
        if self.collide_point(*pos):
            self.color = self.highlight_color
        else:
            self.color = self.original_color

class MyLayout(StackLayout):
    def __init__(self, **kwargs):
        super(MyLayout, self).__init__(**kwargs)

class HighlightApp(App):
    def build(self):
        return MyLayout()

HighlightApp().run()