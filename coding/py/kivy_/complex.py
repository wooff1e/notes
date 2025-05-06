from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.properties import ObjectProperty
from kivy.uix.widget import Widget



class FunkyButton(Button):
    pass

class GameScreen(Widget):
    pass

class GridScreen(Widget):
    pass

class LoginScreen(BoxLayout):
    def __init__(self, **kwargs):
        super(LoginScreen, self).__init__(**kwargs)

        name = ObjectProperty(None)
        pizza = ObjectProperty(None)

        self.add_widget(Label(text='User Name')) 
        self.username = TextInput(multiline=False)
        self.add_widget(self.username)

        self.add_widget(Label(text='password'))
        self.add_widget(TextInput(password=True, multiline=False))
        self.add_widget(GameScreen())

        funky = FunkyButton(
            text='Hello world',
            pos=(50, 50),
            size_hint = (None, None),            
            size = (100, 100),
        )
        funky.bind(on_press=self.do_some_stuff)
        self.add_widget(funky)

        # children objects specified in kivi file are created AFTER the __init__ !!! T_T

    # this method will be called when a widget instance is added or removed from a parent
    def on_parent(self, *largs):
        self.child.some_feature = True
        self.child.update()

    def do_some_stuff(self, value):
        pass

class ComplexApp(App):   
    def build(self):
        return LoginScreen()

ComplexApp().run()
