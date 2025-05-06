# VARIABLES & FUNCTIONS
'''
_leading_underscore             weak "internal use" indicator. import * will avoid these objects
trailing_underscore_            used to avoid conflicts with Python keywords
__double_leading_underscore     name mangling for class attributes (inside class FooBar, __boo becomes _FooBar__boo).
__double_underscores__          "magic" objects or attributes that live in user-controlled namespaces. For example, __init__, __import__ or __file__. You should never invent such names, but only use them as documented.
'''

a = b = 0
a, b = b, a # Shorthand for in-place value swapping

# Input / Output
username = input("Enter username:") 
print(username)
repr()	# Returns a readable version of an object


'''
bitwise operators:
& 	|	^	~	<<	>>
'''

for x in range(1, 7, 2): 	# including 1, excluding 7, incrementing by 2
    print(x)


# Arbitrary Arguments
def my_function(*kids):
    print("The youngest child is " + kids[2])
my_function("Emil", "Tobias", "Linus")

def my_function(**kid):
    print("His last name is " + kid["lname"])
my_function(fname = "Tobias", lname = "Refsnes")


# Function argument unpacking
def myfunc(x, y, z):
    print(x, y, z)

tuple_vec = (1, 0, 1)
myfunc(*tuple_vec)				# 1, 0, 1

dict_vec = {'x': 1, 'y': 0, 'z': 1}
myfunc(**dict_vec)				# 1, 0, 1


# GLOBAL VARIABLES
def myfunc():
    global x # create
    x = "fantastic"

x = "awesome"
def myfunc():
    global x # change
    x = "fantastic"

print(globals())	# Returns the current global symbol table as a dictionary
locals()	        # Returns an updated dictionary of the current local symbol table


### Lambda
func = lambda x : x + 10
print(func(5))

def myfunc(n):
    return lambda a : a * n

mydoubler = myfunc(2)
print(mydoubler(11))

mytripler = myfunc(3)
print(mytripler(11))


### DECORATORS - wrap a function, modifying its behavior
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Something is happening before the function is called.")
        func(*args, **kwargs)
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_whee():
    print("Whee!")

# call
say_whee()   # instead of my_decorator(say_whee)

# decorator with parameters
def outer_decorator(access_level):
    def my_decorator(func):
        def wrapper(*args, **kwargs):
            if access_level == 'guest':
                print("access denied")
            else:
                func(*args, **kwargs)
            return wrapper
        return my_decorator


@outer_decorator('admin')
def say_whee():
    print("Whee!")

 
@outer_decorator('guest')
def say_whee():
    print("Whee!")

 

# STRINGS
'''
formats:
:b      binary
:d      decimal
:x, :X  hex
:o      octal 
:n      number
:f, :F  fix point
:e, :E  scientific 
:g, :G  general
:%      percentage
'''
epoch = 2
f'{epoch:03d}'      # 002
f'{3:.2f} dollars'  # 3.00 dollars
f'{.3:%} percent'   # 30.000000% percent

# :< :> :^      alignment within the available space

txt = "We have {:<8} chickens." # set the available space to 8 characters
txt.format(49) 
# 'We have 49       chickens.'

txt = "We have {:^8} chickens."
txt.format(49) 
# 'We have    49    chickens.'

# := :+ :- :    formats the sign to the left most position
txt = "The temperature is between {:+} and {:+} degrees celsius."
txt.format(-3, 7) # ... between -3 and +7

# :, :_         thousand separator
txt = "The universe is {:,} years old."
txt.format(13800000000) # 13,800,000,000

# convert into the corresponding unicode character
char = 1570
f'{char:c}' # 'آ'

# Escape Characters
'\" \' \\ \n \r \t \b'
'\f (form feed)' 
'\ooo (octal value)'
'\xhh (hex value)'

# string methods
s = 'sopKFYa[woi4trqw]'
s.format()
s.format_map()
s.lower()
s.upper()
s.capitalize()
s.casefold()
s.swapcase()
s.center(30)
s.ljust(30)
s.rjust(30)
s.count('woi')
s.strip()
s.rstrip()
s.lstrip()
s.split()
s.rsplit()
s.splitlines()
s.join()
txt = "Hello Sam!"
mytable = str.maketrans("S", "P")
txt.translate(mytable) # 'Hello Pam!'
s.startswith()
s.endswith()

s.find()
s.rfind()
s.replace()

s.index()
s.rindex()

s.encode()
s.expandtabs()

s.partition()
s.rpartition()

s.title()
s.translate()
s.zfill()
s.islower()
s.isupper()
s.isalnum()
s.isalpha()
s.isdecimal()
s.isdigit()
s.isidentifier()
s.isnumeric()
s.isprintable()
s.isspace()
s.istitle()

# COLLECTIONS
thistuple = ("apple",) # immutable!
thislist = ["apple", "pear"]
thislist.index('pear') # index of the 1st element with the value
thislist.reverse()

# this is not copying, but creating pointers to the same data!!
a = thislist
print(id(thislist))
print(id(a))

'''
Note: The nested loops in list comprehension don't work like normal nested loops. 
`for i in range(2)` is executed before `row[i] for row in matrix`. 
Hence at first, a value is assigned to `i` then item directed by `row[i]` is appended in the `transpose` variable. 
'''
num_list = [y for y in range(100) if y % 2 == 0 if y % 5 == 0]
# [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

obj = ["Even" if i%2==0 else "Odd" for i in range(10)]
# ['Even', 'Odd', 'Even', 'Odd', 'Even', 'Odd', 'Even', 'Odd', 'Even', 'Odd']

matrix = [
    [1,2], 
    [3,4], 
    [5,6], 
    [7,8]
    ]
transpose = [[row[i] for row in matrix] for i in range(2)]
# [[1, 3, 5, 7], [2, 4, 6, 8]]

# SET
this_set = {1, 2, 3}
x = frozenset(this_set)	# make set immutable

second_set = set(num_list)
this_set.isdisjoint(second_set)
this_set.issubset(second_set)
this_set.issuperset(second_set)
this_set.union(second_set)	
this_set.difference(second_set)
this_set.difference_update(second_set)
this_set.intersection(second_set)
this_set.intersection_update(second_set)
this_set.symmetric_difference(second_set)
this_set.symmetric_difference_update(second_set)

# DICT
this_dict = {'year': 1999, 'name': 'aaa'}

# Dictionary Methods
this_dict.items()		# list of tuples (key-value pairs)
this_dict.keys()		# list of keys
this_dict.values()	    # list of values

this_dict.pop('year')		    # Removes the element with the specified key
this_dict.popitem()	    # Removes the last inserted key-value pair (random item in versions before 3.7)
this_dict.setdefault('nom', 'n')	# value of key, if doesn’t exist – insert with specified value

# ITERATOR
this_dict = {'year': 1999, 'name': 'aaa'}

# Dictionary Methods
this_dict.items()		# list of tuples (key-value pairs)
this_dict.keys()		# list of keys
this_dict.values()	    # list of values

this_dict.pop('year')		    # Removes the element with the specified key
this_dict.popitem()	    # Removes the last inserted key-value pair (random item in versions before 3.7)
this_dict.setdefault('nom', 'n')	# Returns value of specified key, if doesn’t exist – insert with specified value


# OOP
# "smart" access to a class' methods / properties
class Person:
    # any change here will be reflected in all objects
    name = "John"
    age = 36
    country = "Norway"

    def __init__(self, name, age):
        self.name = name
        self.age = age

    # string representation of an object (eg. used by print())
    def __str__(self):
        return f'{self.name}: {self.age}'
    
    # a way to recreate object (eg. for python debugger)
    def __repr__(self):
        return f'Person("{self.name}", {self.age})'

    def instance_method(self):
        pass

    # often used as factories (eg. if we want "named" constructors)
    @classmethod
    def class_method(cls, name, age):
        return cls(name, age)

    @staticmethod
    def static_method():
        pass

                               
vars(Person)
dir(Person)

[f for f in dir(Person) if not f.startswith('_')]

person = Person()

# 1
person.instance_method()
# 2
Person.instance_method(person)


Person.class_method() # python will call it as Person.class_method(Person) behind the scenes
Person.static_method() # more like a separate function placed within the class


f"something {self.name!r}.."       # !r calls __repr__() method (here it will add quotes)


# note that @classmethod is being created during class processing (the class doesn't exist yet),
# so if you use type hinting, you can't do
def class_method(cls, name: str, age: int) -> Person:
    pass
# instead do this:
def class_method(cls, name: str, age: int) -> "Person":
    pass
 

class SomeClass():
    def __init__(self, temperature=0):
        self._temperature = temperature

    def get_temperature(self):
        print("Getting value...")
        return self._temperature
    
    temperature = property(get_temperature)

human = SomeClass(37)
print(human.temperature)    # imitates direct access, but uses getter underneath

# delattr() getattr() setattr() hasattr()

# property(fget, fset, fdel, doc)	# all parameters are optional - names of functions
# getattr
class Name():
    def __init__(self):
        self.name = "John"
    def change_name(self):
        self.name = 'James'


class Person():
    def __init__(self):
        self.name = Name()
        self.country = "Norway"

names = ['name']

p = Person()

for n in names:
    temp = getattr(p, n)
    temp.change_name()

# for part in vars(p):
#     if part in names:
#         print(part)
#         p.part.change_name()

print(p.name.name)

# INHERITANCE
'''
Inheritance: LoggingOD --> LoggingDict --> OrderedDict --> dict --> object. 
This means that the `super()` call in `LoggingDict.__setitem__` 
now dispatches the key/value update to OrderedDict instead of dict.
'''
class LoggingOD(LoggingDict, collections.OrderedDict):
    pass

from typing import NamedTuple
import time

class Date(NamedTuple):
    year: int
    month: int
    day: int


today = time.localtime()
chosen_date = Date(today[0], today[1], today[2])
new_date = Date(*chosen_date)


class Bookshelf:
    def __init__(self, *books):
        self.books = books                          

shelf = Bookshelf('book1', 'book2', 'book3')
 
# all paths where python searches for imports (the $PATH in linux)
import sys
print(sys.path)
 


# DON'T do a parameter with mutable value by default:
class Student:
    def __init__(self, grades=[]): # default parameters are avaluated once when the function is defined
        self.grades = grades # now self.grades of ANY object points to the list created via grades=[]

a = Student()
b = Student()
a.grades.append(90)
# no BOTH students have grades [90] !!!

# here no problem
a = Student([80, 100])
b = Student()

# instead do this:
class Student:
    def __init__(self, grades=None):
        self.grades = grades or []


# DATES
# Use exclusively the system `time` module instead of the `datetime` module 
# to prevent ambiguity issues with daylight savings time (DST)!
import time
# a float representing the time in seconds since the system epoch
t = time.time()
print(t)            # 1652355709.0522957

# Conversion to any time format, including local time, is easy:
local = time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(t))
print(local)        # 2022-05-12 13:42 CEST

gmt = time.strftime('%Y-%m-%d %H:%M %Z', time.gmtime(t))
print(gmt)          # 2022-05-12 11:43 GMT

'''
Format codes:
    %a	Weekday, short version
    %A	Weekday, full version
    %w	Weekday as a number 0-6, 0 is Sunday
    %d	Day of month 01-31
    %b	Month name, short version
    %B	Month name, full version
    %m	Month as a number 01-12
    %y	Year, short version, without century
    %Y	Year, full version
    %H	Hour 00-23
    %I	Hour 00-12
    %p	AM/PM
    %M	Minute 00-59
    %S	Second 00-59
    %f	Microsecond 000000-999999
    %z	UTC offset
    %Z	Timezone
    %j	Day number of year 001-366
    %U	Week number of year, Sunday as the first day of week, 00-53
    %W	Week number of year, Monday as the first day of week, 00-53
    %c	Local version of date and time
    %x	Local version of date
    %X	Local version of time
    %%	A % character
'''

# EXCEPTIONS
# Assertion
assert(w.shape == (dim, 1))
assert(isinstance(b, float) or isinstance(b, int))    

## Exception Handling
try:
    f = open("demofile.txt")
    f.write('bla bla bla')
except NameError:
    print("Variable x is not defined")
except:
    print("Something else went wrong")
else:
    # optional
    # executed if no errors were raised:
    print("Nothing went wrong")
finally:
    # optional
    # executed regardless if the try block raises an error or not.
    f.close()	#The program can continue, without leaving the file object open.

# Raise an exception
x = -1
if x < 0:
    raise Exception("Sorry, no numbers below zero")

# You can define what kind of error to raise, and the text to print to the user.
x = "hello"

if not type(x) is int:
    raise TypeError("Only integers are allowed")

def smth_unreliable():
    pass

try:
    x = int(x)
except ValueError:
    print(ValueError)
    
try:
    smth_unreliable()
except Exception as e:  # catch any exception
    print(e)


# FILES
'''
'r'    Read	    Default value. ERROR if not exists
'x'    Create 	Creates the specified file, returns an ERROR if exists

'a'    Append   Opens a file for appending, creates file if not exists
'w'    Write 	Opens a file for writing, creates file if not exists

't'    Text 	Default value. Text mode
'b'    Binary	Binary mode (e.g. images)
'''

path = 'demofile.txt'

f = open(path)		
# this is the same as:
f = open(path, 'rt')

f.close() # sometimes, due to buffering, changes may not show until you close the file

with open(path, 'rt') as f:
    pass

# WRITE
with open(path, 'w') as f:
    f.write('Woops! I have deleted the content!')

with open(path, 'a') as f:
    f.write('Now the file has more content!')


f.writelines()	# Writes a list of strings to the file

f.writable()	# Returns whether the file can be written to or not

f.readable()	# Returns whether the file stream can be read or not

# READ
print(f.read())			    # read the whole file
print(f.read(5)) 			  # Return the 5 first characters of the file:
print(f.readline())			# read one line
print(f.readline())			# read the next line

print(f.readlines())	# Returns a list of lines from the file

with open(path) as f:
    for x in f:
        print(x)

    # or:
    lines = f.readlines() # ['0.000000\n', '2.000000\n', '4.000000\n']
    lines = [line.rstrip() for line in lines] # ['0.000000', '2.000000', '4.000000']

f.detach()	    # Returns the separated raw stream from the buffer
f.fileno()	    # Returns number (representing the stream, from the OS's perspective
f.flush()	    # Flushes the internal buffer
f.isatty()	    # Returns whether the file stream is interactive or not
f.seek()	    # Change the file position
f.seekable()	# Returns whether the file allows us to change the file position
f.tell()	    # Returns the current file position
f.truncate()	# Resizes the file to a specified size


# CSV
import csv
inputs_paths = []
target_paths = []

# WRITE
with open('some_csv.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    for i in range(len(inputs_paths)):
        csvwriter.writerow([inputs_paths[i], target_paths[i]])
        

# READ
with open('some_csv.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        inputs_paths.append(row[0])
        target_paths.append(row[1])


# PATHS
import os
paths = os.listdir('some_dir')
path = paths[0] # 'img001.png'
path_elements = os.path.splitext(path)  # ['img001', '.png']



from shutil import copyfile
source = 'data/logs.txt'
destination = 'data/logs_copy.txt'
copyfile(source, destination)


from os import chdir
chdir('..')
chdir('code_fragments_py')


# COMMAND LINE ARGS
# bash train.sh
'''
#!/bin/sh
python3 main.py \
--image_path some/path
'''

import argparse
import os

parser = argparse.ArgumentParser(description='Synthetic Defocussing Using Depth Estimation')

parser.add_argument('--image_path', type=str, help='path to input image', default='images/sample2.png')
parser.add_argument('--model_path', type=str, help='path to saved model', default='blah')
parser.add_argument('--blur_method', type=str, help='the type of blur to be applied', default='gaussian')

args = parser.parse_args()

img_path = os.path.abspath(args.image_path)
model_path = os.path.abspath(args.model_path)
blur_method = args.blur_method

os.system("python ./depth/depth_simple.py --model_path " + model_path + " --image_path " + img_path)
os.system("python ./defocus/defocus.py --image_path " + img_path + " --blur_method " + blur_method)


# REGEX
## Regular Expressions
import re

# findall() – returns a list containing all matches in the order they are found (if no matches  empty list)
str = "The rain in Spain"
x = re.findall("ai", str)

# search() – returns Match object of the first match (if no matches  the value None)
x = re.search(r"\bS\w+", str) 	# looks for any words that starts with an upper case "S"
print("The first match is located in position:", x.start()) 
print(x.span()) 	# tuple containing the start-, and end positions of the first match 
print(x.string) 	# the string passed into the function
print(x.group())	# the part of the string where there was a match

# split() – returns a list where the string has been split at each match
x = re.split("\s", str) 		# split at each white-space character
x = re.split("\s", str, 1)		# maxsplit parameter - the number of occurrences

# sub() – replaces the matches with the text of your choice.
x = re.sub("\s", "9", str) 	# replace every white-space character with the number 9
x = re.sub("\s", "9", str, 2) 	# replace the first 2 occurrences

'''
    *[]* - A set of characters
        [arn]       one of the specified characters (a, r, or n) are present
        [a-n]	any lower case character, alphabetically between a and n
        [^arn]	any character EXCEPT a, r, and n
        [0123]	any of the specified digits (0, 1, 2, or 3) are present
        [0-9]	any digit between 0 and 9
        [0-5][0-9]	any two-digit numbers from 00 and 59
        [a-zA-Z]	any character alphabetically between a and z, lower OR upper case
        [+]		any character with no special meaning: 	+ * . | () $ {}

    *\* - Special sequence (can also be used to escape special characters)
        \A	specified characters are at the beginning of the string	("\AThe")
        \b	specified characters are at the beginning or at the end of a word (r"\bain"  r"ain\b")
        \B	specified characters are present, but NOT at the beginning/end of a word
        \d	string contains digits (numbers from 0-9)	
        \D	string DOES NOT contain digits	
        \s	string contains a white space character	
        \S	string DOES NOT contain a white space character
        \w	string contains any word characters (a-Z, 0-9, _ )
        \W	string DOES NOT contain any word characters
        \Z	specified characters are at the end of the string ("Spain\Z")

    .	Any character (except newline character)		"he..o"
    ^	Starts with						"^hello"
    $	Ends with						"world$"
    *	Zero or more occurrences				"aix*"
    +	One or more occurrences				        "aix+"
    {}	Exactly the specified number of occurrences		"al{2}"
    |	Either or						"falls|stays"
    ()	Capture and group	
'''