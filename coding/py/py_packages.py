
# print(dir(package))			# list all the function names (or variable names) in a module

'''
structure:

--parent_dir
    |--module
        |--some_code.py
        |--__main__.py
        |--__init__.py

'''

# __init__.py
from .some_code import SomeStuff

# __main__.py
from .some_code import SomeStuff
# python3 -m parent_dir.module




import math
dir(math)       # ['__doc__', ..., 'nan', 'pi', 'pow', ...]

# global namespace
dir()           # ['__annotations__', '__builtins__', ..., 'math']


# import * is used for a module, all objects from the module are imported into the local symbol table, 
# except those whose names begin with an underscore


def fact(n):
    return 1 if n == 1 else n * fact(n-1)

if (__name__ == '__main__'):
    import sys
    if len(sys.argv) > 1:
        print(fact(int(sys.argv[1])))

# C:\Users\john\Documents>python fact.py 6
# 720


'''
if you have a package  structured like this:

package
    |-module1
    |-module2

you need to use any version of
from package import ...

if you import the package itself, it would be synthactically correct, but not really usefull,
since it does not place any of the modules into the local namespace

package.module1     # error

unless you do some package initialization...
'''

# Package Initialization
'''
If a file named __init__.py is present in a package directory, it is invoked when the package or a module in the package is imported. 
This can be used for execution of package initialization code, such as initialization of package-level data.

__init__.py:
import package.module1, package.module2

__init__.py file used to be required for a package (even if empty), 
since python3.3, it is deduced implicitely, but this created another trap:
if a subdirectory encountered on sys.path as part of a package import contains an __init__.py file, 
then the Python interpreter will create a single directory package containing only modules from that directory, 
rather than finding all appropriately named subdirectories.
This happens even if there are other preceding subdirectories on sys.path that match the desired package name, but do not include an __init__.py file.


if the __init__.py file in the package directory contains a list named __all__, 
it is taken to be a list of modules that should be imported when the statement from <package_name> import * is encountered:

__init__.py:
__all__ = [
        'mod1',
        'mod2',
        'mod3',
        'mod4'
        ]
_________________
from pkg import *


__all__ can be defined in a module as well and serves the same purpose: to control what is imported with import *

pkg/mod1.py:
__all__ = ['foo']

def foo():
    print('[mod1] foo()')

class Foo:
    pass


NOTE:
For a package, when __all__ is not defined, import * does not import anything.
For a module, when __all__ is not defined, import * imports everything (except—you guessed it—names starting with an underscore).

'''

# referencing objects in a sibling subpackage

# 1. absolute import
# pkg/sub__pkg2/mod3.py:
from pkg.sub_pkg1.mod1 import foo

# 2. relative import
# pkg/sub__pkg2/mod3.py:
from ..sub_pkg1.mod1 import foo