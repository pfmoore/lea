'''
--------------------------------------------------------------------------------

    leapp_compiler.py

--------------------------------------------------------------------------------
Copyright 2013-2018 Pierre Denis

This file is part of Lea.

Lea is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Lea is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Lea.  If not, see <http://www.gnu.org/licenses/>.
--------------------------------------------------------------------------------
'''

from . import license
from .leapp_translator import LeappTranslator
from .toolbox import input

PY_FILE_HEADER = '''
# ----------------------------------------------------------------------------- 
# %s
# ----------------------------------------------------------------------------- 
# file generated by leapp %s, from compilation of %s
# WARNING: any update will be overwritten at next generation!
# ----------------------------------------------------------------------------- 

'''[1:]

import os
try:
    # try to activate terminal history 
    import readline
except:
    # OS dependent failure -> no terminal history
    pass            

class LeappCompiler(object):
    
    ''' 
    LeappCompiler provides static methods to compile source leapp files into
    Python files (.py). Alternatively, the generated Python code can be directly
    executed, without being dumped in a .py file
    '''

    @staticmethod
    def compile_lea_file(lea_filename):
        f = open(lea_filename)
        lea_multiline_statement = f.read()
        f.close()
        p_multiline_statement = 'from lea import *\n\n'
        p_multiline_statement += LeappTranslator.get_target00(lea_multiline_statement)
        return p_multiline_statement

    @staticmethod
    def exec_python_statement(p_statement,lea_filename):
        code = compile(p_statement+'\n',lea_filename,'exec')
        exec(code)
            
    @staticmethod
    def compile_and_exec_lea_file(lea_filename):
        p_multiline_statement = LeappCompiler.compile_lea_file(lea_filename)
        LeappCompiler.exec_python_statement(p_multiline_statement,lea_filename)

    @staticmethod
    def compile_and_write_lea_file(lea_filename,force=False):
        p_multiline_statement = LeappCompiler.compile_lea_file(lea_filename)
        python_filename = os.path.splitext(lea_filename)[0] + '.py'
        if force or not os.path.exists(python_filename):
            can_write = True
        else:
            res = input("overwrite %s (y/n)? "%python_filename)
            can_write = res.strip().lower() == 'y'
        if can_write:         
            p_multiline_statement = PY_FILE_HEADER%(python_filename,lea.license.VER,lea_filename) + p_multiline_statement
            f = open(python_filename,'w')
            f.write(p_multiline_statement)
            f.close()
            print("'%s' written"%python_filename)
