'''
--------------------------------------------------------------------------------

    leapp_console.py

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

# leapp console configuration parameters
PROMPT1 = 'lea> '
PROMPT2 = ' ... '
PROMPTD = ' py# '
DEBUG = False

CONTINUATION_LINE_CHARS = '(,:\\'

from .leapp_translator import LeappTranslator
from .toolbox import input

import traceback

try:
    # try to activate terminal history 
    import readline
except:
    # OS dependent failure -> no terminal history
    pass
            
class LeappConsole(object):
    
    ''' 
    A LeappConsole instance allows to create an interactive console where the user
    can type leapp statements, which are translated into Python statements using
    Lea module and executed on the-fly.
    '''
        
    __slots__ = ( 'locals', 'debug', 'prompt1', 'prompt2', 'prompt_d')
    
    def __init__(self):
        self.locals = locals()
        self.debug = DEBUG
        self.prompt1 = PROMPT1
        self.prompt2 = PROMPT2
        self.prompt_d = PROMPTD
        self.exec_python_statement('_leapp = self')
        self.exec_python_statement('del self')
        self.exec_python_statement('from lea import *')

    class Error(Exception):
        pass

    def exec_leapp_translator_multiline_statement(self,r_multiline_statement):
        p_multiline_statement = LeappTranslator.get_target00(r_multiline_statement)
        if self.debug:
            print (self.prompt_d+('\n'+self.prompt_d).join(p_multiline_statement.split('\n')))
        self.exec_python_statement(p_multiline_statement)

    def exec_python_statement(self,p_statement):
        if len(p_statement) > 0:
            code = compile(p_statement+'\n','<leapp>','single')
            exec (code,self.locals)

    def input_multiline_statement(self):
        read_lines = []
        prompt = self.prompt1
        while True:
            read_line = input(prompt).rstrip()
            if len(read_line) == 0:
                break
            last_char = read_line[-1]
            if last_char == ':' and len(read_line) > 1:
                read_line += '\n'
            elif last_char == '\\':
                read_line = read_line[:-1]
            read_lines.append(read_line)
            if len(read_line) <= 1 or last_char not in CONTINUATION_LINE_CHARS:
                break
            prompt = self.prompt2
        return ''.join(read_lines)    

    def start_cmd_loop(self):
        from .lea import Lea
        while True:
            try:
                r_multiline_statement = self.input_multiline_statement()
            except EOFError:
                break
            try:
                try:
                    self.exec_leapp_translator_multiline_statement(r_multiline_statement)
                except:
                    if self.debug:
                        traceback.print_exc()
                    raise  
            except Lea.Error as exc:
                print ("Lea error: %s"%exc)
            except LeappTranslator.Error as exc:
                print ("Leapp syntax error: %s"%exc)
            except Exception as exc:
                print ("Python error: %s"%exc)

