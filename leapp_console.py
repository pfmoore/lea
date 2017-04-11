'''
--------------------------------------------------------------------------------

    leapp_console.py

--------------------------------------------------------------------------------
Copyright 2013-2017 Pierre Denis

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
        
    __slots__ = ( 'locals', 'debug', 'prompt1', 'prompt2', 'promptD')
    
    def __init__(self):
        self.locals = locals()
        self.debug = DEBUG
        self.prompt1 = PROMPT1
        self.prompt2 = PROMPT2
        self.promptD = PROMPTD
        self.execPythonStatement('_leapp = self')
        self.execPythonStatement('del self')
        self.execPythonStatement('from lea import *')

    class Error(Exception):
        pass

    def execLeappTranslatorMultilineStatement(self,rMultilineStatement):
        pMultilineStatement = LeappTranslator.getTarget00(rMultilineStatement)
        if self.debug:
            print (self.promptD+('\n'+self.promptD).join(pMultilineStatement.split('\n')))
        self.execPythonStatement(pMultilineStatement)

    def execPythonStatement(self,pStatement):
        if len(pStatement) > 0:
            code = compile(pStatement+'\n','<leapp>','single')
            exec (code,self.locals)

    def inputMultilineStatement(self):
        readLines = []
        prompt = self.prompt1
        while True:
            readLine = input(prompt).rstrip()
            if len(readLine) == 0:
                break
            lastChar = readLine[-1]
            if lastChar == ':' and len(readLine) > 1:
                readLine += '\n'
            elif lastChar == '\\':
                readLine = readLine[:-1]
            readLines.append(readLine)
            if len(readLine) <= 1 or lastChar not in CONTINUATION_LINE_CHARS:
                break
            prompt = self.prompt2
        return ''.join(readLines)    

    def startCmdLoop(self):
        from .lea import Lea
        while True:
            try:
                rMultilineStatement = self.inputMultilineStatement()
            except EOFError:
                break
            try:
                try:
                    self.execLeappTranslatorMultilineStatement(rMultilineStatement)
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

