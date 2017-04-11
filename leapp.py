'''
--------------------------------------------------------------------------------

    leapp.py

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

import sys
from . import license

''' 
Leapp main program
------------------
If no argument is given, then the Leapp console is open and the command loop is started.
Otherwise, the files refered by the given filename arguments are compiled into Python
statements. If '-c' is given, it writes the Python statements into .py files, otherwise,
it executes directly the Python statements.
'''

if __name__=='__main__':
    if len(sys.argv) == 1:
        # no argument : start leapp console
        from .leapp_console import LeappConsole
        import platform
        leappConsole = LeappConsole()
        print ("[running on Python %s]"%platform.python_version())
        print (license.licenseText)
        print ('Welcome in Leapp console!')
        leappConsole.startCmdLoop()
        print ()
    else:
        # argument(s) present : compile files into Python statements then,
        # if '-c' given, write Python statements .py files
        # otherwise, execute Python statements 
        from .leapp_compiler import LeappCompiler
        firstArg = sys.argv[1]
        if firstArg.startswith('-'):
            if firstArg not in ('-c','-cf'):
                print ("ERROR: unknown option '%s'"%firstArg)
                sys.exit(-1)                
        isCompile = firstArg.startswith('-c')
        force = firstArg == '-cf'
        if isCompile:
            leaFilenamesArgIdx = 2
        else:
            leaFilenamesArgIdx = 1
        leaFilenames = sys.argv[leaFilenamesArgIdx:]
        processLeaFile = LeappCompiler.compileAndWriteLeaFile if isCompile else LeappCompiler.compileAndExecLeaFile
        if isCompile:
            for leaFilename in leaFilenames:
                LeappCompiler.compileAndWriteLeaFile(leaFilename,force)
        else:
            for leaFilename in leaFilenames:
                LeappCompiler.compileAndExecLeaFile(leaFilename)
