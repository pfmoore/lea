'''
--------------------------------------------------------------------------------

    leapp.py

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
        leapp_console = LeappConsole()
        print ("[running on Python %s]"%platform.python_version())
        print (license.license_text)
        print ('Welcome in Leapp console!')
        leapp_console.start_cmd_loop()
        print ()
    else:
        # argument(s) present : compile files into Python statements then,
        # if '-c' given, write Python statements .py files
        # otherwise, execute Python statements 
        from .leapp_compiler import LeappCompiler
        first_arg = sys.argv[1]
        if first_arg.startswith('-'):
            if first_arg not in ('-c','-cf'):
                print ("ERROR: unknown option '%s'"%first_arg)
                sys.exit(-1)                
        is_compile = first_arg.startswith('-c')
        force = first_arg == '-cf'
        if is_compile:
            lea_filenames_arg_idx = 2
        else:
            lea_filenames_arg_idx = 1
        lea_filenames = sys.argv[lea_filenames_arg_idx:]
        process_lea_file = LeappCompiler.compile_and_write_lea_file if is_compile else LeappCompiler.compile_and_exec_lea_file
        if is_compile:
            for lea_filename in lea_filenames:
                LeappCompiler.compile_and_write_lea_file(lea_filename,force)
        else:
            for lea_filename in lea_filenames:
                LeappCompiler.compile_and_exec_lea_file(lea_filename)
