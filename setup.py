'''
--------------------------------------------------------------------------------

    setup.py

--------------------------------------------------------------------------------
Copyright 2013 Pierre Denis

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

from distutils.core import setup

from license import VER

setup( name = 'lea',
       version = VER,
       description = 'Discrete probability distributions in Python',
       author = 'Pierre Denis',
       author_email = 'pie.denis@skynet.be',
       url = 'http://code.google.com/p/lea/',
       license = 'LGPL',
       py_modules = [ 'lea',
                      'clea',
                      'flea',
                      'plea',
                      'dlea',
                      'ilea',
                      'olea',
                      'alea',
                      'tlea',
                      'license' ] ,
       data_files = [('.', [ 'COPYING',
                             'COPYING.LESSER' ] ) ],
      )