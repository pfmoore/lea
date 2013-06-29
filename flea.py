'''
--------------------------------------------------------------------------------

    flea.py

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

from lea import Lea
from clea import Clea

class Flea(Lea):
    '''
        Function Lea, obtained by a function applied on a given sequence of arguments
        The arguments are coerced to Lea instances.
        Any instance forms a tree structure, where the leaves must be Alea instances
        The function shall be applied on all elements of cartesian product of arguments
        Lazy
    '''
    
    __slots__ = ('_f','_cleaArgs')

    def __init__(self,f,cleaArgs):
        Lea.__init__(self)
        self._f = f
        self._cleaArgs = cleaArgs
    
    @staticmethod
    def build(f,args):
        return Flea(f,Clea(*args))
    
    def clone(self):
        return Flea(self._f,self._cleaArgs)

    def reset(self):
        Lea.reset(self)
        self._cleaArgs.reset()

    def _genVPs(self,condLea):
        f = self._f
        if isinstance(f,Lea):
            for ((f,args),p) in Clea(f,self._cleaArgs)._genVPs(condLea):
                yield (f(*args),p)            
        else:
            for (args,p) in self._cleaArgs._genVPs(condLea):
                yield (f(*args),p)

