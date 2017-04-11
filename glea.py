'''
--------------------------------------------------------------------------------

    glea.py

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

from .lea import Lea
from .clea import Clea

class Glea(Lea):
    
    '''
    Glea is a Lea subclass, which instance is defined by a Lea instance having functions as values applied
    on a given sequence of arguments. The arguments are coerced to Lea instances. All functions are applied
    on all elements of cartesian product of all arguments (see Clea class). This results in a new probability
    distribution for all the values returned by calls to all the functions.
    '''
    
    __slots__ = ('_cleaFuncAndArgs',)

    def __init__(self,cleaFuncAndArgs):
        Lea.__init__(self)
        self._cleaFuncAndArgs = cleaFuncAndArgs

    @staticmethod
    def build(leaFunc,args):
        return Glea(Clea(leaFunc,Clea(*args)))

    def _getLeaChildren(self):
        return (self._cleaFuncAndArgs,)

    def _clone(self,cloneTable):
        return Glea(self._cleaFuncAndArgs.clone(cloneTable))

    def _genVPs(self):
        for ((f,args),p) in self._cleaFuncAndArgs.genVPs():
            yield (f(*args),p)

    def _genOneRandomMC(self):
        for (f,args) in self._cleaFuncAndArgs._genOneRandomMC():
            yield f(*args)
