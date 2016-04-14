'''
--------------------------------------------------------------------------------

    flea2.py

--------------------------------------------------------------------------------
Copyright 2013-2016 Pierre Denis

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

class Flea2(Lea):
    
    '''
    Flea2 is a Lea subclass, which instance is defined by a given function applied on two given arguments.
    The function is applied on all elements of cartesian product of the arguments. This results in a new
    probability distribution for all the values returned by the function.
    '''
    
    __slots__ = ('_f','_leaArg1','_leaArg2')

    def __init__(self,f,arg1,arg2):
        Lea.__init__(self)
        self._f = f
        self._leaArg1 = Lea.coerce(arg1)
        self._leaArg2 = Lea.coerce(arg2)

    def _getLeaChildren(self):
        return (self._leaArg1,self._leaArg2)

    def _clone(self,cloneTable):
        return Flea2(self._f,self._leaArg1.clone(cloneTable),self._leaArg2.clone(cloneTable))    

    def _genVPs(self):
        f = self._f
        for (v1,p1) in self._leaArg1.genVPs():
            for (v2,p2) in self._leaArg2.genVPs():
                yield (f(v1,v2),p1*p2)

    def _genOneRandomMC(self):
        for v1 in self._leaArg1._genOneRandomMC():
            for v2 in self._leaArg2._genOneRandomMC():
                yield self._f(v1,v2)
