'''
--------------------------------------------------------------------------------

    flea1.py

--------------------------------------------------------------------------------
Copyright 2013, 2014, 2015 Pierre Denis

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

class Flea1(Lea):
    
    '''
    Flea1 is a Lea subclass, which instance is defined by a function applied on one given Lea argument.
    The function is applied on all values of the argument. This results in a new probability distribution
    for all the values returned by the function.
    '''
    
    __slots__ = ('_f','_leaArg')

    def __init__(self,f,leaArg):
        Lea.__init__(self)
        self._f = f
        self._leaArg = leaArg

    def _getLeaChildren(self):
        return (self._leaArg,)

    def _clone(self,cloneTable):
        return Flea1(self._f,self._leaArg.clone(cloneTable))    

    def _genVPs(self):
        f = self._f
        for (v,p) in self._leaArg._genVPs():
            yield (f(v),p)

    def _genOneRandomMC(self):
        for v in self._leaArg._genOneRandomMC():
            yield self._f(v)
