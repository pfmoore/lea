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

from .flea2 import Flea2

class Flea2a(Flea2):
    
    '''
    Flea2a is a Flea2 subclass, which instance is defined by a given function applied on two given Lea arguments
    with a given "right-absorber" value (i.e. f(x,absorber) = absorber). This gives equivalent results as
    Flea2 (without absorber) but these could be more efficient by pruning the tree search. 
    The function is applied on all elements of cartesian product of the arguments. This results in a new
    probability distribution for all the values returned by the function.
    '''
    
    __slots__ = ('_absorber',)

    def __init__(self,f,arg1,arg2,absorber):
        Flea2.__init__(self,f,arg1,arg2)
        self._absorber = absorber

    def _clone(self,cloneTable):
        return Flea2a(self._f,self._leaArg1.clone(cloneTable),self._leaArg2.clone(cloneTable),self._absorber)    

    def _genVPs(self):
        f = self._f
        absorber = self._absorber
        for (v2,p2) in self._leaArg2.genVPs():
            if v2 is absorber:
                yield (absorber,self._leaArg1._getCount()*p2)
            else:
                for (v1,p1) in self._leaArg1.genVPs():
                    yield (f(v1,v2),p1*p2)
