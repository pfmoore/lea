'''
--------------------------------------------------------------------------------

    ilea.py

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

class Ilea(Lea):
    
    '''
    Ilea is a Lea subclass, which instance represents a probability distribution obtained
    by filtering the values Vi of a given Lea instance that verify a given boolean condition C(Vi).
    In the context of a conditional probability table (CPT), each Ilea instance represents
    a given distibution <Vi,p(Vi|C)>, assuming that a given condition C is verified (see Blea class).
    '''

    __slots__ = ('_lea1','_condLea')

    def __init__(self,lea1,condLea):
        Lea.__init__(self)
        self._lea1 = lea1
        self._condLea = condLea

    def _getLeaChildren(self):
        return (self._lea1,self._condLea)
    
    def _clone(self,cloneTable):
        return Ilea(self._lea1.clone(cloneTable),self._condLea.clone(cloneTable))
        
    def _genVPs(self):
        for (cv,cp) in self._condLea.genVPs():
            if cv is True:
                # the condition is true, for some binding of variables
                # yield value-probability pairs of _lea1, given this binding
                for (v,p) in self._lea1.genVPs():
                    yield (v,cp*p)
            elif cv is False:
                pass
            else:
                raise Lea.Error("boolean expression expected")
