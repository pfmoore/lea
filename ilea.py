'''
--------------------------------------------------------------------------------

    ilea.py

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

class Ilea(Lea):
    
    '''
    Ilea is a Lea subclass defined.
    An Ilea instance represents a distribution obtained by constraining a given
    Lea instance by a given boolean condition.
    '''

    __slots__ = ('_lea1','_condLea')

    def __init__(self,lea1,condLea):
        Lea.__init__(self)
        self._lea1 = lea1
        self._condLea = condLea

    def clone(self):
        ilea = Ilea(self._lea1,self._condLea)
        ilea._alea = self._alea
        return ilea

    def reset(self):
        Lea.reset(self)
        self._lea1.reset()
        if self._condLea is not None:
            self._condLea.reset()

    def _genVPs(self,condLea):
        if condLea is None:
            condLea = self._condLea
        elif self._condLea is not None:            
            condLea &= self._condLea
        for (v,p) in self._lea1.genVPs(condLea):
            if condLea is None or condLea.isFeasible():
                yield (v,p)
