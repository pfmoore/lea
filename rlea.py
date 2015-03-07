'''
--------------------------------------------------------------------------------

    rlea.py

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
from alea import Alea
from toolbox import dict

class Rlea(Lea):
    
    '''
    Rlea is a Lea subclass, which instance has other Lea instances as values.
    '''
    
    __slots__ = ('_leaOfLeas','_factors')

    def __init__(self,leaOfLeas):
        Lea.__init__(self)
        self._leaOfLeas = leaOfLeas
        leaCounts = tuple((lea_,sum(p1 for (v,p1) in lea_._genVPs())) for (lea_,p2) in leaOfLeas._genVPs())
        pcount = 1
        for (lea1_,count) in leaCounts:
            pcount *= count
        self._factors = dict((lea_,pcount//count) for (lea_,count) in leaCounts)
         
    def _getLeaChildren(self):
        return (self._leaOfLeas,)

    def _clone(self,cloneTable):
        return Rlea(self._leaOfLeas.clone(cloneTable))    

    def _genVPs(self):
        for (lea1,p1) in self._leaOfLeas.genVPs():
            factor = self._factors[lea1]
            p1 *= factor
            for (v,p2) in lea1.genVPs():
                yield (v,p1*p2)
