'''
--------------------------------------------------------------------------------

    rlea.py

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
from .toolbox import zip

class Rlea(Lea):
    
    '''
    Rlea is a Lea subclass, which instance has other Lea instances as values.
    '''
    
    __slots__ = ('_leaOfLeas','_factors')

    def __init__(self,leaOfLeas):
        Lea.__init__(self)
        self._leaOfLeas = leaOfLeas
        leaOfLeas._initCalc()
        leas = tuple(lea_ for (lea_,_) in leaOfLeas.genVPs())
        for lea_ in leas:
            lea_._initCalc()
        leaCounts = tuple(sum(p1 for (_,p1) in lea_.genVPs()) for lea_ in leas)
        pcount = 1
        for count in leaCounts:
            pcount *= count
        self._factors = tuple(pcount//count for count in leaCounts)
         
    def _getLeaChildren(self):
        return (self._leaOfLeas,)

    def _clone(self,cloneTable):
        return Rlea(self._leaOfLeas.clone(cloneTable))

    def _genVPs(self):
        for ((lea1,p1),factor) in zip(self._leaOfLeas.genVPs(),self._factors):
            p1 *= factor
            for (v,p2) in lea1.genVPs():
                yield (v,p1*p2)

    def _genOneRandomMC(self):
        for leaArg in self._leaOfLeas._genOneRandomMC():
            for v in leaArg._genOneRandomMC():
                yield v
