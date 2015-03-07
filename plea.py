'''
--------------------------------------------------------------------------------

    plea.py

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

class Plea(Lea):

    '''
    Plea is a Lea subclass, which instance represents a probability distribution obtained by making 
    a cartesian product of given Lea instancea with itself, a given number of times. Each value is 
    a tuple having the given number as size.
    '''

    __slots__ = ('_lea1','_lea1Tuple','_nTimes')

    def __init__(self,lea1,nTimes=2):
        Lea.__init__(self)
        self._lea1 = lea1
        self._lea1Tuple = lea1.map(lambda v: (v,))
        self._nTimes = nTimes
        if nTimes <= 0:
            raise Lea.Error("cprodTimes method requires a strictly positive integer")

    def _getLeaChildren(self):
        return (self._lea1,)

    def _clone(self,cloneTable):
        return Plea(self._lea1.clone(cloneTable),self._nTimes)
    
    def _genVPs(self,nTimes=None):
        if nTimes is None:
            nTimes = self._nTimes
        if nTimes == 1:
            return self._lea1Tuple.genVPs()
        # nTimes >= 2 : use dichotomic algorithm
        nTimes1 = nTimes // 2
        plea = Plea(self._lea1,nTimes1)
        alea = plea.getAlea()
        flea = alea + alea.clone()
        if nTimes%2 == 1:
            # nTimes is odd : nTimes = 2*nTimes1 + 1
            # operate with one more lea1 on the current result 
            #flea = Flea.build(add,(flea,self._lea1Tuple))
            flea += self._lea1Tuple
        return flea.genVPs()
