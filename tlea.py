'''
--------------------------------------------------------------------------------

    tlea.py

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
from flea import Flea

class Tlea(Lea):

    '''
    Tlea is a Lea subclass, which instance represents a probability distribution obtained by applying
    a given 2-ary function repeatedly on given Lea instances, a given number of times. It allows to
    avoid tedious typing or explcit loops; also, it makes the calculation faster by using a dichotomic
    algorithm.
    '''

    __slots__ = ('_op','_lea1','_nTimes')

    def __init__(self,op,lea1,nTimes=2):
        Lea.__init__(self)
        self._op = op
        self._lea1 = lea1
        self._nTimes = nTimes
        if nTimes <= 0:
            raise Lea.Error("times method requires a strictly positive integer")

    def _getLeaChildren(self):
        return (self._lea1,)

    def _clone(self,cloneTable):
        return Tlea(self._op,self._lea1.clone(cloneTable),self._nTimes)
    
    def _genVPs(self,nTimes=None):
        if nTimes is None:
            nTimes = self._nTimes
        lea1 = self._lea1.getAleaClone()
        if nTimes == 1:
            return lea1.genVPs()
        # nTimes >= 2 : use dichotomic algorithm
        nTimes1 = nTimes // 2
        tlea = Tlea(self._op,lea1,nTimes1)
        alea = tlea.getAlea()
        # alea = tlea
        flea = Flea.build(self._op,(alea,alea.clone()))
        if nTimes%2 == 1:
            # nTimes is odd : nTimes = 2*nTimes1 + 1
            # operate with one more lea1 on the current result 
            flea = Flea.build(self._op,(flea,self._lea1))
        return flea.genVPs()
