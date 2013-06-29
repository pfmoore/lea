'''
--------------------------------------------------------------------------------

    tlea.py

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
from flea import Flea

class Tlea(Lea):

    __slots__ = ('_op','_lea1','_nTimes')

    def __init__(self,op,lea1,nTimes=2):
        Lea.__init__(self)
        self._op = op
        self._lea1 = lea1
        self._nTimes = nTimes
        if nTimes <= 0:
            raise Exception("Tlea requires that nTimes > 0")

    def _genVPs(self,condLea,nTimes=None):
        if nTimes is None:
            nTimes = self._nTimes
        if nTimes == 1:
            if condLea is None:
               # optimisation if no condition to evaluate:
               # can call _genVP that does not bind variables 
               return self._lea1._genVPs(None)
            # there is a condition to evaluate: call genVp
            # the tuple(...) is essential here to simulate event independency
            # it exhausts the genVPs generator, so the transient variable binding
            # are removed before yielding results
            return iter(tuple(self._lea1.genVPs(condLea)))
        # nTimes >= 2
        nTimes1 = nTimes / 2
        alea = Tlea(self._op,self._lea1,nTimes1).getAlea(condLea)
        flea = Flea.build(self._op,(alea,alea.clone()))
        if nTimes%2 == 1:
            # nTimes is odd : nTimes = 2*nTimes + 1
            # operate one more lea1 on the current result 
            flea = Flea.build(self._op,(flea,self._lea1))
        return flea._genVPs(None)
