'''
--------------------------------------------------------------------------------

    clea.py

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

from operator import mul


class Clea(Lea):
    ''' cartesian product
    '''
    
    __slots__ = ('_leaArgs',)

    def __init__(self,*args):
        Lea.__init__(self)
        self._leaArgs = tuple(Lea.coerce(arg) for arg in args)

    def reset(self):
        Lea.reset(self)
        for leaArg in self._leaArgs:
            leaArg.reset()

    def clone(self):
        return Clea(*self._leaArgs)

    @staticmethod
    def prod(arg,gs):
        if len(gs) == 0:
            return iter(((),))
        return (xs+(x,) for xs in Clea.prod(arg,gs[:-1]) for x in gs[-1](arg))

    def _genVPs(self,condLea):
        ''' generate tuples
        '''
        for vps in Clea.prod(condLea,tuple(leaArg.genVPs for leaArg in self._leaArgs)):
            if condLea is None or condLea.isFeasible():
                v = tuple(v for (v,p) in vps)
                p = reduce(mul,(p for (v,p) in vps),1)
                yield (v,p)
