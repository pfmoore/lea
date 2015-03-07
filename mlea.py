'''
--------------------------------------------------------------------------------

    clea.py

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
from prob_fraction import ProbFraction
from toolbox import calcLCM, zip

class Mlea(Lea):
    
    '''
    Mlea is a Lea subclass, which instance is defined by a given sequence (L1,...Ln)
    of Lea instances; it represents a probability distribution made up by merging
    L1,...,Ln together,i.e. P(v) = (P1(v) + ... + Pn(v)) / n
    where Pi(v) is the probability of value v in Li 
    '''
    
    __slots__ = ('_leaArgs','_factors')

    def __init__(self,*args):
        Lea.__init__(self)
        self._leaArgs = tuple(Lea.coerce(arg) for arg in args)
        counts = tuple(leaArg.getAlea()._count for leaArg in self._leaArgs) 
        lcm = calcLCM(counts)
        self._factors = tuple(lcm//count for count in counts)        

    def _getLeaChildren(self):
        return self._leaArgs

    def _clone(self,cloneTable):
        return Mlea(*(leaArg.clone(cloneTable) for leaArg in self._leaArgs))

    def _genVPs(self):
        for (leaArg,factor) in zip(self._leaArgs,self._factors):
            for (v,p) in leaArg.genVPs():
                yield (v,p*factor)
