'''
--------------------------------------------------------------------------------

    plea.py

--------------------------------------------------------------------------------
Copyright 2013-2020 Pierre Denis

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
from .alea import Alea

from math import exp

class Plea(Lea):
    
    '''
    Plea is a Lea subclass, which instance represents a Poisson probability
    distribution having the given mean; the distribution is approximated by
    the finite set of values that have probability > precision(i.e. low/high
    values with too small probabilities are dropped)
    '''
    
    __slots__ = ('_mean','precision')

    def __init__(self,mean,precision=1e-20):
        Lea.__init__(self)
        self._mean = mean
        self.precision = precision
    
    def _get_lea_children(self):
        return ()

    def _clone_by_type(self,clone_table):
        return Plea(self._mean,self.precision)

    def _gen_vp(self):
        mean = self._mean
        precision = self.precision
        p = exp(-mean)
        v = 0
        t = 0.0
        while p >= precision or v <= mean:
            if p >= precision:
                yield (v,p)
            t += p
            v += 1
            p = (p*mean) / v

    def _gen_one_random_mc(self):
        for v in self.get_alea()._gen_one_random_mc():
            yield v

    def _em_step(self,model_lea,cond_lea,obs_pmf_tuple,conversion_dict):
        if cond_lea is True:
            vps = obs_pmf_tuple
        else:
            vps = tuple((vx,px*(cond_lea.given(model_lea==vx))._p(True)) for (vx,px) in obs_pmf_tuple)
        mean = sum(px*vx for (vx,px) in vps) / sum(px for (_,px) in vps)
        return Plea(mean,self.precision)
