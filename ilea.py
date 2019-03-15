'''
--------------------------------------------------------------------------------

    ilea.py

--------------------------------------------------------------------------------
Copyright 2013-2019 Pierre Denis

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
from operator import and_

class Ilea(Lea):
    
    '''
    Ilea is a Lea subclass, which instance represents a probability distribution obtained
    by filtering the values Vi of a given Lea instance that verify a boolean condition C,
    which is the AND of given boolean conditions Cj(Vi).
    In the context of a conditional probability table (CPT), each Ilea instance represents
    a given distribution <Vi,p(Vi|C)>, assuming that a given condition C is verified (see Blea class).
    An Ilea instance is a "conditional pex", as defined in the paper on Statues algorithm
    (see http://arxiv.org/abs/1806.09997).
    '''

    __slots__ = ('_lea1','_cond_leas')

    def __init__(self,lea1,cond_leas):
        Lea.__init__(self)
        self._lea1 = lea1
        self._cond_leas = tuple(cond_leas)

    def _get_lea_children(self):
        return (self._lea1,) + self._cond_leas
    
    @staticmethod
    def _gen_true_p(cond_leas):
        ''' generates probabilities of True for ANDing the given conditions 
            this uses short-circuit evaluation
        '''
        if len(cond_leas) == 0:
            # empty condition: evaluated as True (seed of recursion)
            yield 1
        else:
            for (cv0,p0) in cond_leas[0].gen_vp():
                if cv0 is True:
                    # the first condition is true, for some binding of variables
                    for p1 in Ilea._gen_true_p(cond_leas[1:]):
                        # the full condition is true, for some binding of variables
                        yield p0*p1
                elif cv0 is False:
                    # short-circuit: do not go further since the AND is false
                    pass
                else:
                    # neither True, nor False -> error
                    raise Lea.Error("boolean expression expected")
    
    def _gen_vp(self):
        for cp in Ilea._gen_true_p(self._cond_leas):
            # the AND of conditions is true, for some binding of variables
            # yield value-probability pairs of _lea1, given this binding
            for (v,p) in self._lea1.gen_vp():
                yield (v,cp*p)

    def _gen_one_random_true_cond(self,cond_leas,with_exception):
        if len(cond_leas) == 0:
            # empty condition: evaluated as True (seed of recursion)
            yield None
        else:
            for cv in cond_leas[0]._gen_one_random_mc():
                if cv is True:
                    for v in self._gen_one_random_true_cond(cond_leas[1:],with_exception):
                        yield v
                elif cv is False:
                    if with_exception:
                        raise Lea._FailedRandomMC()
                    yield self
                else:
                    raise Lea.Error("boolean expression expected")

    def _gen_one_random_mc(self):
        for _ in self._gen_one_random_true_cond(self._cond_leas,True):
            for v in self._lea1._gen_one_random_mc():
                yield v

    def _gen_one_random_mc_no_exc(self):
        for u in self._gen_one_random_true_cond(self._cond_leas,False):
            if u is not self: 
                for v in self._lea1._gen_one_random_mc():
                    yield v

    def lr(self):
        ''' returns a float giving the likelihood ratio (LR) of an 'evidence' E,
            which is self's unconditional probability distribution, for a given
            'hypothesis' H, which is self's condition; it is calculated as 
                  P(E | H) / P(E | not H)
            both E and H must be boolean probability distributions, otherwise
            an exception is raised;
            an exception is raised also if H is certainly true or certainly false      
        '''
        lr_n = self.P
        lr_d = self._lea1.given(~Lea.reduce_all(and_,self._cond_leas,False)).P
        if lr_d == 0:
            if lr_n == 0:
                raise Lea.Error("undefined likelihood ratio")
            return float('inf') 
        return lr_n / lr_d
