'''
--------------------------------------------------------------------------------

    dlea.py

--------------------------------------------------------------------------------
Copyright 2013-2022 Pierre Denis

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

from .lea import Lea, Alea, Clea


class Dlea(Lea):
    
    '''
    Dlea is a Lea subclass, which instance represents the probablity distribution
    that all arguments have different values. The arguments are coerced to Lea instances.
    A Dlea instance is a special case of "functional pex", as defined in the paper on
    Statues algorithm (see http://arxiv.org/abs/1806.09997).
    '''

    __slots__ = ('_clea',)

    def __init__(self,clea1):
        Lea.__init__(self)
        self._clea = clea1
    
    @staticmethod
    def build(args):
        return Dlea(Clea(*args))

    def _get_lea_children(self):
        return (self._clea,)

    def _clone_by_type(self,clone_table):
        return Dlea(self._clea._clone(clone_table))
    
    @staticmethod
    def _gen_all_different_vp(arg_leas,forbidden_values=()):
        ''' static method, generates (v,P(v)) atoms where v is True iff 
            all values of arg_leas are different each from each other
            and absent from forbidden_values; binding are made for each
            value yielded
        '''
        if len(arg_leas) == 0:
            # no argument: evaluated as True with probability 1 (seed of recursion)
            yield (True,1)
        else:
            tail_lea_args = arg_leas[1:]
            for (v0,p0) in arg_leas[0].gen_vp():
                if v0 in forbidden_values:
                    # short-cut evaluation
                    yield (False,p0)
                else:
                    for (v1,p1) in Dlea._gen_all_different_vp(tail_lea_args, forbidden_values+(v0,)):
                        yield (v1,p0*p1)
    
    def _gen_vp(self):
        for vp in Dlea._gen_all_different_vp(self._clea._lea_args):
            yield vp
    
    def _gen_one_random_mc(self):
        for args in self._clea.gen_one_random_mc():
            yield len(frozenset(args)) == len(args)

    def _em_step(self,model_lea,cond_lea,obs_pmf_tuple,conversion_dict):
        return Dlea(self._clea.em_step(model_lea,cond_lea,obs_pmf_tuple,conversion_dict))
