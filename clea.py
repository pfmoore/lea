'''
--------------------------------------------------------------------------------

    clea.py

--------------------------------------------------------------------------------
Copyright 2013-2018 Pierre Denis

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

class Clea(Lea):
    
    '''
    Clea is a Lea subclass, which instance is defined by a given sequence (L1,...Ln)
    of Lea instances; it represents a probability distribution made up from the
    cartesian product L1 x ... x Ln; assuming independency of events, it associates
    each (v1,...,vn) tuple with probability product P1(v1)...Pn(vn).
    '''
    
    __slots__ = ('_lea_args',)

    def __init__(self,*args):
        Lea.__init__(self)
        self._lea_args = tuple(Lea.coerce(arg) for arg in args)

    def _get_lea_children(self):
        return self._lea_args
    
    def _clone(self,clone_table):
        return Clea(*(lea_arg.clone(clone_table) for lea_arg in self._lea_args))

    @staticmethod
    def prod(gs):
        if len(gs) == 0:
            yield ()
        else:
            for xs in Clea.prod(gs[:-1]):
                for x in gs[-1]():
                    yield xs + (x,)

    def _gen_vp(self):
        for vps in Clea.prod(tuple(lea_arg.gen_vp for lea_arg in self._lea_args)):
            v = tuple(v for (v,p) in vps)
            p = 1
            for (v1,p1) in vps:
                p *= p1
            yield (v,p)

    def _gen_one_random_mc(self):
        for v in Clea.prod(tuple(lea_arg._gen_one_random_mc for lea_arg in self._lea_args)):
            yield v
