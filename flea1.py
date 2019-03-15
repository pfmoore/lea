'''
--------------------------------------------------------------------------------

    flea1.py

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

class Flea1(Lea):
    
    '''
    Flea1 is a Lea subclass, which instance is defined by a given function applied on one given Lea argument.
    The function is applied on all values of the argument. This results in a new probability distribution
    for all the values returned by the function.
    A Flea1 instance is a "functional pex", as defined in the paper on Statues algorithm
    (see http://arxiv.org/abs/1806.09997).
    '''
    
    __slots__ = ('_f','_lea_arg')

    def __init__(self,f,lea_arg):
        Lea.__init__(self)
        self._f = f
        self._lea_arg = lea_arg

    def _get_lea_children(self):
        return (self._lea_arg,)

    def _gen_vp(self):
        f = self._f
        for (v,p) in self._lea_arg.gen_vp():
            yield (f(v),p)

    def _gen_one_random_mc(self):
        for v in self._lea_arg._gen_one_random_mc():
            yield self._f(v)
