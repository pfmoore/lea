'''
--------------------------------------------------------------------------------

    flea2.py

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
from .alea import Alea

class Flea2(Lea):
    
    '''
    Flea2 is a Lea subclass, which instance is defined by a given function applied on two given arguments.
    The function is applied on all elements of the joint of the arguments. This results in a new
    probability distribution for all the values returned by the function.
    A Flea2 instance is a "functional pex", as defined in the paper on Statues algorithm
    (see http://arxiv.org/abs/1806.09997).
    '''
    
    __slots__ = ('_f','_lea_arg1','_lea_arg2')

    def __init__(self,f,arg1,arg2):
        Lea.__init__(self)
        self._f = f
        self._lea_arg1 = Alea.coerce(arg1)
        self._lea_arg2 = Alea.coerce(arg2)

    def _get_lea_children(self):
        return (self._lea_arg1,self._lea_arg2)

    def _gen_vp(self):
        f = self._f
        for (v1,p1) in self._lea_arg1.gen_vp():
            for (v2,p2) in self._lea_arg2.gen_vp():
                yield (f(v1,v2),p1*p2)

    def _gen_one_random_mc(self):
        for v1 in self._lea_arg1._gen_one_random_mc():
            for v2 in self._lea_arg2._gen_one_random_mc():
                yield self._f(v1,v2)
