'''
--------------------------------------------------------------------------------

    flea.py

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
from .clea import Clea

class Flea(Lea):
    
    '''
    Flea is a Lea subclass, which instance is defined by a given function applied on a given sequence
    of arguments. The arguments are coerced to Lea instances. The function is applied on all elements
    of cartesian product of all arguments (see Clea class). This results in a new probability
    distribution for all the values returned by the function.
    A Flea instance is a "functional pex", as defined in the paper on Statues algorithm
    (see http://arxiv.org/abs/1806.09997).
    '''
    
    __slots__ = ('_f','_clea_args')

    def __init__(self,f,clea_args):
        Lea.__init__(self)
        self._f = f
        self._clea_args = clea_args
    
    @staticmethod
    def build(f,args):
        return Flea(f,Clea(*args))

    def _get_lea_children(self):
        return (self._clea_args,)

    def _gen_vp(self):
        f = self._f
        for (args,p) in self._clea_args.gen_vp():
            yield (f(*args),p)

    def _gen_one_random_mc(self):
        f = self._f
        for args in self._clea_args._gen_one_random_mc():
            yield f(*args)
