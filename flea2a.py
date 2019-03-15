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

from .flea2 import Flea2

class Flea2a(Flea2):
    
    '''
    Flea2a is a Flea2 subclass, which instance is defined by a given function applied on two given Lea arguments
    with a given "left-absorber" value (i.e. f(absorber,x) = absorber). This gives equivalent results as
    Flea2 (without absorber) but these could be more efficient by pruning the tree search. 
    The function is applied on all elements of the joint of the arguments. This results in a new
    probability distribution for all the values returned by the function.
    A Flea2a instance is a "functional pex", as defined in the paper on Statues algorithm
    (see http://arxiv.org/abs/1806.09997).
    '''
    
    __slots__ = ('_absorber',)

    def __init__(self,f,arg1,arg2,absorber):
        Flea2.__init__(self,f,arg1,arg2)
        self._absorber = absorber

    def _gen_vp(self):
        f = self._f
        absorber = self._absorber
        for (v1,p1) in self._lea_arg1.gen_vp():
            if v1 is absorber:
                yield (absorber,p1)
            else:
                for (v2,p2) in self._lea_arg2.gen_vp():
                    yield (f(v1,v2),p1*p2)
