'''
--------------------------------------------------------------------------------

    rlea.py

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

class Rlea(Lea):
    
    '''
    Rlea is a Lea subclass, which instance has other Lea instances as values.
    An Rlea instance is a "mixture pex", as defined in the paper on Statues algorithm
    (see http://arxiv.org/abs/1806.09997).
    '''
    
    __slots__ = ('_lea_of_leas',)

    def __init__(self,lea_of_leas):
        Lea.__init__(self)
        self._lea_of_leas = lea_of_leas

    def _get_lea_children(self):
        return (self._lea_of_leas,) + self._lea_of_leas.support

    def _gen_vp(self):
        for (lea1,p1) in self._lea_of_leas.gen_vp():
            for (v,p2) in lea1.gen_vp():
                yield (v,p1*p2)

    def _gen_one_random_mc(self):
        for lea_arg in self._lea_of_leas._gen_one_random_mc():
            for v in lea_arg._gen_one_random_mc():
                yield v
