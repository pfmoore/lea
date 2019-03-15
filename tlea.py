'''
--------------------------------------------------------------------------------

    tlea.py

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
from .toolbox import dict, defaultdict

class Tlea(Lea):
    
    '''
    Tlea is a Lea subclass, which instance represents a conditional probability
    table (CPT) giving a Lea instance C and a dictionary associating each
    possible value of C to a specific Lea instance.
    A Tlea instance is a "table pex", as defined in the paper on Statues algorithm
    (see http://arxiv.org/abs/1806.09997).
    '''

    __slots__ = ('_lea_c','_lea_dict','_default_lea')

    def __init__(self,lea_c,lea_dict,default_lea=Lea._DUMMY_VAL):
        if isinstance(lea_dict,defaultdict):
            raise Lea.Error('defaultdict not supported for Tlea, use default_lea argument instead')
        Lea.__init__(self)
        self._lea_c = Alea.coerce(lea_c)
        self._lea_dict = dict((c,Alea.coerce(lea1)) for (c,lea1) in lea_dict.items())
        if default_lea is Lea._DUMMY_VAL:
            self._default_lea = Lea._DUMMY_VAL
        else:
            self._default_lea = Alea.coerce(default_lea)
            self._lea_dict = defaultdict(lambda:self._default_lea,self._lea_dict)

    def _get_lea_children(self):
        lea_children = [self._lea_c]
        for lea1 in self._lea_dict.values():
            lea_children.append(lea1)
        if self._default_lea is not Lea._DUMMY_VAL:
            lea_children.append(self._default_lea)
        return lea_children

    def _gen_vp(self):
        lea_dict = self._lea_dict
        for (vc,pc) in self._lea_c.gen_vp():
            try:
                lea_v = lea_dict[vc]
            except KeyError:
                raise Lea.Error("missing value '%s' in CPT"%vc)
            for (vd,pd) in lea_v.gen_vp():
                yield (vd,pc*pd)

    def _gen_one_random_mc(self):
        lea_dict = self._lea_dict
        for vc in self._lea_c._gen_one_random_mc():
            try:
                lea_v = lea_dict[vc]
            except KeyError:
                raise Lea.Error("missing value '%s' in CPT"%vc)
            for vd in lea_v._gen_one_random_mc():
                yield vd
