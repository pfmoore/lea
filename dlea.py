'''
--------------------------------------------------------------------------------

    dlea.py

--------------------------------------------------------------------------------
Copyright 2013, 2014, 2015 Pierre Denis

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

from lea import Lea

class Dlea(Lea):
    
    '''
    Dlea is a Lea subclass, which instance represents a probability distribution of the
    sequences of values obtained by a given number of draws without replacement from a
    given Lea instance.
    '''
    
    __slots__ = ('_lea1','_nbValues')

    def __init__(self,lea1,nbValues):
        if nbValues <= 0:
            raise Lea.Error("draw method requires a strictly positive integer")
        Lea.__init__(self)
        self._lea1 = lea1
        self._nbValues = nbValues

    def _getLeaChildren(self):
        return (self._lea1,)

    def _clone(self,cloneTable):
        return Dlea(self._lea1.clone(cloneTable),self._nbValues)
    
    def _genVPs(self,nbValues=None):
        if nbValues is None:
            nbValues = self._nbValues
        if nbValues == 1:
            for (v,p) in self._lea1.genVPs():
                yield ((v,),p)
        else:     
            for (v,p) in self._lea1.genVPs():
                lea2 = self._lea1.clone()
                dlea = Dlea(lea2.given(lea2!=v).getAlea(),nbValues-1)
                for (vt,pt) in dlea.genVPs():
                    yield ((v,)+vt,p*pt)
