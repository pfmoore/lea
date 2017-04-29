'''
--------------------------------------------------------------------------------

    tlea.py

--------------------------------------------------------------------------------
Copyright 2013-2017 Pierre Denis

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
from .alea import Alea
from .toolbox import dict, defaultdict

class Tlea(Lea):
    
    '''
    Tlea is a Lea subclass, which instance represents a conditional probability
    table (CPT) giving a Lea instance C and a dictionary associating each
    possible value of C to a specific Lea instance.
    '''

    __slots__ = ('_leaC','_leaDict','_defaultLea')

    def __init__(self,leaC,leaDict,defaultLea=Lea._DUMMY_VAL,fast=False):
        if isinstance(leaDict,defaultdict):
            raise Lea.Error('defaultdict not supported for Tlea, use defaultLea argument instead')
        Lea.__init__(self)
        self._leaC = Lea.coerce(leaC)
        self._leaDict = dict((c,Lea.coerce(lea1)) for (c,lea1) in leaDict.items())
        if defaultLea is Lea._DUMMY_VAL:
            self._defaultLea = Lea._DUMMY_VAL
        else:
            self._defaultLea = Lea.coerce(defaultLea)
            self._leaDict = defaultdict(lambda:self._defaultLea,self._leaDict)

    def _genDependingAleas(self):
        for aleaLeaf in self._leaC.getAleaLeavesSet():
            yield aleaLeaf
        for lea1 in self._leaDict.values():
            for aleaLeaf in lea1.getAleaLeavesSet():
                yield aleaLeaf
        if self._defaultLea is not Lea._DUMMY_VAL:
            for aleaLeaf in self._defaultLea.getAleaLeavesSet():
                yield aleaLeaf

    def _getLeaChildren(self):
        leaChildren = [self._leaC,]
        for lea1 in self._leaDict.values():
            leaChildren.append(lea1)
        if self._defaultLea is not Lea._DUMMY_VAL:
            leaChildren.append(self._defaultLea)
        return leaChildren

    def _clone(self,cloneTable):
        return Tlea(self._leaC.clone(cloneTable),dict((v,lea1.clone(cloneTable)) for (v,lea1) in leaDict),self._defaultLea.clone(cloneTable))

    def _genVPs(self):
        leaDict = self._leaDict
        for (vc,pc) in self._leaC.genVPs():
            try:
                leaV = leaDict[vc]
            except KeyError:
                raise Lea.Error("missing value '%s' in CPT"%vc)
            for (vd,pd) in leaV.genVPs():
                yield (vd,pc*pd)

    def _genOneRandomMC(self):
        leaDict = self._leaDict
        for vc in self._leaC._genOneRandomMC():
            try:
                leaV = leaDict[vc]
            except KeyError:
                raise Lea.Error("missing value '%s' in CPT"%vc)
            for vd in leaV._genOneRandomMC():
                yield vd
