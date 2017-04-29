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
from .alea import Alea
from .toolbox import dict, defaultdict

class Tlea(Lea):
    
    '''
    Tlea is a Lea subclass, which instance represents a conditional probability
    table (CPT) giving a Lea instance C and a dictionary associating each
    possible value of C to a specific Lea instance.
    '''

    __slots__ = ('_leaC','_leaDict','_defaultLea','_factorsDict')

    def __init__(self,leaC,leaDict,defaultLea=Lea._DUMMY_VAL):
        if isinstance(leaDict,defaultdict):
            raise Lea.Error('defaultdict not supported for Tlea, use defaultLea argument instead')
        Lea.__init__(self)
        self._leaC = Lea.coerce(leaC)
        self._leaDict = dict((c,Lea.coerce(lea1)) for (c,lea1) in leaDict.items())
        if not all(isinstance(lea1,Alea) for lea1 in self._leaDict.values()):
            raise Lea.Error('all Tlea dictionary values shall be Alea instances or constants')
        leaDictItems = list(self._leaDict.items())
        if defaultLea is Lea._DUMMY_VAL:
            self._defaultLea = Lea._DUMMY_VAL
        else:
            self._defaultLea = Lea.coerce(defaultLea)
            self._leaDict = defaultdict(lambda:self._defaultLea,self._leaDict)
            leaDictItems.append((Lea._DUMMY_VAL,self._defaultLea))
        for (_,lea1) in leaDictItems:
            lea1._initCalc()
        leaCounts = [(c,sum(p for (_,p) in lea1.genVPs())) for (c,lea1) in leaDictItems]
        pcount = 1
        for (_,count) in leaCounts:
            pcount *= count
        if defaultLea is not Lea._DUMMY_VAL:
            (_,count) = leaCounts.pop()
            defaultFactor = pcount // count
        self._factorsDict = dict((c,pcount//count) for (c,count) in leaCounts)
        if defaultLea is not Lea._DUMMY_VAL:
            self._factorsDict = defaultdict(lambda:defaultFactor,self._factorsDict)

    def _getLeaChildren(self):
        leaChildren = [self._leaC]
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
            f = self._factorsDict[vc]
            for (vd,pd) in leaV.genVPs():
                yield (vd,pc*pd*f)

    def _genOneRandomMC(self):
        leaDict = self._leaDict
        for vc in self._leaC._genOneRandomMC():
            try:
                leaV = leaDict[vc]
            except KeyError:
                raise Lea.Error("missing value '%s' in CPT"%vc)
            for vd in leaV._genOneRandomMC():
                yield vd
