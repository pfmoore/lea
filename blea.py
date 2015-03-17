'''
--------------------------------------------------------------------------------

    blea.py

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
from clea import Clea
from mlea import Mlea
from ilea import Ilea
from prob_fraction import ProbFraction
from toolbox import dict, genPairs

from operator import or_
from itertools import chain
            
class Blea(Lea):
    
    '''
    Blea is a Lea subclass, which instance represents a conditional probability
    table (CPT), as a set of Ilea instances. Each Ilea instance represents a
    given distibution <Vi,p(Vi|C)>, assuming a given condition C is verified,
    in the sense of a conditional probability.
    The set of conditions shall form a partition of the "certain true", i.e.
     ORing  all conditions shall give a "certain true" distribution
     ANDing all conditions pairwise shall give "certain false" distributions
    '''
    
    __slots__ = ('_iLeas','_ctxClea')
    
    def __init__(self,*iLeas):
        Lea.__init__(self)
        self._iLeas = tuple(iLeas)
        # the following treatment is needed only if some clauses miss variables present 
        # in other clauses (e.g. CPT with context-specific independence)
        # a rebalancing is needed if there are such missing variables and if these admit multiple
        # values (total probability weight > 1)
        aleaLeavesSet = frozenset(aleaLeaf for ilea in iLeas                       \
                                           for aleaLeaf in ilea.getAleaLeavesSet() \
                                           if aleaLeaf._count > 1                  )
        self._ctxClea = Clea(*aleaLeavesSet)

    @staticmethod
    def build(*clauses,**kwargs):
        priorLea = kwargs.get('priorLea',None)
        # TODO: check no other args !!
        # PY3: def build(*clauses,priorLea=None):
        elseClauseResults = tuple(result for (cond,result) in clauses if cond is None)
        if len(elseClauseResults) > 1:
            raise Lea.Error("impossible to define more than one 'other' clause")
        if len(elseClauseResults) == 1:
            if priorLea is not None:
                raise Lea.Error("impossible to define together prior probabilities and 'other' clause")
            elseClauseResult = elseClauseResults[0]
        else:
            elseClauseResult = None
        normClauseLeas = tuple((Lea.coerce(cond),Lea.coerce(result)) for (cond,result) in clauses if cond is not None)
        condLeas = tuple(condLea for (condLea,resultLea) in normClauseLeas)
        # check that conditions are disjoint
        for (condLea1,condLea2) in genPairs(condLeas):
            if (condLea1&condLea2).isFeasible():
                raise Lea.Error("clause conditions are not disjoint")
        # build the OR of all given conditions
        orCondsLea = Lea.reduce(or_,condLeas)
        isClauseSetComplete = orCondsLea.isTrue()
        if priorLea is not None:
            # prior distribution: determine elseClauseResult
            if isClauseSetComplete:
                # TODO check priorLea equivalent to self
                raise Lea.Error("forbidden to define prior probabilities for complete clause set")
            (pTrue,count) = orCondsLea._p(True)
            pFalse = count - pTrue
            priorAleaDict = dict(priorLea.getAlea().genVPs())
            priorAleaCount = sum(priorAleaDict.values())
            normAleaDict = dict(Mlea(*(resultLea for (condLea,resultLea) in normClauseLeas)).getAlea().genVPs())
            normAleaCount = sum(normAleaDict.values())
            valuesSet = frozenset(chain(priorAleaDict.keys(),normAleaDict.keys()))
            vps = []
            for value in valuesSet:
                 priorP = priorAleaDict.get(value,0)
                 condP = normAleaDict.get(value,0)
                 p = priorP*count*normAleaCount - condP*pTrue*priorAleaCount
                 if not(0 <= p <= pFalse*normAleaCount*priorAleaCount):
                     # Infeasible : probability represented by p goes outside range from 0 to 1
                     priorPFraction = ProbFraction(priorP,priorAleaCount)
                     lowerPFraction = ProbFraction(condP*pTrue,count*normAleaCount)
                     upperPFraction = ProbFraction(condP*pTrue+pFalse*normAleaCount,count*normAleaCount)
                     raise Lea.Error("prior probability of '%s' is %s, outside the range [ %s , %s ]"%(value,priorPFraction,lowerPFraction,upperPFraction))
                 vps.append((value,p))
            elseClauseResult = Lea.fromValFreqs(*vps)
        elif elseClauseResult is None:
            # check that clause set is complete
            if not isClauseSetComplete:
                # TODO? : assume a uniform prior distribution ? ... which values ? 
                raise Lea.Error("incomplete clause set requires 'other' clause or prior probabilities")
        if elseClauseResult is not None:
            elseCondLea = ~orCondsLea
            normClauseLeas += ((elseCondLea,Lea.coerce(elseClauseResult)),)
            # note that orCondsLea is NOT extended with rCondsLea |= elseCondLea
            # so, in case of else clause (and only in this case), orCondsLea is NOT certainly true
        return Blea(*(Ilea(resultLea,condLea) for (condLea,resultLea) in normClauseLeas))    
    
    def _getLeaChildren(self):
        return self._iLeas + (self._ctxClea,)
    
    def _clone(self,cloneTable):
        return Blea(*(iLea.clone(cloneTable) for iLea in self._iLeas))

    def _genVPs(self):
        for iLea in self._iLeas:
            for (v,p) in iLea.genVPs():
                for (_,p2) in self._ctxClea:
                    yield (v,p*p2)
