'''
--------------------------------------------------------------------------------

    blea.py

--------------------------------------------------------------------------------
Copyright 2013, 2014 Pierre Denis

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
from ilea import Ilea
from mlea import Mlea
from prob_fraction import ProbFraction
from toolbox import calcLCM, zip, dict

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
    
    __slots__ = ('_iLeas','_factors')
    
    def __init__(self,*iLeas):
        Lea.__init__(self)
        self._iLeas = iLeas
        counts = tuple(iLea.getCount() for iLea in iLeas)
        lcm = calcLCM(counts)
        self._factors = tuple(lcm//count for count in counts)

    @staticmethod
    def __genPairs(seq):
        tuple1 = tuple(seq)
        length = len(tuple1)
        if length < 2:
            return
        if length == 2:
            yield tuple1
        else:
            head = tuple1[0]
            tail = tuple1[1:]
            for a in tail:
                yield (head,a)
            for pair in Blea.__genPairs(tail):
                yield pair

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
        for (condLea1,condLea2) in Blea.__genPairs(condLeas):
            if (condLea1&condLea2).isFeasible():
                raise Lea.Error("clause conditions are not disjoint")
        orCondsLea = None
        for condLea in condLeas:
            # check that all conditions are feasible        
            if not condLea.isFeasible():
                raise Lea.Error("some clause condition is not feasible")
            if orCondsLea is None:
                orCondsLea = condLea
            else:       
                orCondsLea |= condLea
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
        elif elseClauseResult is not None:
            if isClauseSetComplete:
                raise Lea.Error("forbidden to define 'other' clause for complete clause set")
        else:
            # check that clause set is complete
            if not isClauseSetComplete:
                # TODO? : assume a uniform prior distribution ? ... which values ? 
                raise Lea.Error("incomplete clause set requires 'other' clause or prior probabilities")
        if elseClauseResult is not None:
            elseCondLea = ~orCondsLea
            normClauseLeas += ((elseCondLea,Lea.coerce(elseClauseResult)),)
            # note that orCondsLea is NOT extended with rCondsLea |= elseCondLea
            # so, in case of else clause (and only in this case), orCondsLea is NOT certainly true  
        # the following treatment is needed only in case of context-specific independence
        # which occurs if at least one given conditions miss variable(s) present in other conditions
        # and if such missing variables have multiple values (e.g. not values coerced to Lea)  
        aleaLeavesSet = frozenset(aleaLeaf for aleaLeaf in orCondsLea.getAleaLeavesSet() if aleaLeaf._count > 1)
        ileas = []
        for (condLea,resultLea) in normClauseLeas:
            missingAleaLeavesSet = aleaLeavesSet - condLea.getAleaLeavesSet()
            for missingAleaLeaf in missingAleaLeavesSet:
                # the current given condition misses variable(s) present in other conditions
                # add dummy condition(s) with missing variable(s), always true,
                # just to have the right balance of probability weights
                condLea &= (missingAleaLeaf==missingAleaLeaf)
            ileas.append(Ilea(resultLea,condLea))
        return Blea(*ileas)    
    
    def _getLeaChildren(self):
        return self._iLeas
    
    def _clone(self,cloneTable):
        return Blea(*(iLea.clone(cloneTable) for iLea in self._iLeas))

    def _genVPs(self):
        for (iLea,f) in zip(self._iLeas,self._factors):
            for (v,p) in iLea.genVPs():
                yield (v,f*p)
