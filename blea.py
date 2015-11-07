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

    __slots__ = ('_ileas','_ctxClea','_condClea')
    
    def __init__(self,*ileas):
        Lea.__init__(self)
        self._ileas = tuple(ileas)
        # the following treatment is needed only if some clauses miss variables present 
        # in other clauses (e.g. CPT with context-specific independence)
        # a rebalancing is needed if there are such missing variables and if these admit
        # multiple possible values (total probability weight > 1)
        # _ctxClea is a cartesian product of all Alea leaves present in CPT and having
        # multiple possible values
        aleaLeavesSet = frozenset(aleaLeaf for ilea in ileas                       \
                                           for aleaLeaf in ilea.getAleaLeavesSet() \
                                           if aleaLeaf._count > 1                  )
        self._ctxClea = Clea(*aleaLeavesSet)
        # _condAlea is a cartesian product of all Alea leaves present in CPT conditions;
        # it is needed only by _genOneRandomMC method
        condAleaLeavesSet = frozenset(aleaLeaf for ilea in ileas                                   \
                                               for aleaLeaf in ilea._condLeas[0].getAleaLeavesSet())
        self._condClea = Clea(*condAleaLeavesSet)

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
        if any(v.count(True) > 1 for (v,_) in Clea(*condLeas)._genVPs()):
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
            priorAleaDict = dict(priorLea.getAlea()._genVPs())
            priorAleaCount = sum(priorAleaDict.values())
            normAleaDict = dict(Lea.fromSeq(resultLea for (condLea,resultLea) in normClauseLeas).flat().getAlea()._genVPs())
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
        return Blea(*(Ilea(resultLea,(condLea,)) for (condLea,resultLea) in normClauseLeas))    

    def _getLeaChildren(self):
        return self._ileas + (self._ctxClea,)
    
    def _clone(self,cloneTable):
        return Blea(*(iLea.clone(cloneTable) for iLea in self._ileas))

    def _genVPs(self):
        for iLea in self._ileas:
            for (v,p) in iLea._genVPs():
                for (_,p2) in self._ctxClea:
                    yield (v,p*p2)

    def _genOneRandomMC(self):
        # the first for loop binds a random value on each Alea instances refered in CPT conditions
        for _ in self._condClea._genOneRandomMC():
            # here, there will be at most one ilea having condition that evaluates to True,
            # regarding the random binding that has been made 
            for iLea in self._ileas:
                for v in iLea._genOneRandomMCNoExc():
                    if v is not iLea:
                        # the current ilea is the one having the condition that evaluates to True
                        yield v
