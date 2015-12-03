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
from alea import Alea
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
    
    def __init__(self,*ileas,**kwargs):
        Lea.__init__(self)
        self._ileas = ileas
        self._ctxClea = kwargs.get('ctxClea')
        # _condLea is used only by _genOneRandomMC method
        self._condClea = None

    ## in PY3, could use: def build(*clauses,priorLea=None,autoElse=False,check=True,requiresCtx=True):
    _argNamesOfBuildMeth = frozenset(('priorLea','autoElse','check','requiresCtx'))

    @staticmethod
    def build(*clauses,**kwargs):
        argNames = frozenset(kwargs.keys())
        unknownArgNames = argNames - Blea._argNamesOfBuildMeth
        if len(unknownArgNames) > 0:
            raise Lea.Error("unknown argument keyword '%s'; shall be only among %s"%(next(iter(unknownArgNames)),tuple(Blea._argNamesOfBuildMeth)))
        priorLea = kwargs.get('priorLea',None)
        autoElse = kwargs.get('autoElse',False)
        check = kwargs.get('check',True)
        requiresCtx = kwargs.get('requiresCtx',True)
        elseClauseResults = tuple(result for (cond,result) in clauses if cond is None)
        if len(elseClauseResults) > 1:
            raise Lea.Error("impossible to define more than one 'else' clause")
        if len(elseClauseResults) == 1:
            if priorLea is not None:
                raise Lea.Error("impossible to define together prior probabilities and 'else' clause")
            if autoElse:
                raise Lea.Error("impossible to have autoElse=True and 'else' clause")                
            elseClauseResult = elseClauseResults[0]
        elif autoElse:
            if priorLea is not None:
                raise Lea.Error("impossible to define together prior probabilities and autoElse=True")
            # take uniform distribution on all values found in clause's results (principle of indifference)
            elseClauseResult = Lea.fromVals(*frozenset(val for (cond,result) in clauses for val in Lea.coerce(result).vals()))
        else:
            elseClauseResult = None
        # note that the getAlea call limits what can be done by the caller (e.g. no way to have "cascaded" CPT)
        normClauses = ((Lea.coerce(cond),Lea.coerce(result).getAlea()) for (cond,result) in clauses if cond is not None)
        ## alternative (NOK):
        ##  normClauses = ((Lea.coerce(cond),Lea.coerce(result).given(cond).getAlea()) for (cond,result) in clauses if cond is not None)
        (condLeas,resAleas) = tuple(zip(*normClauses))
        # check that conditions are disjoint
        if check:
            if any(v.count(True) > 1 for (v,_) in Clea(*condLeas)._genVPs()):
                raise Lea.Error("clause conditions are not disjoint")
        # build the OR of all given conditions
        orCondsLea = Lea.reduce(or_,condLeas,True)
        if priorLea is not None:
            # prior distribution: determine elseClauseResult
            if check and orCondsLea.isTrue():
                # TODO check priorLea equivalent to self
                raise Lea.Error("forbidden to define prior probabilities for complete clause set")
            (pTrue,count) = orCondsLea._p(True)
            pFalse = count - pTrue
            priorAleaDict = dict(priorLea.getAlea()._genVPs())
            priorAleaCount = sum(priorAleaDict.values())
            normAleaDict = dict(Lea.fromSeq(resAleas).flat().getAlea()._genVPs())
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
            # no 'else' clause: check that clause set is complete
            if check and not orCondsLea.isTrue():
                raise Lea.Error("incomplete clause set requires 'else' clause or autoElse=True or priorLea=...")
        if elseClauseResult is not None:
            # add the else clause
            elseCondLea = ~orCondsLea
            ## other equivalent statement: elseCondLea = Lea.reduce(and_,(~condLea for condLea in condLeas))
            condLeas += (elseCondLea,)
            resAleas += (Lea.coerce(elseClauseResult).getAlea(),)
            # note that orCondsLea is NOT extended with orCondsLea |= elseCondLea
            # so, in case of else clause (and only in this case), orCondsLea is NOT certainly true
        if requiresCtx:
            # the caller cannot guarantee that all CPT conditions refer to the same set of variables
            # (e.g. CPT with context-specific independence); to handle this, we define _ctxLea as a 
            # cartesian product of all Alea leaves present in CPT conditions and having multiple
            # possible values; a rebalancing of probability weights is needed if there are such
            # missing variables and if these admit multiple possible values (total probability
            # weight > 1)
            aleaLeavesSet = frozenset(aleaLeaf for condLea in condLeas                    \
                                               for aleaLeaf in condLea.getAleaLeavesSet() \
                                               if aleaLeaf._count > 1                     )
            ctxClea = Clea(*aleaLeavesSet)
        else:
            # the caller guarantees that all CPT conditions refer to the same set of variables
            # e.g. each condition is of the form someLeaVar == v
            ctxClea = None
        # make a probability weight balancing, in the case where Alea results have different 
        # probability weight total
        # 1. calculate the common denominator from probability weight totals of Alea results;
        # note that it would be sensible to calculate the LCM but this could be time-consuming,
        # the technique below (multiplication of unique values) is the fastest
        commonDenominator = 1
        for aleaCount in frozenset(resAlea._count for resAlea in resAleas):
            commonDenominator *= aleaCount
        # 2. transform the Alea results into equivalent Alea having ALL the same total probability
        # weight, using the "non-reduction" Alea constructor (i.e. the given probability weights
        # remain unchanged)
        resAleasNR = []
        for resAlea in resAleas:
            normFactor = commonDenominator // resAlea._count
            resAleasNR.append(Alea.fromValFreqsNR(*((v,p*normFactor) for (v,p) in zip(resAlea._vs,resAlea._ps))))
        # build a Blea, providing a sequence of new Ileas for each of the clause 
        return Blea(*(Ilea(resAleaNR,(condLea,)) for (resAleaNR,condLea) in zip(resAleasNR,condLeas)),ctxClea=ctxClea)    

    def _getLeaChildren(self):
        leaChildren = self._ileas
        if self._ctxClea is not None:
            leaChildren += (self._ctxClea,)
        return leaChildren 
    
    def _clone(self,cloneTable):
        return Blea(*(iLea.clone(cloneTable) for iLea in self._ileas),ctxClea=self._ctxClea.clone(cloneTable))

    def _genCtxFreeVPs(self):
        for iLea in self._ileas:
            for vp in iLea._genVPs():
                yield vp

    def _genVPs(self):
        if self._ctxClea is None:
            for vp in self._genCtxFreeVPs():
                yield vp
        else:
            ctxClea = self._ctxClea
            for (v,p) in self._genCtxFreeVPs():
                for (_,p2) in ctxClea:
                    yield (v,p*p2)

    def _genOneRandomMC(self):
        if self._condClea is None:
            # _condAlea is a cartesian product of all Alea leaves present in CPT conditions;
            condAleaLeavesSet = frozenset(aleaLeaf for ilea in self._ileas                             \
                                                   for aleaLeaf in ilea._condLeas[0].getAleaLeavesSet())
            self._condClea = Clea(*condAleaLeavesSet)
        # the first for loop binds a random value on each Alea instances refered in CPT conditions
        for _ in self._condClea._genOneRandomMC():
            # here, there will be at most one ilea having condition that evaluates to True,
            # regarding the random binding that has been made 
            for iLea in self._ileas:
                for v in iLea._genOneRandomMCNoExc():
                    if v is not iLea:
                        # the current ilea is the one having the condition that evaluates to True
                        yield v
