'''
--------------------------------------------------------------------------------

    blea.py

--------------------------------------------------------------------------------
Copyright 2013-2016 Pierre Denis

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
from .clea import Clea
from .ilea import Ilea
from .prob_fraction import ProbFraction
from .toolbox import dict, zip
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

    __slots__ = ('_ileas','_condClea')
    
    def __init__(self,*ileas,**kwargs):
        Lea.__init__(self)
        self._ileas = ileas
        # _condLea is used only by _genOneRandomMC method
        self._condClea = None

    __argNamesOfBuildMeth = frozenset(('priorLea','autoElse','check'))

    @staticmethod
    def build(*clauses,**kwargs):
        ''' see Lea.buildCPT method        
        '''
        argNames = frozenset(kwargs.keys())
        unknownArgNames = argNames - Blea.__argNamesOfBuildMeth
        if len(unknownArgNames) > 0:
            raise Lea.Error("unknown argument keyword '%s'; shall be only among %s"%(next(iter(unknownArgNames)),tuple(Blea._argNamesOfBuildMeth)))
        priorLea = kwargs.get('priorLea',None)
        autoElse = kwargs.get('autoElse',False)
        check = kwargs.get('check',True)
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
        # get clause conditions and results, excepting 'else' clause, after coercion to Lea instances
        normClauses = ((Lea.coerce(cond),Lea.coerce(result)) for (cond,result) in clauses if cond is not None)
        (condLeas,resLeas) = tuple(zip(*normClauses))
        # check that conditions are disjoint
        if check:
            clea_ = Clea(*condLeas)
            clea_._initCalc()
            if any(v.count(True) > 1 for (v,_) in clea_.genVPs()):
                raise Lea.Error("clause conditions are not disjoint")
        # build the OR of all given conditions, excepting 'else'
        orCondsLea = Lea.reduce(or_,condLeas,True)
        if priorLea is not None:
            # prior distribution: determine elseClauseResult
            if check and orCondsLea.isTrue():
                # TODO check priorLea equivalent to self
                raise Lea.Error("forbidden to define prior probabilities for complete clause set")
            (pTrue,count) = orCondsLea._p(True)
            pFalse = count - pTrue
            priorAleaDict = dict(priorLea.getAlea().vps())
            priorAleaCount = sum(priorAleaDict.values())
            normAleaDict = dict(Lea.fromSeq(resLeas).flat().getAlea().vps())
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
            elseClauseResult = Lea.coerce(elseClauseResult)
            resLeas += (elseClauseResult,)
            condLeas += (elseCondLea,)
            # note that orCondsLea is NOT extended with orCondsLea |= elseCondLea
            # so, in case of else clause (and only in this case), orCondsLea is NOT certainly true
        # build a Blea, providing a sequence of new Ileas for each of the clause 
        return Blea(*(Ilea(resLea,(condLea,)) for (resLea,condLea) in zip(resLeas,condLeas)))

    def _getLeaChildren(self):
        return self._ileas
    
    def _clone(self,cloneTable):
        return Blea(*(iLea.clone(cloneTable) for iLea in self._ileas))

    def _genVPs(self):
        for iLea in self._ileas:
            for vp in iLea.genVPs():
                yield vp

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
