'''
--------------------------------------------------------------------------------

    markov.py

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
from .blea import Blea
from .toolbox import zip, dict
from itertools import islice, tee

class Chain(object):
    '''
    A Chain instance represents a Markov chain, with a given set of states
    and given probabilities of transition from state to state.    
    '''

    __slots__ = ('_stateObjs','_stateAleaDict','_state','_nextStateBlea')
    
    def __init__(self,nextStateLeaPerState):
        ''' initializes Chain instance's attributes; 
            nextStateLeaPerState is a sequence of tuples (stateObj,nextStateLea)
            where stateObj is a state object (e.g. a string) and nextStateLea is a Lea instance
            giving probabilities of transition from stateObj to each state object 
        '''
        object.__init__(self)
        self._stateObjs = tuple(stateObj for (stateObj,nextStateLea) in nextStateLeaPerState)
        self._stateAleaDict = dict((stateObj,StateAlea(Lea.coerce(stateObj),self)) for stateObj in self._stateObjs)
        self._state = StateAlea(Lea.fromVals(*self._stateObjs),self)
        iterNextStateData = ((self._state==stateObj,nextStateLea) for (stateObj,nextStateLea) in nextStateLeaPerState)
        self._nextStateBlea = Blea.build(*iterNextStateData)

    @staticmethod
    def fromMatrix(stateObjs,*transProbsPerState):
        ''' returns a new Chain instance from given arguments
            stateObjs is a sequence of objects representing states (strings, usually);
            transProbsPerState arguments contain the transiiton probability weights;
            there is one such argument per state, it is a tuple (stateObj,transProbs)
            where transProbs is the sequence of probability weights of transition from
            stateObj to each declared state, in the order of their declarations 
        '''
        nextStateLeas = (Alea.fromValFreqs(*zip(stateObjs,transProbs)) for (stateObj,transProbs) in transProbsPerState)
        nextStateLeaPerState = tuple(zip(stateObjs,nextStateLeas))
        return Chain(nextStateLeaPerState)

    @staticmethod
    def fromSeq(stateObjSeq):
        ''' returns a new Chain instance from given sequence of state objects
            the probabilities of state transitions are set according to transition
            frequencies in the given sequence 
        '''
        (fromStateObjIter,toStateObjIter) = tee(stateObjSeq);
        for _ in toStateObjIter:
            break
        nextStateObjsDict = dict()
        for (fromStateObj,toStateObj) in zip(fromStateObjIter,toStateObjIter):
            nextStateObjs = nextStateObjsDict.get(fromStateObj)
            if nextStateObjs is None:
                nextStateObjs = []
                nextStateObjsDict[fromStateObj] = nextStateObjs
            nextStateObjs.append(toStateObj)
        nextStateNameAndObjs = list(nextStateObjsDict.items())
        nextStateNameAndObjs.sort()
        nextStateLeaPerState = tuple((stateObj,Alea.fromVals(*nextStateObjs)) for (stateObj,nextStateObjs) in nextStateNameAndObjs)
        return Chain(nextStateLeaPerState)        

    def __str(self,formatFunc):
        ''' returns a string representation of the Markov chain
            with each state S followed by the indented representation of probability distribution
            of transition from S to next state
            formatFunc is a function to represent probability distribution, taking a Lea instance
            as argument and returning a string 
        '''
        nextStateLeas = (nextStateILea._lea1 for nextStateILea in self._nextStateBlea._iLeas)
        formattedNextStateLeas = ('  -> ' + nextStateLea.map(str) for nextStateLea in nextStateLeas)
        return '\n'.join('%s\n%s'%(stateObj,formatFunc(formattedNextStateLea)) for (stateObj,formattedNextStateLea) in zip(self._stateObjs,formattedNextStateLeas))

    def asFloat(self):
        ''' same as __str__ but the probabilities are given in decimal representation 
        '''        
        return self.__str(Lea.asFloat)

    def asPct(self):
        ''' same as __str__ but the probabilities are given in percentage representation 
        '''        
        return self.__str(Lea.asPct)

    def __str__(self):
        ''' returns a string representation of the Markov chain
            with each state S followed by the indented representation of probability distribution
            of transition from S to next state
            the probabilities are given in fraction representation 
        '''        
        return self.__str(Lea.__str__)
        
    __repr__ = __str__

    def getStates(self):
        ''' returns a tuple containing one StateAlea instance per state declared in the chain,
            in the order of their declaration; each instance represents a certain, unique, state
        ''' 
        #return tuple(StateAlea(Lea.coerce(stateObj),self) for stateObj in self._stateObjs)
        return tuple(self._stateAleaDict[stateObj] for stateObj in self._stateObjs)

    def getState(self,stateObjLea):
        ''' returns a StateAlea instance corresponding to the probability distribution
            given in stateObjLea
            if stateObjLea is not a Lea instance, then it is assumed to be a certain state
        '''
        if isinstance(stateObjLea,Lea):
            return StateAlea(stateObjLea,self)
        # stateObjLea is not Lea instance: assume that it is a certain state object
        return self._stateAleaDict[stateObjLea]

    def nextState(self,fromState=None,n=1):
        ''' returns the StateAlea instance obtained after n transitions from initial state
            defined by the given fromState, instance of StateAlea
            if fromState is None, then the initial state is the uniform distribution of the declared states
            if n = 0, then this initial state is returned
        '''
        if n < 0:
            raise Lea.Error("nextState method requires a positive value for argument 'n'")
        if fromState is None:
            fromState = self._state
        stateN = Lea.coerce(fromState).getAlea()
        while n > 0:
            n -= 1
            stateN = self._nextStateBlea.given(self._state==stateN).getAlea()
        return StateAlea(stateN,self)

    def stateGiven(self,condLea):
        ''' returns the StateAlea instance verifying the given cond Lea 
        '''
        return StateAlea(self._state.given(condLea),self)

    def nextStateGiven(self,condLea,n=1):
        ''' returns the StateAlea instance obtained after n transitions from initial state
            defined by the state distribution verifying the given cond Lea 
            if n = 0, then this initial state is returned
        '''
        fromState = self._state.given(condLea)
        return self.nextState(fromState,n)


class StateAlea(Alea):
    '''
    A StateAlea instance represents a probability distribution of states, for a given Markov chain
    '''
    
    __slots__ = ('_chain',)
    
    def __init__(self,stateObjLea,chain):
        ''' initializes StateAlea instance's attributes
            corresponding to the probability distribution given in stateObjLea
            and referring to the given chain, instance of Chain 
        '''
        Alea.__init__(self,*zip(*stateObjLea.getAlea().vps()))
        self._chain = chain

    def nextState(self,n=1):
        ''' returns the StateAlea instance obtained after n transitions from initial state self
            if n = 0, then self is returned
        '''
        return self._chain.nextState(self,n)

    def genRandomSeq(self):
        ''' generates an infinite sequence of random state objects,
            starting from self and obeying the transition probabilities defined in the chain
        '''
        state = self
        while True:
            stateObj = state.nextState().random()
            yield stateObj
            state = self._chain.getState(stateObj)

    def randomSeq(self,n):
        ''' returns a tuple containing n state objects representing a random sequence
            starting from self and obeying the transition probabilities defined in the chain
        '''
        return tuple(islice(self.genRandomSeq(),n))
