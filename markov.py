'''
--------------------------------------------------------------------------------

    markov.py

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
from alea import Alea 
from blea import Blea
from itertools import islice

class Chain(object):
    '''
    A Chain instance represents a Markov chain, with a given set of states and
    and given probabilities of transition from state to state.    
    '''

    __slots__ = ('_stateObjs','_state','_nextStateBlea')

    def __init__(self,stateObjs,*transProbsPerState):
        ''' initializes Chain instance's attributes; 
            stateObjs is a sequence of objects representing states (strings, usually);
            transProbsPerState arguments contain the transiiton probability weights;
            there is one such argument per state, it is a tuple (stateObj,transProbs)
            where transProbs is the sequence of probability weights of transition from
            stateObj to each declared state, in the order of their declarations 
        '''
        object.__init__(self)
        # TODO check consistency of arguments
        # self._state is declared as a uniform distribution on the states
        # it is used to build self._nextStateBlea, which encompasses the transition probabilities
        self._stateObjs = stateObjs
        self._state = StateAlea(Lea.fromVals(*stateObjs),self)
        iterNextStateData = ( (self._state==stateObj, Alea.fromValFreqs(*zip(stateObjs,transProbs)))
                              for (stateObj,transProbs) in transProbsPerState )
        self._nextStateBlea = Blea.build(*iterNextStateData)

    def getStates(self):
        ''' returns a tuple containing one State instance per state declared in the chain,
            in the order of their declaration; each instance represents a certain 
            unique state
        ''' 
        return tuple(StateAlea(Lea.coerce(stateObj),self) for stateObj in self._stateObjs)

    def nextState(self,fromState=None,n=1):
        ''' returns the State instance obtained after n transitions from initial state
            defined by the given fromState
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
    
    __slots__ = ('_chain',)
    
    def __init__(self,stateLea,chain):
        '''
        '''
        Alea.__init__(self,stateLea.getAlea().genVPs())
        self._chain = chain

    def nextState(self,n=1):
        '''
        '''
        return self._chain.nextState(self,n)

    def randomSeqIter(self):
        '''
        '''
        state = self
        while True:
            yield state.random()
            state = state.nextState()

    def randomSeq(self,n):
        '''
        '''
        return tuple(islice(self.randomSeqIter(),n))
