'''
--------------------------------------------------------------------------------

    markov.py

--------------------------------------------------------------------------------
Copyright 2013-2018 Pierre Denis

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
from .tlea import Tlea
from .toolbox import zip, dict
from itertools import islice, tee

class Chain(object):
    '''
    A Chain instance represents a Markov chain, with a given set of states
    and given probabilities of transition from state to state.    
    '''

    __slots__ = ('_next_state_lea_per_state','_state_objs','_state_alea_dict','_state','_next_state_tlea')
    
    def __init__(self,next_state_lea_per_state):
        ''' initializes Chain instance's attributes; 
            next_state_lea_per_state is a sequence of tuples (state_obj,next_state_lea)
            where state_obj is a state object (e.g. a string) and next_state_lea is a Lea instance
            giving probabilities of transition from state_obj to each state object 
        '''
        object.__init__(self)
        self._next_state_lea_per_state = tuple(next_state_lea_per_state)
        self._state_objs = tuple(state_obj for (state_obj,_) in self._next_state_lea_per_state)
        self._state_alea_dict = dict((state_obj,StateAlea(Alea.coerce(state_obj),self)) for state_obj in self._state_objs)
        self._state = StateAlea(Alea.vals(*self._state_objs),self)
        self._next_state_tlea = Tlea(self._state,dict(self._next_state_lea_per_state))

    @staticmethod
    def from_matrix(state_objs,*trans_probs_per_state):
        ''' returns a new Chain instance from given arguments
            state_objs is a sequence of objects representing states (strings, usually);
            trans_probs_per_state arguments contain the transiiton probability weights;
            there is one such argument per state, it is a tuple (state_obj,trans_probs)
            where trans_probs is the sequence of probability weights of transition from
            state_obj to each declared state, in the order of their declarations 
        '''
        next_state_leas = (Alea.pmf(zip(state_objs,trans_probs)) for (state_obj,trans_probs) in trans_probs_per_state)
        next_state_lea_per_state = tuple(zip(state_objs,next_state_leas))
        return Chain(next_state_lea_per_state)

    @staticmethod
    def from_seq(state_obj_seq):
        ''' returns a new Chain instance from given sequence of state objects
            the probabilities of state transitions are set according to transition
            frequencies in the given sequence 
        '''
        (from_state_obj_iter,to_state_obj_iter) = tee(state_obj_seq);
        for _ in to_state_obj_iter:
            break
        next_state_objs_dict = dict()
        for (from_state_obj,to_state_obj) in zip(from_state_obj_iter,to_state_obj_iter):
            next_state_objs = next_state_objs_dict.get(from_state_obj)
            if next_state_objs is None:
                next_state_objs = []
                next_state_objs_dict[from_state_obj] = next_state_objs
            next_state_objs.append(to_state_obj)
        next_state_name_and_objs = list(next_state_objs_dict.items())
        next_state_name_and_objs.sort()
        next_state_lea_per_state = tuple((state_obj,Alea.vals(*next_state_objs)) for (state_obj,next_state_objs) in next_state_name_and_objs)
        return Chain(next_state_lea_per_state)        

    def __str__(self):
        ''' returns a string representation of the Markov chain
            with each state S followed by the indented representation of probability distribution
            of transition from S to next state
        '''        
        next_state_leas = (next_state for (_,next_state) in self._next_state_lea_per_state)
        formatted_next_state_leas = ('  -> ' + next_state_lea.map(str) for next_state_lea in next_state_leas)
        return '\n'.join('%s\n%s'%(state_obj,formatted_next_state_lea)
                         for (state_obj,formatted_next_state_lea)
                         in zip(self._state_objs,formatted_next_state_leas))
        
    __repr__ = __str__

    def get_states(self):
        ''' returns a tuple containing one StateAlea instance per state declared in the chain,
            in the order of their declaration; each instance represents a certain, unique, state
        '''
        return tuple(self._state_alea_dict[state_obj] for state_obj in self._state_objs)

    def get_state(self,state_obj_lea):
        ''' returns a StateAlea instance corresponding to the probability distribution
            given in state_obj_lea
            if state_obj_lea is not a Lea instance, then it is assumed to be a certain state
        '''
        if isinstance(state_obj_lea,Lea):
            return StateAlea(state_obj_lea,self)
        # state_obj_lea is not Lea instance: assume that it is a certain state object
        return self._state_alea_dict[state_obj_lea]

    def next_state(self,from_state=None,n=1):
        ''' returns the StateAlea instance obtained after n transitions from initial state
            defined by the given from_state, instance of StateAlea
            if from_state is None, then the initial state is the uniform distribution of the declared states
            if n = 0, then this initial state is returned
        '''
        if n < 0:
            raise Lea.Error("next_state method requires a positive value for argument 'n'")
        if from_state is None:
            from_state = self._state
        state_n = Alea.coerce(from_state).get_alea()
        while n > 0:
            n -= 1
            state_n = self._next_state_tlea.given(self._state==state_n).get_alea()
        return StateAlea(state_n,self)

    def state_given(self,cond_lea):
        ''' returns the StateAlea instance verifying the given cond Lea 
        '''
        return StateAlea(self._state.given(cond_lea),self)

    def next_state_given(self,cond_lea,n=1):
        ''' returns the StateAlea instance obtained after n transitions from initial state
            defined by the state distribution verifying the given cond Lea 
            if n = 0, then this initial state is returned
        '''
        from_state = self._state.given(cond_lea)
        return self.next_state(from_state,n)


class StateAlea(Alea):
    '''
    A StateAlea instance represents a probability distribution of states, for a given Markov chain
    '''
    
    __slots__ = ('_chain',)
    
    def __init__(self,state_obj_lea,chain):
        ''' initializes StateAlea instance's attributes
            corresponding to the probability distribution given in state_obj_lea
            and referring to the given chain, instance of Chain 
        '''
        Alea.__init__(self,*zip(*state_obj_lea.get_alea()._gen_vp()))
        self._chain = chain

    def next_state(self,n=1):
        ''' returns the StateAlea instance obtained after n transitions from initial state self
            if n = 0, then self is returned
        '''
        return self._chain.next_state(self,n)

    def gen_random_seq(self):
        ''' generates an infinite sequence of random state objects,
            starting from self and obeying the transition probabilities defined in the chain
        '''
        state = self
        while True:
            state_obj = state.next_state().random()
            yield state_obj
            state = self._chain.get_state(state_obj)

    def random_seq(self,n):
        ''' returns a tuple containing n state objects representing a random sequence
            starting from self and obeying the transition probabilities defined in the chain
        '''
        if n is not None:
            n = int(n)
        return tuple(islice(self.gen_random_seq(),n))

# convenience aliases
chain_from_matrix = Chain.from_matrix
chain_from_seq = Chain.from_seq
