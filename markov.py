'''
--------------------------------------------------------------------------------

    markov.py

--------------------------------------------------------------------------------
Copyright 2013-2019 Pierre Denis

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
from .toolbox import zip, dict, defaultdict
from itertools import islice, tee

# try to import optional numpy module
try:
    import numpy as np
    # numpy module installed
except:
    # numpy module not installed
    np = None


class Chain(object):
    '''
    A Chain instance represents a Markov chain, with a given set of states
    and given probabilities of transition from state to state.
    Two instance attributes are public:
    - states: tuple containing the states of the Markov chain, without any probability
    - state : StateAlea instance representing an equiprobable distribution of states
              of the Markov chain; it can be used to express a condition to pass to
              state_given or next_state_given methods
    '''

    __slots__ = ('_next_state_lea_per_state','states','_state_alea_dict','state','_next_state_tlea')
    
    def __init__(self,next_state_lea_per_state):
        ''' initializes Chain instance's attributes; 
            next_state_lea_per_state is a sequence of tuples (S,next_state_lea)
            where S is an object representing a state (typically a string) and
            next_state_lea is a Lea instance giving probabilities of transition
            from S to the next state
        '''
        object.__init__(self)
        self._next_state_lea_per_state = tuple(next_state_lea_per_state)
        # states of the Markov chain, without any probability
        self.states = tuple(state for (state,_) in self._next_state_lea_per_state)
        # dictionary associating each given state S to StateAlea instance where S has probability 1
        self._state_alea_dict = dict((state,StateAlea(Alea.coerce(state),self))
                                     for state in self.states)
        # StateAlea instance where all states are equiprobable
        self.state = StateAlea(Alea.vals(*self.states),self)
        # CPT associating each state with the probability distribution of transition to next state
        self._next_state_tlea = Tlea(self.state,dict(self._next_state_lea_per_state))

    @staticmethod
    def from_matrix(states,*trans_probs_per_state):
        ''' returns a new Chain instance from given arguments;
            states is a sequence of objects representing states (typically strings);
            trans_probs_per_state arguments contain the transition probabilities;
            there is one such argument per state, it is a tuple (state,trans_probs)
            where trans_probs is the sequence of probabilities of transition from
            state to each declared state, in the order of their declarations 
        '''
        next_state_leas = (Alea.pmf(zip(states,trans_probs),ordered=True)
                           for (state,trans_probs) in trans_probs_per_state)
        next_state_lea_per_state = tuple(zip(states,next_state_leas))
        return Chain(next_state_lea_per_state)

    @staticmethod
    def from_seq(state_seq):
        ''' returns a new Chain instance from given sequence of state objects;
            the probabilities of state transitions are set according to transition
            frequencies in the given sequence;
            if last state of state_seq does not occur elsewhere in state_seq,
            then this state is defined arbitrarily as an absorbing state (i.e.
            its next state is itself with probability 1)
        '''
        (from_state_iter,to_state_iter) = tee(state_seq);
        for _ in to_state_iter:
            break
        next_states_dict = defaultdict(list)
        for (from_state,to_state) in zip(from_state_iter,to_state_iter):
            next_states_dict[from_state].append(to_state)
        if to_state not in next_states_dict:
            next_states_dict[to_state].append(to_state)
        next_state_name_and_objs = list(next_states_dict.items())
        next_state_name_and_objs.sort()
        next_state_lea_per_state = tuple((state,Alea.vals(*next_states))
                                         for (state,next_states) in next_state_name_and_objs)
        return Chain(next_state_lea_per_state)        

    def __str__(self):
        ''' returns a string representation of the Markov chain
            with each state S followed by the indented representation of probability distribution
            of transition from S to next state
        '''        
        ## note: the simpler expression ('  -> ' + next_state_lea.map(str)) works but it reorders the states
        formatted_next_state_leas = (Alea.pmf(('  -> ' + next_state_lea.map(str))._calc(),ordered=True)
                                     for (_,next_state_lea) in self._next_state_lea_per_state)
        return '\n'.join('%s\n%s'%(state,formatted_next_state_lea)
                         for (state,formatted_next_state_lea)
                         in zip(self.states,formatted_next_state_leas))
        
    __repr__ = __str__

    def get_states(self):
        ''' returns a tuple containing one StateAlea instance per state declared in the chain,
            in the order of their declaration; each instance represents a certain, unique, state
        '''
        return tuple(self._state_alea_dict[state] for state in self.states)

    def get_state(self,state_lea):
        ''' returns a StateAlea instance corresponding to the probability distribution
            given in state_lea;
            if state_lea is not a Lea instance, then it is assumed to be a certain state
        '''
        if isinstance(state_lea,Lea):
            return StateAlea(state_lea,self)
        # state_lea is not Lea instance: assume that it is a certain state object
        return self._state_alea_dict[state_lea]

    def next_state(self,from_state=None,n=1):
        ''' returns the StateAlea instance obtained after n transitions from an initial
            state defined by the given from_state, which either a given certain state
            or a Lea instance giving the probability distribution of states;
            if from_state is None, then the initial state is the uniform probability
            distribution of the declared states;
            if n = 0, then this initial state is returned
        '''
        if n < 0:
            raise Lea.Error("next_state method requires a positive value for argument 'n'")
        if from_state is None:
            from_state = self.state
        state_n = Alea.coerce(from_state).get_alea()
        while n > 0:
            n -= 1
            state_n = self._next_state_tlea.given(self.state==state_n).get_alea()
        return StateAlea(state_n,self)

    def state_given(self,cond_lea):
        ''' returns the StateAlea instance verifying the given cond_lea, a Lea instance
            expressing a condition using the 'state' instance attribute
        '''
        return StateAlea(self.state.given(cond_lea),self)

    def next_state_given(self,cond_lea,n=1):
        ''' returns the StateAlea instance obtained after n transitions from initial state
            defined by the state distribution verifying the given cond_lea, a Lea instance
            expressing a condition using the 'state' instance attribute;
            if n = 0, then this initial state is returned
        '''
        from_state = self.state.given(cond_lea)
        return self.next_state(from_state,n)
    
    def matrix(self,from_states=None,to_states=None,as_array=False):
        ''' returns the probability matrix of transition from given iterable from_states to
            given iterable to_states;
            if from_states or to_states is None (default), then it is replaced by all states
            of Markov chain (so, without arguments, the full transition matrix is returned);
            if as_array is False (default), then the matrix is returned as a tuple of tuples
            otherwise, it is returned as a numpy array (an exception is raised if numpy is
            not installed)
        '''
        if from_states is None:
            from_states = self.states
        if to_states is None:
            to_states = self.states
        res_matrix = tuple(tuple(next_state_alea._p(to_state) for to_state in to_states)
                     for (state,next_state_alea) in self._next_state_lea_per_state
                     if state in from_states)
        if as_array:
            if np is None:
                raise Lea.Error("the matrix() method requires the numpy package")
            res_matrix = np.array(res_matrix)
        return res_matrix
    
    def reachable_states(self,from_state,_cur_reachable_states=None):
        ''' returns a tuple containing the states that can be reached starting from the
            given from_state; the returned states are ordered as defined in the 'states'
            attribute of the MC
        '''
        if _cur_reachable_states is None:
            new_reachable_states = set()
        else:
            new_reachable_states = _cur_reachable_states
        new_reachable_states.add(from_state)
        for (to_state,p) in self._state_alea_dict[from_state].next_state()._gen_raw_vps():
            if p > 0 and to_state not in new_reachable_states:
                self.reachable_states(to_state,new_reachable_states)
        if _cur_reachable_states is None:
            return tuple(state for state in self.states if state in new_reachable_states)
    
    def absorbing_mc_info(self,as_array=False):
        ''' returns a tuple
            (is_absorbing, transient_states, absorbing_states, q_matrix, r_matrix, n_matrix)
            where
            * is_absorbing     is a boolean telling whether the Markov chain is absorbing
            * transient_states is a tuple containing the t transient states
            * absorbing_states is a tuple containing the r absorbing states
            * q_matrix         is the t x t probability matrix from transient to transient states
            * r_matrix         is the t x r probability matrix from transient to absorbing states
            * n_matrix         is the t x t fundamental matrix, defined as inv(I-q_matrix),
                               calculated only if as_array is True (=None, otherwise)
            notes:
            - t > 0 iff the Markov chain is absorbing;
            - the returned states in transient_states and absorbing_states are ordered as
              defined in the 'states' attribute of the MC; the q_matrix and r_matrix matrices
              follow the same order
            - if as_array is False (default), then the matrices are returned as a tuple of tuples
              otherwise, they are returned as numpy arrays (an exception is raised if numpy is
              not installed)
        '''
        reachable_states_by_state = tuple((state,self.reachable_states(state))
                                          for (state,next_state_alea) in self._next_state_lea_per_state)
        absorbing_states = tuple(state for (state,reachable_states) in reachable_states_by_state
                                            if len(reachable_states) == 1 and reachable_states[0] is state)
        absorbing_states_set = frozenset(absorbing_states)
        transient_states = tuple(state for (state,reachable_states) in reachable_states_by_state
                                           if state not in absorbing_states
                                           and len(absorbing_states_set.intersection(reachable_states)) > 0)
        is_absorbing = len(self.states) == len(absorbing_states) + len(transient_states)
        q_matrix = self.matrix(from_states=transient_states,to_states=transient_states,as_array=as_array)
        r_matrix = self.matrix(from_states=transient_states,to_states=absorbing_states,as_array=as_array)
        if as_array:
            n_matrix = np.linalg.inv(np.identity(len(transient_states))-q_matrix)
        else:
            n_matrix = None
        return (is_absorbing,transient_states,absorbing_states,q_matrix,r_matrix,n_matrix)


class StateAlea(Alea):
    '''
    A StateAlea instance represents a probability distribution of states, for a given Markov chain
    '''
    
    __slots__ = ('_chain',)
    
    def __init__(self,state_lea,chain):
        ''' initializes StateAlea instance's attributes
            corresponding to the probability distribution given in state_lea
            and referring to the given chain, instance of Chain
        '''
        # order the states to follow the order given in chain.states, for user-friendliness
        ## if not needed, the following statement is sufficient:
        ## Alea.__init__(self,*zip(*state_lea.get_alea()._gen_vp()))
        vs = []
        ps = []
        state_lea_pmf_dict = state_lea.pmf_dict
        for state in chain.states:
            p = state_lea_pmf_dict.get(state,self)
            if p is not self:
                vs.append(state)
                ps.append(p)
        Alea.__init__(self,vs,ps)
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
            state = state.next_state().random()
            yield state
            state = self._chain.get_state(state)

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
