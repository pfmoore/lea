'''
--------------------------------------------------------------------------------

    qgate.py

--------------------------------------------------------------------------------
Copyright 2013-2020 Pierre Denis

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

from .lea import Lea, Alea
from .toolbox import zip, log2
import itertools
from collections import OrderedDict

# 2-tuple giving the values of quantum bits
# other candidates are: ('0', '1'), (False, True), etc.
qbit_values = (0, 1)


class QGate(object):
    '''
    A QGate instance represents a quantum gate acting on n qubits;
    it is defined by a (2^n) x (2^n) matrix associating each possible
    combination of n qubits to the probability amplitude of getting
    another combination of n qubits; more precisely, provided the qubit
    combinations are numbered in lexical order, the element (i,j) of the
    matrix defines the probability amplitude of getting the combination
    j of qubits if they were in combination i before application of the gate;
    an instance of QGate is used a function, passing as arguments either
    the n Lea instances representing input qubits or the joint of these Lea
    instances (a Clea instance); the returned object represents the qubits
    (possibly entangled) after application of the gate, in one of the
    following forms:
    - a Lea instance representing the modified qubit if gate applies on
      one single qubit;
    - a tuple of n Lea instances representing the n modified qubits if gate
      applies on n >= 2 qubits and if input are passed as n arguments;
    - a joint (Clea instance) giving the tuples representing the n quibts
      if gate applies on n >= 2 qubits and if input is passed as a joint
    '''

    __slots__ = ('cpt_dict',)
    
    def __init__(self,cpt_dict):
        ''' initializes QGate instance's attributes;
            cpt_dict is the CPT associating each qubit combination to an Alea
            instance giving the probability amplitudes of each new qubit
            combination after QGate application
        '''
        self.cpt_dict = cpt_dict

    @staticmethod
    def build_gate(matrix):
        ''' static method, returns a new QGate instance from the given matrix;
            matrix is an iterable (row) of iterables (column) such that
            matrix[i][j] is the probability amplitude of getting combination j
            of qubits if they were in combination i before application of the
            gate assuming that Qbit combinations are numbered in lexical order;
            assumes that matrix has dimensions (2^n) x (2^n) if the gate applies
            on n qubits
        '''
        n = int(log2(len(matrix)))
        qubit_tuples = tuple(itertools.product(*(n*(qbit_values,))))
        try:
            cpt_dict = OrderedDict((v0, Alea.pmf(zip(qubit_tuples,matrix_row),prob_type='c',remove_zeroes=True))
                                   for (v0,matrix_row) in zip(qubit_tuples,matrix))
        except:
            # retry with simpler treatment, assuming that exception is due to probabilities as SymPy expressions
            cpt_dict = OrderedDict((v0, Alea.pmf(zip(qubit_tuples,matrix_row),prob_type='sc',remove_zeroes=False,normalization=False))
                                   for (v0,matrix_row) in zip(qubit_tuples,matrix))
        return QGate(cpt_dict)

    def dim(self):
        ''' returns the number of qubits handled by the QGate
        '''
        return int(log2(len(self.cpt_dict)))

    def matrix(self):
        ''' returns the matrix of probability amplitudes of the QGate as
            a tuple of tuples (see build_gate method)
        '''
        values = tuple(self.cpt_dict.keys())
        try:
            return tuple(tuple(complex(alea1.p(v)) for v in values)
                         for alea1 in self.cpt_dict.values())
        except:
            # retry with simpler treatment, assuming that exception is due to probabilities as SymPy expressions
            return tuple(tuple(alea1.p(v) for v in values)
                         for alea1 in self.cpt_dict.values())

    def __call__(self,*args):
        ''' returns new qubit resulting from application of self gate on the
            given args;
            an instance of QGate is used a function, passing as arguments either
            the n Lea instances representing input qubits or the joint of these Lea
            instances (a Clea instance); the returned object represents the qubits
            (possibly entangled) after application of the gate, in one of the
            following forms:
            - a Lea instance representing the modified qubit if gate applies on
              one single qubit;
            - a tuple of n Lea instances representing the n modified qubits if gate
              applies on n >= 2 qubits and if input are passed as n arguments;
            - a joint (Clea instance) giving the tuples representing the n quibts
              if gate applies on n >= 2 qubits and if input is passed as a joint
        ''' 
        nb_args = len(args)
        gate_dim = self.dim()
        args = (Alea.coerce(a) for a in args)
        # Alea instances shall be copied, otherwise they would be shared betwwen all gate calls
        cpt_dict = dict((k,alea1.new()) for (k,alea1) in self.cpt_dict.items())
        if nb_args == 1 and gate_dim > 1:
            arg = next(args)
            joint_qb_out = arg.switch(cpt_dict)
        else:
            if nb_args != gate_dim:
                raise Lea.Error("the gate requires %d qubit(s) instead of %d"%(gate_dim,nb_args))
            joint_qb_out = Lea.joint(*args).switch(cpt_dict)
        if gate_dim == 1:
            return joint_qb_out[0]
        if nb_args == 1:
            return joint_qb_out
        return tuple(joint_qb_out[i] for i in range(gate_dim))

# Predefined gates
# note: it is not mandatory to normalize the matrices, the normalization is done in constructor
#        (e.g. matrix for gate 'H' will have a 1/sqrt(2) factor)

# gates for 1 qubit
#                      out:   0   ,   1
x     = QGate.build_gate(((  0+0j ,  1+0j ),    # in: 0
                          (  1+0j ,  0+0j )))   # in: 1
y     = QGate.build_gate(((  0+0j ,  0-1j ),    # in: 0
                          (  0+1j ,  0+0j )))   # in: 1
z     = QGate.build_gate(((  1+0j ,  0+0j ),    # in: 0
                          (  0+0j , -1+0j )))   # in: 1
h     = QGate.build_gate(((  1+0j ,  1+0j ),    # in: 0
                          (  1+0j , -1+0j )))   # in: 1
s     = QGate.build_gate(((  1+0j ,  0+0j ),    # in: 0
                          (  0+0j ,  0+1j )))   # in: 1    
t     = QGate.build_gate(((  1+0j ,  0+0j ),    # in: 0
                          (  0+0j ,  1+1j )))   # in: 1
i     = QGate.build_gate(((  1+0j ,  0+0j ),    # in: 0
                          (  0+0j ,  1+0j )))   # in: 1
# WARNING not in line with Google complementary information
sx    = QGate.build_gate(((  1+1j ,  1-1j ),    # in: 0
                          (  1-1j ,  1+1j )))   # in: 1
sy    = QGate.build_gate(((  1+1j , -1-1j ),    # in: 0
                          (  1+1j ,  1+1j )))   # in: 1
sz    = QGate.build_gate(((  1+0j ,  0+0j ),    # in: 0
                          (  0+0j ,  0+1j )))   # in: 1
try:
    #requires sympy
    sym1 = QGate.build_gate((('a' ,   'b' ),    # in: 0
                             ('c' ,   'd' )))   # in: 1
except:
    sym1 = None

# gates for 2 qubits
#                      out: (0,0) , (0,1) , (1,0) , (1,1)
cnot  = QGate.build_gate(((  1+0j ,  0+0j ,  0+0j ,  0+0j ),    # in: (0,0)
                          (  0+0j ,  1+0j ,  0+0j ,  0+0j ),    # in: (0,1)
                          (  0+0j ,  0+0j ,  0+0j ,  1+0j ),    # in: (1,0)
                          (  0+0j ,  0+0j ,  1+0j ,  0+0j )))   # in: (1,1)
swap  = QGate.build_gate(((  1+0j ,  0+0j ,  0+0j ,  0+0j ),    # in: (0,0)
                          (  0+0j ,  0+0j ,  1+0j ,  0+0j ),    # in: (0,1)
                          (  0+0j ,  1+0j ,  0+0j ,  0+0j ),    # in: (1,0)
                          (  0+0j ,  0+0j ,  0+0j ,  1+0j )))   # in: (1,1)
sswap = QGate.build_gate(((  1+0j ,  0+0j ,  0+0j ,  0+0j ),    # in: (0,0)
                          (  0+0j ,  1+1j ,  1-1j ,  0+0j ),    # in: (0,1)
                          (  0+0j ,  1-1j ,  1+1j ,  0+0j ),    # in: (1,0)
                          (  0+0j ,  0+0j ,  0+0j ,  1+0j )))   # in: (1,1)
