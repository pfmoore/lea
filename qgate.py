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

from .lea import Alea
from .toolbox import zip, log2 
import itertools
from collections import OrderedDict

# 2-tuple giving the values of quantum bits
# other candidates: ('0','1'), (False,True), ...
qbit_values = (0, 1)


class QGate(object):
    '''
    A QGate instance represents a quantum gate acting on n QBits;
    it is defined by a (n^2) x (n^2) matrix associating each possible
    combination of n QBits to the probability amplitude of getting
    another combination of n QBits; more precisely, provided the Qbit
    combinations are numbered in lexical order, the element (i,j) of the
    matrix defines the probability amplitude of getting the combination
    j of QBits if they were in combination i before application of the gate  
    '''

    class Error(Exception):
        ''' exception representing any violation of requirements of QGate methods  
        '''
        pass

    # dictionary for retrieving a QGate instance by its name 
    _instances_by_name = dict()

    def __init__(self,name,cpt_dict):
        ''' initializes QGate instance's attributes;
            name is the name to be assigned on created QGate;
            cpt_dict is the CPT associating each QBit combination to an Alea
            instance giving the probability amplitudes of each new QBit
            combination after QGate application;
            requires that name is unique
        '''
        self.name = name
        self.cpt_dict = cpt_dict
        if name in QGate._instances_by_name:
            raise QGate.Error("duplicate QGate name '%s'"%(name,))
        QGate._instances_by_name[name] = self

    @staticmethod
    def _build_gate(name,qbit_values,matrix):
        ''' static method, returns a new QGate instance from the given matrix;
            name is the name to be assigned on created QGate;
            qbit_values is an iterable givng the 2^n combinations of n QBit values;
            matrix is an iterable (row) of iterables (column) giving such that
            matrix[i][j] is the probability amplitude of getting qbit_values[j]
            if it was qbit_values[i] before application of the gate 
            requires that name is unique;
            requires that qbit_values has dimension (r^2) if the gate applies on r QBits;
            requires that matrix has dimensions (r^2) x (r^2) if the gate applies
            on r QBits  
        '''
        qbit_values = tuple(qbit_values)
        cpt_dict = OrderedDict((v0, Alea.pmf(zip(qbit_values,matrix_row),prob_type='c',remove_zeroes=True))
                               for (v0,matrix_row) in zip(qbit_values,matrix))
        return QGate(name,cpt_dict)

    @staticmethod
    def build_gate_sn(name,matrix):
        ''' static method, returns a new QGate instance from the given matrix;
            matrix is an iterable (row) of iterables (column) such that
            matrix[i][j] is the probability amplitude of getting combination j
            of QBits if they were in combination i before application of the
            gate assuming that Qbit combinations are numbered in lexical order;
            requires that name is unique;
            requires that matrix has dimensions (r^2) x (r^2) if the gate applies
            on r QBits
        '''
        n = int(log2(len(matrix)))
        gate_slots = itertools.product(*(n*(qbit_values,)))
        return QGate._build_gate(name,gate_slots,matrix)

    @staticmethod
    def retrieve(gate_name):
        ''' static method, returns the QGate instance having the given name;
            requires that QGate with given name exists;
        '''
        gate = QGate._instances_by_name.get(gate_name)
        if gate is None:
            raise QGate.Error("unknown gate '%s'"%gate_name)
        return gate

    @staticmethod
    def get_cpt(gate,n=None):
        ''' returns the CPT dictionary of given gate;
            if gate argument is a string, then the QGate with given name is retrieved;
            requires that if gate argument is a string, then QGate with given name exists;
            requires that, if n is not None, the gate handles n QBits;
        '''
        if isinstance(gate,str):
            gate = QGate.retrieve(gate)
        if n is not None and len(gate.cpt_dict) != 2**n:
            raise QGate.Error("gate '%s' requires %d qubit(s) instead of %d"%(gate.name,int(log2(len(gate.cpt_dict))),n))
        return gate.cpt_dict

    def dim(self):
        ''' returns the number of QBits handled by the QGate
        '''
        return int(log2(len(self.cpt_dict)))

    def get_matrix(self):
        ''' returns the matrix of probability amplitudes of the QGate as
            a tuple of tuples (see __init__)
        '''
        values = tuple(self.cpt_dict.keys())
        return tuple(tuple(complex(alea1.p(v)) for v in values)
                     for alea1 in self.cpt_dict.values())

# Predefined matrices 
# note: it is not mandatory to normalize the matrices, the normalization is done in constructor
#        (e.g. matrix for gate 'H' will have a 1/sqrt(2) factor)

__gate_sn_matrices = (
# gates for 1 QBit
#           out:   0   ,   1
    ( 'X'   , ((  0+0j ,  1+0j ),    # in: 0
               (  1+0j ,  0+0j ))),  # in: 1
    ( 'Y'   , ((  0+0j ,  0-1j ),    # in: 0
               (  0+1j ,  0+0j ))),  # in: 1
    ( 'Z'   , ((  1+0j ,  0+0j ),    # in: 0
               (  0+0j , -1+0j ))),  # in: 1
    ( 'H'   , ((  1+0j ,  1+0j ),    # in: 0
               (  1+0j , -1+0j ))),  # in: 1
    ( 'S'   , ((  1+0j ,  0+0j ),    # in: 0
               (  0+0j ,  0+1j ))),  # in: 1    
    ( 'T'   , ((  1+0j ,  0+0j ),    # in: 0
               (  0+0j ,  1+1j ))),  # in: 1
    ( 'I'   , ((  1+0j ,  0+0j ),    # in: 0
               (  0+0j ,  1+0j ))),  # in: 1
    # WARNING not in line with Google complementary information
    ( 'SX'  , ((  1+1j ,  1-1j ),    # in: 0
               (  1-1j ,  1+1j ))),  # in: 1 
    ( 'SY'  , ((  1+1j , -1-1j ),    # in: 0
               (  1+1j ,  1+1j ))),  # in: 1 
    ( 'SZ'  , ((  1+0j ,  0+0j ),    # in: 0
               (  0+0j ,  0+1j ))),  # in: 1 
# gates for 2 QBits
#           out: (0,0) , (0,1) , (1,0) , (1,1)
    ( 'SHOR' ,((  1+0j ,  0+0j ,  0+0j ,  0+0j ),    # in: (0,0)
               (  0+0j ,  1+0j ,  0+0j ,  0+0j ),    # in: (0,1)
               (  0+0j ,  0+0j ,  1+0j ,  1+0j ),    # in: (1,0)
               (  0+0j ,  0+0j ,  1+0j , -1+0j ))),  # in: (1,1)
    ( 'CNOT' ,((  1+0j ,  0+0j ,  0+0j ,  0+0j ),    # in: (0,0)
               (  0+0j ,  1+0j ,  0+0j ,  0+0j ),    # in: (0,1)
               (  0+0j ,  0+0j ,  0+0j ,  1+0j ),    # in: (1,0)
               (  0+0j ,  0+0j ,  1+0j ,  0+0j ))),  # in: (1,1)
    ( 'SWAP' ,((  1+0j ,  0+0j ,  0+0j ,  0+0j ),    # in: (0,0)
               (  0+0j ,  0+0j ,  1+0j ,  0+0j ),    # in: (0,1)
               (  0+0j ,  1+0j ,  0+0j ,  0+0j ),    # in: (1,0)
               (  0+0j ,  0+0j ,  0+0j ,  1+0j ))),  # in: (1,1)
    ( 'SSWAP',((  1+0j ,  0+0j ,  0+0j ,  0+0j ),    # in: (0,0)
               (  0+0j ,  1+1j ,  1-1j ,  0+0j ),    # in: (0,1)
               (  0+0j ,  1-1j ,  1+1j ,  0+0j ),    # in: (1,0)
               (  0+0j ,  0+0j ,  0+0j ,  1+0j ))))  # in: (1,1)

for (name,matrix) in __gate_sn_matrices:
    QGate.build_gate_sn(name,matrix)

