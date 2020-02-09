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

qbit_values = (0, 1)

class QGate(object):

    class Error(Exception):
        pass
    
    _instances_by_name = dict()

    def __init__(self,name,cpt):
        self.name = name
        self.cpt = cpt
        QGate._instances_by_name[name] = self

    @staticmethod
    def _build_gate(name,qbit_values,matrix):
        cpt = OrderedDict((v0, Alea.pmf(zip(qbit_values,matrix_row),prob_type='c',remove_zeroes=True))
                          for (v0,matrix_row) in zip(qbit_values,matrix))
        # TODO
        #cpt = OrderedDict((v0, Alea.pmf(zip(qbit_values,(abs(p*p) for p in matrix_row)),prob_type='f',remove_zeroes=True,normalization=False))
        #                  for (v0,matrix_row) in zip(qbit_values,matrix))
        return QGate(name,cpt)

    @staticmethod
    def build_gate_s1(name,matrix):
        return QGate._build_gate(name,qbit_values,matrix)

    @staticmethod
    def build_gate_sn(name,matrix):
        n = int(log2(len(matrix)))
        gate_slots = tuple(itertools.product(*(n*(qbit_values,))))
        return QGate._build_gate(name,gate_slots,matrix)

    @staticmethod
    def retrieve(gate_name):
        gate = QGate._instances_by_name.get(gate_name)
        if gate is None:
            raise QGate.Error("unknown gate '%s'"%gate_name)
        return gate

    @staticmethod
    def get_cpt(gate,n=None):
        if isinstance(gate,str):
            gate = QGate.retrieve(gate)
        if len(gate.cpt) != 2**n:
            raise QGate.Error("gate '%s' requires %d qubit(s) instead of %d"%(gate.name,int(log2(len(gate.cpt))),n))
        return gate.cpt

    def get_matrix(self):
        values = tuple(self.cpt.keys())
        return tuple(tuple(complex(alea1.p(v)) for v in values)
                     for alea1 in self.cpt.values())

# warning; matrices can be not normalized, assuming normalization done in constructor
__gate_s1_matrices = (
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
               (  0+0j ,  0+1j ))))  # in: 1 

for (name,matrix) in __gate_s1_matrices:
    QGate.build_gate_s1(name,matrix)

__gate_sn_matrices = (
#           out: (0,0) , (0,1) , (1,0) , (1,1)
    ( 'CNOT', ((  1+0j ,  0+0j ,  0+0j ,  0+0j ),    # in: (0,0)
               (  0+0j ,  1+0j ,  0+0j ,  0+0j ),    # in: (0,1)
               (  0+0j ,  0+0j ,  0+0j ,  1+0j ),    # in: (1,0)
               (  0+0j ,  0+0j ,  1+0j ,  0+0j ))),  # in: (1,1)
    ( 'SWAP', ((  1+0j ,  0+0j ,  0+0j ,  0+0j ),    # in: (0,0)
               (  0+0j ,  0+0j ,  1+0j ,  0+0j ),    # in: (0,1)
               (  0+0j ,  1+0j ,  0+0j ,  0+0j ),    # in: (1,0)
               (  0+0j ,  0+0j ,  0+0j ,  1+0j ))),  # in: (1,1)
    ( 'SSWAP',((  1+0j ,  0+0j ,  0+0j ,  0+0j ),    # in: (0,0)
               (  0+0j ,  1+1j ,  1-1j ,  0+0j ),    # in: (0,1)
               (  0+0j ,  1-1j ,  1+1j ,  0+0j ),    # in: (1,0)
               (  0+0j ,  0+0j ,  0+0j ,  1+0j ))))  # in: (1,1)
               
for (name,matrix) in __gate_sn_matrices:
    QGate.build_gate_sn(name,matrix)

