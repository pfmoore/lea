'''
--------------------------------------------------------------------------------

    qregister.py

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

from .lea import Lea
from .qbit import QBit
from .qgate import QGate

qbit_values = (0, 1)


class QRegister(object):

    '''
    A QRegister instance represents a quantum register, that is a vector of
    n quantum bits represented by QBit instances. Quantum gates (see QGate
    class) can be applied in sequence on given subsets of QBits, following
    a quantum circuit; this results into a QRegister where the inner QBit
    instances are entangled.
    '''

    __slots__ = ('qbit_vec',)

    def __init__(self,*qbit_vec):
        self.qbit_vec = tuple(qbit_vec)

    @staticmethod
    def build(psy_ps=QBit.default_psy_ps,n=None):
        return QRegister(*QBit.build(psy_ps,n))

    def copy(self):
        return QRegister(*(qbit.copy() for qbit in self.qbit_vec))

    def __getitem__(self,slice):
        return self.qbit_vec[slice]
    
    def apply_gate(self,gate,*qb_idxs):
        nb_idxs = len(qb_idxs)
        if nb_idxs == 1:
            self.qbit_vec[qb_idxs[0]].apply_gate(gate)
        else:
            if nb_idxs == 0:
                nb_idxs = len(self.qbit_vec)
                qb_idxs = range(nb_idxs)
            gate_cpt = QGate.get_cpt(gate,nb_idxs)
            joint_qb_in = Lea.joint(*(self.qbit_vec[qb_idx].psy_lea for qb_idx in qb_idxs))
            joint_qb_out = joint_qb_in.switch(gate_cpt)
            for (i,qb_idx) in enumerate(qb_idxs):
                self.qbit_vec[qb_idx].set_psy(joint_qb_out[i])

    def as_joint(self):
        return Lea.joint(*(qbit.psy_lea for qbit in self.qbit_vec))

    def __str__(self):
        return str(self.as_joint())
    
    __repr__ = __str__

    def calq(self,*args,**kwargs):
        return self.as_joint().calc('q',*args,**kwargs)

    def measure(self,n=None):
        return self.calq().random(n)


q = QBit.build()
print (q)
print (q.calq())
q.apply_gate('H')
print (q)
print (q.calq())

print ('--------------------------')
qr = QRegister.build(n=2)
print (qr)
print (qr.calq())
qr.apply_gate('H',1)
print (qr)
print (qr.calq())
#qr.apply_gate_s2('SWAP',0,1)
qr.apply_gate('H',0)
print (qr)
print (qr.calq())
qr.apply_gate('SWAP',0,1)
print (qr)
print (qr.calq())
qr.apply_gate('H',0)
print (qr)
print (qr.calq())
qr.apply_gate('SSWAP',0,1)
print (qr)
print (qr.calq())
qr.apply_gate('SSWAP',1,0)
print (qr)
print (qr.calq())
qr.apply_gate('Y',0)
print (qr)
print (qr.calq())


print ('--------------------------')
qbitx = QBit.build((-1/2**.5,1j/2**.5))
qbit0 = QBit.build((1,0))

qr = QRegister(qbitx,qbit0,qbit0)
print (qr)
#print (qr.calq())
qr.apply_gate('H',1)
qr.apply_gate('CNOT',1,2)
qr.apply_gate('CNOT',0,1)
qr.apply_gate('H',0)
qrm = qr.calq()
r = qrm[1].switch({0: qrm[1],
                   1: qrm[1]}) 
