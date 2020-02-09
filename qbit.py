'''
--------------------------------------------------------------------------------

    qbit.py

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
from .qgate import QGate
import itertools

qbit_values = (0, 1)

class QBit(object):

    '''
    A QBit instance represents a quantum bit, with a superposition of the states
    0 and 1. The quantum state of such instance is represented by a Lea instance
    having probability amplitudes (i.e. complex numbers) instead of usual
    probabilities. A "pure state" is defined by an Alea instance, with probability
    amplitudes for 0 and 1. A "mixed state" is defined by a Tlea instance, obtained
    by application of quantum gates (see QGate class), possibly interacting with
    other quantum bit(s); this Tlea instance represents a DAG refering to other
    Alea / Tlea instances, mimicking quantum bits entangled by a quantum circuit.
    The measured state of a Qbit can be calculated as a normal Alea instance,
    giving probabilities of 0 and 1, by using the usual Statues algorithm followed
    by the Born rule, for converting probability amplitudes to true probabilities.
    Then, random samples made on this Alea instance represent measurements.
    '''

    __slots__ = ('psy_lea',)

    class Error(Exception):
        pass

    #                   0   ,   1
    default_psy_ps = ( 1+0j ,  0+0j)
    #default_psy_ps = ( 1+1j ,  0+0j)

    def __init__(self,psy_lea):
        self.psy_lea = None
        self.set_psy(psy_lea)

    def set_psy(self,psy_lea):
        self.psy_lea = psy_lea

    @staticmethod
    def build(psy_ps=default_psy_ps,n=None):
        if n is None:
            return QBit(Alea.pmf(zip(qbit_values,psy_ps),prob_type='c'))
        return tuple(QBit.build(psy_ps) for _ in range(n))

    def copy(self):
        return QBit(self.psy_lea)

    def __str__(self):
        return str(self.psy_lea)

    __repr__ = __str__

    def calq(self,*args,**kwargs):
        return self.psy_lea.calc('q',*args,**kwargs)

    def measure(self,n=None):
        return self.calq().random(n)

    def apply_gate(self,gate):
        gate_cpt = QGate.get_cpt(gate,1)
        self.set_psy(self.psy_lea.switch(gate_cpt))

