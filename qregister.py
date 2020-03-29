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

from .lea import Lea, Alea
from .qgate import QGate, qbit_values


class QRegister(object):

    '''
    A QRegister instance represents a quantum register, that is a vector of
    n quantum bits. Quantum gates (see QGate class) can be applied in sequence
    on given subsets of QBits, modelling a quantum circuit; this results into
    a QRegister where the inner QBit instances are entangled.
    Each quantum bit is modelled as a Lea instance, modelling a supeposition of
    0 and 1 with respective probability amplitudes (i.e. complex numbers)instead
    of usual probabilities. A "pure state" is defined by an Alea instance.
    A "mixed state" is defined by a Tlea or Flea2 instance, obtained
    by application of quantum gates (see QGate class), possibly interacting with
    other quantum bit(s); such instance represents a DAG refering to other
    Lea instances, mimicking quantum bits entangled by a quantum circuit.
    The measured state of a QRegister can be calculated as a usual Alea instance,
    giving probabilities of each n-tuple of 0 and 1; this is done by using the
    usual Statues algorithm followed by the Born rule, for converting probability
    amplitudes to true probabilities. The obtained Alea instance is a joint
    probability distribution, without quantum features, that models the
    distributios of mesaurements taking into account the entanglement.
    Consequently, random samples made on this Alea instance represent actual
    measurements.
    Note: Unlike Lea instances, a QRegister instance is mutable. New Lea instances
    are created and stored each time a gate is applied. The created Lea instances
    replace the ones modified by the gates, these last being children of the new
    Lea instances.
    '''

    __slots__ = ('qbits',)

    # default probability amplitudes for building QBit (used in QRegistry.build_pure method)
    default_psy_p0 = 1 + 0j
    default_psy_p1 = 0 + 0j

    def __init__(self,qbits):
        ''' initialize QRegister instance with qbits, a vector of n Lea
            instances; each Lea instance represents a quantum bit giving
            probability amplitudes of 0 and 1
        '''
        # a list is used, to be able to replace Qbits after gate application
        self.qbits = [Alea.coerce(qbit,prob_type='c') for qbit in qbits]

    @staticmethod
    def build_pure(n=1,psy_p0=default_psy_p0,psy_p1=default_psy_p1):
        ''' static method, returns a QRegister with n quantum bits in
            "pure state", each of which is initialized with an Alea
            instances giving probability amplitudes {0: psy_p0, 1: psy_p1}
        '''
        qbit_alea = Alea.pmf(zip(qbit_values,(psy_p0, psy_p1)),prob_type='c')
        return QRegister(qbit_alea.new(n))

    @staticmethod
    def build_mixed(jpd_alea):
        ''' static method, returns a QRegister with mixed states, modelled
            as a joint probability distribution giving the probability
            amplitude of each binary n-tuple
            note: the qbit of index i is  defined as the the Flea2 instance
                  jpd_alea[i]
        '''
        n = len(jpd_alea.support[0])
        return QRegister((jpd_alea[i] for i in range(n)))

    def apply_gate(self,gate,*qb_idxs):
        ''' apply the given gate on the quantum bits identified by the given
            indexes; the gate argumemnt can be a QGate instance or a name
            (string) identifying such instance;
            the QBits at given indexes are replaced by new Flea2 instances
        '''
        nb_idxs = len(qb_idxs)
        if nb_idxs == 0:
            nb_idxs = len(self.qbits)
            qb_idxs = range(nb_idxs)
        gate_cpt = QGate.get_cpt(gate,nb_idxs)
        joint_qb_in = Lea.joint(*(self.qbits[qb_idx] for qb_idx in qb_idxs))
        joint_qb_out = joint_qb_in.switch(gate_cpt)
        for (i,qb_idx) in enumerate(qb_idxs):
            self.qbits[qb_idx] = joint_qb_out[i]

    def as_joint(self):
        ''' returns a new Clea instance, representing the joint of the n
            quantum bits of the register
        '''
        return Lea.joint(*self.qbits)
    
    def __str__(self):
        ''' returns, a string representation of the QRegister;
            it contains one line per distinct binary n-tuple, separated by
            a newline character;
            each line contains the string representation of a value with its
            amplitude probability expressed as a complex number 
        '''
        return str(self.as_joint())

    __repr__ = __str__

    def calq(self,*args,**kwargs):
        ''' returns an Alea instance giving the probability distribution
            of measurement of the n quantum bits;
            the support is the set of possible binary n-tuples that can
            be measured from the quantum register;
            each probability amplitude p (complex, float or int types)
            is converted into the float |p|^2 (Born rule);
            note: as any Alea instance, the returned object is a snapshot
            that has loses any link to the originating QRegister instance;
            this last can still evolve by applications of gates
        '''
        return self.as_joint().calq(*args,**kwargs)

    def measure(self,n=None):
        ''' if n is None, returns a random binary n-tuple simulating a
            measurement of the n quantum bits of the register;
            the probability distribution is obtained as described in
            calq method;
            if n is not None, returns a tuple of n such random values
            note: this convenience method is equivalent to
                  return self.calq().random(n)
        '''
        return self.calq().random(n)
