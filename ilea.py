'''
--------------------------------------------------------------------------------

    ilea.py

--------------------------------------------------------------------------------
Copyright 2013, 2014, 2015 Pierre Denis

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

class Ilea(Lea):
    
    '''
    Ilea is a Lea subclass, which instance represents a probability distribution obtained
    by filtering the values Vi of a given Lea instance that verify a boolean condition C,
    which is the AND of given boolean conditions Cj(Vi).
    In the context of a conditional probability table (CPT), each Ilea instance represents
    a given distribution <Vi,p(Vi|C)>, assuming that a given condition C is verified (see Blea class).
    '''

    __slots__ = ('_lea1','_condLeas')

    def __init__(self,lea1,condLeas):
        Lea.__init__(self)
        self._lea1 = lea1
        self._condLeas = tuple(condLeas)

    def _getLeaChildren(self):
        return (self._lea1,) + self._condLeas
    
    def _clone(self,cloneTable):
        return Ilea(self._lea1.clone(cloneTable),(condLea.clone(cloneTable) for condLea in self._condLeas))

    @staticmethod
    def _genTrueP(condLeas):
        ''' generates probabilities of True for ANDing the given conditions 
            this uses short-circuit evaluation
        '''
        if len(condLeas) == 0:
            # empty condition: evaluated as True (seed of recursion)
            yield 1
        else:
            condLea0 = condLeas[0]
            condLeaT = condLeas[1:]
            for (cv0,p0) in condLea0.genVPs():
                if cv0 is True:
                    # the first condition is true, for some binding of variables
                    for p1 in Ilea._genTrueP(condLeaT):
                        # the full condition is true, for some binding of variables
                        yield p0*p1
                elif cv0 is False:
                    # short-circuit: do not go further since the AND is false
                    pass
                else:
                    # neither True, nor False -> error
                    raise Lea.Error("boolean expression expected")
    
    def _genVPs(self):
        for cp in Ilea._genTrueP(self._condLeas):
            # the AND of conditions is true, for some binding of variables
            # yield value-probability pairs of _lea1, given this binding
            for (v,p) in self._lea1.genVPs():
                yield (v,cp*p)
