'''
--------------------------------------------------------------------------------

    prob_fraction.py

--------------------------------------------------------------------------------
Copyright 2013-2021 Pierre Denis

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

from .ext_fraction import ExtFraction
from fractions import Fraction

class ProbFraction(ExtFraction):
    ''' 
    A ProbFraction instance represents a probability as a fraction.
    It inherits ExtFraction, checking the probability range, from 0 to 1
    '''    
    
    def __new__(cls, numerator=0, denominator=None):
         new_prob_fraction = Fraction.__new__(ProbFraction,Fraction(numerator,denominator))
         new_prob_fraction.check()
         return new_prob_fraction

    def _get_base_class(self):
        return Fraction

    @staticmethod
    def coerce(value):
        ''' static method, returns a ProbFraction instance corresponding the given value:
            if the value is a ProbFraction instance, then it is returned
            otherwise, a new ProbFraction instance is returned corresponding to given value
        '''
        if isinstance(value,ProbFraction):
            return value
        return ProbFraction(value)
