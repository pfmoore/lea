'''
--------------------------------------------------------------------------------

    prob_decimal.py

--------------------------------------------------------------------------------
Copyright 2013-2022 Pierre Denis

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

from .ext_decimal import ExtDecimal
from decimal import Decimal

class ProbDecimal(ExtDecimal):
    ''' 
    A ProbDecimal instance represents a probability as a decimal.
    It inherits ExtDecimal, checking the probability range, from 0 to 1.
    '''    
    
    def __new__(cls, val=0):
        is_percentage = False
        if isinstance(val, str):
            val = val.strip()
            if val.endswith('%'):
                val = Decimal(val[:-1])
                is_percentage = True
        decimal = Decimal(val)
        if is_percentage:
            decimal /= 100
        new_prob_decimal = Decimal.__new__(ProbDecimal,decimal)
        new_prob_decimal.check()
        return new_prob_decimal

    def _get_base_class(self):
        return Decimal

    @staticmethod
    def coerce(value):
        ''' static method, returns a ProbDecimal instance corresponding the given value:
            if the value is a ProbDecimal instance, then it is returned
            otherwise, a new ProbDecimal instance is returned corresponding to given value
        '''
        if isinstance(value,ProbDecimal):
            return value
        return ProbDecimal(value)

