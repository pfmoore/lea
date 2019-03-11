'''
--------------------------------------------------------------------------------

    prob_decimal.py

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

from .prob_number import ProbNumber
from decimal import Decimal

class ProbDecimal(ProbNumber,Decimal):

    ''' 
    A ProbDecimal instance represents a probability as a decimal
    It inherits ProbNumber and Decimal, overloading methods to
    improve useability
    '''
    
    class Error(Exception):
        pass

    def __new__(cls, val=0):
        ''' returns a new instance of ProbDecimal
            following signatures of Decimal constructor
            + allowing a percentage in val as a string 'xxx %'
              with xxx being a float literal
            Note that the constructor does NOT check that the decimal
            is in the range 0 to 1; this is so to allow intermediate
            results in expressions to go beyond that range;
            the range is verified when string representation is required
            (method str) or by explicit call to check() method 
        '''
        is_percentage = False
        if isinstance(val, str):
            val = val.strip()
            if val.endswith('%'):
                val = Decimal(val[:-1])
                is_percentage = True
        decimal = Decimal(val)
        if is_percentage:
            decimal /= 100
        return ProbDecimal._from_decimal(decimal)
    
    @staticmethod         
    def _from_decimal(decimal):
        ''' static method, returns a ProbDecimal numerically equivalent to
            the given Decimal instance;
            if decimal is not an instance of Decimal then it is returned
            as-is
        '''
        if not isinstance(decimal,Decimal):
            return decimal
        return Decimal.__new__(ProbDecimal,decimal)

    @staticmethod
    def coerce(value):
        ''' static method, returns a ProbDecimal instance corresponding the given value:
            if the value is a ProbDecimal instance, then it is returned
            otherwise, a new ProbDecimal instance is returned corresponding to given value
        '''
        if not isinstance(value,ProbDecimal):
            value = ProbDecimal(value)
        return value
        
    def __coerce_func(f):
        ''' internal utility function
            returns a function returning a ProbDecimal
            equivalent to the given function f returning Decimal
        '''
        return lambda *x: ProbDecimal._from_decimal(f(*x))
     
    # overloading arithmetic magic methods of Decimal
    # to convert Decimal result into ProbDecimal result
    # Note: do not overwrite __floordiv__, __rfloordiv__, __pow__
    # since these methods do not return Decimal instances
    __pos__      = __coerce_func(Decimal.__pos__)
    __neg__      = __coerce_func(Decimal.__neg__)
    __pow__      = __coerce_func(Decimal.__pow__)
    __add__      = __coerce_func(Decimal.__add__)
    __radd__     = __coerce_func(Decimal.__radd__)
    __sub__      = __coerce_func(Decimal.__sub__)
    __rsub__     = __coerce_func(Decimal.__rsub__)
    __mul__      = __coerce_func(Decimal.__mul__)
    __rmul__     = __coerce_func(Decimal.__rmul__)
    __truediv__  = __coerce_func(Decimal.__truediv__)
    __rtruediv__ = __coerce_func(Decimal.__rtruediv__)

    # Python 2 compatibility
    __div__ = __truediv__
    __rdiv__ = __rtruediv__

# constant unity instance to ease definition of other instances by multiplication
ProbDecimal.one = ProbDecimal(1)
