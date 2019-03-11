'''
--------------------------------------------------------------------------------

    prob_fraction.py

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
from fractions import Fraction

class ProbFraction(ProbNumber,Fraction):

    ''' 
    A ProbFraction instance represents a probability as a fraction
    It inherits ProbFraction and Fraction, overloading methods to
    improve useability
    '''
    
    class Error(Exception):
        pass

    def __new__(cls, numerator=0, denominator=None):
        ''' returns a new instance of ProbFraction
            following signatures of Fraction constructor
            Note that the constructor does NOT check that the fraction
            is in the range 0 to 1; this is so to allow intermediate
            results in expressions to go beyond that range;
            the range is verified when string representation is required
            (method str) or by explicit call to check() method 
        '''
        return ProbFraction._from_fraction(Fraction(numerator,denominator))
    
    @staticmethod         
    def _from_fraction(fraction):
        ''' static method, returns a ProbFraction numerically equivalent to
            the given Fraction instance;
            if fraction is not an instance of Fraction then it is returned
            as-is
        '''
        if not isinstance(fraction,Fraction):
            return fraction
        return Fraction.__new__(ProbFraction,fraction)

    @staticmethod
    def coerce(value):
        ''' static method, returns a ProbFraction instance corresponding the given value:
            if the value is a ProbFraction instance, then it is returned
            otherwise, a new ProbFraction instance is returned corresponding to given value
        '''
        if not isinstance(value,ProbFraction):
            value = ProbFraction(value)
        return value
        
    def __coerce_func(f):
        ''' internal utility function
            returns a function returning a ProbFraction
            equivalent to the given function f returning Fraction
        '''
        return lambda *x: ProbFraction._from_fraction(f(*x))
     
    # overloading arithmetic magic methods of Fraction
    # to convert Fraction result into ProbFraction result
    # Note: do not overwrite __floordiv__, __rfloordiv__, __pow__
    # since these methods do not return Fraction instances
    __pos__      = __coerce_func(Fraction.__pos__)
    __neg__      = __coerce_func(Fraction.__neg__)
    __pow__      = __coerce_func(Fraction.__pow__)
    __add__      = __coerce_func(Fraction.__add__)
    __radd__     = __coerce_func(Fraction.__radd__)
    __sub__      = __coerce_func(Fraction.__sub__)
    __rsub__     = __coerce_func(Fraction.__rsub__)
    __mul__      = __coerce_func(Fraction.__mul__)
    __rmul__     = __coerce_func(Fraction.__rmul__)
    __truediv__  = __coerce_func(Fraction.__truediv__)
    __rtruediv__ = __coerce_func(Fraction.__rtruediv__)

    # Python 2 compatibility
    __div__ = __truediv__
    __rdiv__ = __rtruediv__


    @staticmethod
    def calc_lcm(values):
        ''' returns the least common multiple among the given sequence of integers;
            assumes that all values are strictly positive
        '''
        values0 = tuple(frozenset(values))
        values1 = list(values0)
        while len(set(values1)) > 1:
            min_val = min(values1)
            idx = values1.index(min_val)
            values1[idx] += values0[idx]
        return values1[0]

    @staticmethod
    def convert_to_same_denom(fractions):
        ''' static method, returns a tuple of integers
            which are the numerators of given sequence of fractions,
            after conversion to a common denominator  
        '''
        denominators = tuple(fraction.denominator for fraction in fractions)
        if len(denominators) == 0:
            raise ProbFraction.Error('get_prob_weights requires at least one fraction')
        lcm = ProbFraction.calc_lcm(denominators)
        return (tuple(fraction.numerator*(lcm//fraction.denominator) for fraction in fractions), lcm)

