'''
--------------------------------------------------------------------------------

    prob_decimal.py

--------------------------------------------------------------------------------
Copyright 2013-2017 Pierre Denis

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


class ProbNumber(object):

    ''' 
    A ProbDecimal instance represents a probability as a decimal
    It inherits Decimal, overloading methods to improve useability
    '''

    class Error(Exception):
        pass

    def check(self):
        ''' raises an Exception if the decimal is not in the probability range, from 0 to 1
        '''
        if not (0 <= self <= 1):
            raise ProbNumber.Error("%s is not a valid probability (between 0 and 1)"%self.getBaseClass().__str__(self))

    def __str__(self):
        ''' returns a string representation "numerator/denominator" of self
            raises an Exception if the decimal is not in the probability range, from 0 to 1
        '''
        self.check()
        return self.getBaseClass().__str__(self)
    
    # overwrites representation method
    __repr__ = __str__

    def simplify(self,toFloat=False):
        if toFloat:
            return float(self)
        return self

    def getBaseClass(self):
        return self.__class__.__bases__[1]

    def asBaseClassInstance(self):
        return self.getBaseClass()(self)

    def asFloat(self):
        ''' returns float string representation of self
            note: to get float number, use float(self) 
        '''
        return str(float(self))

    def asPct(self):
        ''' returns float percentage string representation of self
        '''
        return "%f %%" % float(self*100)

