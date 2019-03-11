'''
--------------------------------------------------------------------------------

    prob_number.py

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


class ProbNumber(object):

    ''' 
    A ProbNumber is an abstract class for representing a probability;
    Before displaying, it checks that value is between 0 and 1
    '''

    class Error(Exception):
        pass

    def check(self):
        ''' raises an Exception if the decimal is not in the probability range, from 0 to 1
        '''
        if not (0 <= self <= 1):
            raise ProbNumber.Error("%s is not a valid probability (between 0 and 1)"%self._get_base_class().__str__(self))

    def __str__(self):
        ''' returns a string representation of self
            raises an Exception if the value is not in the probability range, from 0 to 1
        '''
        self.check()
        return self._get_base_class().__str__(self)
    
    # overwrites representation method
    __repr__ = __str__

    def simplify(self,to_float=False):
        ''' if to_float is False (default): returns  self
            if to_float is True: returns self converted to float
        '''
        if to_float:
            return float(self)
        return self

    def _get_base_class(self):
        ''' returns the second parent class of self,
            assuming that self's class inherits also from ProbNumber
            (multiple inheritance); the returned class is then a sibling
            class of ProbNumber
        '''
        return self.__class__.__bases__[1]

    def as_base_class_instance(self):
        ''' returns self converted to the parent base class of self
            assuming that self's class inherits also from ProbNumber
            (multiple inheritance); the class of returned instance
            is then a sibling class of ProbNumber
        '''
        return self._get_base_class()(self)

    def as_float(self):
        ''' returns float string representation of self
            note: to get float number, use float(self) 
        '''
        return str(float(self))

    def as_pct(self):
        ''' returns float percentage string representation of self
        '''
        return "%f %%" % float(self*100)

