'''
--------------------------------------------------------------------------------

    toolbox.py

--------------------------------------------------------------------------------
Copyright 2013, 2014 Pierre Denis

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

'''
The module toolbox provides general functions and constants needed by Lea classes 
'''

from math import log

def calcGCD(a,b):
    ''' returns the greatest common divisor between the given integer arguments
    '''
    while a > 0:
        (a,b) = (b%a,a)
    return b
    
def calcLCM(values):
    ''' returns the greatest least common multiple among the given sequence of integers
        requires that all values are strictly positive (not tested) 
    ''' 
    values1 = list(values)
    while len(set(values1)) > 1:
        minVal = min(values1)
        idx = values1.index(minVal)
        values1[idx] += values[idx]
    return values1[0]
    
LOG2 = log(2.)

def log2(x):
    ''' returns a float number that is the logarithm in base 2 of the given float x
    '''
    return log(x)/LOG2

# standard input function
try:
    # Python 2.x
    inputFunc = raw_input
except NameError:
    # Python 3.x
    inputFunc = input
