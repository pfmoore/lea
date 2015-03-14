'''
--------------------------------------------------------------------------------

    toolbox.py

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

'''
The module toolbox provides general functions and constants needed by Lea classes 
'''

from math import log
from functools import wraps
import sys

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

def makeTuple(v):
    ''' returns a tuple with v as unique element
    '''    
    return (v,)
    
def easyMin(*args):
    ''' returns the minimum element of given args
        Note: if only one arg, it is returned (unlike Python's min function)
    '''
    if len(args) == 1:
        return args[0]
    return min(args)

def easyMax(*args):
    ''' returns the maximum element of given args
        Note: if only one arg, it is returned (unlike Python's max function)
    '''
    if len(args) == 1:
        return args[0]
    return max(args)

def genPairs(seq):
    ''' generates as tuples all the pairs from the elements of given sequence seq
    '''
    tuple1 = tuple(seq)
    length = len(tuple1)
    if length < 2:
        return
    if length == 2:
        yield tuple1
    else:
        head = tuple1[0]
        tail = tuple1[1:]
        for a in tail:
            yield (head,a)
        for pair in genPairs(tail):
            yield pair
                
# Python 2 / 3 dependencies
# the following redefines / rebinds the following objects in Python2: 
#  input
#  zip
#  next
#  dict
#  defaultdict
# these shall be imported by all modules that uses such names

# standard input function, zip and dictionary methods as iterators
from collections import defaultdict
if sys.version_info.major == 2:
    # Python 2.x
    # the goal of this part is to mimic a Python3 env in a Python2 env
    # rename raw_input method
    input = raw_input
    # zip as iterator shall be imported
    from itertools import izip as zip
    # next method shall be accessible as function
    def next(it):
        return it.next()
    # the dictionary classes shall have keys, values, items methods
    # wich are iterators; note that dictionaries must be created
    # with dict() instead of {}
    class dict(dict):
        keys = dict.iterkeys
        values = dict.itervalues
        items = dict.iteritems
    class defaultdict(defaultdict):
        keys = defaultdict.iterkeys
        values = defaultdict.itervalues
        items = defaultdict.iteritems
else:
    # Python 3.x
    # the following trick is needed to be able to import the names
    input = input
    zip = zip
    next = next
    dict = dict

def memoize(f):
   ''' returns a memoized version of the given instance method f;
       requires that the instance has a _cachesByFunc attribute
       referring to a dictionary;
       can be used as a decorator
       note: not usable on functions and static methods
   '''
   @wraps(f)
   def wrapper(self,*args):
       cache = self._cachesByFunc.get(f)
       if cache is None:
           # first call to self.f(...) -> build a new cache for f
           cache = self._cachesByFunc[f] = dict()
       if args in cache:
           # first call to self.f(*args) -> returns the cached result
           return cache[args]
       # first call to self.f(*args) -> put self.f(*args) in the cache    
       res = cache[args] = f(self,*args)
       return res
   return wrapper
