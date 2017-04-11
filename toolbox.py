'''
--------------------------------------------------------------------------------

    toolbox.py

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

'''
The module toolbox provides general functions and constants needed by Lea classes 
'''

from math import log
from functools import wraps
import sys
import csv

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
    values0 = tuple(frozenset(values))
    values1 = list(values0)
    while len(set(values1)) > 1:
        minVal = min(values1)
        idx = values1.index(minVal)
        values1[idx] += values0[idx]
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
   def wrapper(obj,*args):
       # retrieve the cache for method f
       cache = obj._cachesByFunc.get(f)
       if cache is None:
           # first call to obj.f(...) -> build a new cache for f
           cache = obj._cachesByFunc[f] = dict()
       elif args in cache:
           # obj.f(*args) already called in the past -> returns the cached result
           return cache[args]
       # first call to obj.f(*args) -> calls obj.f(*args) and store the result in the cache    
       res = cache[args] = f(obj,*args)
       return res
   return wrapper

def strToBool(bStr):
    ''' returns True  if bStr is '1', 't', 'true', 'y' or 'yes' (case insentive)
                False if bStr is '0', 'f', 'false', 'n' or 'no' (case insentive)
        raise ValueError exception in other cases
    '''
    bStr = bStr.lower()
    if bStr in ('t','true','1','y','yes'):
        return True
    if bStr in ('f','false','0','n','no'):
        return False
    raise ValueError("invalid boolean litteral '%s'"%bStr)

def readCSVFilename(csvFilename,colNames=None,dialect='excel',**fmtparams):
    ''' same as readCSVFile method, except that it takes a filename instead
        of an open file (i.e. the method opens itself the file for reading);
        see readCSVFile doc for more details
    '''
    with open(csvFilename,'rU') as csvFile:
        return readCSVFile(csvFile,colNames,dialect,**fmtparams)

def readCSVFile(csvFile,colNames=None,dialect='excel',**fmtparams):
    ''' returns a tuple (attrNames,dataFreq) from the data read in the given CSV file
        * attrNames is a tuplewith the attribute names found in the header row 
        * dataFreq is a list of tuples (tupleValue,count) for each CSV row 
          with tupleValue containing read fields and count the positive integer
          giving the probability weight of this row;
        the arguments follow the same semantics as those of Python's csv.reader
        method, which supports different CSV formats
        see doc in https://docs.python.org/2/library/csv.html
        * if colNames is None, then the fields found in the first read row of the CSV
          file provide information on the attributes: each field is made up of a name,
          which shall be a valid identifier, followed by an optional 3-characters type
          code among  
            {b} -> boolean
            {i} -> integer
            {f} -> float
            {s} -> string
            {#} -> count   
          if the type code is missing for a given field, the type string is assumed for
          this field; for example, using the comma delimiter (default), the first row
          in the CSV file could be:
              name,age{i},heigth{f},married{b}
        * if colNames is not None, then colNames shall be a sequence of strings giving
          attribute information as described above, e.g.
              ('name','age{i}','heigth{f}','married{b}')
          it assumed that there is NO header row in the CSV file
        the type code defines the conversion to be applied to the fields read on the
        data lines; if the read value is empty, then it is converted to Python's None,
        except if the type is string, then, the value is the empty string; 
        if the read value is not empty and cannot be parsed for the expected type, then
        an exception is raised; for boolean type, the following values (case
        insensitive):
          '1', 't', 'true', 'y', 'yes' are interpreted as Python's True,
          '0', 'f', 'false', 'n', 'no' are interpreted as Python's False;
        the {#} code identifies a field that provides a count number of the row,
        representing the probability of the row or its frequency as a positive integer;
        such field is NOT included as attribute of the joint distribution; it is useful
        to define non-uniform probability distribution, as alternative to repeating the
        same row multiple times
    '''
    # read the CSV file
    attrNames = []
    convFunctions = []
    countAttrIdx = None
    fieldsPerRowIter = csv.reader(csvFile,dialect,**fmtparams)
    if colNames is None:
        # parse the header row
        colNames = next(fieldsPerRowIter)
    # if colNames is not None, it is assumed that there is no header row in the CSV file
    for (colIdx,colName) in enumerate(colNames):
        colName = colName.strip()
        if colName.endswith('{#}'):
            if countAttrIdx is not None:
                raise ValueError("count column ('{#}') must be unique in CSV header line")
            countAttrIdx = colIdx
        else:
            hasSuffix = True
            convFunction = None    
            if colName.endswith('{b}'):
                convFunction = strToBool
            elif colName.endswith('{i}'):
                convFunction = int
            elif colName.endswith('{f}'):
                convFunction = float
            elif not colName.endswith('{s}'):
                hasSuffix = False
            if hasSuffix:
                attrName = colName[:-3].strip()
            else:
                attrName = colName
            attrNames.append(attrName)
            convFunctions.append(convFunction)
    # parse the data rows
    fieldsPerRow = tuple(fieldsPerRowIter)
    dataFreq = []
    for fields in fieldsPerRow:
        if countAttrIdx is None:
            # no 'count' field: each read row has a count of 1 
            count = 1
        else:
            # 'count' field: extract the count value from the fields
            count = int(fields.pop(countAttrIdx))
        # conversion of read fields according to optional given types
        convFields = []
        for (field,convFunction) in zip(fields,convFunctions):
            if convFunction is None:
                convField = field
            else:
                if field == '':
                    # empty value translated as Python's None 
                    convField = None
                else:
                    convField = convFunction(field)
            convFields.append(convField)
        dataFreq.append((tuple(convFields),count))
    return (attrNames,dataFreq)

