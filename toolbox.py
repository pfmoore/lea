'''
--------------------------------------------------------------------------------

    toolbox.py

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

'''
The module toolbox provides general functions and constants needed by Lea classes 
'''

import sys
import csv
from functools import wraps
from collections import defaultdict

try:
    # log2 function available in Python 3.3+
    from math import log2
except ImportError:
    # Python ver < 3.3 does not have log2 function
    from math import log
    def log2(x):
        ''' returns a float number that is the logarithm in base 2 of the given number x
        '''
        return log(x,2)

def make_tuple(v):
    ''' returns a tuple with v as unique element
    '''    
    return (v,)
    
def min2(a,b):
    ''' returns the minimum element between given a and b;
        note: this is slightly more efficient than using standard Python min function
    '''
    if a <= b:
        return a
    return b

def max2(a,b):
    ''' returns the maximum element between given a and b;
        note: this is slightly more effcient than using standard Python max function
    '''
    if a >= b:
        return a
    return b

def gen_pairs(seq):
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
        for pair in gen_pairs(tail):
            yield pair
            
## storing standard dict in __std_dict is needed for Python 2.x
## because dict is rebound (see below)
__std_dict = dict

def is_dict(d):
    ''' returns True iff given d is a dictionary
    '''
    return isinstance(d,__std_dict)

numeric_types = (float,int,complex)
                
# Python 2 / 3 dependencies
# the following redefines / rebinds the following objects in Python2: 
#  input
#  zip
#  next
#  dict
#  defaultdict
# these shall be imported by all modules that uses such names

# standard input function, zip and dictionary methods as iterators
## note don't use sys.version_info.major because NOK if <= 2.6
if sys.version_info[0] == 2:
    # Python 2.x
    if sys.version_info[1] < 6:
        raise Exception("Lea requires Python 2.6+ or 3 to run")
    # Python 2.6+
    # the goal of this part is to mimic a Python 3 env in a Python 2.6+ env
    # add long type to numeric types 
    numeric_types += (long,)
    # rename raw_input method
    input = raw_input
    # zip as iterator shall be imported
    from itertools import izip as zip
    # next method shall be accessible as function
    def next(it):
        return it.next()
    # the dictionary classes shall have keys, values, items methods
    # which are iterators; note that dictionaries must be created
    # with dict() instead of {}
    class dict(dict):
        keys = dict.iterkeys
        values = dict.itervalues
        items = dict.iteritems
    class defaultdict(defaultdict):
        keys = defaultdict.iterkeys
        values = defaultdict.itervalues
        items = defaultdict.iteritems
    # function replacing str.isidentifier, missing in Python 2.x
    import re, tokenize, keyword
    def is_identifier(s):
        return re.match(tokenize.Name + '$', s) and not keyword.iskeyword(s)
else:
    # Python 3.x
    # the following trick is needed to be able to import the names
    input = input
    zip = zip
    next = next
    dict = dict
    # function replacing str.isidentifier method (see above)    
    def is_identifier(s):
        return s.isidentifier()

def indent(str_func,obj,width):
    ''' returns a string representation of given object obj obtained
        by applying given function str_func and justifying on given
        width; the string is left-justified except if obj is a number
    '''
    ## note that bool is a subtype of int, although it shall not be right-justified as a number
    if isinstance(obj,numeric_types) and not isinstance(obj,bool):
        return str_func(obj).rjust(width)
    return str_func(obj).ljust(width)

def memoize(f):
   ''' returns a memoized version of the given instance method f;
       requires that the instance has a _caches_by_func attribute
       referring to a dictionary;
       can be used as a decorator
       note: not usable on functions and static methods
   '''
   @wraps(f)
   def wrapper(obj,*args):
       # retrieve the cache for method f
       cache = obj._caches_by_func.get(f)
       if cache is None:
           # first call to obj.f(...) -> build a new cache for f
           cache = obj._caches_by_func[f] = dict()
       elif args in cache:
           # obj.f(*args) already called in the past -> returns the cached result
           return cache[args]
       # first call to obj.f(*args) -> calls obj.f(*args) and store the result in the cache    
       res = cache[args] = f(obj,*args)
       return res
   return wrapper

def str_to_bool(b_str):
    ''' returns True  if b_str is '1', 't', 'true', 'y' or 'yes' (case insensitive)
                False if b_str is '0', 'f', 'false', 'n' or 'no' (case insensitive)
        raise ValueError exception in other cases
    '''
    b_str = b_str.lower()
    if b_str in ('t','true','1','y','yes'):
        return True
    if b_str in ('f','false','0','n','no'):
        return False
    raise ValueError("invalid boolean litteral '%s'"%b_str)

def read_csv_filename(csv_filename,col_names=None,dialect='excel',**fmtparams):
    ''' same as read_csv_file method, except that it takes a filename instead
        of an open file (i.e. the method opens itself the file for reading);
        see read_csv_file doc for more details
    '''
    with open(csv_filename,'rU') as csv_file:
        return read_csv_file(csv_file,col_names,dialect,**fmtparams)

def read_csv_file(csv_file,col_names=None,dialect='excel',**fmtparams):
    ''' returns a tuple (attr_names,data_freq) from the data read in the given CSV file
        * attr_names is a tuple with the attribute names found in the header row 
        * data_freq is a list of tuples (tuple_value,count) for each CSV row 
          with tuple_value containing read fields and count the positive integer
          giving the probability weight of this row;
        the arguments follow the same semantics as those of Python's csv.reader
        method, which supports different CSV formats
        see doc in http://docs.python.org/3.7/library/csv.html
        * if col_names is None, then the fields found in the first read row of the CSV
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
        * if col_names is not None, then col_names shall be a sequence of strings giving
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
    attr_names = []
    conv_functions = []
    count_attr_idx = None
    fields_per_row_iter = csv.reader(csv_file,dialect,**fmtparams)
    if col_names is None:
        # parse the header row
        col_names = next(fields_per_row_iter)
    # if col_names is not None, it is assumed that there is no header row in the CSV file
    for (col_idx,col_name) in enumerate(col_names):
        col_name = col_name.strip()
        if col_name.endswith('{#}'):
            if count_attr_idx is not None:
                raise ValueError("count column ('{#}') must be unique in CSV header line")
            count_attr_idx = col_idx
        else:
            has_suffix = True
            conv_function = None    
            if col_name.endswith('{b}'):
                conv_function = str_to_bool
            elif col_name.endswith('{i}'):
                conv_function = int
            elif col_name.endswith('{f}'):
                conv_function = float
            elif not col_name.endswith('{s}'):
                has_suffix = False
            if has_suffix:
                attr_name = col_name[:-3].strip()
            else:
                attr_name = col_name
            attr_names.append(attr_name)
            conv_functions.append(conv_function)
    # parse the data rows
    fields_per_row = tuple(fields_per_row_iter)
    data_freq = []
    for fields in fields_per_row:
        if count_attr_idx is None:
            # no 'count' field: each read row has a count of 1 
            count = 1
        else:
            # 'count' field: extract the count value from the fields
            count = int(fields.pop(count_attr_idx))
        # conversion of read fields according to optional given types
        conv_fields = []
        for (field,conv_function) in zip(fields,conv_functions):
            if conv_function is None:
                conv_field = field
            else:
                if field == '':
                    # empty value translated as Python's None 
                    conv_field = None
                else:
                    conv_field = conv_function(field)
            conv_fields.append(conv_field)
        data_freq.append((tuple(conv_fields),count))
    return (attr_names,data_freq)

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    ''' returns True iff float a and b are almost equal
    '''
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
