'''
--------------------------------------------------------------------------------

    alea.py

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

from .lea import Lea
from .prob_number import ProbNumber
from .prob_fraction import ProbFraction
from .prob_decimal import ProbDecimal
from .toolbox import log2, memoize, zip, next, dict, defaultdict, make_tuple, read_csv_file, \
                     read_csv_filename, is_dict, indent, is_identifier
from fractions import Fraction
from decimal import Decimal
from random import random
from bisect import bisect_left, bisect_right
import itertools
from math import exp, factorial
from operator import truediv
import collections

# try to import optional modules:
# - Matplotlib
try:
    import matplotlib.pyplot as plt
    # matplotlib module installed
except:
    # matplotlib module not installed
    plt = None
# - SymPy
try:
    import sympy
    # sympy module installed
except:
    # sympy module not installed
    sympy = None
# - NumPy
try:
    # see usage of np_bool_type in Alea._p method
    from numpy import bool_ as np_bool_type
    # NumPy module installed  
except:
    # NumPy module not installed
    np_bool_type = ()

    
class Alea(Lea):
    
    '''
    Alea is a Lea subclass, which instance is defined by explicit probability distribution data.
    An Alea instance is defined by given value-probability pairs, that is an explicit probability
    mass function (pmf). The probabilities can be expressed as any object with arithmetic semantic.
    The main candidates are float, fraction or symbolic expressions.
    An Alea instance is an "elementary pex", as defined in the paper on Statues algorithm
    (see http://arxiv.org/abs/1806.09997).
    '''

    __slots__ = ('_vs','_ps','_cumul','_inv_cumul','_random_iter','_caches_by_func')
   
    # class or function used by default to convert each probability given in
    # an Alea constructor method; if None and if no prob_type arg is
    # specified in the constructore, then each probability is stored as-is
    ## this is embedded in a list to circumvent a problem occuring only in Python 2.7
    ## i.e. functions assigned directly to this attribute becomes instance methods 
    _prob_type = [None]

    # function used to simplify symbolic probability expressions to be
    # displayed (see _simplify and Alea.__init__ methods)
    _symbolic_simplify_function = sympy and staticmethod(sympy.factor)

    # dictionary used in _downcast method
    __downcast_prob_class = dict( {Fraction : ProbFraction,
                                   Decimal  : ProbDecimal })

    @staticmethod
    def prob_symbol(arg):
        ''' static method, if given arg is a string, then
               returns a sympy Symbol, having arg as name, possibly embedded
               in parentheses if arg is not a valid identifier;
            otherwise,
               returns arg as-is (which could incidentally be a sympy Symbol)
        '''
        if isinstance(arg,str):
            if not is_identifier(arg):
                arg = "(%s)" % arg
            return sympy.Symbol(arg)
        return arg

    @staticmethod
    def prob_any(arg):
        ''' static method, returns a probability object corresponding to the
            given arg:
            if arg is not a string, then it is returned as-is;
            if arg is a string, then it is tried to be interpreted as
            Decimal, Fraction or sympy symbol, in that order; the object
            of the first successful type is returned;
        '''
        if not isinstance(arg,str):
            return arg
        # arg is a string, convert it into an object representing a number
        try:
            return Decimal(arg)
        except:            
            pass
        try:
            return Fraction(arg)
        except:            
            pass
        if sympy is None:
            raise Lea.Error("probability expression '%s' requires the sympy module, which does not seem to be installed"%arg)
        return Alea.prob_symbol(arg)
   
    @staticmethod
    def _simplify(v,to_float=False):
        ''' static method, tries to simplify the given value v
            if given v is a sympy Expr, then
                returns the simplified probability using
                    Alea._symbolic_simplify_function
            otherwise, if to_float is True, then
                returns v converted to float
            otherwise,
                returns v unchanged
        '''
        if sympy is not None and isinstance(v,sympy.Expr):
            return Alea._symbolic_simplify_function(v)
        if to_float:
            return float(v)
        return v

    @staticmethod
    def get_prob_type(prob_type):
        ''' static method, returns the class or function associated to
            given code, this class or function is applied to convert
            each probability given in an Alea constructor method;
            if prob_type is -1, then None is returned;
            if prob_type is a callable object, then it is returned as-is;
            if prob_type is None, then current prob_type configured by
               Alea.set_prob_type is returned;
            otherwise, the given prob_type is a code interpreted as follows:
            - 'f' -> float (instance of Python's float)
            - 'd' -> decimal (instance of Python's decimal.Decimal)
            - 'r' -> rational (instance of Python's fractions.Fraction)
            - 's' -> symbolic (instance of a sympy Symbol)
                     - see Alea.prob_symbol method
            - 'x' -> any: if probability given in a string, then determines
                     the type from it (decimal, rational or symbol) and
                     converts into that type;
                     otherwise, takes the object as-is
                     - see Alea.prob_any method
            requires that prob_type is -1 or None or a callable or a code
            among the ones given above
        '''
        if prob_type is None:
            return Alea._prob_type[0]
        if prob_type == -1:
            return None
        if not isinstance(prob_type,str):
            if not callable(prob_type):
                raise Lea.Error("given prob_type '%s' is not a probability type code and it is not callable"%prob_type)
            return prob_type
        if prob_type == 'f':
            return float
        if prob_type == 'r':
            return Fraction
        if prob_type == 'd':
            return Decimal
        if prob_type == 's':
            if sympy is None:
                raise Lea.Error("prob_type 's' requires the installation of SymPy module")
            return Alea.prob_symbol
        if prob_type == 'x':
            return Alea.prob_any
        raise Lea.Error("unknown probability type code '%s', should be 'f', 'd', 'r', 's' or 'x'"%prob_type)

    @staticmethod
    def set_prob_type(prob_type):
        ''' static method allowing to change the representation of probability
            values for newly created Lea instances, according to the given
            prob_type;
            if prob_type is a callable object, then it is set as such;
            otherwise, the given prob_type is a code interpreted as follows:
            - 'f' -> float (instance of Python's float) - default
            - 'd' -> decimal (instance of Python's decimal.Decimal)
            - 'r' -> rational (instance of Python's fractions.Fraction)
            - 's' -> symbolic (instance of a sympy Symbol)
                     - see Alea.prob_symbol method
            - 'x' -> any: if probability given in a string, then determines
                     the type from it (decimal, rational or symbol) and
                     convert into it;
                     otherwise, takes the object as-is
                     - see Alea.prob_any method
            requires that a prob_type is a callable or a code among the ones
            given above
        '''
        if prob_type is None or prob_type == -1:
            raise Lea.Error("Alea.set_prob_type does not allow %s as argument"%prob_type)
        Alea._prob_type[0] = Alea.get_prob_type(prob_type)

    def __init__(self,vs,ps,normalization=True,prob_type=-1):
        ''' initializes Alea instance's attributes
            vs is a sequence of values
            ps is a sequence of probabilities (same length and order as ps)
            if normalization argument is True (default), then probabilities ps are
            updated before being stored to ensure that their sum equals 1:
            * if one probability is None, then it is replaced by
               1 - sum(not None elements of ps)
            * otherwise, each probability p is replaced by p / sum(ps)
              (in such case, it's not mandatory to have true probabilities for ps
               elements; these could be simple counters for example)
            if prob_type is different from -1, then the given probabilities ps
            are converted using prob_type, as documented in Alea.set_prob_type;
            requires that there is maximum one None probability if normalization;
            requires that vs and ps have same length
        '''
        Lea.__init__(self)
        # for an Alea instance, the alea cache is itself
        self._alea = self
        self._vs = tuple(vs)
        prob_type_func = Alea.get_prob_type(prob_type)
        if prob_type_func is not None:
            ps = (None if p is None else prob_type_func(p) for p in ps)
        if normalization:
            ps = tuple(ps)
            nb_none = ps.count(None)
            if nb_none > 1:
                raise Lea.Error("for normalization, no more than one single probability can be None")
            if nb_none == 1:
                p_sum = sum(p for p in ps if p is not None)
                Alea._check_prob(p_sum)
                idx_none = ps.index(None)
                ps = ps[:idx_none] + (1-p_sum,) + ps[idx_none+1:]
            else:
                p_sum = sum(ps)
                ps = (truediv(p,p_sum) for p in ps)
            if sympy is not None and Alea._symbolic_simplify_function is not None and isinstance(p_sum,sympy.Expr):
                ps = (Alea._symbolic_simplify_function(p) for p in ps)
        self._ps = tuple(ps)
        if len(self._vs) != len(self._ps):
            raise Lea.Error("number of values (%d) different from number of probabilities (%d)"%(len(self._vs),len(self._ps)))
        self._cumul = [0]
        self._inv_cumul = []
        self._random_iter = self._create_random_iter()
        self._caches_by_func = dict()

    
    # constructor methods
    # -------------------

    # call tree of Alea constructor methods ([S] means static method):
    #
    #   __init__
    #    <-- coerce [S]
    #    <-- pmf [S]
    #         <-- vals [S]
    #              <-- read_pandas_df [S]
    #         <-- interval [S]
    #         <-- read_csv_file [S]
    #         <-- _selections
    #              <-- draw_sorted_with_replacement
    #              <-- draw_sorted_without_replacement
    #         <-- poisson
    #    <-- _pmf_ordered [S]
    #         <-- pmf [S]
    #              <-- ... (see above)
    #         <-- draw_without_replacement
    #              <-- draw_sorted_without_replacement
    #    <-- _binary_distribution [S]
    #         <-- event [S]
    #         <-- bernoulli [S]
    #              <-- binom [S]

    @staticmethod
    def coerce(value,prob_type=-1):
        ''' static method, returns a Lea instance corresponding to the
            given value:
            if the value is a Lea instance, then it is returned as-is
            otherwise, a new Alea instance is returned, with given value
            as unique value, with a probability of 1.
            if prob_type is -1,
               then the returned Alea instance has integer 1 as probability;
               otherwise, the returned Alea instance has probability 1
               converted according to prob_type (see doc of Alea.set_prob_type)
        '''
        if not isinstance(value,Lea):
            # build a singleton value, with probability 1
            ## note: do not put something else than 1, as an integer,
            ## which is the highest arithmetic class in class hierarchy
            return Alea((value,),(1,),normalization=False,prob_type=prob_type)
        return value

    def new(self,n=None,prob_type=-1,sorting=False):
        ''' returns a new Alea instance, which represents the same probability
            distribution as self but for another event, independent from the
            event represented by self;
            * if n is not None,
               then a tuple containing n new independent Alea instances is returned
            * if prob_type is -1,
               then the returned Alea instance is a shallow copy of self
                    (values and probabilities data are shared);
               otherwise, the returned Alea instance has shared values data
                    but has new probabilities converted according to prob_type
                    (see doc of Alea.set_prob_type);
            * sorting allows sorting the value of the returned Alea instance
              (see Alea.pmf method);
            note that the present method overloads Lea.new to be more efficient;
        '''
        if sorting:
            # for sorting, relay to Lea.new, which is less efficient but handles the case correctly 
            return Lea.new(self,prob_type=prob_type,sorting=True)
        normalization = prob_type != -1
        new_alea = Alea(self._vs,self._ps,normalization=normalization,prob_type=prob_type)
        if prob_type == -1:
            ## note that the new Alea instance shares the immutable _vs and _ps attributes of self
            ## it can share also the mutable _cumul and _inv_cumul attributes of self (lists)
            new_alea._cumul = self._cumul
            new_alea._inv_cumul = self._inv_cumul
        if n is not None:
            return tuple(new_alea.new() for _ in range(n))
        return new_alea

    # keyword arguments available for Alea constructor
    __contructor_arg_names = frozenset(('ordered', 'sorting', 'normalization', 'check', 'prob_type'))

    @staticmethod
    def _parsed_kwargs(kwargs):
        ''' static method, returns (ordered,sorting,normalization,check,prob_type) tuple,
            with values found in the given kwargs dictionary (keywords);
            for missing keywords, the default values are
            (False, True, True, True, -1), respectively, except
            if ordered=True and sorting is missing, then sorting=False;
            requires that the given kwargs dictionary contains no other
            keywords than those defined above;
            requires that ordered and sorting are not set to True together
        '''
        arg_names = frozenset(kwargs.keys())
        unknown_arg_names = arg_names - Alea.__contructor_arg_names
        if len(unknown_arg_names) > 0:
            raise Lea.Error("unknown argument keyword '%s'; shall be only among %s"%(next(iter(unknown_arg_names)),tuple(Alea.__contructor_arg_names)))
        ordered = kwargs.get('ordered',False)
        normalization = kwargs.get('normalization',True)
        check = kwargs.get('check',True)
        prob_type = kwargs.get('prob_type',-1)
        if ordered and 'sorting' not in kwargs:
            sorting = False
        else:
            sorting = kwargs.get('sorting',True)
            if ordered and sorting:
                raise Lea.Error("ordered and sorting arguments cannot be set to True together")
        return (ordered,sorting,normalization,check,prob_type)

    @staticmethod
    def _check_not_empty(arg):
        ''' static method, verifies that the given arg is not empty;
            otherwise, raises an exception
        '''
        if len(arg) == 0:
            raise Lea.Error("cannot build a probability distribution with no value - maybe due to impossible evidence")        

    @staticmethod
    def _zip_vps(vps):
        ''' static method, returns a tuple (vs,ps) 
            where vs is a tuple with all first items of items of vps,
                  ps is a tuple with all second items of items of vps;
            requires that vps is not empty; 
            requires that vps is an iterable of pairs (v,p) 
        '''
        tuple_vps = tuple(vps)
        Alea._check_not_empty(tuple_vps)
        try:
            (vs,ps) = zip(*tuple_vps)
            return (vs,ps)
        except ValueError:
            pass
        raise Lea.Error("argument shall be a dictionary or shall contain pairs (v,P(v))")
    
    @staticmethod
    def pmf(arg,prob_type=None,**kwargs):
        ''' static method, returns an Alea instance representing a probability
            distribution for a probability mass function specified by the given
            arg, which is
              either a dictionary { v1:p1, ... , vn:pn }
              or an iterable of pairs (v1,p1), ... , (vn,pn)
            pi is the probability of occurrence of vi or a number proportional
            to it (see normalization argument below);
            in the iterable case, if the same value v occurs multiple times,
            then the associated p are summed together;
            * prob_type argument allows converting the given probabilities:
              -1: no conversion;
              None (default): default conversion, as set by Alea.set_prob_type;
              other: see doc of Alea.get_prob_type;
            the method admits three other optional boolean argument (in kwargs):
            * sorting (default:True): if True, then the values for displaying
            the distribution or getting the values will be sorted if possible
            (i.e. no exception on sort); otherwise, the order of values is
            unspecified unless ordered=True;
            * ordered (default:False): if ordered is True, then the values for
            displaying the distribution or getting the values will follow the
            given order (requires that the arg is an iterable or a 
            collections.OrderedDict);
            * normalization (default:True): if True, then each element
            of the given ps is divided by the sum of all ps before being stored
            (in such case, it's not mandatory to have true probabilities for ps
            elements; these could be simple counters, for example);
            requires that all the given values vi are hashable;
            requires that prob_dict is not empty;
            requires that ordered and sorting are not set to True together
            requires Python 2.7+ for ordered=True and arg OrderedDict
        '''
        (ordered,sorting,normalization,_,_) = Alea._parsed_kwargs(kwargs)
        if is_dict(arg):
            prob_dict = arg
            if ordered and not isinstance(prob_dict,collections.OrderedDict):
                raise Lea.Error("ordered=True requires to provide a collections.OrderedDict")
            Alea._check_not_empty(prob_dict)
        else:
            if ordered:
                return Alea._pmf_ordered(arg,prob_type=prob_type,**kwargs)
            vps = zip(*Alea._zip_vps(arg))
            prob_dict = defaultdict(int)
            prob_type_func = Alea.get_prob_type(prob_type)
            if prob_type_func is None:
                # no probability conversion required
                for (v,p) in vps:
                    prob_dict[v] += p
            else:
                # probability conversion required
                for (v,p) in vps:
                    prob_dict[v] += prob_type_func(p)
            ## note: since probability conversions have been done (if required),
            ## putting prob_type=-1 avoids unneccessary conversion in the subsequent calls
            prob_type = -1
        vps = prob_dict.items()
        if sorting:
            vps = list(vps)
            try:
                vps.sort()
            except:
                # no ordering relationship on values (e.g. complex numbers)
                pass
        prob_type_func = Alea.get_prob_type(prob_type)
        if prob_type_func is not None:
            vps = ((v,prob_type_func(p)) for (v,p) in vps)
        return Alea(*zip(*vps),normalization=normalization,prob_type=-1)

    @staticmethod
    def vals(*values,**kwargs):
        ''' static method, returns an Alea instance representing a distribution
            for the given values, so that each value occurrence is
            taken as equiprobable; if each value occurs exactly once, then the
            probability distribution is uniform, i.e. the probability of each
            value is equal to 1 / #values; otherwise, the probability of each
            value is equal to its frequency in the sequence;
            the optional arguments (kwargs) are:
              prob_type, ordered, sorting, normalization, check;
            see doc of Alea.pmf static method;
            requires at least one vals argument
        '''
        return Alea.pmf(((val,1) for val in values),**kwargs)
  
    @staticmethod
    def _pmf_ordered(vps,**kwargs):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of (val,freq) tuples, where freq is a natural
            number so that each value is taken with the given frequency;
            the values will be stored and displayed in the given order (no sort);
            the method admits 2 optional boolean argument (kwargs):
            * check (default: True): if True and if a value occurs multiple
            times, then an exception is raised;        
            requires that each value has a unique occurrence;
            requires at least one value_freqs argument;
        '''
        (_,_,normalization,check,prob_type) = Alea._parsed_kwargs(kwargs)
        (vs,ps) = Alea._zip_vps(vps)
        # check duplicates
        if check and len(frozenset(vs)) < len(vs):
            raise Lea.Error("duplicate values are not allowed for ordered=True")
        return Alea(vs,ps,normalization=normalization,prob_type=prob_type)
   
    @staticmethod
    def _check_prob(p):
        ''' static method checking that given p is a valid probability value,
            i.e. in the range [0,1];
            if comparisons are infeasible on p, then p is assumed to be a
            symbolic probability and is considered valid;
            raises Lea.Error exception if invalid
        '''
        try:
            is_valid = 0 <= p <= 1
        except:
            # comparisons are infeasible: we assume that p is a symbolic
            # probability, the range check cannot be enforced
            is_valid = True
        if not is_valid:
            raise Lea.Error("invalid probability value %s"%p)

    @staticmethod
    def _binary_distribution(v1,v2,p2,prob_type=None):
        ''' static method, returns an Alea instance representing a boolean
            probability distribution giving v1 with probability 1-p2
            and v2 with probability p2;
            prob_type argument allows converting the probabilities p2 and 1-p2:
              -1: no conversion;
              None (default): default conversion, as set by Alea.set_prob_type;
              other: see doc of Alea.get_prob_type;
        '''
        prob_type_func = Alea.get_prob_type(prob_type)
        if prob_type_func is not None:
            p2 = prob_type_func(p2)
        Alea._check_prob(p2)
        if p2 == 1:
            ## note: do not replace p2 by 1, in order to keep the given type
            (vs,ps) = ((v2,),(p2,))
        elif p2 == 0:
            ## note: do not replace 1-p2 by 1, in order to keep the given type
            (vs,ps) = ((v1,),(1-p2,))
        else:
            (vs,ps) = ((v1,v2),(1-p2,p2))
        return Alea(vs,ps,normalization=False,prob_type=-1)

    @staticmethod
    def event(p,prob_type=None):
        ''' static method, returns an Alea instance representing a boolean
            probability distribution giving True with probability p and
            False with probability 1-p;
            prob_type argument allows converting the given probability p:
              -1: no conversion;
              None (default): default conversion, as set by Alea.set_prob_type;
              other: see doc of Alea.get_prob_type;
        '''
        return Alea._binary_distribution(False,True,p,prob_type)

    @staticmethod
    def bernoulli(p,prob_type=None):
        ''' static method, returns an Alea instance representing a Bernoulli
            distribution giving 1 with probability p and 0 with probability
            1-p;
            prob_type argument allows converting the given probability p:
              -1: no conversion;
              None (default): default conversion, as set by Alea.set_prob_type;
              other: see doc of Alea.get_prob_type;
        '''
        return Alea._binary_distribution(0,1,p,prob_type)

    @staticmethod
    def binom(n,p,prob_type=None):
        ''' static method, returns an Alea instance representing a binomial
            distribution giving the number of successes among a number n of
            independent experiments, each having probability p of success;
            note: the binom method generalizes the bernoulli method:
              binom(1,p) is the same as bernoulli(p)
            prob_type argument allows converting the given probability p:
              -1: no conversion;
              None (default): default conversion, as set by Alea.set_prob_type;
              other: see doc of Alea.get_prob_type;
        '''
        return Alea.bernoulli(p,prob_type).times(n)

    @staticmethod
    def interval(from_val,to_val,prob_type=None):
        ''' static method, returns an Alea instance representing a uniform probability
            distribution, for all the integers in the interval [from_val,to_val]
            the given prob_type, if not None, allows using a probability
            type different from the default one (float or any one set by
            Alea.set_prob_type) - see doc of Alea.set_prob_type
        '''
        return Alea.vals(*range(from_val,to_val+1),prob_type=prob_type)

    @staticmethod
    def read_csv_file(csv_file,col_names=None,dialect='excel',**fmtparams):
        ''' static method, returns an Alea instance representing the joint probability
            distribution of the data read in the given CSV file;
            if csv_file is a string, then it is interpreted as a filename,
            otherwise, csv_file is interpreted as a file object ready to be read;
            the arguments follow the same semantics as those of Python's csv.reader
            method, which supports different CSV formats;
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
        # Filter out kewords argument targeting Alea constructors (assuming no name collision csv.reader keywords) 
        kwargs = dict((k,v) for (k,v) in fmtparams.items() if k in Alea.__contructor_arg_names)
        fmtparams = dict((k,v) for (k,v) in fmtparams.items() if k not in kwargs)
        if isinstance(csv_file,str):
            (attr_names,vps) = read_csv_filename(csv_file,col_names,dialect,**fmtparams)
        else:            
            (attr_names,vps) = read_csv_file(csv_file,col_names,dialect,**fmtparams)
        return Alea.pmf(vps,**kwargs).as_joint(*attr_names)

    @staticmethod
    def read_pandas_df(dataframe,index_col_name=None,**kwargs):
        ''' static method, returns an Alea instance representing the joint probability
            distribution from the given pandas dataframe;
            the attribute names of the distribution are those of the column of the
            given dataframe; the first field in each item of the dataframe is assumed
            to be the index; its treatment depends on given index_col_name:
            if index_col_name is None, then this index field is ignored;
            otherwise, it is put in the joint distribution with index_col_name as
            attribute name
        '''
        # TODO: retrieve index_col in df, if not 0
        attr_names = tuple(dataframe.columns)
        if index_col_name is None:
            values_iter = (t[1:] for t in dataframe.itertuples())
            attr_names = dataframe.columns
        else:
            values_iter = dataframe.itertuples()
            attr_names = (index_col_name,) + attr_names
        return Alea.vals(*values_iter,**kwargs).as_joint(*attr_names)

    def is_uniform(self):
        ''' returns True  if the probability distribution is uniform,
                    False otherwise
        '''
        p0 = self._ps[0]
        return all(p==p0 for p in self._ps)

    def _selections(self,n,gen_selector):
        ''' returns a new Alea instance representing a probability distribution of
            the n-length tuples yielded by the given combinatorial generator
            gen_selector, applied on the values of self distribution;
            the order of the elements of each built tuple is irrelevant: each tuple
            represents any permutation of its elements; the actual order of the
            elements of each tuple shall be the one defined by gen_selector;
            assumes that n >= 0;
            the efficient combinatorial algorithm is due to Paul Moore
        '''
        # First of all, get the values and weights for the distribution
        vps = dict(self._gen_raw_vps())
        # The total number of permutations of N samples is N!
        permutations = factorial(n)
        # We will calculate the frequency table for the result
        freq_table = []
        # Use gen_selector to get the list of outcomes.
        # as itertools guarantees to give sorted output for sorted input,
        # giving the sorted sequence self._vs ensures our outputs are sorted
        for outcome in gen_selector(self._vs,n):
            # We calculate the weight in 2 stages.
            # First we calculate the weight as if all values were equally
            # likely - in that case, the weight is N!/a!b!c!... where
            # a, b, c... are the sizes of each group of equal values
            weight = permutations
            # We run through the set counting and dividing as we go
            run_len = 0
            prev_roll = None
            for roll in outcome:
                if roll != prev_roll:
                    prev_roll = roll
                    run_len = 0
                run_len += 1
                if run_len > 1:
                    weight //= run_len
            # Now we take into account the relative weights of the values, by
            # multiplying the weight by the product of the weights of the
            # individual elements selected
            for roll in outcome:
                weight *= vps[roll]
            freq_table.append((outcome,weight))
        return Alea.pmf(freq_table)

    def draw_sorted_with_replacement(self,n):
        ''' returns a new Alea instance representing the probability distribution
            of drawing n elements from self WITH replacement, whatever the order
            of drawing these elements; the returned values are tuples with n
            elements sorted by increasing order;
            assumes that n >= 0;
            the efficient combinatorial algorithm is due to Paul Moore;
            requires Python 2.7+            
        '''
        return self._selections(n,itertools.combinations_with_replacement)

    def draw_sorted_without_replacement(self,n):
        ''' returns a new Alea instance representing the probability distribution
            of drawing n elements from self WITHOUT replacement, whatever the order
            of drawing these elements; the returned values are tuples with n
            elements sorted by increasing order;
            assumes that 0 <= n <= number of values of self;
            note: if the probability distribution of self is uniform,
            then the results is produced in an efficient way, thanks to the
            combinatorial algorithm of Paul Moore
        '''
        if self.is_uniform():
            # the probability distribution is uniform,
            # the efficient algorithm of Paul Moore can be used
            return self._selections(n,itertools.combinations)
        else:
            # the probability distribution is not uniform,
            # we use the general algorithm less efficient:
            # make first a draw unsorted then sort (the sort makes the
            # required probability additions between permutations)
            return self.draw_without_replacement(n).map(lambda vs: tuple(sorted(vs))).get_alea()

    def draw_with_replacement(self,n):
        ''' returns a new Alea instance representing the probability distribution
            of drawing n elements from self WITH replacement, taking the order
            of drawing into account; the returned values are tuples with n elements
            put in the order of their drawing;
            assumes that n >= 0
        '''
        if n == 0:
            return Alea.coerce((),prob_type=self._ps[0].__class__)
        return self.map(make_tuple).times(n)

    def draw_without_replacement(self,n):
        ''' returns a new Alea instance representing the probability distribution
            of drawing n elements from self WITHOUT replacement, taking the order
            of drawing into account; the returned values are tuples with n elements
            put in the order of their drawing;
            assumes that n >= 0;
            requires that n <= number of values of self, otherwise an exception
            is raised
        '''
        if n == 0:
            return Alea.coerce((),prob_type=self._ps[0].__class__)
        if len(self._vs) == 1:
            if n > 1:
                raise Lea.Error("number of values to draw exceeds the number of possible values")
            return Alea.coerce((self._vs[0],),prob_type=self._ps[0].__class__)
        alea2s = tuple(Alea._pmf_ordered(tuple((v0, p0) for (v0, p0) in self._gen_raw_vps() if v0 != v), check=False).draw_without_replacement(n - 1) for v in self._vs)
        vps = []
        for (v,p,alea2) in zip(self._vs,self._ps,alea2s):
            for (vt,pt) in alea2._gen_raw_vps():
                vps.append(((v,)+vt,p*pt))
        return Alea._pmf_ordered(vps, check=False)

    @staticmethod
    def poisson(mean,precision=1e-20,prob_type=None):
        ''' static method, returns an Alea instance representing a Poisson probability
            distribution having the given mean; the distribution is approximated by
            the finite set of values that have probability > precision
            (i.e. low/high values with too small probabilities are dropped);
            the probabilities are stored as float, whatever the current probability
            type configured
        '''
        val_freqs = []
        p = exp(-mean)
        v = 0
        t = 0.0
        while p >= precision or v <= mean:
            if p >= precision:
                val_freqs.append((v,p))
            t += p
            v += 1
            p = (p*mean) / v
        return Alea.pmf(val_freqs,prob_type)

    __DISPLAY_KINDS = (None, '/', '.', '%', '-', '/-', '.-', '%-')

    def as_string(self,kind=None,nb_decimals=6,histo_size=100,tabular=True):
        ''' returns a string representation of probability distribution self;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability in a format depending of given kind, which is either None
            (default) or a string among
                '/', '.', '%', '-', '/-', '.-', '%-';
            the probabilities are displayed as
            - if kind is None   : as they are stored
            - if kind[0] is '/' : rational numbers "n/d" or "0" or "1"
            - if kind[0] is '.' : decimals with given nb_decimals digits
            - if kind[0] is '%' : percentage decimals with given nb_decimals digits
            - if kind[0] is '-' : histogram bar made up of repeated '-', such that
                                  a bar length of histo_size represents a probability 1 
            if kind[1] is '-', the histogram bars with '-' are appended after 
                               numerical representation of probabilities;
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used;
            if tabular is True (default), then if values are tuples of same length, then
            these are represented in a tabular format (fixed column width); in the
            specific cases of named tuple, a header line is prepended with the field names
        '''
        if kind not in Alea.__DISPLAY_KINDS:
            raise Lea.Error("invalid display format '%s'; should be among %s"%(kind,Alea.__DISPLAY_KINDS))
        if self._val is self:
            # self is not bound (default)
            vs = self._vs
            ps = self._ps
        else:
            # self is explicitely bound - see observe(...) method or .calc(bindings=...)
            vs = (self._val,)
            ps = (1,)
        header = ''
        lines_iter = None
        v0 = vs[0]
        if tabular and isinstance(v0,tuple):
            v0_class = v0.__class__
            v0_length = len(v0)
            if all((v.__class__ is v0_class and len(v) == v0_length) for v in vs):
                # values are tuples of same length: perform a tabular display, where column widths depend of the longest elements
                repr_vs = tuple(tuple(repr(e) for e in v) for v in vs)
                if hasattr(v0_class,'_fields'):
                    # values are named tuples, of the same class: prepend the field names tuple for header
                    repr_vs = (v0_class._fields,) + repr_vs
                # determine max required length per column
                max_length_per_pos = tuple(max(len(e) for e in a) for a in zip(*repr_vs))
                if hasattr(v0_class,'_fields'):
                    header = " %s\n" % ', '.join(indent(str,e,s) for (e,s) in zip(v0._fields,max_length_per_pos))
                lines_iter = ('(%s)' % ', '.join(indent(repr,e,s) for (e,s) in zip(v,max_length_per_pos)) for v in vs)
        if lines_iter is None:
            # general, non-tabular, display 
            vm = max(len(str(v)) for v in vs)
            lines_iter = (indent(str,v,vm) for v in vs)
        lines_iter = (v+' : ' for v in lines_iter)
        if kind is None:
            lines_iter = (line+str(p) for (line,p) in zip(lines_iter,ps))
        else:
            prob_representation = kind[0]
            with_histo = kind[-1] == '-'
            if prob_representation == '/':
                (pnums,pdenom) = ProbFraction.convert_to_same_denom(tuple(Fraction(p) for p in ps))
                p_strings = tuple(str(pnum) for pnum in pnums)
                pnum_size_max = len(str(max(pnum for pnum in pnums)))
                if pdenom == 1:
                    den = ''
                else:
                    den = '/%d' % pdenom
                lines_iter = (line+p_string.rjust(pnum_size_max)+den for (line,p_string) in zip(lines_iter,p_strings))
            elif prob_representation == '.':
                fmt = "%%s%%.%df" % nb_decimals
                lines_iter = (fmt%(line,p) for (line,p) in zip(lines_iter,ps))
            elif prob_representation == '%':
                fmt = "%%s%%%d.%df %%%%" % (4+nb_decimals,nb_decimals)
                lines_iter = (fmt%(line,100.*p) for (line,p) in zip(lines_iter,ps))
            if with_histo:
                lines_iter = (line+' '+int(0.5+(p)*histo_size)*'-' for (line,p) in zip(lines_iter,ps))
        return header + '\n'.join(lines_iter)

    def __str__(self):
        ''' returns a string representation of probability distribution self;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability expressed according to its type;
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used;
            called on evaluation of "str(self)" and "repr(self)"
        '''
        if all(isinstance(p,Fraction) for p in self._ps):
            kind = '/'
        else:
            kind = None
        return self.as_string(kind)

    __repr__ = __str__
          
    def as_float(self,nb_decimals=6):
        ''' returns a string representation of probability distribution self;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability expressed as decimal with given nb_decimals digits;
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used;
        '''
        return self.as_string('.',nb_decimals)

    def as_pct(self,nb_decimals=2):
        ''' returns a string representation of probability distribution self;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability expressed as percentage with given nb_decimals digits;
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used;
        '''
        return self.as_string('%',nb_decimals)

    def histo(self,size=100):
        ''' returns a string representation of probability distribution self;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability expressed as a histogram bar made up of repeated '-',
            such that a bar length of given size represents a probability 1;
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used;
        '''
        return self.as_string('-',histo_size=size)

    def plot(self,title=None,fname=None,savefig_args=dict(),**bar_args):
        ''' produces a matplotlib bar chart representing the probability distribution self
            with the given title (if not None); the bar chart may be customized by using
            named arguments bar_args, which are relayed to matplotlib.pyplot.bar function
            (see doc in http://matplotlib.org/api/pyplot_api.html);
            * if fname is None, then the chart is displayed on screen, in a matplotlib window;
              the previous chart, if any, is erased
            * otherwise, the chart is saved in a file specified by given fname as specified
              by matplotlib.pyplot.savefig; the file format may be customized by using
              savefig_args argument, which is a dictionary relayed to matplotlib.pyplot.savefig
              function and containing named arguments expected by this function;
              example:
               flip.plot(fname='flip.png',savefig_args=dict(bbox_inches='tight'),color='green')
            the method requires matplotlib package; an exception is raised if it is not installed
        '''
        # try to import matplotlib package, required by plot() method
        if plt is None:
            raise Lea.Error("the plot() method requires the matplotlib package")
        # switch on interactive mode, so the control is back to console as soon as a chart is displayed
        plt.ion()
        if fname is None:
            # no file specified: erase the current chart, if any
            plt.clf()
        else:
            # file specified: switch off interactive mode
            plt.ioff()
        plt.bar(range(len(self._vs)),self._ps,tick_label=self._vs,align='center',**bar_args)
        plt.ylabel('Probability')
        if title is not None:
            plt.title(title)
        if fname is None:
            # no file specified: display the chart on screen
            plt.show()
        else:
            # file specified: save chart on file, using given parameters and switch back interactive mode
            plt.savefig(fname,**savefig_args)
            # TODO needed?
            plt.ion()

    def support(self):
        ''' returns a tuple with values of self
            the sequence follows the increasing order defined on values;
            if order is undefined (e.g. complex numbers), then the order is
            arbitrary but fixed from call to call;
            WARNING: this method is called without parentheses
        '''
        return self._vs

    def ps(self):
        ''' returns a tuple with probability of self
            the sequence follows the increasing order defined on values;
            if order is undefined (e.g. complex numbers), then the order is
            arbitrary but fixed from call to call;
            WARNING: this method is called without parentheses
        '''
        return tuple(Alea._downcast(p) for p in self._ps)

    def pmf_tuple(self):
        ''' returns, after evaluation of the probability distribution self, the probability
            mass function of self, as a tuple with tuples (v,P(v));
            the sequence follows the order defined on values;
            WARNING: this method is called without parentheses
        '''
        return tuple(self._gen_vps())

    def pmf_dict(self):
        ''' returns, after evaluation of the probability distribution self, the probability
            mass function of self, as an OrderedDict with v : P(v)) pairs;
            the sequence follows the order defined on values;
            requires Python 2.7+;
            WARNING: this method is called without parentheses
        '''
        return collections.OrderedDict(self._gen_vps())

    def cdf_tuple(self):
        ''' returns, after evaluation of the probability distribution self, the cumulative
            distribution function of self, as a tuple with tuples (v,P(x<=v));
            the sequence follows the order defined on values;
            WARNING: this method is called without parentheses
        '''
        return tuple((v,Alea._downcast(p)) for (v,p) in zip(self._vs,self.cumul()[1:]))

    def cdf_dict(self):
        ''' returns, after evaluation of the probability distribution self, the cumulative
            distribution function of self, as an OrderedDict with v : P(x<=v)) pairs;
            the sequence follows the order defined on values;
            requires Python 2.7+;
            WARNING: this method is called without parentheses
        '''
        return collections.OrderedDict((v,Alea._downcast(p)) for (v,p) in zip(self._vs,self.cumul()[1:]))

    def get_alea_leaves_set(self):
        ''' returns a set containing all the Alea leaves in the tree having the root self
            in the present case of Alea instance, it returns the singleton set with self as element
        '''
        return frozenset((self,))        
     
    def _get_lea_children(self):
        # Alea instance has no children
        return ()

    def _gen_vp(self):
        ''' generates tuples (v,p) where v is a value of self
            and p is the associated probability;
            the sequence follows the order defined on values
        '''
        return zip(self._vs,self._ps)

    def _gen_one_random_mc(self):
        ''' generates one random value from the current probability distribution,
            WITHOUT precalculating the exact probability distribution (contrarily to 'random' method);
            this obeys the "binding" mechanism, so if the same variable is referred multiple times in
            a given expression, then same value will be yielded at each occurrence; 
            before yielding the random value v, this value v is bound to the current instance;
            then, if the current calculation requires to get again a random value on the current
            instance, then the bound value is yielded;
            the instance is rebound to a new value at each iteration, as soon as the execution
            is resumed after the yield;
            the instance is unbound at the end;
            the method calls the _gen_one_random_mc method implemented in Lea subclasses;
        '''
        if self._val is not self:
            # distribution already bound to a value, because gen_one_random_mc has been called already on self 
            # yield the bound value, in order to be consistent
            yield self._val
        else:
            try:
                # bind value v: this is important if an object calls gen_one_random_mc on the same 
                # instance before resuming the present generator (see above)
                self._val = self.random_val()
                # yield the bound value v
                yield self._val
            finally:
                # unbind value, after the random value has been bound or if an exception has been raised
                self._val = self

    def observe(self,v):
        ''' (re)bind self with given value v;
            requires that self is an Alea instance (i.e. not dependent of other Lea instances);
            requires that v is present in the domain of self
        '''
        if v not in self._vs:
            raise Lea.Error("impossible to bind %s with '%s' because it is out of domain"%(self._id(),v))
        self._val = v

    def free(self,check=True):
        ''' unbind self;
            requires that self is an Alea instance (i.e. not dependent of other Lea instances);
            if check is True, then requires that self is bound
        '''
        if check and self._val is self:
            raise Lea.Error("%s already unbound"%self._id())
        self._val = self

    def _p(self,val,check_val_type=False):
        ''' returns the probability p of the given value val;
            if check_val_type is True, then raises an exception if some value
            in the distribution has a type incompatible with val's
        '''
        p1 = None
        if check_val_type:
            err_val = self  # dummy value
            type_to_check = type(val)
        for (v,p) in self._gen_vp():
            # check that all values have a common ancestor with val's type;
            # to avoid spurious error when using numpy, the numpy's bool_ is considered compatible with Python's bool
            if check_val_type and not isinstance(v,type_to_check) \
               and not (type_to_check is bool and isinstance(v,np_bool_type)):
                err_val = v
            if p1 is None and v == val:
                p1 = p
        if check_val_type and err_val is not self:
            raise Lea.Error("found <%s> value although <%s> is expected"%(type(err_val).__name__,type_to_check.__name__))
        if p1 is None:
            # val is absent form self: the probability is null, casted in the type of the last probability found
            p1 = 0 * p
        return Alea._simplify(p1)

    def cumul(self):
        ''' returns a list with the probabilities p that self <= value ;
            there is one element more than number of values; the first element is 0, then
            the sequence follows the order defined on values; if an order relationship is defined
            on values, then the tuples follows their increasing order; otherwise, an arbitrary
            order is used, fixed from call to call;
            Note: the returned list is cached
        '''
        if len(self._cumul) == 1:
            cumul_list = self._cumul
            p_sum = 0
            for p in self._ps:
                p_sum += p
                cumul_list.append(p_sum)
        return self._cumul

    def inv_cumul(self):
        ''' returns a tuple with the probabilities p that self >= value ;
            there is one element more than number of values; the first element is 0, then
            the sequence follows the order defined on values; if an order relationship is defined
            on values, then the tuples follows their increasing order; otherwise, an arbitrary
            order is used, fixed from call to call;
            Note: the returned list is cached
        '''
        if len(self._inv_cumul) == 0:
            inv_cumul_list = self._inv_cumul
            p_sum = 1
            for p in self._ps:
                inv_cumul_list.append(p_sum)
                p_sum -= p
            inv_cumul_list.append(0)
        return self._inv_cumul
            
    def random_val(self):
        ''' returns a random value among the values of self, according to their probabilities
        '''
        return next(self._random_iter)
        
    def _create_random_iter(self):
        ''' generates an infinite sequence of random values among the values of self,
            according to their probabilities
        '''
        try:
            probs = tuple(map(float,self.cumul()[1:]))
        except:
            raise Lea.Error("random sampling impossible because given probabilities cannot be converted to float")
        vals = self._vs
        while True:
            yield vals[bisect_right(probs,random())]
        
    def random_draw(self,n=None,sorted=False):
        ''' if n is None, returns a tuple with all the values of the distribution,
            in a random order respecting the probabilities;
            (the higher probability of a value, the more likely the value will be in the
             beginning of the sequence);
            if n > 0, then only n different values will be drawn;
            if sorted is True, then the returned tuple is sorted
        '''
        if n is None:
           n = len(self._vs)
        elif n < 0:
            raise Lea.Error("random_draw method requires a positive integer")    
        if n == 0:
            return ()
        lea1 = self
        res = []
        while True:
            lea1 = lea1.get_alea(sorting=False)
            x = lea1.random()
            res.append(x)
            n -= 1
            if n == 0:
                break
            lea1 = lea1.given(lea1!=x)
        if sorted:
            res.sort()
        return tuple(res)
    
    @memoize
    def p_cumul(self,val):
        ''' returns, as an integer, the probability that self <= val;
            note that it is not required that val is in the support of self
        '''
        return self.cumul()[bisect_right(self._vs,val)] 

    @memoize
    def p_inv_cumul(self,val):
        ''' returns, as an integer, the probability that self >= val;
            note that it is not required that val is in the support of self
        '''
        return self.inv_cumul()[bisect_left(self._vs,val)] 

    @staticmethod
    def fast_extremum(cumul_func,*alea_args):
        ''' static method, returns a new Alea instance giving the probabilities
            to have the extremum value (min or max) of each combination of the
            given Alea args;
            cumul_func is the cumul function that determines whether max or min is
            used : respectively, Alea.p_cumul or Alea.p_inv_cumul;
            the method uses an efficient algorithm (linear complexity), which is
            due to Nicky van Foreest
        '''
        if len(alea_args) == 1:
            return alea_args[0]
        if len(alea_args) == 2:
            (alea_arg1,alea_arg2) = alea_args
            pmf_dict = defaultdict(int)
            for (v,p) in alea_arg1._gen_raw_vps():
                p1 = cumul_func(alea_arg2,v)
                if p1 != 0:
                    pmf_dict[v] = p * p1
            for (v,p) in alea_arg2._gen_raw_vps():
                p1 = cumul_func(alea_arg1,v) - alea_arg1._p(v)
                if p1 != 0:
                    pmf_dict[v] +=  p1 * p
            return Alea.pmf(pmf_dict)
        return Alea.fast_extremum(cumul_func,alea_args[0],Alea.fast_extremum(cumul_func,*alea_args[1:]))

    # WARNING: the following methods are called without parentheses (see Lea.__getattr__)

    indicator_method_names = ('P', 'Pf', 'mean', 'mean_f', 'var', 'var_f',
                              'std', 'std_f', 'mode', 'entropy',
                              'rel_entropy', 'redundancy', 'information',
                              'support', 'ps', 'p_sum', 'pmf_tuple', 'pmf_dict',
                              'cdf_tuple', 'cdf_dict',)

    @staticmethod
    def _downcast(p):
        ''' static method, returns p or an object equivalent to p, more convenient
            to display:
             Fraction -> ProbFraction,
             Decimal  -> ProbDecimal
        '''
        downcast_prob_class = Alea.__downcast_prob_class.get(p.__class__)
        if downcast_prob_class is not None:
            p = downcast_prob_class(p)
        return p

    def p_sum(self):
        ''' returns the sum of all probabilities of self;
            the result is expressed in the probability type used in self,
            possibly downcasted for convenience (Fraction -> ProbFraction,
            Decimal -> ProbDecimal);
            note: the result is supposed to be 1 (expressed in some type);
            BUT it could be different:
            - due to float rounding-errors
            - due to an explicit normalization=False argument;
            WARNING: this method is called without parentheses
        '''
        ## note that the following expression is NOK for unorderable types (e.g. complex)
        ##   self.p_cumul(self._vs[-1])
        return Alea._downcast(Alea._simplify(sum(self._ps)))

    def P(self):
        ''' returns the probability that self is True;
            the probability is expressed in the probability type used in self,
            possibly downcasted for convenience (Fraction -> ProbFraction,
            Decimal -> ProbDecimal);
            raises an exception if some value in the distribution is not boolean
            (note that this is NOT the case with self.p(True));
            WARNING: this method is called without parentheses
        '''
        return Alea._downcast(self._p(True,check_val_type=True))

    def Pf(self):
        ''' returns the probability that self is True;
            the probability is expressed as a float between 0.0 and 1.0;
            raises an exception if the probability type is no convertible to float;
            raises an exception if some value in the distribution is not boolean;
            (this is NOT the case with self.p(True));
            WARNING: this method is called without parentheses
        '''
        return float(self._p(True,check_val_type=True))

    def _mean(self):
        ''' same as mean method but without conversion nor simplification
        '''
        res = None
        v0 = None
        for (v,p) in self._gen_raw_vps():
            if v0 is None:
                v0 = v
            elif res is None:
                res = p * (v-v0)
            else:
                res += p * (v-v0)
        if res is not None:
            v0 += res
        return v0

    def mean(self):
        ''' returns the mean value of the probability distribution, which is the
            probability weighted sum of the values;
            requires that
            1 - the values can be subtracted together,
            2 - the differences of values can be multiplied by integers,
            3 - the differences of values multiplied by integers can be
                added to the values,
            4 - the sum of values calculated in 3 can be divided by a float
                or an integer;
            if any of these conditions is not met, then the result depends of the
            value class implementation (likely, raised exception);
            WARNING: this method is called without parentheses
        '''
        return Alea._simplify(self._mean(),False)

    def mean_f(self):
        ''' same as mean method but with conversion to float or simplification of symbolic expression;
            WARNING: this method is called without parentheses
        '''
        return Alea._simplify(self._mean(),True)

    def _var(self):
        ''' same as var method but without conversion nor simplification
        '''
        res = 0
        m = self._mean()
        for (v,p) in self._gen_raw_vps():
            res += p*(v-m)**2
        return res

    def var(self):
        ''' returns the variance of the probability distribution;
            requires that
            1 - the requirements of the mean() method are met,
            2 - the values can be subtracted to the mean value,
            3 - the differences between values and the mean value can be squared;
            if any of these conditions is not met, then the result depends of the
            value implementation (likely, raised exception)
            WARNING: this method is called without parentheses
        '''
        return Alea._simplify(self._var(),False)

    def var_f(self):
        ''' same as var method but with conversion to float or simplification of symbolic expression;
            WARNING: this method is called without parentheses
        '''
        return Alea._simplify(self._var(),True)

    def _std(self):
        ''' same as std method but without conversion nor simplification
        '''
        var = self._var()
        sqrt_exp = var.__class__(0.5)
        return var ** sqrt_exp

    def std(self):
        ''' returns the standard deviation of the probability distribution
            requires that the requirements of the var method are met;
            WARNING: this method is called without parentheses
        '''
        return Alea._simplify(self._std(),False)

    def std_f(self):
        ''' same as std method but with conversion to float or simplification
            of symbolic expression;
            WARNING: this method is called without parentheses
        '''
        return Alea._simplify(self._std(),True)

    def _cov(self,lea1):
        ''' same as cov method but without conversion nor simplification
        '''
        alea1 = Alea.coerce(lea1)
        return ((self-self._mean())*(alea1-alea1.get_alea()._mean())).get_alea()._mean()

    def cov(self,lea1):
        ''' returns the covariance between self and given lea1 probability distributions;
            requires that, for self and lea1,
            1 - the requirements of the mean() method are met,
            2 - the values can be subtracted to the mean value,
            3 - the differences between values and the mean value can be multiplied together;
            if any of these conditions is not met, then the result depends of the
            value implementation (likely, raised exception)
        '''            
        return Alea._simplify(self._cov(lea1),False)

    def cov_f(self,lea1):
        ''' same as cov method but with conversion to float or simplification
            of symbolic expression;
        '''
        return Alea._simplify(self._cov(lea1),True)

    def mode(self):
        ''' returns a tuple with the value(s) of the probability distribution
            having the highest probability;
            WARNING: this method is called without parentheses
        '''
        max_p = max(self._ps)
        return tuple(v for (v,p) in self._gen_raw_vps() if p == max_p)

    def information_of(self,val):
        ''' returns a float number representing the information of given val,
            expressed in bits:
               log2(P(self==val))
            assuming that probability of val is (convertible to) float;
            if probability of val is a sympy expression, then the returned
            object is the information of val as a sympy expression;
            raises an exception if given val is impossible;
            raises an exception if probability of given val is neither;
            convertible to float nor a sympy expression
        '''
        p = self._p(val)
        try:
            if p == 0:
                raise Lea.Error("no information from impossible value")
            return -log2(p)
        except TypeError:
            try:
                from sympy import log
                return -log(p,2)
            except:
                raise Lea.Error("cannot calculate logarithm of %s"%p)

    def information(self):
        ''' returns the information of self being true, expressed in bits
            assuming that self is a boolean distribution;
            the returned type is a float or a sympy expression (see doc of
            Alea.entropy);
            raises an exception if self is certainly false;
            WARNING: this method is called without parentheses
        '''
        return self.information_of(True)

    def entropy(self):
        ''' returns the entropy of self in bits;
            if all probabilities are (convertible to) float, then the entropy
            is returned as a float;
            if any probability is a sympy expression, then the entropy is
            returned as a sympy expression;
            raises an exception if some probabilities are neither convertible
            to float nor a sympy expression;
            WARNING: this method is called without parentheses
        '''
        res = 0
        try:
            for (v,p) in self._gen_raw_vps():
                if p > 0:
                    res -= p*log2(p)
            return res
        except TypeError:
            try:
                from sympy import log
                for (v,p) in self._gen_raw_vps():
                    res -= p*log(p)
                return res / log(2)
            except:
                raise Lea.Error("cannot calculate logarithm on given probability types")

    def rel_entropy(self):
        ''' returns the relative entropy of self;
            if all probabilities are (convertible to) float, then the relative
            entropy is returned as a float between 0.0 and 1.0;
            if any probability is a sympy expression, then the relative entropy
            is returned as a sympy expression;
            raises an exception if some probabilities are neither convertible
            to float nor a sympy expression;
            WARNING: this method is called without parentheses
        '''
        n = len(self._vs)
        if n == 1:
            return 0.0
        res = self.entropy / log2(n)
        try:
            return min(1.0,res)
        except TypeError:
            return res

    def redundancy(self):
        ''' returns the redundancy of self;
            if all probabilities are (convertible to) float, then the
            redundancy is returned as a float between 0.0 and 1.0;
            if any probability is a sympy expression, then the redundancy
            is returned as a sympy expression;
            raises an exception if some probabilities are neither convertible
            to float nor a sympy expression;
            WARNING: this method is called without parentheses
        '''
        return 1.0 - self.rel_entropy

    def internal(self,indent='',refs=None):
        ''' returns a string representing the inner definition of self;
            if the same lea child appears multiple times, it is expanded only
            on the first occurrence, the other ones being marked with
            reference id; the arguments are used only for recursive calls
            from Lea.internal method, they can be ignored for a normal usage
        '''
        if refs is None:
            refs = set()
        if self in refs:
            return self._id() + '*'
        refs.add(self)
        vps = tuple(self._gen_raw_vps())
        res = "%s <%s" % (self._id(),vps[0])
        if len(vps) >= 2:
            res += ', ...'
        res += '>'
        return res
