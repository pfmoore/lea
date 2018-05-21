'''
--------------------------------------------------------------------------------

    lea.py

--------------------------------------------------------------------------------
Copyright 2013-2018 Pierre Denis

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

import operator
import sys
from itertools import islice
import collections
from .prob_fraction import ProbFraction
from .toolbox import easy_min, easy_max, read_csv_filename, read_csv_file, dict, zip, isclose

class Lea(object):
    
    '''
    Lea is an abstract class representing discrete probability distributions.

    Each instance of concrete Lea's subclasses (called simply a "Lea instance" in the following)
    represents a discrete probability distribution, which associates each value of a set of
    values with the probability that such value occurs.

    A Lea instance can be defined by a sequence of (value,probability), giving the probability 
    of each value. A Lea instance can be defined also by a sequence of values, the probability
    of a given value being its frauency in the sequence.

    Lea instances can be combined in arithmetic expressions resulting in new Lea instances, by
    obeying the following rules:

    - Lea instances can be added, subtracted, multiplied and divided together,
    through +, -, *, /, // operators. The resulting distribution's values and probabilities
    are determined by combination of operand's values with a sumed by probability
    products (the operation known as 'convolution', for the adition case).
    - Other supported binary arithmetic operators are power (**), modulo (%) and
    divmod function.
    - Unary operators +, - and abs function are supported also.
    - The Python's operator precedence rules, with the parenthesis overrules, are fully
    respected.
    - Any object X, which is not a Lea instance, involved as argument of an
    expression containing a Lea instance, is coerced to a Lea instance
    having X has sole value, with probability 1 (i.e. occurrence of X is certain).
    - Lea instances can be compared together, through ==, !=, <, <=, >, >= operators.
    The resulting distribution is a boolean distribution, giving probability of True result
    and complementary probability of False result.
    - Boolean distributions can be combined together with AND, OR, XOR, through &, |, ^
    operators, respectively.

    WARNING: the Python's 'and', 'or' and 'not' operators shall NOT be used on Lea instances;
    this raises an exception. Replace:
           a and b    by    a & b
           a or b     by    a | b
           not a      by    ~ a

    WARNING: in boolean expression involving arithmetic comparisons, the parenthesis
    shall be used, e.g. (a < b) & (b < c)

    WARNING: the augmented comparison (a < b < c) expression shall NOT be used.; this raises
    an exception (reason: it has the same limitation as 'and' operator).

    Lea instances can be used to generate random values, respecting the given probabilities.
    There are two Lea methods for this purpose:
    - random:   calculates the exact probabiliy distribution, then takes random values 
    - random_mc: takes random values from atomic probabiliy distribution, then makes the 
                required calculations (Monte-Carlo algorithm)
    The random_mc is suited for complex distributions, when calculation of exact probability
    distribution is intractable. This could be used to provide an estimation of the probability
    distribution (see estimate_mc method).

    There are nine concrete subclasses to Lea class, namely:
      Alea, Clea, Flea, Flea1, Flea2, Glea, Ilea, Rlea and Blea.
    
    Each subclass represents a "definition" of discrete probability distribution, with its own data
    or with references to other Lea instances to be combined together through a given operation.
    Each subclass defines what are the (value,probability) pairs or how they can be generated (see
    _gen_vp method implemented in each Lea subclass). The Lea class acts as a facade, by providing
    different sconstrucors (static methods) to instantiate these subclasses, so it is usually not
    needed to instantiate Lea subcasses explicitely. Here is an overview on these subclasses, with
    their relationships.

    - An Alea instance is defined by explicit value-probability pairs, that is a probability mass
    function (p.m.f.) defined by extension.

    Instances of other Lea's subclasses represent probability distributions obtained by operations
    done on existing Lea instance(s). Any such instance forms a direct acyclic graph (DAG) structure,
    having other Lea instances as nodes and Alea instances as leaves. This uses "lazy evaluation":
    actual (value,probability) pairs are calculated only at the time they are required (e.g. display,
    queryprobability of a given value, etc); then, these are aggregated in a new Alea instance. This 
    Alea instance is then cached, as an attribute of the queried Lea instance, for speeding up next
    queries.
    
    Here is a brief presentation of these Lea's subclasses: 

    - Clea   provides the joint of a given sequence of Lea instances
    - Flea   applies a given n-ary function to a given sequence of n Lea instances
    - Flea1  applies a given 1-ary function to a given Lea instance
    - Flea2  applies a given 2-ary function to two given Lea instances
    - Glea   applies n-ary functions present in a given Lea instance to a given sequence of n Lea
             instances
    - Ilea   filters the values of a given Lea instance according to a given Lea instance
             representing a boolean condition (conditional probabilities)
    - Rlea   embeds Lea instances as values of a parent Lea instance 
    - Blea   defines CPT, providing Lea instances corresponding to given conditions
             (used for bayesian networks)

    Note that Flea1 and Flea2 subclasses have more efficient implementation than Flea subclass'.

    WARNING: The following methods are called without parentheses (for the sake of ease of use):
      P, Pf, mean, mean_f, var, var_f, std, std_f, mode, entropy, rel_entropy, redundancy, information,
      support, ps, pmf_tuple, pmf_dict, cdf_tuple, cdf_dict
    These are applicable on any Lea instance; these are implemented and documented in the Alea class.

    Short design notes:
    
    Lea uses the "template method" design pattern: the Lea base abstract class calls the following methods,
    which are implemented in each Lea's subclass: _get_lea_children, _gen_vp and _gen_one_random_mc.
    Excepting the afore-mentioned estimate_mc method, Lea performs EXACT calculation of probability
    distributions.
    It implements an original algorithm, called the "Statues" algorithm, by reference to the game of the
    same name; this uses a variable binding mechanism that relies on Python's generators. To learn more,
    see doc of Alea._gen_vp method as well as other Xlea._gen_vp methods implemented in Lea's subclasses. 
    '''

    class Error(Exception):
        ''' exception representing any violation of requirements of Lea methods  
        '''
        pass
        
    class _FailedRandomMC(Exception):
        ''' internal exception representing a failure to get a set of random values that
            satisfy a given condition in a given number of trials (see methods having '..._mc' suffix) 
        '''
        pass

    # Lea attributes
    __slots__ = ('_alea', '_val', 'gen_vp')

    # a mutable object, which cannnot appear in Lea's values (not hashable)
    _DUMMY_VAL = []

    # constructor methods
    # -------------------

    def __init__(self):
        ''' initializes Lea instance's attributes
        '''
        # alea instance acting as a cache when actual value-probability pairs have been calculated
        self._alea = None
        # _val is the value temporarily bound to the instance, during evaluation (see _gen_bound_vp method)
        # note: self is used as a sentinel value to express that no value is currently bound; Python's
        # None is not a good sentinel value since it prevents using None as value in a distribution
        self._val = self
        # when evaluation is needed, gen_vp shall be bound on _gen_vp or _gen_bound_vp method
        # (see _init_calc method)
        self.gen_vp = None

    def _id(self):
        ''' returns a unique id, containing the concrete Lea class name as prefix
        '''
        return '%s#%s'%(self.__class__.__name__,id(self))

    def get_alea_leaves_set(self):
        ''' returns a set containing all the Alea leaves in the tree having the root self;
            this calls _get_lea_children() method implemented in Lea's subclasses;
            this method is overloaded in Alea subclass to stop the recursion
        '''
        return frozenset(alea_leaf for lea_child in self._get_lea_children()
                                   for alea_leaf in lea_child.get_alea_leaves_set())

    # constructor methods
    # -------------------

    def _gen_bound_vp(self):
        ''' generates tuple (v,p) where v is a value of the current probability distribution
            and p is the associated probability  (integer > 0);
            this obeys the "binding" mechanism, so if the same variable is referred multiple
            times in a given expression, then same value will be yielded at each occurrence;
            "Statues" algorithm:
            before yielding a value v, this value v is bound to the current instance;
            then, if the current calculation requires to get again values on the current
            instance, then the bound value is yielded with probability 1;
            the instance is rebound to a new value at each iteration, as soon as the execution
            is resumed after the yield;
            it is unbound at the end;
            the method calls the _gen_vp method implemented in Lea subclasses;
            the present Alea._gen_vp method is called by the _gen_vp methods implemented in
            other Lea subclasses; these methods are themselves called by Lea.new and,
            indirectly, by Lea.get_alea
        '''
        if self._val is not self:
            # distribution already bound to a value because gen_vp has been called already on self
            # it is yielded as a certain distribution (unique yield)
            yield (self._val,1)
        else:
            # distribution not yet bound to a value
            try:
                # browse all (v,p) tuples
                for (v,p) in self._gen_vp():
                    # bind value v: this is important if an object calls gen_vp on the same instance
                    # before resuming the present generator (see above)
                    self._val = v
                    # yield the bound value v with probability p
                    yield (v,p)
            finally:
                # unbind value v, after all values have been bound or if an exception has been raised
                self._val = self

    def _reset_gen_vp(self):
        ''' set gen_vp = None on self and all Lea descendants
        '''
        self.gen_vp = None
        # treat children recursively, up to Alea instances
        for lea_child in self._get_lea_children():
            lea_child._reset_gen_vp()

    def _set_gen_vp(self):
        ''' prepare calculation of probability distribution by binding self.gen_vp to the most adequate method:
            self.gen_vp is bound
             either on self._gen_vp method, if no binding is required (single occurrence of self in expression)
             or on self._gen_bound_vp method, if binding is required (multiple occurrences of self in expression)
            note: self._gen_bound_vp works in any case but perform unnecessary binding job if self occurrence is
            unique in the evaluated expression
            requires that gen_vp = None for self and all Lea descendants
        '''
        if self.gen_vp is None:
            # first occurrence of self in the expression: use the simplest _gen_vp method
            # this may be overwritten if a second occurrence is found
            self.gen_vp = self._gen_vp
        elif self.gen_vp == self._gen_vp:
            # second occurrence of self in the expression: use the _gen_bound_vp method
            self.gen_vp = self._gen_bound_vp
        # treat children recursively, up to Alea instances
        for lea_child in self._get_lea_children():
            lea_child._set_gen_vp()

    def _init_calc(self):
        ''' prepare calculation of probability distribution by binding self.gen_vp to the most adequate method;
            see _set_gen_vp method
        '''
        self._reset_gen_vp()
        self._set_gen_vp()

    def given_prob(self,cond_lea,p):
        ''' returns a new Lea instance from current distribution,
            such that given p is the probability that cond_lea is true
        '''
        cur_cond_lea = Alea.coerce(cond_lea)
        req_cond_lea = Alea.event(p)
        if req_cond_lea.is_true():
            return self.given(cur_cond_lea)
        elif not req_cond_lea.is_feasible():
            return self.given(~cur_cond_lea)
        return Lea.if_(req_cond_lea,self.given(cur_cond_lea).get_alea(sorting=False),self.given(~cur_cond_lea).get_alea(sorting=False))

    def given(self,*evidences):
        ''' returns a new Ilea instance representing the current distribution
            updated with the given evidences, which are each either a boolean or
            a Lea instance with boolean values; the values present in the returned
            distribution are those and only those compatible with the given AND of
            evidences;
            the resulting (value,probability) pairs are calculated when the
            returned Ilea instance is evaluated;
            an exception is raised if the evidences contain a non-boolean or
            if they are unfeasible
        '''
        return Ilea(self,(Alea.coerce(evidence) for evidence in evidences))    
    
    def times(self,n,op=operator.add):
        ''' returns, after evaluation of the probability distribution self, a new
            Alea instance representing the current distribution operated n times
            with itself, through the given binary operator op;
            if n = 1, then a copy of self is returned;
            requires that n is strictly positive; otherwise, an exception is
            raised;
            note that the implementation uses a fast dichotomic algorithm,
            instead of a naive approach that scales up badly as n grows
        '''
        alea1 = self.get_alea()
        if n <= 0:
            raise Lea.Error("times method requires a strictly positive integer")
        if n == 1:
            return alea1.new()
        (n2,r) = divmod(n,2)
        alea2 = alea1.times(n2,op)
        res_flea2 = Flea2(op,alea2,alea2.new())
        if r == 1:
            res_flea2 = Flea2(op,res_flea2,alea1)
        return res_flea2.get_alea()

    def times_tuple(self,n):
        ''' returns a new Alea instance with tuples of length n, containing
            the joint of self with itself repeated n times
            note: equivalent to self.draw(n,sorted=False,replacement=True)
        '''
        return self.get_alea().draw_with_replacement(n)

    def joint(self,*args):
        ''' returns a new Clea instance, representing the joint of all
            arguments, coerced to Lea instances, including self as first argument 
        '''
        return Clea(self,*args)

    @staticmethod
    def reduce_all(op,args,absorber=None):
        ''' static method, returns a new Flea2 instance that join the given args with
            the given function op, from left to right;
            requires that op is a 2-ary function, accepting self's values as arguments;
            requires that args contains at least one element
            if absorber is not None, then it is considered as a "left-absorber" value
            (i.e. op(absorber,x) = absorber); this activates a more efficient algorithm
            which prunes the tree search as soon as the absorber is met.
        '''
        args_rev_iter = iter(reversed(tuple(args)))
        res = next(args_rev_iter)
        if absorber is None:
            for arg in args_rev_iter:
                res = Flea2(op,arg,res)
        else:
            for arg in args_rev_iter:
                res = Flea2a(op,arg,res,absorber)
        return res

    def merge(self,*lea_args):
        ''' returns a new Blea instance, representing the merge of self and given lea_args, i.e.
                  P(v) = (P1(v) + ... + Pn(v)) / n
            where P(v)  is the probability of value v in the merge result 
                  Pi(v) is the probability of value v in ((self,)+lea_args)[i]
        '''
        leas = (self,) + lea_args
        lea = Alea.vals(*range(len(leas)))
        return Blea.build(*((lea==i,lea_arg) for (i,lea_arg) in enumerate(leas)))

    def map(self,f,*args):
        ''' returns a new Flea instance representing the distribution obtained
            by applying the given function f, taking values of self distribution
            as first argument and optional given args as following arguments;
            requires that f is a n-ary function with 1 <= n = len(args)+1;
            note: f can be also a Lea instance, with functions as values
        '''
        return Flea.build(f,(self,)+args)

    def map_seq(self,f,*args):
        ''' returns a new Flea instance representing the distribution obtained
            by applying the given function f on each element of each value
            of self distribution; optional given args are added as f's
            following arguments;
            requires that f is a n-ary function with 1 <= n = len(args)+1;
            requires that self's values are sequences;
            the values of returned distribution are tuples
            note: f can be also a Lea instance, with functions as values
        '''
        return self.map(lambda v: tuple(f(e,*args) for e in v))

    @staticmethod
    def func_wrapper(f):
        ''' returns a wrapper function on given f function, mimicking f with
            Lea instances as arguments;
            the returned wrapper function has the same number of arguments
            as f and expects for argument #i
            - either an object of the type expected by f for argument #i
            - or a Lea instance with values of that type;
            the returned wrapper function, when called, returns a Lea instance
            having values of the type returned by f;
            note: Lea.func_wrapper can be used as a function decorator
        '''
        def wrapper(*args):
            return Flea.build(f,args)
        wrapper.__name__ = 'lea_wrapper_on__' + f.__name__
        wrapper.__doc__ = ("" if f.__doc__ is None else f.__doc__) \
                          + "\nThe present function wraps '%s' so to work with Lea instances as arguments." % f.__name__
        return wrapper

    def as_joint(self,*attr_names):
        ''' returns a new Alea instance representing a joint probability distribution
            from the current distribution supposed to have n-tuples as values,
            to be associated with the given n attribute names
        '''
        NTClass = collections.namedtuple('_',attr_names)
        return self.map(lambda a_tuple: NTClass(*a_tuple)).get_alea()

    def is_uniform(self):
        ''' returns, after evaluation of the probability distribution self,
            True  if the probability distribution is uniform,
            False otherwise
        '''
        return self.get_alea().is_uniform()

    def draw(self,n,sorted=False,replacement=False):
        ''' returns, after evaluation of the probability distribution self,
            a new Alea instance representing the probability distribution
            of drawing n elements from self;
            the returned values are tuples with n elements;
            * if sorted is True, then the order of drawing is irrelevant and
                 the tuples are arbitrarily sorted by increasing order;
                 (the efficient algorithm used is due to Paul Moore)
              otherwise, the order of elements of each tuple follows the order
                 of the drawing
            * if replacement is True, then the drawing is made WITH replacement,
                 so the same element may occur several times in each tuple
              otherwise, the drawing is made WITHOUT replacement,
                 so an element can only occur once in each tuple;
                 this last case requires that 0 <= n <= number of values of self,
                 otherwise an exception is raised
            Note: if the order of drawing is irrelevant, it is strongly advised to
             use sorted=True because the processing can be far more efficient thanks
             to a combinatorial algorithm proposed by Paul Moore; however, this
             algorithm is NOT used if replacement is False AND the probability
             distribution is NOT uniform.
            requires Python 2.7+ for the case replacement = sorted = True
        '''
        if n < 0:
            raise Lea.Error("draw method requires a positive integer")
        alea1 = self.get_alea()
        if replacement:
            if sorted:
                # draw sorted with replacement
                return alea1.draw_sorted_with_replacement(n)
            else:
                # draw sorted without replacement
                return alea1.draw_with_replacement(n)
        else:
            if len(alea1._vs) < n:
                raise Lea.Error("number of values to draw without replacement (%d) exceeds the number of possible values (%d)"%(n,len(alea1._vs)))
            if sorted:
                # draw sorted without replacement
                return alea1.draw_sorted_without_replacement(n)
            else:
                # draw unsorted without replacement
                return alea1.draw_without_replacement(n)

    def flat(self):
        ''' assuming that self's values are themselves Lea instances,
            returns a new Rlea instance representing a probability distribution of
            inner values of these Lea instances  
        '''
        return Rlea(self)

    def equiv(self,other,prec=None):
        ''' returns True iff self and other represent the same probability distribution,
            i.e. they have the same probability for each of their value;
            returns False otherwise;
            the probabilities are compared strictly (see Lea.equiv_f method for
            comparisons tolerant to rounding errors)
        '''
        other = Alea.coerce(other)
        # absolute equality required
        # frozenset(...) is used to avoid any dependency on the order of values
        return frozenset(self._gen_raw_vps()) == frozenset(other._gen_raw_vps())

    def equiv_f(self,other):
        ''' returns True iff self and other represent the same probability distribution,
            i.e. they have the same probability for each of their value;
            returns False otherwise;
            the probabilities are compared using the math.isclose function,
            in order to be tolerant to rounding errors
        '''
        other = Alea.coerce(other)
        vps1 = tuple(self._gen_raw_vps())
        vps2Dict = dict(other._gen_raw_vps())
        if len(vps1) != len(vps2Dict):
            return False
        for (v1,p1) in vps1:
            p2 = vps2Dict.get(v1)
            if p2 is None:
                return False
            if not isclose(p1,p2):
                return False
        return True

    def p(self,val):
        ''' returns the probability of given value val
        '''
        return Alea._downcast(self._p(val))

    def _gen_raw_vps(self):
        ''' generates, after evaluation of the probability distribution self,
            tuples (v,p) where v is a value of self
            and p is the associated probability (integer > 0);
            the sequence follows the order defined on values;
            note that there is NO binding, contrarily to _gen_vp method
        '''
        return self.get_alea()._gen_vp()
    
    def _gen_vps(self):
        ''' generates, after evaluation of the probability distribution self,
            tuples (v,p) where v is a value of self and p is the associated
            probability;
            the sequence follows the order defined on values;
            note that there is NO binding, contrarily to _gen_vp method
        '''
        return ((v,Alea._downcast(p)) for (v,p) in self._gen_raw_vps())

    def _p(self,val,check_val_type=False):
        ''' returns the probability of the given value val
            if check_val_type is True, then raises an exception if some value in the
            distribution has a type different from val's
        '''
        return self.get_alea()._p(val,check_val_type)

    def sort_by(self,*ordering_leas):
        ''' returns an Alea instance representing the same probability distribution as self
            but having values ordered according to given ordering_leas;
            requires that self doesn't contain duplicate values, otherwise an exception is
            raised; note that it is NOT required that all ordering_leas appear in self 
        '''
        # after prepending ordering_leas to self, the Alea returned by new() is sorted with ordering_leas;
        # then, extracting self (index -1) allows generating self's (v,p) pairs in the expected order;
        # these shall be used to create a new Alea, keeping the values in that order (no sort)
        sorted_lea = Lea.joint(*ordering_leas).joint(self).new()[-1]
        sorted_lea._init_calc()
        return Alea._pmf_ordered(sorted_lea.gen_vp())

    def is_any_of(self,*values):
        ''' returns a boolean probability distribution
            indicating the probability that a value is any of the values passed as arguments
        '''
        return Flea1(lambda v: v in values,self)

    def is_none_of(self,*values):
        ''' returns a boolean probability distribution
            indicating the probability that a value is none of the given values passed as arguments 
        '''
        return Flea1(lambda v: v not in values,self)

    def subs(self,*args):
        ''' returns a new Alea instance, equivalent to self, where probabilities have been converted
            by applying subs(*args) on them;
            this is useful for substituying variables when probabilities are expressesd as sympy
            expressions (see doc of sympy.Expreesion.subs method);
            requires that all self's probabiliies have a 'subs' method available
        '''
        return Alea(*zip(*((v,p.subs(*args)) for (v,p) in self._gen_raw_vps())),normalization=False)

    @staticmethod
    def if_(cond_lea,then_lea,else_lea):
        ''' static method, returns an instance of Tlea representing the
            conditional probability table
            giving then_lea  if cond_lea is true
                   else_lea  otherwise
            this is a convenience method equivalent to 
              cond_lea.switch({True:then_lea,False:else_lea})
        '''
        return Tlea(cond_lea,{True:then_lea,False:else_lea})

    def switch(self,lea_dict,default_lea=_DUMMY_VAL):
        ''' returns an instance of Tlea representing a conditional probability table (CPT)
            defined by the given dictionary lea_dict associating each value of self to a
            specific Lea instance;
            if default_lea is given, then it provides the Lea instance associated to the
            value(s) of self missing in lea_dict;
            all dictionary's values and default_lea (if defined) are cast to Lea instances
        '''
        return Tlea(self,lea_dict,default_lea)

    ## note: in PY3, could use:
    ## def cpt(*clauses,prior_lea=None,auto_else=False,check=True):
    @staticmethod
    def cpt(*clauses,**kwargs):
        ''' static method, returns an instance of Blea representing the conditional
            probability table (e.g. a node in a Bayes network) from the given clauses;
            each clause is a tuple (condition,result)
            where condition is a boolean or a Lea boolean distribution
              and result is a value or Lea distribution representing the result
                   assuming that condition is true
            the conditions from all clauses shall be mutually exclusive
            if a clause contains None as condition, then it is considered as a 'else'
            condition;
            the method supports three optional named arguments:
             'prior_lea', 'auto_else', 'check' and 'ctx_type';
            'prior_lea' and 'auto_else' are mutually exclusive, they require the absence
            of an 'else' clause (otherwise, an exception is raised); 
            * if prior_lea argument is specified, then the 'else' clause is calculated
            so that the prior_lea is returned for the unconditional case
            * if auto_else argument is specified as True, then the 'else' clause is
            calculated so that a uniform probability distribution is returned for
            the condition cases not covered in given clauses (principle of indifference);
            the values are retrieved from the results found in given clauses
            * if check argument is specified as False, then NO checks are made on the given
            clauses (see below); this can significantly increase performances, as the 
            set of clauses or variables become large; 
            by default (check arg absent or set to True), checks are made on clause
            conditions to ensure that they form a partition:
              1) the clause conditions shall be mutually disjoint, i.e. no subset
                 of conditions shall be true together;
              2) if 'else' is missing and not calculated through 'prior_lea' nor 'auto_else',
                 then the clause conditions shall cover all possible cases, i.e. ORing
                 them shall be certainly true;
            an exception is raised if any of such conditions is not verified;
        '''
        return Blea.build(*clauses,**kwargs)

    def revised_with_cpt(self,*clauses):
        ''' returns an instance of Blea representing the conditional probability table
            (e.g. a node in a Bayes network) from the given clauses;
            each clause is a tuple (condition,result)
            where condition is a boolean or a Lea boolean distribution
              and result is a value or Lea distribution representing the result
                   assuming that condition is true
            the conditions from all clauses shall be mutually exclusive
            no clause can contain None as condition
            the 'else' clause is calculated so that the returned Blea if no condition is
            given is self
        ''' 
        return Blea.build(*clauses,prior_lea=self)

    def build_bn_from_joint(self,*bn_definition):
        ''' returns a named tuple of Lea instances (Alea or Tlea) representing a Bayes
            network with variables stored in attributes A1, ... , An, assuming that self
            is a Lea joint probability distribution having, as values, named tuples
            with the same set of attributes A1, ... , An (such Lea instance is
            returned by as_joint method, for example);
            each argument of given bn_definition represents a dependency relationship
            from a set of given variables to one given variable; this is expressed as
            a tuple (src_var_names, tgt_var_name) where src_var_names is a sequence of
            attribute names (strings) identifying 'from' variables and tgt_name is the
            attribute name (string) identifying the 'to' variable;
            the method builds up the 'to' variable of the BN as a CPT calculated from
            each combination of 'from' variables in the joint probability distribution:
            for each such combination C, the distribution of 'to' variable is calculated
            by marginalisation on the joint probability distribution given the C condition;
            possible missing combinations are covered in an 'else' clause on the CPT
            that is defined as a uniform distribution of the values of 'to' variable,
            which are found in the other clauses (principle of indifference);
            the variables that are never refered as 'to' variable are considered
            as independent in the BN and are calculated by unconditional marginalisation
            on the joint probability distribution;
            if a variable appears in more than one 'to' variable, then an exception is
            raised (error)
        '''
        joint_alea = self.get_alea()
        # retrieve the named tuple class from the first value of the joint distribution,
        NamedTuple = joint_alea._vs[0].__class__
        vars_dict = dict((var_name,self.__getattribute__(var_name)) for var_name in NamedTuple._fields)
        # all BN variables initialized as independent (maybe overwritten below, according to given relationships)
        vars_bn_dict = dict((var_name,var.get_alea(sorting=False)) for (var_name,var) in vars_dict.items())
        for (src_var_names,tgt_var_name) in bn_definition:
            if not isinstance(vars_bn_dict[tgt_var_name],Alea):
                raise Lea.Error("'%s' is defined as target in more than one BN relationship"%tgt_var_name)
            tgt_var = vars_dict[tgt_var_name]
            joint_src_vars = Lea.joint(*(vars_dict[src_var_name] for src_var_name in src_var_names))
            joint_src_vars_bn = Lea.joint(*(vars_bn_dict[src_var_name] for src_var_name in src_var_names))
            # build CPT clauses (condition,result) from the joint probability distribution
            joint_src_vals = joint_src_vars.support
            clauses = tuple((joint_src_val,tgt_var.given(joint_src_vars==joint_src_val).get_alea(sorting=False)) \
                             for joint_src_val in joint_src_vals)
            # determine missing conditions in the CPT, if any
            all_vals = Lea.joint(*(vars_dict[src_var_name].get_alea(sorting=False) for src_var_name in src_var_names)).support
            missing_vals = frozenset(all_vals) - frozenset(joint_src_vals)
            if len(missing_vals) > 0:
                # there are missing conditions: add c   lauses with each of these conditions associating
                # them with a uniform distribution built on the values found in results of other clauses
                # (principle of indifference)
                else_result = Alea.vals(*frozenset(val for (cond,result) in clauses for val in result.support))
                clauses += tuple((missing_val,else_result) for missing_val in missing_vals)
            # overwrite the target BN variable (currently independent Alea instance), with a CPT built
            # up from the clauses determined from the joint probability distribution
            # the check is deactivated for the sake of performance; this is safe since, by construction,
            # the clauses conditions verify the "truth partioning" rules
            # the ctx_type is 2 for the sake of performance; this is safe since, by construction, the
            # clauses results are Alea instances and clause conditions refer to the same variable,
            # namely joint_src_vars_bn
            vars_bn_dict[tgt_var_name] = joint_src_vars_bn.switch(dict(clauses))
        # return the BN variables as attributes of a new named tuple having the same attributes as the
        # values found in self
        return NamedTuple(**vars_bn_dict)

    @staticmethod
    def make_vars(obj,tgt_dict,prefix='',suffix=''):
        ''' retrieve attributes names A1, ... , An of obj and put associations 
            {V1 : obj.A1, ... , Vn : obj.An} in tgt_dict dictionary
            where Vi is a variable name string built as prefix + Ai + suffix;
            obj is
            (a) either a named tuple with attributes A1, ... , An (as returned
            by build_b_nfrom_joint, for example)
            (b) or a Lea instances representing a joint probability distribution
            with the attributes A1, ... , An (such Lea instance is returned by
            as_joint method, for example);
            note: if the caller passes globals() as tgt_dict, then the variables
            named Vi, refering to obj.Ai, shall be created in its scope, as
            a side-effect (this is the purpose of the method);
            warning: the method may silently overwrite caller's variables
        '''
        if isinstance(obj,Lea):
            # case (b)
            # retrieve the named tuple class from the first value of the joint distribution
            NamedTuple = obj.get_alea()._vs[0].__class__
        else:
            # case (a)
            NamedTuple = obj.__class__
        tgt_dict.update((prefix+var_name+suffix,obj.__getattribute__(var_name)) for var_name in NamedTuple._fields)       
    
    def __call__(self,*args):
        ''' returns a new Flea instance representing the probability distribution
            of values returned by invoking functions of current distribution on 
            given arguments (assuming that the values of current distribution are
            functions);
            called on evaluation of "self(*args)"
        '''
        return Glea.build(self,args)

    def __getitem__(self,index):
        ''' returns a new Flea instance representing the probability distribution
            obtained by indexing or slicing each value with index
            called on evaluation of "self[index]"
        '''
        return Flea2(operator.getitem,self,index)

    def __iter__(self):
        ''' raises en error exception
            called on evaluation of "iter(self)", "tuple(self)", "list(self)"
                or on "for x in self"
        '''
        raise Lea.Error("cannot iterate on a Lea instance")

    def __getattribute__(self,attr_name):
        ''' returns the attribute with the given name in the current Lea instance;
            if the attribute name is a distribution indicator, then the distribution
            is evaluated and the indicator method is called; 
            if the attribute name is unknown as a Lea instance's attribute,
            then returns a Flea instance that shall retrieve the attibute in the
            values of current distribution; 
            called on evaluation of "self.attr_name"
            WARNING: the following methods are called without parentheses:
                         P, Pf, mean, mean_f, var, var_f, std, std_f, mode,
                         entropy, rel_entropy, redundancy, information,
                         support, ps, pmf_tuple, pmf_dict, cdf_tuple, cdf_dict
                     these are applicable on any Lea instance
                     and these are documented in the Alea class
        '''
        try:
            if attr_name in Alea.indicator_method_names:
                # indicator methods are called implicitely
                return object.__getattribute__(self.get_alea(),attr_name)()
            # return Lea's instance attribute
            return object.__getattribute__(self,attr_name)
        except AttributeError:
            # return new Lea made up of attributes of inner values
            return Flea2(getattr,self,attr_name)

    ## in PY3, could use
    ## def max_of(*args,fast=False):
    @staticmethod
    def max_of(*args,**kwargs):
        ''' static method, returns a new Flea instance giving the probabilities to
            have the maximum value of each combination of the given args;
            if some elements of args are not Lea instances, then these are coerced
            to a Lea instance with probability 1;
            if optional arg fast is False (default),
                the returned distribution keeps dependencies with args but the 
                calculation could be prohibitively slow (exponential complexity);
            if optional arg fast is True,
                the method uses an efficient algorithm (linear complexity), which is
                due to Nicky van Foreest; for explanations, see
                http://nicky.vanforeest.com/scheduling/cpm/stochastic_makespan.html
                however, unlike most of Lea methods, the distribution returned loses
                any dependency with given args; this could be important if some args
                appear in the same expression as Lea.max(...) but outside it, e.g.
                conditional probability expressions
        '''
        fast = kwargs.get('fast',False)
        if not fast:
            return Flea.build(easy_max,args)
        alea_args = tuple(Alea.coerce(arg).get_alea() for arg in args)
        return Alea.fast_extremum(Alea.p_cumul,*alea_args)

    ## in PY3, could use
    ## def min_of(*args,fast=False):
    @staticmethod
    def min_of(*args,**kwargs):
        ''' static method, returns a new Flea instance giving the probabilities to
            have the minimum value of each combination of the given args;
            if some elements of args are not Lea instances, then these are coerced
            to a Lea instance with probability 1;
            if optional arg fast is False (default),
                the returned distribution keeps dependencies with args but the 
                calculation could be prohibitively slow (exponential complexity);
            if optional arg fast is True,
                the method uses an efficient algorithm (linear complexity), which is
                due to Nicky van Foreest; for explanations, see
                http://nicky.vanforeest.com/scheduling/cpm/stochastic_makespan.html
                however, unlike most of Lea methods, the distribution returned loses
                any dependency with given args; this could be important if some args
                appear in the same expression as Lea.max(...) but outside it, e.g.
                conditional probability expressions
        '''
        fast = kwargs.get('fast',False)
        if not fast:
            return Flea.build(easy_min,args)
        alea_args = tuple(Alea.coerce(arg).get_alea() for arg in args)
        return Alea.fast_extremum(Alea.p_inv_cumul,*alea_args)

    def __lt__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            that the values of self are less than the values of other;
            called on evaluation of "self < other"
        '''
        return Flea2(operator.lt,self,other)

    def __le__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            that the values of self are less than or equal to the values of other;
            called on evaluation of "self <= other"
        '''
        return Flea2(operator.le,self,other)

    def __eq__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            that the values of self are equal to the values of other;
            called on evaluation of "self == other"
        '''
        return Flea2(operator.eq,self,other)

    def __hash__(self):
        return id(self)

    def __ne__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            that the values of self are different from the values of other;
            called on evaluation of "self != other"
        '''
        return Flea2(operator.ne,self,other)

    def __gt__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            that the values of self are greater than the values of other;
            called on evaluation of "self > other"
        '''
        return Flea2(operator.gt,self,other)

    def __ge__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            that the values of self are greater than or equal to the values of other;
            called on evaluation of "self >= other"
        '''
        return Flea2(operator.ge,self,other)
    
    def __add__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the addition of the values of self with the values of other;
            called on evaluation of "self + other"
        '''
        return Flea2(operator.add,self,other)

    def __radd__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the addition of the values of other with the values of self;
            called on evaluation of "other + self"
        '''
        return Flea2(operator.add,other,self)

    def __sub__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the subtraction of the values of other from the values of self;
            called on evaluation of "self - other"
        '''
        return Flea2(operator.sub,self,other)

    def __rsub__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the subtraction of the values of self from the values of other;
            called on evaluation of "other - self"
        '''
        return Flea2(operator.sub,other,self)

    def __pos__(self):
        ''' returns a Flea instance representing the probability distribution
            resulting from applying the unary positive operator on the values of self;
            called on evaluation of "+self"
        '''
        return Flea1(operator.pos,self)

    def __neg__(self):
        ''' returns a Flea instance representing the probability distribution
            resulting from negating the values of self;
            called on evaluation of "-self"
        '''
        return Flea1(operator.neg,self)

    def __mul__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the multiplication of the values of self by the values of other;
            called on evaluation of "self * other"
        '''
        return Flea2(operator.mul,self,other)

    def __rmul__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the multiplication of the values of other by the values of self;
            called on evaluation of "other * self"
        '''
        return Flea2(operator.mul,other,self)

    def __pow__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the powering the values of self with the values of other;
            called on evaluation of "self ** other"
        '''
        return Flea2(operator.pow,self,other)

    def __rpow__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the powering the values of other with the values of self;
            called on evaluation of "other ** self"
        '''
        return Flea2(operator.pow,other,self)

    def __truediv__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the division of the values of self by the values of other;
            called on evaluation of "self / other"
        '''
        return Flea2(operator.truediv,self,other)

    def __rtruediv__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the division of the values of other by the values of self;
            called on evaluation of "other / self"
        '''
        return Flea2(operator.truediv,other,self)

    def __floordiv__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the floor division of the values of self by the values of other;
            called on evaluation of "self // other"
        '''
        return Flea2(operator.floordiv,self,other)

    def __rfloordiv__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the floor division of the values of other by the values of self;
            called on evaluation of "other // self"
        '''
        return Flea2(operator.floordiv,other,self)

    # Python 2 compatibility
    __div__ = __truediv__
    __rdiv__ = __rtruediv__

    def __mod__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the modulus of the values of self with the values of other;
            called on evaluation of "self % other"
        '''
        return Flea2(operator.mod,self,other)

    def __rmod__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the modulus of the values of other with the values of self;
            called on evaluation of "other % self"
        '''
        return Flea2(operator.mod,other,self)

    def __divmod__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from applying the function divmod on the values of self and the values of other;
            called on evaluation of "divmod(self,other)"
        '''
        return Flea2(divmod,self,other)

    def __rdivmod__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from applying the function divmod on the values of other and the values of self;
            called on evaluation of "divmod(other,self)"
        '''
        return Flea2(divmod,other,self)

    def __floordiv__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the integer division of the values of self by the values of other;
            called on evaluation of "self // other"
        '''
        return Flea2(operator.floordiv,self,other)
    
    def __rfloordiv__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the integer division of the values of other by the values of self;
            called on evaluation of "other // self"
        '''
        return Flea2(operator.floordiv,other,self)

    def __abs__(self):
        ''' returns a Flea instance representing the probability distribution
            resulting from applying the abs function on the values of self;
            called on evaluation of "abs(self)"
        '''
        return Flea1(abs,self)
    
    def __and__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical AND between the values of self and the values of other;
            called on evaluation of "self & other"
        '''
        return Flea2(Lea._safe_and,self,other)

    def __rand__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical AND between the values of other and the values of self;
            called on evaluation of "other & self"
        '''
        return Flea2(Lea._safe_and,other,self)

    def __or__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical OR between the values of self and the values of other;
            called on evaluation of "self | other"
        '''
        return Flea2(Lea._safe_or,self,other)

    def __ror__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical OR between the values of other and the values of self;
            called on evaluation of "other | self"
        '''
        return Flea2(Lea._safe_or,other,self)

    def __xor__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical XOR between the values of self and the values of other;
            called on evaluation of "self ^ other"
        '''
        return Flea2(Lea._safe_xor,self,other)

    def __invert__(self):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical NOT of the values self;
            called on evaluation of "~self"
        '''
        return Flea1(Lea._safe_not,self)

    def __bool__(self):
        ''' raises an exception telling that Lea instance cannot be evaluated as a boolean
            called on evaluation of "bool(self)", "if self:", "while self:", etc
        '''
        raise Lea.Error("Lea instance cannot be evaluated as a boolean")

    # Python 2 compatibility
    __nonzero__ = __bool__

    @staticmethod
    def _check_booleans(op_msg,*vals):
        ''' static method, raise an exception if any of vals arguments is not boolean;
            the exception messsage refers to the name of a logical operation given in the op_msg argument
        '''
        for val in vals:
            if not isinstance(val,bool):
                raise Lea.Error("non-boolean object involved in %s logical operation (maybe due to a lack of parentheses)"%op_msg) 

    @staticmethod
    def _safe_and(a,b):
        ''' static method, returns a boolean, which is the logical AND of the given boolean arguments; 
            raises an exception if any of arguments is not boolean
        '''
        Lea._check_booleans('AND',a,b)
        return operator.and_(a,b)

    @staticmethod
    def _safe_or(a,b):
        ''' static method, returns a boolean, which is the logical OR of the given boolean arguments; 
            raises an exception if any of arguments is not boolean
        '''
        Lea._check_booleans('OR',a,b)
        return operator.or_(a,b)

    @staticmethod
    def _safe_xor(a,b):
        ''' static method, returns a boolean, which is the logical XOR of the given boolean arguments; 
            raises an exception if any of arguments is not boolean
        '''
        Lea._check_booleans('XOR',a,b)
        return operator.xor(a,b)

    @staticmethod
    def _safe_not(a):
        ''' static method, returns a boolean, which is the logical NOT of the given boolean argument; 
            raises an exception if the argument is not boolean
        '''
        Lea._check_booleans('NOT',a)
        return operator.not_(a)    

    def _get_lea_children(self):
        ''' returns a tuple containing all the Lea instances children of the current Lea;
            Lea._get_lea_children method is abstract: it is implemented in all Lea's subclasses
        '''
        raise NotImplementedError("missing method '%s._get_lea_children(self)'"%(self.__class__.__name__))
       
    def _gen_vp(self):
        ''' generates tuple (v,p) where v is a value of the current probability distribution
            and p is the associated probability (integer > 0);
            this obeys the "binding" mechanism, so if the same variable is refered multiple times in
            a given expression, then same value will be yielded at each occurrence;
            Lea._gen_vp method is abstract: it is implemented in all Lea's subclasses
        '''
        raise NotImplementedError("missing method '%s._gen_vp(self)'"%(self.__class__.__name__))
         
    def _gen_one_random_mc(self):
        ''' generates one random value from the current probability distribution,
            WITHOUT precalculating the exact probability distribution (contrarily to 'random' method);
            this obeys the "binding" mechanism, so if the same variable is refered multiple times in
            a given expression, then same value will be yielded at each occurrence; 
            before yielding the random value v, this value v is bound to the current instance;
            then, if the current calculation requires to get again a random value on the current
            instance, then the bound value is yielded;
            the instance is rebound to a new value at each iteration, as soon as the execution
            is resumed after the yield;
            the instance is unbound at the end;
            Lea._gen_one_random_mc method is abstract: it is implemented in all Lea's subclasses
        '''
        raise NotImplementedError("missing method '%s._gen_one_random_mc(self)'"%(self.__class__.__name__))
            
    def gen_random_mc(self,n,nb_tries=None):
        ''' generates n random value from the current probability distribution,
            without precalculating the exact probability distribution (contrarily to 'random' method);
            nb_tries, if not None, defines the maximum number of trials in case a random value
            is incompatible with a condition; this happens only if the current Lea instance
            is (referring to) an Ilea or Blea instance, i.e. 'given' or 'cpt' methods;
            WARNING: if nb_tries is None, any infeasible condition shall cause an infinite loop
        '''
        for _ in range(n):
            remaining_nb_tries = 1 if nb_tries is None else nb_tries
            v = self
            while remaining_nb_tries > 0:
                try:
                    for v in self._gen_one_random_mc():
                        yield v
                    remaining_nb_tries = 0
                except Lea._FailedRandomMC:
                    if nb_tries is not None:
                        remaining_nb_tries -= 1        
            if v is self:
                raise Lea.Error("impossible to validate given condition(s), after %d random trials"%nb_tries) 
        
    def random_mc(self,n=None,nb_tries=None):
        ''' if n is None, returns a random value with the probability given by the distribution
            without precalculating the exact probability distribution (contrarily to 'random' method);
            otherwise, returns a tuple of n such random values;
            nb_tries, if not None, defines the maximum number of trials in case a random value
            is incompatible with a condition; this happens only if the current Lea instance
            is (referring to) an Ilea or Blea instance, i.e. 'given' or 'cpt' methods;
            WARNING: if nb_tries is None, any infeasible condition shall cause an infinite loop
        '''
        n1 = 1 if n is None else n
        random_mc_tuple = tuple(self.gen_random_mc(n1,nb_tries))
        if n is None:
            return random_mc_tuple[0]
        return random_mc_tuple
        
    def estimate_mc(self,n,nb_tries=None): 
        ''' returns an Alea instance, which is an estimation of the current distribution from a sample
            of n random values; this is a true Monte-Carlo algorithm, which does not precalculate the
            exact probability distribution (contrarily to 'random' method); 
            the method is suited for complex distributions, when calculation of exact probability
            distribution is intractable; the larger the value of n, the better the returned estimation;
            nb_tries, if not None, defines the maximum number of trials in case a random value
            is incompatible with a condition; this happens only if the current Lea instance
            is (referring to) an Ilea or Blea instance, i.e. 'given' or 'cpt' methods;
            WARNING: if nb_tries is None, any infeasible condition shall cause an infinite loop
        '''
        return Alea.vals(*self.random_mc(n,nb_tries))
    
    def nb_cases(self):
        ''' returns the number of atomic cases evaluated to build the exact probability distribution;
            this provides a measure of the complexity of the probability distribution 
        '''
        self._init_calc()
        return sum(1 for vp in self.gen_vp())

    def is_true(self):
        ''' returns True iff the value True has probability 1;
                    False otherwise;
            raises exception if some value is not boolean
        '''
        return self._p(True,check_val_type=True) == 1

    def is_feasible(self):
        ''' returns True iff the value True has a non-null probability;
                    False otherwise;
            raises exception if some value is not boolean
        '''
        return self._p(True,check_val_type=True) > 0
        
    def as_string(self,kind='/',nb_decimals=6,chart_size=100):
        ''' returns, after evaluation of the probability distribution self, a string
            representation of it;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability in a format depending of given kind, which is string among
            '/', '.', '%', '-', '/-', '.-', '%-'; 
            the probabilities are displayed as
            - if kind[0] is '/' : rational numbers "n/d" or "0" or "1"
            - if kind[0] is '.' : decimals with given nb_decimals digits
            - if kind[0] is '%' : percentage decimals with given nb_decimals digits
            - if kind[0] is '-' : histogram bar made up of repeated '-', such that
                                  a bar length of histo_size represents a probability 1 
            if kind[1] is '-', the histogram bars with '-' are appended after 
                               numerical representation of probabilities
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used
        '''        
        return self.get_alea().as_string(kind,nb_decimals,chart_size)

    def __str__(self):
        ''' returns, after evaluation of the probability distribution self, a string
            representation of it;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value  with its
            probability expressed as a rational number "n/d" or "0" or "1";
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used;
            called on evalution of "str(self)" and "repr(self)"
        '''        
        return self.get_alea().__str__()

    __repr__ = __str__

    def as_float(self,nb_decimals=6):
        ''' returns, after evaluation of the probability distribution self, a string
            representation of it;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability expressed as decimal with given nb_decimals digits;
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used;
        '''
        return self.get_alea().as_float(nb_decimals)

    def as_pct(self,nb_decimals=1):
        ''' returns, after evaluation of the probability distribution self, a string
            representation of it;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability expressed as percentage with given nb_decimals digits;
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used;
        '''
        return self.get_alea().as_pct(nb_decimals)
    
    def histo(self,size=100):
        ''' returns, after evaluation of the probability distribution self, a string
            representation of it;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability expressed as a histogram bar made up of repeated '-',
            such that a bar length of given size represents a probability 1
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used;
        '''
        return self.get_alea().histo(size)

    def plot(self,title=None,fname=None,savefig_args=dict(),**bar_args):
        ''' produces, after evaluation of the probability distribution self,
            a matplotlib bar chart representing it with the given title (if not None);
            the bar chart may be customised by using named arguments bar_args, which are
            relayed to matplotlib.pyplot.bar function
            (see doc in http://matplotlib.org/api/pyplot_api.html)
            * if fname is None, then the chart is displayed on screen, in a matplotlib window;
              the previous chart, if any, is erased
            * otherwise, the chart is saved in a file specified by given fname as specified
              by matplotlib.pyplot.savefig; the file format may be customised by using
              savefig_args argument, which is a dictionary relayed to matplotlib.pyplot.savefig
              function and containing named arguments expected by this function;
              example:
               flip.plot(fname='flip.png',savefig_args=dict(bbox_inches='tight'),color='green')
            the method requires matplotlib package; an exception is raised if it is not installed
        '''
        self.get_alea().plot(title,fname,savefig_args,**bar_args)

    def get_alea(self,**kwargs):
        ''' returns an Alea instance representing the distribution after it has been evaluated;
            if self is an Alea instance, then it returns itself,
            otherwise the newly created Alea is cached : the evaluation occurs only for the first
            call; for successive calls, the cached Alea instance is returned, which is faster 
        '''
        if self._alea is None:
            self._alea = self.new(**kwargs)
        return self._alea

    def new(self,prob_type=-1,**kwargs):
        ''' returns a new Alea instance representing the distribution after it has been evaluated;
            if self is an Alea, then it returns a clone of itself representing an independent event;
            the probability type used in the returned instance depends on given prob_type:
            if prob_type is -1, then the probability type is the same as self's
            otherwise, the probability type is defined using prob_type (see doc of Alea.set_prob_type);
            note that the present method is overloaded in Alea class, to be more efficient
        '''
        self._init_calc()
        return Alea.pmf(tuple(self.gen_vp()),prob_type=prob_type,**kwargs)

    def cumul(self):
        ''' evaluates the distribution, then,
            returns a tuple with probabilitys p that self <= value ;
            the sequence follows the order defined on values (if an order relationship is defined
            on values, then the tuples follows their increasing order; otherwise, an arbitrary
            order is used, fixed from call to call
            Note : the returned value is cached
        '''
        return tuple(Alea._downcast(p) for p in self.get_alea().cumul())
        
    def inv_cumul(self):
        ''' evaluates the distribution, then,
            returns a tuple with the probabilitys p that self >= value ;
            the sequence follows the order defined on values (if an order relationship is defined
            on values, then the tuples follows their increasing order; otherwise, an arbitrary
            order is used, fixed from call to call
            Note : the returned value is cached
        '''
        return tuple(Alea._downcast(p) for p in self.get_alea().inv_cumul())
        
    def random_iter(self):
        ''' evaluates the distribution, then,
            generates an infinite sequence of random values among the values of self,
            according to their probabilities
        '''
        return self.get_alea()._random_iter
        
    def random(self,n=None):
        ''' evaluates the distribution, then, 
            if n is None, returns a random value with the probability given by the distribution
            otherwise, returns a tuple of n such random values
        '''
        if n is None:
            return self.get_alea().random_val()
        return tuple(islice(self.random_iter(),int(n)))

    def random_draw(self,n=None,sorted=False):
        ''' evaluates the distribution, then,
            if n=None, then returns a tuple with all the values of the distribution, in a random order
                       respecting the probabilities (the higher probability of a value, the most likely
                       the value will be in the beginning of the sequence)
            if n > 0,  then returns only n different drawn values
            if sorted is True, then the returned tuple is sorted
        '''
        return self.get_alea().random_draw(n,sorted)

    @staticmethod
    def joint_entropy(*args):
        ''' returns the joint entropy of arguments, expressed in bits;
            the returned type is a float or a sympy expression (see doc of
            Alea.entropy)
        '''
        return Clea(*args).entropy

    def cond_entropy(self,other):
        ''' returns the conditional entropy of self given other, expressed in
            bits; note that this value is also known as the equivocation of
            self about other;
            the returned type is a float or a sympy expression (see doc of
            Alea.entropy)
        '''
        other = Alea.coerce(other)
        ce = Clea(self,other).entropy - other.entropy
        try:
            return max(0.0,ce)
        except:
            return ce

    @staticmethod
    def mutual_information(lea1,lea2):
        ''' returns the mutual information between given arguments, expressed
            in bits;
            the returned type is a float or a sympy expression (see doc of
            Alea.entropy)
        '''
        lea1 = Alea.coerce(lea1)
        lea2 = Alea.coerce(lea2)
        mi = lea1.entropy + lea2.entropy - Clea(lea1,lea2).entropy
        try:
            return max(0.0,mi)
        except:
            return mi

    def information_of(self,val):
        ''' returns the information of given val, expressed in bits;
            the returned type is a float or a sympy expression (see doc of
            Alea.entropy);
            raises an exception if given val has a null probability
        '''
        return self.get_alea().information_of(val)

    def lr(self,*hyp_leas):
        ''' returns a float giving the likelihood ratio (LR) of an 'evidence' E,
            which is self, for a given 'hypothesis' H, which is the AND of given
            hyp_leas arguments; it is calculated as 
                  P(E | H) / P(E | not H)
            both E and H must be boolean probability distributions, otherwise
            an exception is raised;
            an exception is raised also if H is certainly true or certainly false      
        '''
        return self.given(*hyp_leas).lr()

    def internal(self,indent='',refs=None):
        ''' returns a string representing the inner definition of self, with
            children leas recursively up to Alea leaves; if the same lea child
            appears multiple times, it is expanded only on the first occurrence,
            the other ones being marked with reference id;
            the arguments are used only for recursive calls, they can be ignored
            for a normal usage;
            note: this method is overloaded in Alea class
        '''
        if refs is None:
            refs = set()
        if self in refs:
            args = [self._id()+'*']
        else:
            refs.add(self)
            args = [self._id()]
            for attr_name in self.__slots__:
                attr_val = getattr(self,attr_name)
                if isinstance(attr_val,Lea):
                    args.append(attr_val.internal(indent+'  ',refs))
                elif isinstance(attr_val,tuple):
                    args1 = ['(']
                    for lea1 in attr_val:
                        args1.append(lea1.internal(indent+'    ',refs))
                    args.append(('\n'+indent+'    ').join(args1)+'\n'+indent+'  )')
                elif hasattr(attr_val,'__call__'):
                    args.append(attr_val.__module__+'.'+attr_val.__name__)
        return ('\n'+indent+'  ').join(args)

# import modules with Lea subclasses
from .alea import Alea
from .clea import Clea
from .ilea import Ilea
from .rlea import Rlea
from .blea import Blea
from .flea import Flea
from .flea1 import Flea1
from .flea2 import Flea2
from .flea2a import Flea2a
from .glea import Glea
from .tlea import Tlea

# init Alea class with default 'x' type code: if a probability is expressed as
# a string, then the target type is determined from its content
# - see Alea.prob_any method
Alea.set_prob_type('x')

# convenience functions

def P(lea1):
    ''' returns a ProbFraction instance representing the probability for
        lea1 to be True, expressed in the type used in lea1 definition;
        raises an exception if some value in the distribution is not boolean
        (this is NOT the case with lea1.p(True));
        this is a convenience function equivalent to lea1.P
    '''
    return lea1.P

def Pf(lea1):
    ''' returns a ProbFraction instance representing the probability for
        lea1 to be True, expressed as a float between 0.0 and 1.0;
        raises an exception if some value in the distribution is not boolean
        (this is NOT the case with lea1.p(True));
        this is a convenience function equivalent to lea1.Pf
    '''
    return lea1.Pf
