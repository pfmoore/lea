'''
--------------------------------------------------------------------------------

    lea.py

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

import operator
import sys
from itertools import islice
from collections import namedtuple
from .prob_fraction import ProbFraction
from .toolbox import calcGCD, easyMin, easyMax, readCSVFilename, readCSVFile, dict, zip, isclose

class Lea(object):
    
    '''
    Lea is an abstract class representing discrete probability distributions.

    Each instance of concrete Lea's subclasses (called simply a "Lea instance" in the following)
    represents a discrete probability distribution, which associates each value of a set of
    values with the probability that such value occurs.

    A Lea instance can be defined by a sequence of (value,weight), giving the probability weight 
    of each value. Such probability weights are natural numbers. The actual probability of a
    given value can be calculated by dividing a weight by the sum of all weights. A Lea instance
    can be defined also by a sequence of values, their probability weight being their number of
    occurences in the sequence.

    Lea instances can be combined in arithmetic expressions resulting in new Lea instances, by
    obeying the following rules:

    - Lea instances can be added, subtracted, multiplied and divided together,
    through +, -, *, /, // operators. The resulting distribution's values and probabilities
    are determined by combination of operand's values with a sum weighted by probability
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

    WARNING: the Python's and, or, not, operators shall NOT be used on Lea instances because
    they do not return any sensible result. Replace:
           a and b    by    a & b
           a or b     by    a | b
           not a      by    ~ a

    WARNING: in boolean expression involving arithmetic comparisons, the parenthesis
    shall be used, e.g. (a < b) & (b < c)

    WARNING: the augmented comparison (a < b < c) expression shall NOT be used.; it does
    not return any sensible result (reason: it has the same limitation as 'and' operator).

    Lea instances can be used to generate random values, respecting the given probabilities.
    There are two Lea methods for this purpose:
    - random:   calculates the exact probabiliy distribution, then takes random values 
    - randomMC: takes random values from atomic probabiliy distribution, then makes the 
                required calculations (Monte-Carlo algorithm)
    The randomMC is suited for complex distributions, when calculation of exact probability
    distribution is intractable. This could be used to provide an estimation of the probability
    distribution (see estimateMC method).

    There are nine concrete subclasses to Lea, namely:
      Alea, Clea, Flea, Flea1, Flea2, Glea, Ilea, Rlea and Blea.
    
    Each subclass represents a "definition" of discrete probability distribution, with its own data
    or with references to other Lea instances to be combined together through a given operation.
    Each subclass defines what are the (value,probability) pairs or how they can be generated (see
    _genVPs method implemented in each Lea subclass). The Lea class acts as a facade, by providing
    different methods to instantiate these subclasses, so it is usually not needed to instantiate
    them explicitely. Here is an overview on these subclasses, with their relationships.

    - An Alea instance is defined by explicit value-probability pairs, that is a probability mass
    function.

    Instances of other Lea's subclasses represent probability distributions obtained by operations
    done on existing Lea instance(s). Any such instance forms a tree structure, having other Lea
    instances as nodes and Alea instances as leaves. This uses "lazy evaluation": actual (value,
    probability) pairs are calculated only at the time they are required (e.g. display, query 
    probability of a given value, etc); then, these are aggregated in a new Alea instance. This 
    Alea instance is then cached, as an attribute of the queried Lea instance, for speeding up next
    queries.
    
    Here is a brief presentation of these Lea's subclasses: 

    - Clea provides the cartesian product of a given sequence of Lea instances
    - Flea applies a given n-ary function to a given sequence of n Lea instances
    - Flea1 applies a given 1-ary function to a given Lea instance
    - Flea2 applies a given 2-ary function to two given Lea instances
    - Glea applies n-ary functions present in a given Lea instance to a given sequence of n Lea instances
    - Ilea filters the values of a given Lea instance according to a given Lea instance representing a boolean condition (conditional probabilities)
    - Rlea embeds Lea instances as values of a parent Lea instance 
    - Blea defines CPT, providing Lea instances corresponding to given conditions (used for bayesian networks)

    Note that Flea1 and Flea2 are more efficient alternatives to Flea-based implementation.

    WARNING: The following methods are called without parentheses:
        mean, var, std, mode, entropy, information
    These are applicable on any Lea instance; these are implemented and documented in the Alea class.

    Short design notes:
    Lea uses the "template method" design pattern: the Lea base abstract class calls the following methods,
    which are implemented in each Lea's subclass: _clone, _getLeaChildren, _genVPs and _genOneRandomMC.
    Excepting the afore-mentioned estimateMC method, Lea performs EXACT calculation of probability distributions.
    It implements an original algorithm, called the "Statues" algorithm, by reference to the game of the same name;
    this uses a variable binding mechanism that relies on Python's generators. To learn more, see doc of
    Alea._genVPs method as well as other Xlea._genVPs methods implemented in Lea's subclasses. 
    '''

    class Error(Exception):
        ''' exception representing any violation of requirements of Lea methods  
        '''
        pass
        
    class _FailedRandomMC(Exception):
        ''' internal exception representing a failure to get a set of random values that
            satisfy a given condition in a given number of trials (see '...MC' methods) 
        '''
        pass

    # Lea attributes
    __slots__ = ('_alea','_val','genVPs')

    # a mutable object, which cannnot appear in Lea's values (not hashable)
    _DUMMY_VAL = []

    # constructor methods
    # -------------------

    def __init__(self):
        ''' initializes Lea instance's attributes
        '''
        # alea instance acting as a cache when actual value-probability pairs have been calculated
        self._alea = None
        # _val is the value temporarily bound to the instance, during evaluation (see _genBoundVPs method)
        # note: self is used as a sentinel value to express that no value is currently bound; Python's
        # None is not a good sentinel value since it prevents using None as value in a distribution
        self._val = self
        # when evaluation is needed, genVPs shall be bound on _genVPs or _genBoundVPs method
        # (see _initCalc method)
        self.genVPs = None

    def _id(self):
        ''' returns a unique id, containing the concrete Lea class name as prefix
        '''
        return '%s#%s'%(self.__class__.__name__,id(self))

    def getAleaLeavesSet(self):
        ''' returns a set containing all the Alea leaves in the tree having the root self;
            this calls _getLeaChildren() method implemented in Lea's subclasses;
            this method is overloaded in Alea subclass to stop the recursion
        '''
        return frozenset(aleaLeaf for leaChild in self._getLeaChildren() for aleaLeaf in leaChild.getAleaLeavesSet())

    # constructor methods
    # -------------------
             
    def clone(self,cloneTable=None):
        ''' returns a deep copy of current Lea, without any value binding;
            if the Lea tree contains multiple references to the same Lea instance,
            then it is cloned only once and the references are copied in the cloned tree
            (the cloneTable dictionary serves this purpose);
            the method calls the _clone() method implemented in Lea's subclasses
        '''
        if cloneTable is None:
            cloneTable = dict()
        clonedLea = cloneTable.get(self)
        if clonedLea is None:
            clonedLea = self._clone(cloneTable)
            cloneTable[self] = clonedLea
            if self._alea is not None:
                clonedLea._alea = self._alea.clone(cloneTable)
        return clonedLea

    def _genBoundVPs(self):
        ''' generates tuple (v,p) where v is a value of the current probability distribution
            and p is the associated probability weight (integer > 0);
            this obeys the "binding" mechanism, so if the same variable is referred multiple times in
            a given expression, then same value will be yielded at each occurrence;
            "Statues" algorithm:
            before yielding a value v, this value v is bound to the current instance;
            then, if the current calculation requires to get again values on the current
            instance, then the bound value is yielded with probability 1;
            the instance is rebound to a new value at each iteration, as soon as the execution
            is resumed after the yield;
            it is unbound at the end;
            the method calls the _genVPs method implemented in Lea subclasses;
            the present Alea._genVPs method is called by the _genVPs methods implemented in
            other Lea subclasses; these methods are themselves called by Lea.new and,
            indirectly, by Lea.getAlea
        '''
        if self._val is not self:
            # distribution already bound to a value because genVPs has been called already on self
            # it is yielded as a certain distribution (unique yield)
            yield (self._val,1)
        else:
            # distribution not yet bound to a value
            try:
                # browse all (v,p) tuples
                for (v,p) in self._genVPs():
                    # bind value v: this is important if an object calls genVPs on the same instance
                    # before resuming the present generator (see above)
                    self._val = v
                    # yield the bound value v with probability weight p
                    yield (v,p)
            finally:
                # unbind value v, after all values have been bound or if an exception has been raised
                self._val = self

    def _resetGenVPs(self):
        ''' set genVPs = None on self and all Lea descendants
        '''
        self.genVPs = None
        # treat children recursively, up to Alea instances
        for leaChild in self._getLeaChildren():
            leaChild._resetGenVPs()

    def _setGenVPs(self):
        ''' prepare calculation of probability distribution by binding self.genVPs to the most adequate method:
            self.genVPs is bound
             either on self._genVPs method, if no binding is required (single occurrence of self in expression)
             or on self._genBoundVPs method, if binding is required (multiple occurrences of self in expression)
            note: self._genBoundVPs works in any case but perform unnecessary binding job if self occurrence is
            unique in the evaluated expression
            requires that genVPs = None for self and all Lea descendants
        '''
        if self.genVPs is None:
            # first occurrence of self in the expression: use the simplest _genVPs method
            # this may be overwritten if a second occurrence is found
            self.genVPs = self._genVPs
        elif self.genVPs == self._genVPs:
            # second occurrence of self in the expression: use the _genBoundVPs method
            self.genVPs = self._genBoundVPs
        # treat children recursively, up to Alea instances
        for leaChild in self._getLeaChildren():
            leaChild._setGenVPs()

    def _initCalc(self):
        ''' prepare calculation of probability distribution by binding self.genVPs to the most adequate method;
            see _setGenVPs method
        '''
        self._resetGenVPs()
        self._setGenVPs()

    def withProb(self,condLea,pNum,pDen=None):
        ''' returns a new Alea instance from current distribution,
            such that pNum/pDen is the probability that condLea is true
            if pDen is None, then pNum expresses the probability as a Fraction
        '''
        curCondLea = Lea.coerce(condLea)
        reqCondLea = Lea.boolProb(pNum,pDen)
        if reqCondLea.isTrue():
            lea1 = self.given(curCondLea)
        elif not reqCondLea.isFeasible():
            lea1 = self.given(~curCondLea)
        else:    
            lea1 = Blea.if_(reqCondLea,self.given(curCondLea).getAlea(sorting=False),self.given(~curCondLea).getAlea(sorting=False))
        return lea1.getAlea()
        
    def withCondProb(self,condLea,givenCondLea,pNum,pDen):
        ''' [DEPRECATED: use Lea.revisedWithCPT instead]
            returns a new Alea instance from current distribution,
            such that pNum/pDen is the probability that condLea is true
            given that givenCondLea is True, under the constraint that
            the returned distribution keeps prior probabilities of condLea
            and givenCondLea unchanged
        '''
        if not (0 <= pNum <= pDen):
            raise Lea.Error("%d/%d is outside the probability range [0,1]"%(pNum,pDen))
        condLea = Lea.coerce(condLea)
        givenCondLea = Lea.coerce(givenCondLea)
        # max 2x2 distribution (True,True), (True,False), (False,True), (True,True)
        # prior joint probabilities, non null probability
        d = self.map(lambda v:(condLea.isTrue(),givenCondLea.isTrue())).getAlea()
        e = dict(d._genVPs())
        eTT = e.get((True,True),0)
        eFT = e.get((False,True),0)
        eTF = e.get((True,False),0)
        eFF = e.get((False,False),0)
        nCondLeaTrue = eTT + eTF
        nCondLeaFalse = eFT + eFF
        nGivenCondLeaTrue = eTT + eFT
        # new joint probabilities
        nTT = nGivenCondLeaTrue*pNum
        nFT = nGivenCondLeaTrue*(pDen-pNum)
        nTF = nCondLeaTrue*pDen - nTT
        nFF = nCondLeaFalse*pDen - nFT
        # feasibility checks
        if eTT == 0 and nTT > 0:
            raise Lea.Error("unfeasible: probability shall remain 0")
        if eFT == 0 and nFT > 0:
            raise Lea.Error("unfeasible: probability shall remain 1")
        if eTF == 0 and nTF > 0:
            raise Lea.Error("unfeasible: probability shall remain %d/%d"%(nCondLeaTrue,nGivenCondLeaTrue)) 
        if eFF == 0 and nFF > 0:
            msg = "unfeasible"
            if nGivenCondLeaTrue >= nCondLeaTrue:
                msg += ": probability shall remain %d/%d"%(nGivenCondLeaTrue-nCondLeaTrue,nGivenCondLeaTrue)
            raise Lea.Error(msg)
        if nTF < 0 or nFF < 0:
            pDenMin = nGivenCondLeaTrue
            pNumMin = max(0,nGivenCondLeaTrue-nCondLeaFalse)
            pDenMax = nGivenCondLeaTrue
            pNumMax = min(pDenMax,nCondLeaTrue)
            gMin = calcGCD(pNumMin,pDenMin)
            gMax = calcGCD(pNumMax,pDenMax)
            pNumMin //= gMin 
            pDenMin //= gMin 
            pNumMax //= gMax 
            pDenMax //= gMax
            raise Lea.Error("unfeasible: probability shall be in the range [%d/%d,%d/%d]"%(pNumMin,pDenMin,pNumMax,pDenMax))
        w = { (True  , True ) : nTT,
              (True  , False) : nTF,
              (False , True ) : nFT,
              (False , False) : nFF }
        m = 1
        for r in e.values():
            m *= r      
        # factors to be applied on current probabilities
        # depending on the truth value of (condLea,givenCondLea) on each value
        w2 = dict((cg,w[cg]*(m//ecg)) for (cg,ecg) in e.items())
        return Alea.fromValFreqs(*((v,p*w2[(condLea.isTrue(),givenCondLea.isTrue())]) for (v,p) in self.genVPs()))

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
        return Ilea(self,(Lea.coerce(evidence) for evidence in evidences))    
    
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
        return self.getAlea().times(n,op)

    def timesTuple(self,n):
        ''' returns a new Alea instance with tuples of length n, containing
            the cartesian product of self with itself repeated n times
            note: equivalent to self.draw(n,sorted=False,replacement=True)
        '''
        return self.getAlea().drawWithReplacement(n)

    def cprod(self,*args):
        ''' returns a new Clea instance, representing the cartesian product of all
            arguments (coerced to Lea instances), including self as first argument 
        '''
        return Clea(self,*args)

    @staticmethod
    def reduce(op,args,absorber=None):
        ''' static method, returns a new Flea2 instance that join the given args with
            the given function op, from left to right;
            requires that op is a 2-ary function, accepting self's values as arguments;
            requires that args contains at least one element
            if absorber is not None, then it is considered as a "right-absorber" value
            (i.e. op(x,absorber) = absorber); this activates a more efficient algorithm
            which prunes the tree search as soon as the absorber is met.
        '''
        argsIter = iter(args)
        res = next(argsIter)
        if absorber is None:
            for arg in argsIter:
                res = Flea2(op,res,arg)
        else:
            for arg in argsIter:
                res = Flea2a(op,res,arg,absorber)
        return res

    def merge(self,*leaArgs):
        ''' returns a new Blea instance, representing the merge of self and given leaArgs, i.e.
                  P(v) = (P1(v) + ... + Pn(v)) / n
            where P(v)  is the probability of value v in the merge result 
                  Pi(v) is the probability of value v in ((self,)+leaArgs)[i]
        '''
        leas = (self,) + leaArgs
        lea = Lea.fromSeq(range(len(leas)))
        return Blea.build(*((lea==i,leaArg) for (i,leaArg) in enumerate(leas)))
    
    def map(self,f,args=()):
        ''' returns a new Flea instance representing the distribution obtained
            by applying the given function f, taking values of self distribution
            as first argument and given args tuple as following arguments (expanded);
            requires that f is a n-ary function with 1 <= n = len(args)+1 
            note: f can be also a Lea instance, with functions as values
        '''
        return Flea.build(f,(self,)+args)

    def mapSeq(self,f,args=()):
        ''' returns a new Flea instance representing the distribution obtained
            by applying the given function f on each element of each value
            of self distribution; if args is not empty, then it is expanded
            and added as f arguments
            requires that f is a n-ary function with 1 <= n = len(args)+1 
            requires that self's values are sequences
            returned distribution values are tuples
            note: f can be also a Lea instance, with functions as values
        '''
        return self.map(lambda v: tuple(f(e,*args) for e in v))

    def asJoint(self,*attrNames):
        ''' returns a new Alea instance representing a joint probability distribution
            from the current distribution supposed to have n-tuples as values,
            to be associated with the given n attribute names
        '''
        NTClass = namedtuple('_',attrNames)
        return self.map(lambda aTuple: NTClass(*aTuple)).getAlea()

    def isUniform(self):
        ''' returns, after evaluation of the probability distribution self,
            True  if the probability distribution is uniform,
            False otherwise
        '''
        return self.getAlea().isUniform()

    def draw(self,n,sorted=False,replacement=False):
        ''' returns, after evaluation of the probability distribution self,
            a new Alea instance representing the probability distribution
            of drawing n elements from self
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
        '''
        if n < 0:
            raise Lea.Error("draw method requires a positive integer")
        alea1 = self.getAlea()
        if replacement:
            if sorted:
                # draw sorted with replacement
                return alea1.drawSortedWithReplacement(n)
            else:
                # draw sorted without replacement
                return alea1.drawWithReplacement(n)
        else:
            if len(alea1._vs) < n:
                raise Lea.Error("number of values to draw without replacement (%d) exceeds the number of possible values (%d)"%(n,len(alea1._vs)))
            if sorted:
                # draw sorted without replacement
                return alea1.drawSortedWithoutReplacement(n)
            else:
                # draw unsorted without replacement
                return alea1.drawWithoutReplacement(n)

    def flat(self):
        ''' assuming that self's values are themselves Lea instances,
            returns a new Rlea instance representing a probability distribution of
            inner values of these Lea instances  
        '''
        return Rlea(self)
    '''
    @staticmethod
    def coerce(value):
        '' static method, returns a Lea instance corresponding the given value:
            if the value is a Lea instance, then it is returned
            otherwise, an Alea instance is returned, with given value
            as unique (certain) value
        ''
        if not isinstance(value,Lea):
            return Alea((value,),(1,))
        return value
    '''

    def equiv(self,other,prec=None):
        ''' returns True iff self and other represent the same probability distribution,
            i.e. they have the same probability for each of their value
            returns False otherwise
        '''
        other = Lea.coerce(other)
        # absolute equality required -> quick test
        # frozenset(...) is used to avoid any dependency on the order of values
        return frozenset(self.vps()) == frozenset(other.vps())

    def equivF(self,other): #prec=sys.float_info.epsilon):
        ''' returns True iff self and other represent the same probability distribution,
            i.e. they have the same probability for each of their value
            returns False otherwise
        '''
        other = Lea.coerce(other)
        vps1 = tuple(self.vps())
        vps2Dict = dict(other.vps())
        if len(vps1) != len(vps2Dict):
            return False
        for (v1,p1) in vps1:
            p2 = vps2Dict.get(v1)
            if p2 is None:
                return False
            #if abs(p1-p2) > prec:
            if not isclose(p1,p2):
                return False
        return True

    def p(self,val=None):
        ''' returns a ProbFraction instance representing the probability of given value val,
            from 0/1 to 1/1
            if val is None, then a tuple is returned with the probabilities of each value,
            in the same order as defined on values (call vals method to get this 
            ordered sequence)
        '''
        if val is None:
            return tuple(self.ps())
        return self._p(val)
 
    def vps(self):
        ''' generates, after evaluation of the probability distribution self,
            tuples (v,p) where v is a value of self
            and p is the associated probability weight (integer > 0);
            the sequence follows the order defined on values
            not that there is NO binding, contrarily to _genVPs method
        '''
        return self.getAlea().vps()

    def vals(self):
        ''' returns a tuple with values of self
            the sequence follows the increasing order defined on values
            if order is undefined (e.g. complex numbers), then the order is
            arbitrary but fixed from call to call
        '''
        return self.getAlea()._vs

    def ps(self):
        ''' returns a tuple with probability weights (integer > 0) of self
            the sequence follows the increasing order defined on values
            if order is undefined (e.g. complex numbers), then the order is
            arbitrary but fixed from call to call
        '''
        return self.getAlea()._ps
        
    def support(self):
        ''' same as vals method
        '''
        return self.vals()
              
    def pmf(self,val=None):
        ''' probability mass function
            returns the probability of the given value val, as a floating point number
            from 0.0 to 1.0
            if val is None, then a tuple is returned with the probabilities of each value,
            in the same order as defined on values (call vals method to get this 
            ordered sequence)
        '''
        # TODO: float vs frac , see cdf
        if val is None:
            return tuple(float(p) for p in self.ps())
        return self._p(val)

    def cdf(self,val=None):
        ''' cumulative distribution function
            returns the probability that self's value is less or equal to the given value val,
            as a floating point number from 0.0 to 1.0
            if val is None, then a tuple is returned with the probabilities of each value,
            in the same order as defined on values (call vals method to get this 
            ordered sequence); the last probability is always 1.0 
        '''
        if val is None:
            return tuple(self.cumul()[1:])
        return self.getAlea().pCumul(val)

    def _p(self,val,checkValType=False):
        ''' returns the probability p/s of the given value val, as a tuple of naturals (p,s)
            where
            s is the sum of the probability weights of all values 
            p is the probability weight of the given value val (from 0 to s)
            note: the ratio p/s is not reduced
            if checkValType is True, then raises an exception if some value in the
            distribution has a type different from val's
        '''
        return self.getAlea()._p(val,checkValType)

    def sortBy(self,*orderingLeas):
        ''' returns an Alea instance representing the same probability distribution as self
            but having values ordered according to given orderingLeas;
            requires that self doesn't contain duplicate values, otherwise an exception is
            raised; note that it is NOT required that all orderingLeas appear in self 
        '''
        # after prepending orderingLeas to self, the Alea returned by new() is sorted with orderingLeas;
        # then, extracting self (index -1) allows generating self's (v,p) pairs in the expected order;
        # these shall be used to create a new Alea, keeping the values in that order (no sort)
        return Alea._fromValFreqsOrdered(*Lea.cprod(*orderingLeas).cprod(self).new()[-1].genVPs())

    def isAnyOf(self,*values):
        ''' returns a boolean probability distribution
            indicating the probability that a value is any of the values passed as arguments
        '''
        return Flea1(lambda v: v in values,self)

    def isNoneOf(self,*values):
        ''' returns a boolean probability distribution
            indicating the probability that a value is none of the given values passed as arguments 
        '''
        return Flea1(lambda v: v not in values,self)

    @staticmethod
    def buildCPTFromDict(aCPTDict,priorLea=None):
        ''' static method, same as buildCPT, with clauses specified in the aCPTDict dictionary
            {condition:result}
        '''
        return Blea.build(*(aCPTDict.items()),priorLea=priorLea)

    @staticmethod
    def if_(condLea,thenLea,elseLea):
        ''' static method, returns an instance of Tlea representing the
            conditional probability table
            giving thenLea  if condLea is true
                   elseLea  otherwise
            this is a convenience method equivalent to 
              condLea.switch({True:thenLea,False:elseLea})
            Note: before ver 2.3, it was equivalent to
              Lea.buildCPT((condLea,thenLea),(None,elseLea))
        '''
        return Tlea(condLea,{True:thenLea,False:elseLea})
        ## before version 2.3: Blea.build((condLea,thenLea),(None,elseLea))

    def switch(self,leaDict,defaultLea=_DUMMY_VAL):
        ''' returns an instance of Tlea representing a conditional probability table (CPT)
            defined by the given dictionary leaDict associating each value of self to a
            specific Lea instance;
            if defaultLea is given, then it provides the Lea instance associated to the
            value(s) of self missing in leaDict;
            all dictionary's values and defaultLea (if defined) are cast to Lea instances
        '''
        return Tlea(self,leaDict,defaultLea)

    ## note: in PY3, could use:
    ## def buildCPT(*clauses,priorLea=None,autoElse=False,check=True):
    @staticmethod
    def buildCPT(*clauses,**kwargs):
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
             'priorLea', 'autoElse', 'check' and 'ctxType';
            'priorLea' and 'autoElse' are mutually exclusive, they require the absence
            of an 'else' clause (otherwise, an exception is raised); 
            * if priorLea argument is specified, then the 'else' clause is calculated
            so that the priorLea is returned for the unconditional case
            * if autoElse argument is specified as True, then the 'else' clause is
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
              2) if 'else' is missing and not calculated through 'priorLea' nor 'autoElse',
                 then the clause conditions shall cover all possible cases, i.e. ORing
                 them shall be certainly true;
            an exception is raised if any of such conditions is not verified;
        '''
        return Blea.build(*clauses,**kwargs)

    def revisedWithCPT(self,*clauses):
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
        return Blea.build(*clauses,priorLea=self)
    
    def buildBNfromJoint(self,*bnDefinition):
        ''' returns a named tuple of Blea instances representing a Bayes network
            with variables stored in attributes A1, ... , An, assuming that self
            is a Lea joint probability distribution having, as values, named tuples
            with the same set of attributes A1, ... , An (such Lea instance is
            returned by asJoint method, for example);
            each argument of given bnDefinition represent a dependency relationship
            from a set of given variables to one given variable; this is expressed as
            a tuple (srcVarNames, tgtVarName) where srcVarNames is a sequence of
            attribute names (strings) identifying 'from' variables and tgtName is the
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
        jointAlea = self.getAlea()
        # retrieve the named tuple class from the first value of the joint distribution,
        NamedTuple = jointAlea._vs[0].__class__
        varsDict = dict((varName,self.__getattribute__(varName)) for varName in NamedTuple._fields)
        # all BN variables initialized as independent (maybe overwritten below, according to given relationships)
        varsBNDict = dict((varName,var.getAlea(sorting=False)) for (varName,var) in varsDict.items())
        for (srcVarNames,tgtVarName) in bnDefinition:
            if not isinstance(varsBNDict[tgtVarName],Alea):
                raise Lea.Error("'%s' is defined as target in more than one BN relationship"%tgtVarName)
            tgtVar = varsDict[tgtVarName]
            cprodSrcVars = Lea.cprod(*(varsDict[srcVarName] for srcVarName in srcVarNames))
            cprodSrcVarsBN = Lea.cprod(*(varsBNDict[srcVarName] for srcVarName in srcVarNames))
            ## Note: the following statement is functionnaly equivalent to statements after
            ## BUT it is far slower in case of missing conditions on the CPT ('else' clause)
            ## because it relies on the (too) generic 'autoElse' algorithm of Blea, which makes 
            ## a negated OR on all given conditions;
            ## varsBNDict[tgtVarName] = Blea.build(
            ##                *((cprodSrcVarsBN==cprodSrcVal,tgtVar.given(cprodSrcVars==cprodSrcVal).getAlea()) \
            ##                   for cprodSrcVal in cprodSrcVars.vals()), autoElse=True)
            # build CPT clauses (condition,result) from the joint probability distribution
            cprodSrcVals = cprodSrcVars.vals()
            clauses = tuple((cprodSrcVarsBN==cprodSrcVal,tgtVar.given(cprodSrcVars==cprodSrcVal).getAlea(sorting=False)) \
                             for cprodSrcVal in cprodSrcVals)
            # determine missing conditions in the CPT, if any
            allVals = Lea.cprod(*(varsDict[srcVarName].getAlea(sorting=False) for srcVarName in srcVarNames)).vals()
            missingVals = frozenset(allVals) - frozenset(cprodSrcVals)
            if len(missingVals) > 0:
                # there are missing conditions: add clauses with each of these conditions associating  
                # them with a uniform distribution built on the values found in results of other clauses
                # (principle of indifference)
                elseResult = Lea.fromVals(*frozenset(val for (cond,result) in clauses for val in result.vals()))
                clauses += tuple((cprodSrcVarsBN==missingVal,elseResult) for missingVal in missingVals)
                ## Note: the following statements are functionally equivalent to the statement above
                ## these aggregate missing condtions with an OR; the tests show similar performance,
                ## albeit a small decrease : that's why the former has been preferred;
                ## elseCond = Lea.reduce(operator.or_,(cprodSrcVarsBN==missingVal for missingVal in missingVals),True)
                ## clauses += ((elseCond,elseResult),)
            # overwrite the target BN variable (currently independent Alea instance), with a CPT built
            # up from the clauses determined from the joint probability distribution
            # the check is deactivated for the sake of performance; this is safe since, by construction,
            # the clauses conditions verify the "truth partioning" rules 
            # the ctxType is 2 for the sake of performance; this is safe since, by construction, the
            # clauses results are Alea instances and clause conditions refer to the same variable,
            # namely cprodSrcVarsBN  
            varsBNDict[tgtVarName] = Blea.build(*clauses,check=False,ctxType=2)
        # return the BN variables as attributes of a new named tuple having the same attributes as the
        # values found in self
        return NamedTuple(**varsBNDict)
    
    @staticmethod
    def makeVars(obj,tgtDict,prefix='',suffix=''):
        ''' retrieve attributes names A1, ... , An of obj and put associations 
            {V1 : obj.A1, ... , Vn : obj.An} in tgtDict dictionary
            where Vi is a variable name string built as prefix + Ai + suffix;
            obj is
            (a) either a named tuple with attributes A1, ... , An (as returned
            by buildBNfromJoint, for example)
            (b) or a Lea instances representing a joint probability distribution
            with the attributes A1, ... , An (such Lea instance is returned by
            asJoint method, for example);
            note: if the caller passes globals() as tgtDict, then the variables
            named Vi, refering to obj.Ai, shall be created in its scope, as
            a side-effect (this is the purpose of the method);
            warning: the method may silently overwrite caller's variables
        '''
        if isinstance(obj,Lea):
            # case (b)
            # retrieve the named tuple class from the first value of the joint distribution
            NamedTuple = obj.getAlea()._vs[0].__class__
        else:
            # case (a)
            NamedTuple = obj.__class__
        tgtDict.update((prefix+varName+suffix,obj.__getattribute__(varName)) for varName in NamedTuple._fields)       
    
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

    def __getattribute__(self,attrName):
        ''' returns the attribute with the given name in the current Lea instance;
            if the attribute name is a distribution indicator, then the distribution
            is evaluated and the indicator method is called; 
            if the attribute name is unknown as a Lea instance's attribute,
            then returns a Flea instance that shall retrieve the attibute in the
            values of current distribution; 
            called on evaluation of "self.attrName"
            WARNING: the following methods are called without parentheses:
                         mean, var, std, mode, entropy, information
                     these are applicable on any Lea instance
                     and these are documented in the Alea class
        '''
        try:
            if attrName in Alea.indicatorMethodNames:
                # indicator methods are called implicitely
                return object.__getattribute__(self.getAlea(),attrName)()
            # return Lea's instance attribute
            return object.__getattribute__(self,attrName)
        except AttributeError:
            # return new Lea made up of attributes of inner values
            return Flea2(getattr,self,attrName)

    @staticmethod
    def fastMax(*args):
        ''' static method, returns a new Alea instance giving the probabilities to
            have the maximum value of each combination of the given args;
            if some elements of args are not Lea instance, then these are coerced
            to an Lea instance with probability 1;
            the method uses an efficient algorithm (linear complexity), which is
            due to Nicky van Foreest; for explanations, see
            http://nicky.vanforeest.com/scheduling/cpm/stochasticMakespan.html
            Note: unlike most of Lea methods, the distribution returned by Lea.fastMax
            loses any dependency with given args; this could be important if some args
            appear in the same expression as Lea.max(...) but outside it, e.g.
            conditional probability expressions; this limitation can be avoided by
            using the Lea.max method; however, this last method can be
            prohibitively slower (exponential complexity)
        '''
        aleaArgs = tuple(Lea.coerce(arg).getAlea() for arg in args)
        return Alea.fastExtremum(Alea.pCumul,*aleaArgs)
    
    @staticmethod
    def fastMin(*args):
        ''' static method, returns a new Alea instance giving the probabilities to have
            the minimum value of each combination of the given args;
            if some elements of args are not Lea instances, then these are coerced
            to an Alea instance with probability 1;
            the method uses an efficient algorithm (linear complexity), which is
            due to Nicky van Foreest; for explanations, see
            http://nicky.vanforeest.com/scheduling/cpm/stochasticMakespan.html
            Note: unlike most of Lea methods, the distribution returned by Lea.fastMin
            loses any dependency with given args; this could be important if some args
            appear in the same expression as Lea.min(...) but outside it, e.g.
            conditional probability expressions; this limitation can be avoided by
            using the Lea.min method; however, this last method can be prohibitively
            slower (exponential complexity)
        '''
        aleaArgs = tuple(Lea.coerce(arg).getAlea() for arg in args)
        return Alea.fastExtremum(Alea.pInvCumul,*aleaArgs)
        
    @staticmethod
    def max(*args):
        ''' static method, returns a new Flea instance giving the probabilities to
            have the maximum value of each combination of the given args;
            if some elements of args are not Lea instances, then these are coerced
            to a Lea instance with probability 1;
            the returned distribution keeps dependencies with args but the 
            calculation could be prohibitively slow (exponential complexity);
            for a more efficient implemetation, assuming that dependencies are not
            needed, see Lea.fastMax method
        '''
        return Flea.build(easyMax,args)

    @staticmethod
    def min(*args):
        ''' static method, returns a new Flea instance giving the probabilities to
            have the minimum value of each combination of the given args;
            if some elements of args are not Lea instances, then these are coerced
            to a Lea instance with probability 1;
            the returned distribution keeps dependencies with args but the 
            calculation could be prohibitively slow (exponential complexity);
            for a more efficient implemetation, assuming that dependencies are not
            needed, see Lea.fastMin method
        '''
        return Flea.build(easyMin,args)

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
        return Flea2(Lea._safeAnd,self,other)

    def __rand__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical AND between the values of other and the values of self;
            called on evaluation of "other & self"
        '''
        return Flea2(Lea._safeAnd,other,self)

    def __or__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical OR between the values of self and the values of other;
            called on evaluation of "self | other"
        '''
        return Flea2(Lea._safeOr,self,other)

    def __ror__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical OR between the values of other and the values of self;
            called on evaluation of "other | self"
        '''
        return Flea2(Lea._safeOr,other,self)

    def __xor__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical XOR between the values of self and the values of other;
            called on evaluation of "self ^ other"
        '''
        return Flea2(Lea._safeXor,self,other)

    def __invert__(self):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical NOT of the values self;
            called on evaluation of "~self"
        '''
        return Flea1(Lea._safeNot,self)

    def __bool__(self):
        ''' raises an exception telling that Lea instance cannot be evaluated as a boolean
            called on evaluation of "bool(self)", "if self:", "while self:", etc
        '''
        raise Lea.Error("Lea instance cannot be evaluated as a boolean")

    # Python 2 compatibility
    __nonzero__ = __bool__

    @staticmethod
    def _checkBooleans(opMsg,*vals):
        ''' static method, raise an exception if any of vals arguments is not boolean;
            the exception messsage refers to the name of a logical operation given in the opMsg argument
        '''
        for val in vals:
            if not isinstance(val,bool):
                raise Lea.Error("non-boolean object involved in %s logical operation (maybe due to a lack of parentheses)"%opMsg) 

    @staticmethod
    def _safeAnd(a,b):
        ''' static method, returns a boolean, which is the logical AND of the given boolean arguments; 
            raises an exception if any of arguments is not boolean
        '''
        Lea._checkBooleans('AND',a,b)
        return operator.and_(a,b)

    @staticmethod
    def _safeOr(a,b):
        ''' static method, returns a boolean, which is the logical OR of the given boolean arguments; 
            raises an exception if any of arguments is not boolean
        '''
        Lea._checkBooleans('OR',a,b)
        return operator.or_(a,b)

    @staticmethod
    def _safeXor(a,b):
        ''' static method, returns a boolean, which is the logical XOR of the given boolean arguments; 
            raises an exception if any of arguments is not boolean
        '''
        Lea._checkBooleans('XOR',a,b)
        return operator.xor(a,b)

    @staticmethod
    def _safeNot(a):
        ''' static method, returns a boolean, which is the logical NOT of the given boolean argument; 
            raises an exception if the argument is not boolean
        '''
        Lea._checkBooleans('NOT',a)
        return operator.not_(a)    

    def _getLeaChildren(self):
        ''' returns a tuple containing all the Lea instances children of the current Lea;
            Lea._getLeaChildren method is abstract: it is implemented in all Lea's subclasses
        '''
        raise NotImplementedError("missing method '%s._getLeaChildren(self)'"%(self.__class__.__name__))

    def _clone(self,cloneTable):
        ''' returns a deep copy of current Lea, without any value binding;
            if the Lea tree contains multiple references to the same Lea instance,
            then it is cloned only once and the references are copied in the cloned tree
            (the cloneTable dictionary serves this purpose);
            Lea._clone method is abstract: it is implemented in all Lea's subclasses
        '''
        raise NotImplementedError("missing method '%s._clone(self,cloneTable)'"%(self.__class__.__name__))
        
    def _genVPs(self):
        ''' generates tuple (v,p) where v is a value of the current probability distribution
            and p is the associated probability weight (integer > 0);
            this obeys the "binding" mechanism, so if the same variable is refered multiple times in
            a given expression, then same value will be yielded at each occurrence;
            Lea._genVPs method is abstract: it is implemented in all Lea's subclasses
        '''
        raise NotImplementedError("missing method '%s._genVPs(self)'"%(self.__class__.__name__))
         
    def _genOneRandomMC(self):
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
            Lea._genOneRandomMC method is abstract: it is implemented in all Lea's subclasses
        '''
        raise NotImplementedError("missing method '%s._genOneRandomMC(self)'"%(self.__class__.__name__))
            
    def genRandomMC(self,n,nbTries=None):
        ''' generates n random value from the current probability distribution,
            without precalculating the exact probability distribution (contrarily to 'random' method);
            nbTries, if not None, defines the maximum number of trials in case a random value
            is incompatible with a condition; this happens only if the current Lea instance
            is (referring to) an Ilea or Blea instance, i.e. 'given' or 'buildCPT' methods;
            WARNING: if nbTries is None, any infeasible condition shall cause an infinite loop
        '''
        for _ in range(n):
            remainingNbTries = 1 if nbTries is None else nbTries
            v = self
            while remainingNbTries > 0:
                try:
                    for v in self._genOneRandomMC():
                        yield v
                    remainingNbTries = 0
                except Lea._FailedRandomMC:
                    if nbTries is not None:
                        remainingNbTries -= 1        
            if v is self:
                raise Lea.Error("impossible to validate given condition(s), after %d random trials"%nbTries) 
        
    def randomMC(self,n=None,nbTries=None):
        ''' if n is None, returns a random value with the probability given by the distribution
            without precalculating the exact probability distribution (contrarily to 'random' method);
            otherwise, returns a tuple of n such random values;
            nbTries, if not None, defines the maximum number of trials in case a random value
            is incompatible with a condition; this happens only if the current Lea instance
            is (referring to) an Ilea or Blea instance, i.e. 'given' or 'buildCPT' methods;
            WARNING: if nbTries is None, any infeasible condition shall cause an infinite loop
        '''
        n1 = 1 if n is None else n
        randomMCTuple = tuple(self.genRandomMC(n1,nbTries))
        if n is None:
            return randomMCTuple[0]
        return randomMCTuple
        
    def estimateMC(self,n,nbTries=None): 
        ''' returns an Alea instance, which is an estimation of the current distribution from a sample
            of n random values; this is a true Monte-Carlo algorithm, which does not precalculate the
            exact probability distribution (contrarily to 'random' method); 
            the method is suited for complex distributions, when calculation of exact probability
            distribution is intractable; the larger the value of n, the better the returned estimation;
            nbTries, if not None, defines the maximum number of trials in case a random value
            is incompatible with a condition; this happens only if the current Lea instance
            is (referring to) an Ilea or Blea instance, i.e. 'given' or 'buildCPT' methods;
            WARNING: if nbTries is None, any infeasible condition shall cause an infinite loop
        '''
        return Lea.fromSeq(self.randomMC(n,nbTries))
    
    def nbCases(self):
        ''' returns the number of atomic cases evaluated to build the exact probability distribution;
            this provides a measure of the complexity of the probability distribution 
        '''
        self._initCalc()
        return sum(1 for vp in self.genVPs())
    
    def isTrue(self):
        ''' returns True iff the value True has probability 1;
                    False otherwise;
            raises exception if some value is not boolean
        '''
        (n,d) = self._p(True,checkValType=True) 
        return n == d

    def isFeasible(self):
        ''' returns True iff the value True has a non-null probability;
                    False otherwise;
            raises exception if some value is not boolean
        '''
        (n,d) = self._p(True,checkValType=True)
        return n > 0
        
    def asString(self,kind='/',nbDecimals=6,chartSize=100):
        ''' returns, after evaluation of the probability distribution self, a string
            representation of it;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability in a format depending of given kind, which is string among
            '/', '.', '%', '-', '/-', '.-', '%-'; 
            the probabilities are displayed as
            - if kind[0] is '/' : rational numbers "n/d" or "0" or "1"
            - if kind[0] is '.' : decimals with given nbDecimals digits
            - if kind[0] is '%' : percentage decimals with given nbDecimals digits
            - if kind[0] is '-' : histogram bar made up of repeated '-', such that
                                  a bar length of histoSize represents a probability 1 
            if kind[1] is '-', the histogram bars with '-' are appended after 
                               numerical representation of probabilities
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used
        '''        
        return self.getAlea().asString(kind,nbDecimals,chartSize)

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
        return self.getAlea().__str__()

    __repr__ = __str__

    def asFloat(self,nbDecimals=6):
        ''' returns, after evaluation of the probability distribution self, a string
            representation of it;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability expressed as decimal with given nbDecimals digits;
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used;
        '''
        return self.getAlea().asFloat(nbDecimals)

    def asPct(self,nbDecimals=1):
        ''' returns, after evaluation of the probability distribution self, a string
            representation of it;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability expressed as percentage with given nbDecimals digits;
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used;
        '''
        return self.getAlea().asPct(nbDecimals)
    
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
        return self.getAlea().histo(size)

    def plot(self,title=None,fname=None,savefigArgs=dict(),**barArgs):
        ''' produces, after evaluation of the probability distribution self,
            a matplotlib bar chart representing it with the given title (if not None);
            the bar chart may be customised by using named arguments barArgs, which are
            relayed to matplotlib.pyplot.bar function
            (see doc in http://matplotlib.org/api/pyplot_api.html)
            * if fname is None, then the chart is displayed on screen, in a matplotlib window;
              the previous chart, if any, is erased
            * otherwise, the chart is saved in a file specified by given fname as specified
              by matplotlib.pyplot.savefig; the file format may be customised by using
              savefigArgs argument, which is a dictionary relayed to matplotlib.pyplot.savefig
              function and containing named arguments expected by this function;
              example:
               flip.plot(fname='flip.png',savefigArgs=dict(bbox_inches='tight'),color='green')
            the method requires matplotlib package; an exception is raised if it is not installed
        '''
        self.getAlea().plot(title,fname,savefigArgs,**barArgs)

    def getAlea(self,**kwargs):
        ''' returns an Alea instance representing the distribution after it has been evaluated;
            if self is an Alea instance, then it returns itself,
            otherwise the newly created Alea is cached : the evaluation occurs only for the first
            call; for successive calls, the cached Alea instance is returned, which is faster 
        '''
        if self._alea is None:
            self._alea = self.new(**kwargs)
        return self._alea

    def new(self,**kwargs):
        ''' returns a new Alea instance representing the distribution after it has been evaluated;
            if self is an Alea, then it returns a clone of itself (independent)
            note that the present method is overloaded in Alea class, to be more efficient
        '''
        self._initCalc()
        return Alea._fromValFreqs(*tuple(self.genVPs()),**kwargs)

    def cumul(self):
        ''' evaluates the distribution, then,
            returns a tuple with probability weights p that self <= value ;
            the sequence follows the order defined on values (if an order relationship is defined
            on values, then the tuples follows their increasing order; otherwise, an arbitrary
            order is used, fixed from call to call
            Note : the returned value is cached
        '''
        return self.getAlea().cumul()
        
    def invCumul(self):
        ''' evaluates the distribution, then,
            returns a tuple with the probability weights p that self >= value ;
            the sequence follows the order defined on values (if an order relationship is defined
            on values, then the tuples follows their increasing order; otherwise, an arbitrary
            order is used, fixed from call to call
            Note : the returned value is cached
        '''
        return self.getAlea().invCumul()
        
    def randomIter(self):
        ''' evaluates the distribution, then,
            generates an infinite sequence of random values among the values of self,
            according to their probabilities
        '''
        return self.getAlea()._randomIter
        
    def random(self,n=None):
        ''' evaluates the distribution, then, 
            if n is None, returns a random value with the probability given by the distribution
            otherwise, returns a tuple of n such random values
        '''
        if n is None:
            return self.getAlea().randomVal()
        return tuple(islice(self.randomIter(),n))

    def randomDraw(self,n=None,sorted=False):
        ''' evaluates the distribution, then,
            if n=None, then returns a tuple with all the values of the distribution, in a random order
                       respecting the probabilities (the higher probability of a value, the most likely
                       the value will be in the beginning of the sequence)
            if n > 0,  then returns only n different drawn values
            if sorted is True, then the returned tuple is sorted
        '''
        return self.getAlea().randomDraw(n,sorted)

    @staticmethod
    def jointEntropy(*args):
        ''' returns the joint entropy of arguments, expressed in bits;
            the returned type is a float or a sympy expression (see doc of
            Alea.entropy)
        '''
        return Clea(*args).entropy

    def condEntropy(self,other):
        ''' returns the conditional entropy of self given other, expressed in
            bits; note that this value is also known as the equivocation of
            self about other;
            the returned type is a float or a sympy expression (see doc of
            Alea.entropy)
        '''
        other = Lea.coerce(other)
        ce = Clea(self,other).entropy - other.entropy
        try:
            return max(0.0,ce)
        except:
            return ce

    def mutualInformation(self,other):
        ''' returns the mutual information between self and other, expressed
            in bits;
            the returned type is a float or a sympy expression (see doc of
            Alea.entropy)
        '''
        other = Lea.coerce(other)
        mi = self.entropy + other.entropy - Clea(self,other).entropy
        try:
            return max(0.0,mi)
        except:
            return mi

    def informationOf(self,val):
        ''' returns the information of given val, expressed in bits;
            the returned type is a float or a sympy expression (see doc of
            Alea.entropy);
            raises an exception if given val has a null probability
        '''
        return self.getAlea().informationOf(val)

    def lr(self,*hypLeas):
        ''' returns a float giving the likelihood ratio (LR) of an 'evidence' E,
            which is self, for a given 'hypothesis' H, which is the AND of given
            hypLeas arguments; it is calculated as 
                  P(E | H) / P(E | not H)
            both E and H must be boolean probability distributions, otherwise
            an exception is raised;
            an exception is raised also if H is certainly true or certainly false      
        '''
        return self.given(*hypLeas).lr()

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
            for attrName in self.__slots__:
                attrVal = getattr(self,attrName)
                if isinstance(attrVal,Lea):
                    args.append(attrVal.internal(indent+'  ',refs))
                elif isinstance(attrVal,tuple):
                    args1 = ['(']
                    for lea1 in attrVal:
                        args1.append(lea1.internal(indent+'    ',refs))
                    args.append(('\n'+indent+'    ').join(args1)+'\n'+indent+'  )')
                elif hasattr(attrVal,'__call__'):
                    args.append(attrVal.__module__+'.'+attrVal.__name__)
        return ('\n'+indent+'  ').join(args)

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

# init Alea class, with float type by default
Alea.setProbType('f')

# Lea static methods exported from Alea
Lea.setProbType = Alea.setProbType
Lea.coerce = Alea.coerce
Lea.fromVals = Alea.fromVals
Lea.fromSeq = Alea.fromSeq
Lea.fromValFreqs = Alea.fromValFreqs
Lea.interval = Alea.interval
Lea.boolProb = Alea.boolProb
Lea.fromValFreqsDict = Alea.fromValFreqsDict
Lea.fromValFreqsOrdered = Alea._fromValFreqsOrdered
Lea.fromCSVFile = Alea.fromCSVFile
Lea.fromCSVFilename = Alea.fromCSVFilename
Lea.fromPandasDF = Alea.fromPandasDF
Lea.bernoulli = Alea.bernoulli
Lea.binom = Alea.binom
Lea.poisson = Alea.poisson


# Lea convenience functions (see __init__.py)
V  = Lea.fromVals
VP = Lea.fromValFreqs
B  = Lea.boolProb
X  = Lea.cprod
f_1 = ProbFraction.one

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

