'''
--------------------------------------------------------------------------------

    lea.py

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

import operator
from itertools import islice
from math import log
from prob_fraction import ProbFraction
from toolbox import calcGCD, log2, makeTuple, easyMin, easyMax

class Lea(object):
    
    '''
    Lea is an abstract class representing discrete probability distributions.

    Each instance of concrete Lea subclasses (called simply a "Lea instance" in the following)
    represents a discrete probability distribution, which associates each value of a set of
    values with the probability that such value occurs.

    A Lea instance can be defined by a sequence of (value,weight), giving the probability weight 
    of each value. Such probability weights are natural numbers. The actual probability of a
    given value can be calculated by dividing a weight by the sum of all weights. 
     
    A Lea instance can be defined also by a sequence of values, their probability weight being 
    their number of occurences in the sequence.

    Lea instances can be combined in arithmetic expressions resulting in new Lea instances, by
    obeying the following rules:

    - Lea instances can be added, subtracted, multiplied and divided together,
    through +, -, *, / operators. The resulting distribution's values and probabilities
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

    - WARNING: the Python's and, or, not, operators shall not be used because they do not return
    any sensible result. Replace:
           a and b    by    a & b
           a or b     by    a | b
           not a      by    ~ a

    - WARNING: in boolean expression involving arithmetic comparisons, the parenthesis
    shall be used, e.g. (a < b) & (b < c)

    - WARNING: the augmented comparison (a < b < c) expression shall not be used.; it does
    not return any sensible result (reason: it has the same limtation as 'and' operator).

    Lea instances can be used to generate random values, respecting the given probabilities.
            
    There are ten concrete subclasses to Lea, namely: Alea, Clea, Plea, Flea, Tlea, Ilea, Olea,
     Dlea, Mlea, Rlea and Blea.
    
    Each subccass represents a special kind of discrete probability distribution, with its own data
    or with references to other Lea instances to be combined together through a given operation.
    Each subclass defines what are the (value,probability) pairs or how they can be generated (see
    _genVPs method implemented in each Lea subclass, which is called by Lea.genVPs method). 
    The Lea class acts as a facade, by providing different methods to instantiate these subclasses,
    so it is usually not needed to instantiate them explicitely. 
    
    Here is an overview on these subclasses, with their relationships.

    - An Alea instance is defined by explicit value-probability pairs. Each probability is
    defined as a positive "counter" integer, without upper limit. The actual
    probability is calculated by dividing the counter by the sum of all counters.

    Instances of other Lea subclasses represent probability distributions obtained by
    operations done on existing Lea instance(s), assuming that represented events are independent.
    Any instance of such subclasses forms a tree structure, having other Lea instances as nodes
    and Alea instances as leaves. These use lazy evaluation: actual value-probability pairs are
    calculated only at the time they are required (e.g. display); then, these are cached in an
    Alea instance, itself attribute of the Lea instance, to speed up next accesses.
    
    Here is a brief presentation of these subclasses: 

    - the Clea subclass provides the cartesian product of a given sequence of Lea instances;
    - the Plea subclass provides the cartesian product of one Lea instance with itself, a given number of times;
    - the Flea subclass applies a given function to a given sequence of Lea instances;
    - the Tlea subclass applies a given 2-ary function, a given number of times, on a given Lea instance;
    - the Ilea subclass filters the values of a given Lea instance by applying a given boolean condition;
    - the Olea subclass builds a joint probability distribution from a Lea instance with tuples as values;
    - the Dlea subclass performs a given number of draws without replacement on a given Lea instance.
    - the Mlea subclass merges several Lea instances together
    - The Rlea subclass manages Lea instances containing other Lea instances as values
    - The Blea subclass defines CPT providing Lea instances corresponding to given conditions

    WARNING: The following methods are called without parentheses:
                         mean, var, std, mode, entropy, information
             These are applicable on any Lea instance; these are documented in the Alea class
    '''

    class Error(Exception):
        ''' exception representing any violation of requirements of Lea methods  
        '''
        pass

    # Lea attributes; name and data are placeholder attributes : these are not used by Lea
    # but could be used by client applications for specific needs
    __slots__ = ('_val','_alea','name','data')

    # constructor methods
    # -------------------

    def __init__(self):
        ''' initializes Lea instance's attributes
        '''
        # _val is the value temporarily bound to the instance, during evaluation (see genVPs method)
        # note: self is used as a sentinel value to express that no value is currently bound 
        # None value is not a good sentinel value since it prevents to be used as value in a distribution 
        self._val = self
        # alea instance acting as a cache when actual value-probability pairs have been calculated
        self._alea = None
        
    def id(self):
        ''' returns a unique id, containing the concrete Lea class name as prefix
        '''
        return '%s#%s'%(self.__class__.__name__,id(self))

    def getAleaLeavesSet(self):
        ''' returns a set containing all the Alea leaves in the tree having the root self;
            this calls _getLeaChildren() method implemented in Lea subclasses;
            this method is overloaded in Alea subclass to stop the recursion
        '''
        return frozenset(aleaLeaf for leaChild in self._getLeaChildren() for aleaLeaf in leaChild.getAleaLeavesSet())

    def reset(self):
        ''' removes current value binding (to be used only in case of brutal halt of genVPs());
            this calls _getLeaChildren() method implemented in Lea subclasses
         '''
        self._val = self
        for leaChild in self._getLeaChildren():
            leaChild.reset()
         
    def clone(self,cloneTable=None):
        ''' returns a deep copy of current Lea, without any value binding;
            if the Lea tree contains multiple references to the same Lea instance,
            then it is cloned only once and the references are copied in the cloned tree
            (the cloneTable dictionary serves this purpose);
            the method calls the _clone() method implemented in Lea subclasses
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

    @staticmethod
    def fromVals(*vals):
        ''' static method, returns an Alea instance representing a distribution
            for the given values passed as arguments, so that each value
            occurrence is taken as equiprobable;
            if each value occurs exactly once, then the distribution is uniform,
            i.e. the probability of each value is equal to 1 / #values;
            if the sequence is empty, then an exception is raised
        '''
        return Alea.fromVals(*vals)

    @staticmethod
    def fromSeq(sequence):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of values (e.g. a list, tuple, iterator,...),
            so that each value occurrence is taken as equiprobable;        
            if each value occurs exactly once, then the distribution is uniform,
            i.e. the probability of each value is equal to 1 / #values;
            if the sequence is empty, then an exception is raised
        '''
        return Lea.fromVals(*sequence)

    @staticmethod
    def fromValFreqs(*valFreqs):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of (val,freq) tuples, where freq is a natural number
            so that each value is taken with the given frequency (or sum of 
            frequencies of that value if it occurs multiple times);
            the frequencies are reduced by dividing them by their GCD
            if the sequence is empty, then an exception is raised
        '''
        return Alea.fromValFreqs(*valFreqs)
    
    @staticmethod
    def fromValFreqsNR(*valFreqs):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of (val,freq) tuples, where freq is a natural number
            so that each value is taken with the given frequency (or sum of 
            frequencies of that value if it occurs multiple times);
            if the sequence is empty, then an exception is raised
        '''
        return Alea.fromValFreqsNR(*valFreqs)

    @staticmethod
    def fromValFreqsDict(probDict):
        ''' static method, returns an Alea instance representing a distribution
            for the given dictionary of {val:prob}, where prob is an integer number
            so that each value val has probability proportional to prob to occur
            if the sequence is empty, then an exception is raised
        '''
        return Alea.fromValFreqsDict(probDict)

    @staticmethod
    def fromValFreqsDictArgs(**probDict):
        ''' static method, same as fromValFreqsDict, excepting that the dictionary
            is passed in a **kwargs style
        '''
        return Lea.fromValFreqsDict(probDict)

    @staticmethod
    def boolProb(pNum,pDen=None):
        ''' static method, returns an Alea instance representing a boolean
            distribution such that probability of True is pNum/pDen
            if pDen is None, then pNum expresses the probability as a Fraction
        '''
        if pDen is None:
            # pNum is expected to be a Fraction
            pDen = pNum.denominator
            pNum = pNum.numerator
        ProbFraction(pNum,pDen).check()
        return Alea.fromValFreqs((True,pNum),(False,pDen-pNum))

    @staticmethod
    def bernoulli(pNum,pDen=None):
        ''' static method, returns an Alea instance representing a bernoulli
            distribution giving 1 with probability pNum/pDen and 0 with
            complementary probability;
            if pDen is None, then pNum expresses the probability as a Fraction
        '''
        if pDen is None:
            # pNum is expected to be a Fraction
            pDen = pNum.denominator
            pNum = pNum.numerator
        ProbFraction(pNum,pDen).check()
        return Alea.fromValFreqs((1,pNum),(0,pDen-pNum))

    @staticmethod
    def binom(n,pNum,pDen=None):
        ''' static method, returns an Alea instance representing a binomial
            distribution giving the number of successes among a number n of 
            independent experiments, each having probability pNum/pDen of success;  
            if pDen is None, then pNum expresses the probability as a Fraction
        '''
        return Lea.bernoulli(pNum,pDen).times(n)

    @staticmethod
    def poisson(mean,precision=1e-20):
        ''' static method, returns an Alea instance representing a Poisson probability
            distribution having the given mean; the distribution is approximated by
            the finite set of values that have probability > precision (= 1e-20 by default)
            (i.e. low/high values with too small probabilities are dropped)
        '''
        return Alea.poisson(mean,precision)

    @staticmethod
    def interval(fromVal,toVal):
        ''' static method, returns an Alea instance representing a uniform probability
            distribution, for all the integers in the interval [fromVal,toVal]
        '''
        return Lea.fromVals(*range(fromVal,toVal+1))


    # constructor methods
    # -------------------

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
            lea1 = Blea.if_(reqCondLea,self.given(curCondLea).getAlea(),self.given(~curCondLea).getAlea())
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
        e = dict(d.genVPs())
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
    
    def given(self,info):
        ''' returns a new Ilea instance representing the current distribution
            updated with the given info, which is either a boolean or a Lea instance
            with boolean values; the values present in the returned distribution 
            are those and only those compatible with the given info
            The resulting (value,probability) pairs are calculated 
            when the returned Ilea instance is evaluated; if no value is found,
            then an exception is raised
        '''
        return Ilea(self,Lea.coerce(info))

    def times(self,n,op=operator.add):
        ''' returns a new Tlea instance representing the current distribution
            operated n times with itself, through the given binary operator
        '''
        return Tlea(op,self,n)

    def timesTuple(self,n):
        ''' returns a new Tlea instance with tuples of length n, containing
            the cartesian product of self with itslef repeated n times
        '''
        return self.map(makeTuple).times(n)

    def cprod(self,*args):
        ''' returns a new Clea instance, representing the cartesian product of all
            arguments (coerced to Lea instances), including self as first argument 
        '''
        return Clea(self,*args)

    def cprodTimes(self,nTimes):
        ''' returns a new Plea instance, representing the cartesian product of self
            with itself, iterated nTimes
        '''
        return Plea(self,nTimes)

    @staticmethod
    def reduce(op,args):
        ''' returns a new Flea instance that join the given args with the given
            function op, from left to right;
            requires that op is a 2-ary function, accepting self's values as arguments;
            requires that args contains at least one element
        '''
        argsIter = iter(args)
        res = next(argsIter)
        for arg in argsIter:
            res = Flea.build(op,(res,arg))
        return res

    def merge(self,*leaArgs):
        ''' returns a new Mlea instance, representing the merge of given leaArgs, i.e.
                  P(v) = (P1(v) + ... + Pn(v)) / n
            where P(v)  is the probability of value v in the merge result 
                  Pi(v) is the probability of value v in leaArgs[i]
        '''
        return Mlea(self,*leaArgs)
    
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
        ''' returns a new Olea instance representing a joint probability distribution
            from the current distribution supposed to have n-tuples as values,
            to be associated with the given n attribute names
        '''
        return Olea(attrNames,self)
          
    def draw(self,nbValues):
        ''' returns a new Dlea instance representing a probability distribution of the
            sequences of values obtained by the given number of draws without
            replacement from the current distribution 
        '''
        return Dlea(self,nbValues)
        
    def flat(self):
        ''' assuming that self's values are themselves Lea instances,
            returns a new Rlea instance representing a probability distribution of
            inner values of these Lea instances  
        '''
        return Rlea(self)      
      
    @staticmethod
    def coerce(value):
        ''' static method, returns a Lea instance corresponding the given value:
            if the value is a Lea instance, then it is returned
            otherwise, an Alea instance is returned, with given value
            as unique (certain) value
        '''
        if not isinstance(value,Lea):
            return Alea(((value,1),))
        return value

    def equiv(self,other):
        ''' returns True iff self and other represent the same probability distribution,
            i.e. they have the same probability for each of their value
            returns False otherwise
        '''
        other = Lea.coerce(other)
        # set(...) is used to avoid any dependency on the order of values
        res = frozenset(self.vps()) == frozenset(other.vps())
        if not res:
            # the previous test assumed that the instances have the same denominator
            # this is not the case if one of them has been created with fromValFreqsNR method
            # make an 'advanced' test, by insuring that both instances have the same denominator
            s = Alea.fromValFreqs(*self.vps())
            o = Alea.fromValFreqs(*other.vps())
            res = frozenset(s.vps()) == frozenset(o.vps()) 
        return res

    def p(self,val=None):
        ''' returns a ProbFraction instance representing the probability of given value val,
            from 0/1 to 1/1
            if val is None, then a tuple is returned with the probabilities of each value,
            in the same order as defined on values (call vals method to get this 
            ordered sequence)
        '''
        if val is None:
            count = self.getAlea()._count
            return tuple(ProbFraction(p,count) for p in self.ps())
        return ProbFraction(*self._p(val))
 
    def vps(self):
        ''' generates tuples (v,p) where v is a value of self
            and p is the associated probability weight (integer > 0);
            the sequence follows the order defined on values
        '''
        return self.getAlea()._genVPs()

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
        if val is None:
            count = float(self.getAlea()._count)
            return tuple(p/count for p in self.ps())
        (p,count) = self._p(val)
        return p / float(count)
        
    def cdf(self,val=None):
        ''' cumulative distribution function
            returns the probability that self's value is less or equal to the given value val,
            as a floating point number from 0.0 to 1.0
            if val is None, then a tuple is returned with the probabilities of each value,
            in the same order as defined on values (call vals method to get this 
            ordered sequence); the last probability is always 1.0 
        '''
        count = float(self.getAlea()._count)
        if val is None:
            return tuple(p/count for p in self.cumul()[1:])
        return self.getAlea().pCumul(val)/count

    def _p(self,val):
        ''' returns the probability p/s of the given value val, as a tuple of naturals (p,s)
            where
            s is the sum of the probability weights of all values 
            p is the probability weight of the given value val (from 0 to s)
            note: the ratio p/s is not reduced
        '''
        return self.getAlea()._p(val)

    def isAnyOf(self,*values):
        ''' returns a boolean probability distribution
            indicating the probability that a value is any of the values passed as arguments
        '''
        return Flea.build(lambda v: v in values,(self,))

    def isNoneOf(self,*values):
        ''' returns a boolean probability distribution
            indicating the probability that a value is none of the given values passed as arguments 
        '''
        return Flea.build(lambda v: v not in values,(self,))

    @staticmethod
    def buildCPTFromDict(aCPTDict,priorLea=None):
        ''' same as buildCPT, with clauses specified in the aCPTDict dictionary
            {condition:result}
        '''
        return Blea.build(*(aCPTDict.items()),priorLea=priorLea)

    @staticmethod
    def if_(condLea,thenLea,elseLea):
        ''' returns an instance of Blea representing the conditional probability table
            giving thenLea  if condLea is true
                   elseLea  otherwise
            this is a convenience method equivalent to 
              Lea.buildCPT((condLea,thenLea),(None,elseLea))
        '''
        return Blea.build((condLea,thenLea),(None,elseLea))
        
    @staticmethod
    def buildCPT(*clauses,**kwargs):
        ''' returns an instance of Blea representing the conditional probability table
            (e.g. a node in a Bayes network) from the given clauses;
            each clause is a tuple (condition,result)
            where condition is a boolean or a Lea boolean distribution
              and result is a value or Lea distribution representing the result
                   assuming that condition is true
            the conditions from all clauses shall be mutually exclusive
            if a clause contains None as condition, then it is considered as a 'else'
            condition
            if a priorLea argument is specified, then the 'else' clause is calculated
            so that the returned Blea if no condition is given is priorLea ; it is an 
            error to specify a 'else' clause if priorLea argument is specified
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
            the 'else' clause is calculated so that the returned Blea if no condition is given 
            is self
        ''' 
        return Blea.build(*clauses,priorLea=self)
    
    def __call__(self,*args):
        ''' returns a new Flea instance representing the probability distribution
            of values returned by invoking functions of current distribution on 
            given arguments (assuming that the values of current distribution are
            functions);
            called on evaluation of "self(*args)"
        '''
        return Flea.build(self,args)
        
    def __getitem__(self,index):
        ''' returns a new Flea instance representing the probability distribution
            obtained by indexing or slicing each value with index
            called on evaluation of "self[index]"
        '''
        return Flea.build(operator.getitem,(self,index))

    def __iter__(self):
        ''' returns the iterator returned by genVP()
            called on evaluation of "iter(self)", "tuple(self)", "list(self" 
                or on "for x in self"
        '''
        return self.genVPs()
        
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
            return Flea.build(getattr,(self,attrName))

    @staticmethod
    def fastMax(*args):
        ''' returns a new Alea instance giving the probabilities to have the maximum
            value of each combination of the given args;
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
        ''' returns a new Alea instance giving the probabilities to have the minimum
            value of each combination of the given args;
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
        ''' returns a new Flea instance giving the probabilities to have the maximum
            value of each combination of the given args;
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
        ''' returns a new Flea instance giving the probabilities to have the minimum
            value of each combination of the given args;
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
        return Flea.build(operator.lt,(self,other))

    def __le__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            that the values of self are less than or equal to the values of other;
            called on evaluation of "self <= other"
        '''
        return Flea.build(operator.le,(self,other))

    def __eq__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            that the values of self are equal to the values of other;
            called on evaluation of "self == other"
        '''
        return Flea.build(operator.eq,(self,other))

    def __hash__(self):
        return id(self)

    def __ne__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            that the values of self are different from the values of other;
            called on evaluation of "self != other"
        '''
        return Flea.build(operator.ne,(self,other))

    def __gt__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            that the values of self are greater than the values of other;
            called on evaluation of "self > other"
        '''
        return Flea.build(operator.gt,(self,other))

    def __ge__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            that the values of self are greater than or equal to the values of other;
            called on evaluation of "self >= other"
        '''
        return Flea.build(operator.ge,(self,other))
    
    def __add__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the addition of the values of self with the values of other;
            called on evaluation of "self + other"
        '''
        return Flea.build(operator.add,(self,other))

    def __radd__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the addition of the values of other with the values of self;
            called on evaluation of "other + self"
        '''
        return Flea.build(operator.add,(other,self))

    def __sub__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the subtraction of the values of other from the values of self;
            called on evaluation of "self - other"
        '''
        return Flea.build(operator.sub,(self,other))

    def __rsub__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the subtraction of the values of self from the values of other;
            called on evaluation of "other - self"
        '''
        return Flea.build(operator.sub,(other,self))

    def __pos__(self):
        ''' returns a Flea instance representing the probability distribution
            resulting from applying the unary positive operator on the values of self;
            called on evaluation of "+self"
        '''
        return Flea.build(operator.pos,(self,))

    def __neg__(self):
        ''' returns a Flea instance representing the probability distribution
            resulting from negating the values of self;
            called on evaluation of "-self"
        '''
        return Flea.build(operator.neg,(self,))

    def __mul__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the multiplication of the values of self by the values of other;
            called on evaluation of "self * other"
        '''
        return Flea.build(operator.mul,(self,other))

    def __rmul__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the multiplication of the values of other by the values of self;
            called on evaluation of "other * self"
        '''
        return Flea.build(operator.mul,(other,self))

    def __pow__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the powering the values of self with the values of other;
            called on evaluation of "self ** other"
        '''
        return Flea.build(operator.pow,(self,other))

    def __rpow__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the powering the values of other with the values of self;
            called on evaluation of "other ** self"
        '''
        return Flea.build(operator.pow,(other,self))

    def __truediv__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the division of the values of self by the values of other;
            called on evaluation of "self / other"
        '''
        return Flea.build(operator.truediv,(self,other))

    def __rtruediv__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the division of the values of other by the values of self;
            called on evaluation of "other / self"
        '''
        return Flea.build(operator.truediv,(other,self))

    def __floordiv__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the floor division of the values of self by the values of other;
            called on evaluation of "self // other"
        '''
        return Flea.build(operator.floordiv,(self,other))

    def __rfloordiv__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the floor division of the values of other by the values of self;
            called on evaluation of "other // self"
        '''
        return Flea.build(operator.floordiv,(other,self))

    # Python 2 compatibility
    __div__ = __truediv__
    __rdiv__ = __rtruediv__

    def __mod__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the modulus of the values of self with the values of other;
            called on evaluation of "self % other"
        '''
        return Flea.build(operator.mod,(self,other))

    def __rmod__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the modulus of the values of other with the values of self;
            called on evaluation of "other % self"
        '''
        return Flea.build(operator.mod,(other,self))

    def __divmod__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from applying the function divmod on the values of self and the values of other;
            called on evaluation of "divmod(self,other)"
        '''
        return Flea.build(divmod,(self,other))

    def __rdivmod__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from applying the function divmod on the values of other and the values of self;
            called on evaluation of "divmod(other,self)"
        '''
        return Flea.build(divmod,(other,self))

    def __floordiv__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the integer division of the values of self by the values of other;
            called on evaluation of "self // other"
        '''
        return Flea.build(operator.floordiv,(self,other))
    
    def __rfloordiv__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the integer division of the values of other by the values of self;
            called on evaluation of "other // self"
        '''
        return Flea.build(operator.floordiv,(other,self))

    def __abs__(self):
        ''' returns a Flea instance representing the probability distribution
            resulting from applying the abs function on the values of self;
            called on evaluation of "abs(self)"
        '''
        return Flea.build(abs,(self,))
    
    def __and__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical AND between the values of self and the values of other;
            called on evaluation of "self & other"
        '''
        return Flea.build(Lea._safeAnd,(self,other))

    def __rand__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical AND between the values of other and the values of self;
            called on evaluation of "other & self"
        '''
        return Flea.build(Lea._safeAnd,(other,self))

    def __or__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical OR between the values of self and the values of other;
            called on evaluation of "self | other"
        '''
        return Flea.build(Lea._safeOr,(self,other))

    def __ror__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical OR between the values of other and the values of self;
            called on evaluation of "other | self"
        '''
        return Flea.build(Lea._safeOr,(other,self))

    def __xor__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical XOR between the values of self and the values of other;
            called on evaluation of "self ^ other"
        '''
        return Flea.build(Lea._safeXor,(self,other))

    def __invert__(self):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical NOT of the values self;
            called on evaluation of "~self"
        '''
        return Flea.build(Lea._safeNot,(self,))

    def __nonzero__(self):
        ''' raises an exception telling that Lea instance cannot be evaluated as a boolean
            called on evaluation of "bool(self)"
        '''
        raise Lea.Error("Lea instance cannot be evaluated as a boolean (maybe due to a lack of parentheses)")

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

    def genVPs(self):
        ''' generates tuple (v,p) where v is a value of the current probability distribution
            and p is the associated probability weight (integer > 0);
            before yielding a value v, this value is bound to the current instance;
            then, if the current calculation requires to get again values on the current
            instance, then the bound value is yielded with probability 1;
            the instance is rebound to a new value at each iteration, as soon as the execution
            is resumed after the yield;
            it is unbound at the end;
            the method calls the _genVPs method implemented in Lea subclasses;
        '''
        if self._val is not self:
            # distribution already bound to a value
            # it is yielded as a certain distribution (unique yield)
            yield (self._val,1)
        else:
            # distribution not yet bound to a value
            try:
                # browse all (v,p) tuples 
                for (v,p) in self._genVPs():
                    # bind value v
                    self._val = v
                    # yield the value with probability weight
                    # if an object calls the genVPs on the same instance before resuming
                    # the present generator, then the present instance is bound to v  
                    yield (v,p)
                # unbind value v
                self._val = self
            except:
                # an exception occurred, the current genVPs generator(s) are aborted
                # and any bound value shall be unbound
                self.reset()
                raise
        
    def isTrue(self):
        ''' returns True iff the value True has probability 1;
            returns False otherwise
        '''
        (p,count) = self._p(True)
        return p > 0 and p == count

    def isFeasible(self):
        ''' returns True iff the value True has a non-null probability;
            returns False otherwise;
            raises exception if some value are not booleans
        '''
        res = False
        for (v,p) in self.genVPs():
            if res is False:
                if not isinstance(v,bool):
                    res = v
                elif v and p > 0:
                    # true or maybe true
                    res = True
        if not isinstance(res,bool):    
            raise Lea.Error("condition evaluated as a %s although a boolean is expected"%type(res))    
        return res

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

    def getAlea(self):
        ''' returns an Alea instance representing the distribution after it has been evaluated;
            Note : the returned value is cached (the evaluation occurs only for the first call,
            for successive calls, a cached Alea instance is returned, which is faster) 
        '''
        if self._alea is None:
            try:
                self._alea = Alea.fromValFreqs(*(tuple(self.genVPs())))
            except:
                # in case of exception, remove any pending binding which could distort 
                # forthcoming calculations
                self.reset()
                raise
        return self._alea

    def getAleaClone(self):
        ''' same as getAlea method, excepting if applied on an Alea instance:
            in this case, a clone of the Alea instance is returned (insead of itself)
        '''
        return self.getAlea()
        
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
        return self.getAlea().randomIter()
        
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
                       respecting the probabilities (the higher count of a value, the most likely
                       the value will be in the beginning of the sequence)
            if n > 0,  then returns only n different drawn values
            if sorted is True, then the returned tuple is sorted
        '''
        return self.getAlea().randomDraw(n,sorted)

    def mutualInformation(self,other):
        ''' returns a float number representing the mutual information between self and other,
            expressed in bits
        '''
        other = Lea.coerce(other)
        return max(0.,self.entropy + other.entropy - Clea(self,other).entropy)

    def information(self):
        ''' returns a float number representing the information of self being true,
            expressed in bits (assuming that self is a boolean distribution)
            raises an exception if self is certainly false
        '''
        return self.informationOf(True)

    def informationOf(self,val):
        ''' returns a float number representing the information of given val,
            expressed in bits
            raises an exception if given val is impossible
        '''        
        (p,count) = self._p(val)
        if p == 0:
            raise Lea.Error("no information from impossible value")
        return log2(count/float(p))

    
from alea import Alea
from clea import Clea
from plea import Plea
from tlea import Tlea
from dlea import Dlea
from ilea import Ilea
from flea import Flea
from olea import Olea
from rlea import Rlea
from mlea import Mlea
from blea import Blea

# Lea constants representing certain values
Lea.true  = Alea(((True ,1),))
Lea.false = Alea(((False,1),))
Lea.zero  = Alea(((0    ,1),))

