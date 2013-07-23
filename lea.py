'''
--------------------------------------------------------------------------------

    lea.py

--------------------------------------------------------------------------------
Copyright 2013 Pierre Denis

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
from random import randrange
from math import log, sqrt


class Lea(object):
    
    '''
    Lea is an abstract class representing discrete probability distributions.

    Each instance of Lea subclasses (called simply a "Lea instance" in the following)
    represents a discrete probability distribution, which associates each value of a set of
    values to the probability that such value occurs.

    Distribution can be defined by enumerating values, their occurence counts
    being translated in probabilities.

    Distribution objects can be combined in arithmetic expressions resulting in new
    distribution objects, with the following rules.

    - Distribution objects can be added, subtracted, multiplied and divided together,
    through +, -, *, / operators. The resulting dstribution's values and probabilities
    are determined by combination of operand's values with a sum weighted by probability
    products (operation known as 'convolution', for the adition case).

    - Other supported binary arithmetic operators are power (**), modulo (%) and
    divmod function.

    - Unary operators +, - and abs function are supported also.

    - The Python's operator precedence rules, with the parenthesis overrules, are fully
    respected.

    - Any object X, which is not a distribution object, involved as argument of an
    expression containing a distribution object, is coerced to a distribution object
    having X has sole value, with probabilty 1 (i.e. occurrence of X is certain).

    - Distribution objects can be compared together, through ==, !=, <, <=, >, >= operators.
    The resulting distribution is a boolean distribution, giving probability of True result
    and (complement) probability of False result.

    - Boolean distributions can be combined together with AND, OR, XOR, through &, |, ^
    operators

    - WARNING: the and, or, not, ~ operators shall not be used because they do not return
    any sensible result. Replace:
           a and b    by    a & b
           a or b     by    a | b
           not a      by    True ^ a
           ~ a        by    True ^ a
    Note that the "True ^ a"  expression could also be replaced by "a == False"

    - WARNING: the augmented comparisons (a < b < c) shall not be used.; they do
    not return any sensible result (reason: it has the same limtation as 'and' operator).

    - Each occurence of a given variable referring to a distribution object in an expression
    represents an independent random variable. For example, if d is a distribution
    representing the throwing of one die, the expression 'd+d' is the distribution of
    the sum of two thrown dice, although '2*d' is the distribution of one thrown die
    multiplied by 2 (which is not the same). 

    Distribution objects can be used to generate random values, respecting the given probabilities.
            
    There are four concrete subclasses to Lea, namely: Alea, Clea, Flea, Tlea and Ilea.
    Here is an overview on these classes, with their relationships.

    An Alea instance is defined by explicit value-probability pairs. Each probability is
    defined as a positive "counter" integer, without upper limit. The actual
    probability is calculated by dividing the counter by the sum of all counters.

    Instances of other Lea subclasses represent probability distributions obtained by transformation
    of existing Lea instance(s), assuming that represented events are independent.
    These form recursive tree structures having Alea instances as leaves.
    These use lazy evaluation: actual value-probability pairs
    are calculated only at the time they are required (e.g. display); then these
    are cached in an Alea instance, instance attribute of Lea, to speed up next accesses. 

    A Clea instance is defined by a given sequence (L1,...Ln) of Lea instances; it represents
    a probability distribution made up from the cartesian product L1 x ... x Ln; it associates
    each (v1,...,vn) tuple with probability product P1(v1)...Pn(vn).

    A Flea instance represents a probability distribution obtained by applying a given n-ary function f on
    arguments taking the values of a given sequence (L1,...Ln) of Lea instances. It is able
    also to treat, in place of a single function f, a distribution having n-ary functions
    as values. The function(s) and arguments are processed with a Clea instance to
    perform the required cartesian product between functions and arguments.
    As special cases, the arithmetic operators used on Lea instance are translated 
    into Flea instances with the corresponding functions of Python operator module
    (add, sub, mul, ...).

    A Tlea instance represents a probability distribution obtained by applying a given 2-ary function
    repeatedly on given Lea instancea, a given number of times. It allows to avoid tedious
    typing or explcit loops; also, it makes the calculation faster by using a dichotomic algorithm.

    An Ilea instance represents a distribution obtained by constraining a given Lea instance by
    a given boolean condition.

    '''

    __slots__ = ('_val','_alea')

    def __init__(self):
        ''' initialize Lea instance's attributes
        '''
        # value temporarily bound to the instance, during operations
        self._val = None
        # alea instance acting as a cache when value-probability pairs have been calculated  
        self._alea = None

    def reset(self):
        ''' remove current binding
        '''
        self._val = None

    @staticmethod
    def fromVals(*vals):
        ''' static method, returning an Alea instance representing a distribution
            for the given sequence of values, so that each value occurrence is
            taken as equiprobable;
            if each value occurs exactly once, then the distribution is uniform,
            i.e. the probability of each value is equal to 1 / #values
            if no value is provided, then an exception is raised
        '''
        return Alea.fromVals(*vals)

    @staticmethod
    def fromValFreqs(*valFreqs):
        ''' static method, returning an Alea instance representing a distribution
            for the given sequence of (val,freq) tuples, where freq is an integer number
            so that each value is taken with the given frequency (or sum of 
            frequencies of that value if it occurs multiple times)
            if the sequence is empty, then an exception is raised
        '''
        return Alea.fromValFreqs(*valFreqs)
    
    @staticmethod
    def fromValFreqsDict(probDict):
        ''' static method, returning an Alea instance representing a distribution
            for the given dictionary of {val:freq}, where freq is an integer number
            so that each value is taken with the given frequency
            if the sequence is empty, then an exception is raised
        '''
        return Alea.fromValFreqsDict(probDict)

    @staticmethod
    def boolProb(pNum,pDen):
        ''' static method, returning an Alea instance representing a boolean
            distribution such that prob(True) = pNum/pDen
        '''
        return Lea.fromValFreqs((True,pNum),(False,pDen-pNum))

    def withProb(self,condLea,pNum,pDen):
        ''' returns a new Alea instance from current distribution,
            such that pNum/pDen is the probability that condLea is true
        '''
        if not (0 <= pNum <= pDen):
            raise Exception("ERROR; %d/%d is outside the probability range [0,1]"%(pNum,pDen))
        condLea = Lea.coerce(condLea)
        d = self.map(lambda v:condLea.isTrue()).getAlea()
        e = dict(d.genVPs(None))
        eT = e.get(True,0)
        eF = e.get(False,0)
        # new probabilities
        nT = pNum
        nF = pDen - pNum
        # feasibility checks
        if eT == 0 and nT > 0:
            raise Exception("ERROR: unfeasible: probability shall remain 0")
        if eF == 0 and nF > 0:
            raise Exception("ERROR: unfeasible: probability shall remain 1")
        w = { True  : nT,
              False : nF }
        m = reduce(operator.mul,e.itervalues(),1)
        # factors to be applied on current probabilities
        # depending on the truth value of condLea on each value
        w2 = dict((cg,w[cg]*(m/ecg)) for (cg,ecg) in e.iteritems())
        return Alea.fromValFreqs(*((v,p*w2[condLea.isTrue()]) for (v,p) in self.genVPs(None)))

    '''
    a1 0   0       3/7  0    3      15     30     15
    a2 0   0            0                         15
    a3 2   2/5     4/7  2    4      20     40     16
    a4 3   3/5          3                         24

                                           70     70
    70 = pDen *  n * nb0

    n * pNum                  if T
    nb0 * ni * (pDen-pNum)    if F


    '''


    def withCondProb(self,condLea,givenCondLea,pNum,pDen):
        ''' returns a new Alea instance from current distribution,
            such that pNum/pDen is the probability that condLea is true
            given that givenCondLea is True, under the constraint that
            the returned sitsribution keeps prior probabilities of condLea
            and givenCondLea unchanged
        '''
        if not (0 <= pNum <= pDen):
            raise Exception("ERROR; %d/%d is outside the probability range [0,1]"%(pNum,pDen))
        condLea = Lea.coerce(condLea)
        givenCondLea = Lea.coerce(givenCondLea)
        # max 2x2 distribution (True,True), (True,False), (False,True), (True,True)
        # prior joint probabilities, non null probability
        d = self.map(lambda v:(condLea.isTrue(),givenCondLea.isTrue())).getAlea()
        e = dict(d.genVPs(None))
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
            raise Exception("ERROR: unfeasible: probability shall remain 0")
        if eFT == 0 and nFT > 0:
            raise Exception("ERROR: unfeasible: probability shall remain 1")
        if eTF == 0 and nTF > 0:
            raise Exception("ERROR: unfeasible: probability shall remain %d/%d"%(nCondLeaTrue,nGivenCondLeaTrue)) 
        if eFF == 0 and nFF > 0:
            msg = "ERROR: unfeasible"
            if nGivenCondLeaTrue >= nCondLeaTrue:
                msg += ": probability shall remain %d/%d"%(nGivenCondLeaTrue-nCondLeaTrue,nGivenCondLeaTrue)
            raise Exception(msg) 
        if nTF < 0 or nFF < 0:
            pDenMin = nGivenCondLeaTrue
            pNumMin = max(0,nGivenCondLeaTrue-nCondLeaFalse)
            pDenMax = nGivenCondLeaTrue
            pNumMax = min(pDenMax,nCondLeaTrue)
            gMin = Lea.gcd(pNumMin,pDenMin)
            gMax = Lea.gcd(pNumMax,pDenMax)
            pNumMin /= gMin 
            pDenMin /= gMin 
            pNumMax /= gMax 
            pDenMax /= gMax
            raise Exception("ERROR: unfeasible: probability shall be in the range [%d/%d,%d/%d]"%(pNumMin,pDenMin,pNumMax,pDenMax))
        w = { (True  , True ) : nTT,
              (True  , False) : nTF,
              (False , True ) : nFT,
              (False , False) : nFF }
        m = reduce(operator.mul,e.itervalues(),1)
        # factors to be applied on current probabilities
        # depending on the truth value of (condLea,givenCondLea) on each value
        w2 = dict((cg,w[cg]*(m/ecg)) for (cg,ecg) in e.iteritems())
        return Alea.fromValFreqs(*((v,p*w2[(condLea.isTrue(),givenCondLea.isTrue())]) for (v,p) in self.genVPs(None)))
    
    def given(self,info):
        ''' returns a new Ilea instance representing the current distribution
            amended with the given info, which is either a boolean
            or a Lea instance with boolean values
            the values present in the returned distribution 
            are those and only those compatible with the given info
            The resulting (value,probability) pairs are calculated 
            when the returned Ilea instance is evaluated; if no value is found,
            then an exception shall be raised
        '''
        return Ilea(self,Lea.coerce(info))

    def times(self,n,op=operator.add):
        ''' returns a new Tlea instance representing the current distribution
            operated n times with itself, through the given binary operator
        '''
        return Tlea(op,self,n)

    def cprod(self,*args):
        ''' returns a new Clea instance, representing the cartesian product of all
            arguments (coerced to Lea instances), including self. 
        '''
        return Clea(self,*args)

    def map(self,f,args=()):
        ''' returns a new Flea instance representing the distribution obtained
            by applying the given function f, taking values of present distribution
            as first argument and given args tuple as following arguments
            note: f can be also a Lea instance, with functions as values
        '''
        return Flea.build(f,(self,)+args)

    def asJoint(self,*attrNames):
        return Olea(attrNames,self)
              
    @staticmethod
    def coerce(value):
        ''' return a Lea instance corresponding the given value:
            if the value is a Lea instance, then it is returned
            otherwise an Alea instance is returned, with given value
            as unique (certain) value
        '''
        if not isinstance(value,Lea):
            value = Alea(((value,1),))
        return value

    @staticmethod
    def gcd(a,b):
        ''' static method returning the greatest common divisoe between the given
            integer arguments
        '''
        while a > 0:
            (a,b) = (b%a,a)
        return b
            
    def p(self,val):
        ''' returns a string representing the probability of the given value val,
            as a reduced rational number, from 0 to 1
        '''
        (p,count) = self._p(val)
        gcd = Lea.gcd(p,count)
        res = '%d' % (p/gcd)
        count /= gcd
        if count > 1:
            res += '/%d' % count
        return res

    def pf(self,val):
        ''' returns the probability of the given value val,
            as a floating point number, from 0. to 1.
        '''
        (p,count) = self._p(val)
        return float(p) / count

    def _p(self,val):
        ''' returns the weight of the given value val,
            as a natural number from 0 to N = sum of all weigths
            such number divided by N gives the probability of val
        '''
        if self._alea is not None:
            return self._alea._p(val)
        count = 0
        p = 0
        for (v1,p1) in self.genVPs(None):
            count += p1
            if v1 == val:
                p += p1
        return (p,count) 

    def __call__(self,*args):
        ''' 
        '''
        return Flea.build(self,args)
    
    def __getattribute__(self,attrName):
        ''' 
        '''
        try:
            return object.__getattribute__(self,attrName)
        except AttributeError:
            return self.buildLeaFromAttr(attrName)

    def buildLeaFromAttr(self,attrName):
        ''' 
        '''
        return Flea.build(getattr,(self,attrName,))
    
    def __lt__(self, other):
        ''' 
        '''
        return Flea.build(operator.lt,(self,other))

    def __le__(self, other):
        ''' 
        '''
        return Flea.build(operator.le,(self,other))

    def __eq__(self, other):
        ''' 
        '''
        return Flea.build(operator.eq,(self,other))

    def __ne__(self, other):
        ''' 
        '''
        return Flea.build(operator.ne,(self,other))

    def __gt__(self, other):
        ''' 
        '''
        return Flea.build(operator.gt,(self,other))

    def __ge__(self, other):
        ''' 
        '''
        return Flea.build(operator.ge,(self,other))
    
    def __add__(self,other):
        ''' 
        '''
        return Flea.build(operator.add,(self,other))

    def __radd__(self,other):
        ''' 
        '''
        return Flea.build(operator.add,(other,self))

    def __sub__(self,other):
        ''' 
        '''
        return Flea.build(operator.sub,(self,other))

    def __rsub__(self,other):
        ''' 
        '''
        return Flea.build(operator.sub,(other,self))

    def __neg__(self):
        ''' 
        '''
        return Flea.build(operator.neg,(self,))

    def __mul__(self,other):
        ''' 
        '''
        return Flea.build(operator.mul,(self,other))

    def __rmul__(self,other):
        ''' 
        '''
        return Flea.build(operator.mul,(other,self))

    def __pow__(self,other):
        ''' 
        '''
        return Flea.build(operator.pow,(self,other))

    def __rpow__(self,other):
        ''' 
        '''
        return Flea.build(operator.pow,(other,self))

    def __div__(self,other):
        ''' 
        '''
        return Flea.build(operator.div,(self,other))

    def __rdiv__(self,other):
        ''' 
        '''
        return Flea.build(operator.div,(other,self))

    def __mod__(self,other):
        ''' 
        '''
        return Flea.build(operator.mod,(self,other))

    def __rmod__(self,other):
        ''' 
        '''
        return Flea.build(operator.mod,(other,self))

    def __divmod__(self,other):
        return Flea.build(divmod,(self,other))

    def __rdivmod__(self,other):
        return Flea.build(divmod,(other,self))

    def __floordiv__(self,other):
        return Flea.build(operator.floordiv,(self,other))
    
    def __rfloordiv__(self,other):
        return Flea.build(operator.floordiv,(other,self))

    def __abs__(self):
        ''' 
        '''
        return Flea.build(abs,(self,))
    
    #NOK : shall return a float : verify that there is one unique value or else raise exception 
    def __float__(self):
        return Flea.build(float,(self,))

    def __complex__(self):
        return Flea.build(complex,(self,))

    def __and__(self,other):
        ''' 
        '''
        return Flea.build(operator.and_,(self,other))

    def __rand__(self,other):
        ''' 
        '''
        return Flea.build(operator.and_,(other,self))

    def __or__(self,other):
        ''' 
        '''
        return Flea.build(operator.or_,(self,other))

    def __ror__(self,other):
        ''' 

        '''
        return Flea.build(operator.or_,(other,self))

    def __invert__(self):
        ''' 
        '''
        return Flea.build(operator.not_,(self,))

    def genVPs(self,condLea):
        ''' 
        '''
        if self._val is not None:
            # distribution already bound to a value
            # if such binding is compatible with the given condition
            # then it is returned as a certain distribution
            if condLea is None or condLea.isFeasible():
                yield (self._val,1)
        else:
            # distribution not yet bound to a value
            if condLea is not None or self._alea is None:
                lea = self
            else:
                lea = self._alea
            try:
                for (v,p) in lea._genVPs(condLea):
                    self._val = v
                    if condLea is None or condLea.isFeasible():
                        yield (v,p)
                    self._val = None
            except:
                self.reset()
                raise

    def isTrue(self):
        (p,count) = self._p(True)
        return p > 0 and p == count

    '''
    def isFeasible(self):
        (p,count) = self._p(True)
        return p > 0
    '''

    def isFeasible(self):
        ''' 
        '''
        res = False
        for (v,p) in self.genVPs(None):
            # TODO: review this !!!!
            if res is False:
                if not isinstance(v,bool):
                    res = v
                elif v and p > 0:
                    # true or maybe true
                    res = True
        if not isinstance(res,bool):    
            raise Exception("condition evaluated as a %s although a boolean is expected"%type(res))    
        return res
    
    def __str__(self):
        ''' 
        '''
        return self.getAlea().__str__()

    def asPct(self,nbDecimals=1):
        '''
        '''
        return self.getAlea().asPct(nbDecimals)
    
    def getAlea(self):
        ''' 
        '''
        if self._alea is None:
            try:
                self._alea = Alea.fromValFreqs(*(tuple(self.genVPs(None))))
            except:
                self.reset()
                raise
        return self._alea

    __repr__ = __str__

    def integral(self):
        ''' returns a tuple with couples (x,p)
            giving the count p of having value <= x,
            if an order is not defined on values,
            then an arbitrary order is defined
        '''
        return self.getAlea().integral()
        
    def random(self,n=None):
        ''' if n is None, returns a random value with the probability given by the distribution
            otherwise returns a tuple of n such random values
        '''
        return self.getAlea().random(n)

    def randomSuite(self,n=None,sorted=False):
        ''' if n=None, returns a tuple with all the values of the distribution,
            in a random order respecting the probabilities
            (the higher count of a value, the most likely the value will be in the
             beginning of the sequence)
            if n>0, then only n different values will be drawn
            if sorted is True, then the returned tuple is sorted
        '''
        return self.getAlea().randomSuite(n,sorted)

    def stdev(self):
        ''' returns the standard deviation of the distribution
            provided that values can be multiplied by probabilities and added
        '''      
        return sqrt(self.variance())

    def mean(self):
        ''' 
        '''
        return self.getAlea().mean()

    def variance(self):
        ''' 
        '''
        return self.getAlea().variance()

    def entropy(self):
        ''' 
        '''
        return self.getAlea().entropy()

from alea import Alea
from clea import Clea
from tlea import Tlea
from ilea import Ilea
from flea import Flea
from olea import Olea

import license
