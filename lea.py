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
from math import log, sqrt, exp


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
    through +, -, *, / operators. The resulting dstribution's values and probabilities
    are determined by combination of operand's values with a sum weighted by probability
    products (the operation known as 'convolution', for the adition case).

    - Other supported binary arithmetic operators are power (**), modulo (%) and
    divmod function.

    - Unary operators +, - and abs function are supported also.

    - The Python's operator precedence rules, with the parenthesis overrules, are fully
    respected.

    - Any object X, which is not a Lea instance, involved as argument of an
    expression containing a Lea instance, is coerced to a Lea instance
    having X has sole value, with probabilty 1 (i.e. occurrence of X is certain).

    - Lea instances can be compared together, through ==, !=, <, <=, >, >= operators.
    The resulting distribution is a boolean distribution, giving probability of True result
    and complementary probability of False result.

    - Boolean distributions can be combined together with AND, OR, XOR, through &, |, ^
    operators

    - WARNING: the Python's and, or, not, operators shall not be used because they do not return
    any sensible result. Replace:
           a and b    by    a & b
           a or b     by    a | b
           not a      by    ~ a

    - WARNING: in boolean expression invloving arithmetic comparisons, the parenthesis
    shall be used, e.g. (a < b) & (b < c)

    - WARNING: the augmented comparison (a < b < c) expression shall not be used.; it does
    not return any sensible result (reason: it has the same limtation as 'and' operator).

    Lea instances can be used to generate random values, respecting the given probabilities.
            
    There are six concrete subclasses to Lea, namely: Alea, Clea, Plea, Flea, Tlea, Ilea, Olea and Dlea.
    
    Each subccass represents a special kind of discrete probability distribution, with its own data
    or with references to other Lea instances to be combined together through a given operation.
    Each subclass defines what are the (value,probability) pairs or how they can be generated. 
    The Lea class acts as a facade, by providing different methods to instantiate these subclasses,
    so it is usually not needed to instiantiate them explicitely. 
    
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

    '''

    __slots__ = ('_val','_alea')

    def __init__(self):
        ''' initializes Lea instance's attributes
        '''
        # value temporarily bound to the instance, during calculations (see genVPs method)
        # note: self is used as a sentinel value to mean that the no value is currently bound 
        # (None is not a good sentinel value since it prevents to be used as value in a distribution) 
        self._val = self
        # alea instance acting as a cache when actual value-probability pairs have been calculated  
        self._alea = None

    def reset(self):
        ''' removes current value binding;
            this calls _reset() method implemented in Lea subclasses
        '''
        self._val = self
        self._reset()
         
    def clone(self,cloneTable=None):
        ''' returns a deep copy of current Lea, without any value binding;
            if the lea tree contains multiple references to the same Lea instance,
            then it is cloned only once and the references are copied in the cloned tree
            (the cloneTable dictionary serves this purpose);
            the method calls the _clone() method implemented in Lea subclasses
        '''
        if cloneTable is None:
            cloneTable = {}
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
            for the given sequence of values, so that each value occurrence is
            taken as equiprobable;
            if each value occurs exactly once, then the distribution is uniform,
            i.e. the probability of each value is equal to 1 / #values;
            if the sequence is empty, then an exception is raised
        '''
        return Alea.fromVals(*vals)

    @staticmethod
    def fromValFreqs(*valFreqs):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of (val,freq) tuples, where freq is a natural number
            so that each value is taken with the given frequency (or sum of 
            frequencies of that value if it occurs multiple times);
            if the sequence is empty, then an exception is raised
        '''
        return Alea.fromValFreqs(*valFreqs)
    
    @staticmethod
    def fromValFreqsDict(probDict):
        ''' static method, returns an Alea instance representing a distribution
            for the given dictionary of {val:freq}, where freq is an integer number
            so that each value is taken with the given frequency
            if the sequence is empty, then an exception is raised
        '''
        return Alea.fromValFreqsDict(probDict)

    @staticmethod
    def boolProb(pNum,pDen):
        ''' static method, returns an Alea instance representing a boolean
            distribution such that probabilty of True is pNum/pDen
        '''
        return Alea.fromValFreqs((True,pNum),(False,pDen-pNum))

    @staticmethod
    def poisson(mean):
        ''' static method, returns a Alea instance representing a Poisson probability
            distribution having the given mean
        '''
        from sys import maxint
        valFreqs = []
        p = exp(-mean)
        v = 0
        t = 0.
        while True:
            n = int(p*maxint)
            if n <= 0:
                break
            valFreqs.append((v,n))
            t += p
            v += 1
            p = (p*mean) / v
        return Alea.fromValFreqs(*valFreqs)

    def withProb(self,condLea,pNum,pDen):
        ''' returns a new Alea instance from current distribution,
            such that pNum/pDen is the probability that condLea is true
        '''
        if not (0 <= pNum <= pDen):
            raise Exception("ERROR; %d/%d is outside the probability range [0,1]"%(pNum,pDen))
        condLea = Lea.coerce(condLea)
        d = self.map(lambda v:condLea.isTrue()).getAlea()
        e = dict(d.genVPs())
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
        return Alea.fromValFreqs(*((v,p*w2[condLea.isTrue()]) for (v,p) in self.genVPs()))

    def withCondProb(self,condLea,givenCondLea,pNum,pDen):
        ''' returns a new Alea instance from current distribution,
            such that pNum/pDen is the probability that condLea is true
            given that givenCondLea is True, under the constraint that
            the returned distribution keeps prior probabilities of condLea
            and givenCondLea unchanged
        '''
        if not (0 <= pNum <= pDen):
            raise Exception("ERROR; %d/%d is outside the probability range [0,1]"%(pNum,pDen))
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
        return Alea.fromValFreqs(*((v,p*w2[(condLea.isTrue(),givenCondLea.isTrue())]) for (v,p) in self.genVPs()))
    
    def given(self,info):
        ''' returns a new Ilea instance representing the current distribution
            updated with the given info, which is either a boolean or a Lea instance
            with boolean values; the values present in the returned distribution 
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
            arguments (coerced to Lea instances), including self as first argument 
        '''
        return Clea(self,*args)

    def cprodTimes(self,nTimes):
        ''' returns a new Plea instance, representing the cartesian product of self
            with itself, iterated nTimes
        '''
        return Plea(self,nTimes)

    def map(self,f,args=()):
        ''' returns a new Flea instance representing the distribution obtained
            by applying the given function f, taking values of current distribution
            as first argument and given args tuple as following arguments;
            note: f can be also a Lea instance, with functions as values
        '''
        return Flea.build(f,(self,)+args)

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
      
    @staticmethod
    def coerce(value):
        ''' static method, returns a Lea instance corresponding the given value:
            if the value is a Lea instance, then it is returned
            otherwise, an Alea instance is returned, with given value
            as unique (certain) value
        '''
        if not isinstance(value,Lea):
            value = Alea(((value,1),))
        return value

    @staticmethod
    def gcd(a,b):
        ''' static method, returns the greatest common divisor between the given
            integer arguments
        '''
        while a > 0:
            (a,b) = (b%a,a)
        return b
            
    def p(self,val):
        ''' returns a string of the form 'n/d' representing the probability of
            the given value val, as a reduced rational number
            if the probability is 0 or 1, then 'O' or '1' is returned, respectively
        '''
        (p,count) = self._p(val)
        gcd = Lea.gcd(p,count)
        res = '%d' % (p/gcd)
        count /= gcd
        if count > 1:
            res += '/%d' % count
        return res

    def pf(self,val):
        ''' returns the probability of the given value val, as a floating point
            number, from 0.0 to 1.0
        '''
        (p,count) = self._p(val)
        return float(p) / count

    def _p(self,val):
        ''' returns a tuple of natural numbers (p,s) where
            s is the sum of the probability weights of all values 
            p is the probability weight of the given value val,
              as a natural number from 0 to s;
            note: the probability of val is p/s but the ratio is not reduced
        '''
        if self._alea is not None:
            return self._alea._p(val)
        count = 0
        p = 0
        for (v1,p1) in self.genVPs():
            count += p1
            if v1 == val:
                p += p1
        return (p,count)

    def isAnyOf(self,*values):
        ''' returns a boolean probability distribution
            indicating the probability that a value is any of the given values 
        '''
        return Flea.build(lambda v: v in values,(self,))

    def isNoneOf(self,*values):
        ''' returns a boolean probability distribution
            indicating the probability that a value is none of the given values 
        '''
        return Flea.build(lambda v: v not in values,(self,))

    def __call__(self,*args):
        ''' returns a new Flea instance representing the probability distribution
            of values returned by invoking functions of current distribution on 
            given arguments (assuming that the values of current distribution are
            functions);
            called on evaluation of "self(*args)"
        '''
        return Flea.build(self,args)
    
    def __getattribute__(self,attrName):
        ''' returns the attribute with the given name in the current Lea instance;
            if the attribute name is unknown as a Lea instance's attribute,
            then returns a Flea instance that shall retrieve the attibute in the
            values of current distribution; 
            called on evaluation of "self.attrName"
        '''
        try:
            return object.__getattribute__(self,attrName)
        except AttributeError:
            return Flea.build(getattr,(self,attrName,))
    
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

    def __div__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the division of the values of self by the values of other;
            called on evaluation of "self / other"
        '''
        return Flea.build(operator.div,(self,other))

    def __rdiv__(self,other):
        ''' returns a Flea instance representing the probability distribution
            resulting from the division of the values of other by the values of self;
            called on evaluation of "other / self"
        '''
        return Flea.build(operator.div,(other,self))

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
        return Flea.build(operator.and_,(self,other))

    def __rand__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical AND between the values of other and the values of self;
            called on evaluation of "other & self"
        '''
        return Flea.build(operator.and_,(other,self))

    def __or__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical OR between the values of self and the values of other;
            called on evaluation of "self | other"
        '''
        return Flea.build(operator.or_,(self,other))

    def __ror__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical OR between the values of other and the values of self;
            called on evaluation of "other | self"
        '''
        return Flea.build(operator.or_,(other,self))

    def __xor__(self,other):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical XOR between the values of self and the values of other;
            called on evaluation of "self ^ other"
        '''
        return Flea.build(operator.xor,(self,other))

    def __invert__(self):
        ''' returns a Flea instance representing the boolean probability distribution
            resulting from the locical NOT of the values self;
            called on evaluation of "~self"
        '''
        return Flea.build(operator.not_,(self,))

    def genVPs(self):
        ''' generates tuple (v,p) where v is a value of the current probability distribution
            and p is the associated probability weight (integer > 0);
            before yielding a value v, this value is bound to the current instance;
            then, if the current calculation requires to get again values on the current
            instance, then the bound value is yielded with probability 1;
            the value is unbound as soon as the execution is resumed after the yield;
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
            raise exception if some value are not booleans
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
            raise Exception("ERROR: condition evaluated as a %s although a boolean is expected"%type(res))    
        return res
        
    def __str__(self):
        ''' returns, after evaluation of the distribution, a string representation of it;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value  with its
            respective probability expressed as a rational number "n/d" or "0" or "1";
            if an order relationship is defined on values, then the values ares sorted by 
            increasing order; otherwise, an arbitrary order is used;            
            called on evalution of "str(self)" and "repr(self)"
        '''
        return self.getAlea().__str__()

    __repr__ = __str__

    def asPct(self,nbDecimals=1):
        ''' returns, after evaluation of the distribution, a string representation of it;
            it is thes ame as __str__ method , but the the probabilities are displayed as percentage
            values, with the given number of decimals (default = 1)
        '''
        return self.getAlea().asPct(nbDecimals)
    
    def getAlea(self):
        ''' returns an Alea instance representing the distribution after it has been evaluated;
            the evaluation occurs only for the first call; for successive calls, a cached
            Alea instance is returned, which is faster. 
        '''
        if self._alea is None:
            try:
                self._alea = Alea.fromValFreqs(*(tuple(self.genVPs())))
            except:
                self.reset()
                raise
        return self._alea

    def integral(self):
        ''' returns, after evaluation of the distribution, a tuple with couples (x,p) giving the
            probability weight p of having a value less than or equal to x;
            if an order relationship is defined on values, then the tuples follows the incresing
            order of x; otherwise, an arbitrary order is used
        '''
        return self.getAlea().integral()
        
    def random(self,n=None):
        ''' evaluates the distribution, then, 
            if n is None, returns a random value with the probability given by the distribution
            otherwise, returns a tuple of n such random values
        '''
        return self.getAlea().random(n)

    def randomDraw(self,n=None,sorted=False):
        ''' evaluates the distribution, then,
            if n=None, then returns a tuple with all the values of the distribution, in a random order
                       respecting the probabilities (the higher count of a value, the most likely
                       the value will be in the beginning of the sequence)
            if n>0, then returns only n different drawn values
            if sorted is True, then the returned tuple is sorted
        '''
        return self.getAlea().randomDraw(n,sorted)

    def stdev(self):
        ''' returns, after evaluation of the distribution, the standard deviation of the distribution
            requires that the requirements of the variance() method are met
        '''      
        return sqrt(self.variance())

    def mean(self):
        ''' returns, after evaluation of the distribution, the mean value of the probability
            distribution, which is the probability weighted sum of the values;
            requires that
            1 - the values can be subtracted together,
            2 - the differences of values can be multiplied by floating-point numbers,
            3 - the differences of values multiplied by floating-point numbers can be
                added to the values;
            if any of these conditions is not met, then the result depends of the
            value class implementation (likely, raised exception)
        '''
        return self.getAlea().mean()

    def variance(self):
        ''' returns, after evaluation of the distribution, the variance of the probability
            distribution;
            requires that
            1 - the requirements of the mean() method are met,
            2 - the values can be subtracted to the mean value,
            3 - the differences between values and the mean value can be squared;
            if any of these conditions is not met, then the result depends of the
            value implementation (likely, raised exception)
        '''
        return self.getAlea().variance()

    def entropy(self):
        ''' returns, after evaluation of the distribution, the entropy of the distribution
        '''
        return self.getAlea().entropy()

from alea import Alea
from clea import Clea
from plea import Plea
from tlea import Tlea
from dlea import Dlea
from ilea import Ilea
from flea import Flea
from olea import Olea

import license
