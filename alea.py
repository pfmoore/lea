'''
--------------------------------------------------------------------------------

    alea.py

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

from lea import Lea
from prob_fraction import ProbFraction
from random import randrange
from bisect import bisect_left, bisect_right
from math import log, sqrt, exp
from toolbox import LOG2, memoize, zip, next, dict, defaultdict
import operator
import sys


class Alea(Lea):
    
    '''
    Alea is a Lea subclass, which instance is defined by explicit probability distribution data.
    An Alea instance is defined by given value-probability pairs. Each probability is
    defined as a positive "counter" or "weight" integer, without upper limit. The actual
    probabilities are calculated by dividing the counters by the sum of all counters.
    Values having null probability counters are dropped.
    '''

    __slots__ = ('_vs','_ps','_count','_val','_cumul','_invCumul','_randomIter','_cachesByFunc')
    
    def __init__(self,vps):
        ''' initializes Alea instance's attributes
        '''
        Lea.__init__(self)
        # for an Alea instance, the alea cache is itself
        self._alea = self
        (self._vs,self._ps) = zip(*vps)
        self._count = sum(self._ps)
        # _val is the value temporarily bound to the instance, during evaluation (see _genVPs method)
        # note: self is used as a sentinel value to express that no value is currently bound; Python's
        #  None is not a good sentinel value since it prevents using None as value in a distribution 
        self._val = self
        self._cumul = None
        self._invCumul = None
        self._randomIter = self._createRandomIter()
        self._cachesByFunc = dict()

    # constructor methods
    # -------------------
    
    def getAleaClone(self):
        ''' same as getAlea method, excepting if applied on an Alea instance:
            in this case, a clone of the Alea instance is returned (insead of itself)
        '''        
        return self.clone()

    @staticmethod
    def fromValFreqsDictGen(probDict):
        ''' static method, returns an Alea instance representing a distribution
            for the given probDict dictionary of {val:prob}, where prob is an integer number,
            a floating-point number or a fraction (Fraction or ProbFraction instance)
            so that each value val has probability proportional to prob to occur
            any value with null probability is ignored (hence not stored)
            if the sequence is empty, then an exception is raised
        '''
        probFractions = tuple(ProbFraction.coerce(probWeight) for probWeight in probDict.values())
        # TODO Check positive
        probWeights = ProbFraction.getProbWeights(probFractions)
        return Alea.fromValFreqsDict(dict(zip(probDict.keys(),probWeights)))
    
    @staticmethod
    def fromValFreqsDict(probDict,reducing=True):
        ''' static method, returns an Alea instance representing a distribution
            for the given probDict dictionary of {val:prob}, where prob is an integer number
            so that each value val has probability proportional to prob to occur
            any value with null probability is ignored (hence not stored)
            if reducing is True, then the probability weights are reduced to have a GCD = 1
            if the sequence is empty, then an exception is raised
        '''
        count = sum(probDict.values())
        if count == 0:
            raise Lea.Error("impossible to build a probability distribution with no value")
        gcd = count
        impossibleValues = []
        # check probabilities, remove null probabilities and calculate GCD
        for (v,p) in probDict.items():
            if p < 0:
                raise Lea.Error("negative probability")
            if p == 0:
                impossibleValues.append(v)
            elif reducing and gcd > 1:
                while p != 0:
                    if gcd > p:
                        (gcd,p) = (p,gcd)
                    p %= gcd
        for impossibleValue in impossibleValues:
            del probDict[impossibleValue]
        vpsIter = probDict.items()
        if reducing:
            vpsIter = ((v,p//gcd) for (v,p) in vpsIter)
        vps = list(vpsIter)
        try:            
            vps.sort()
        except:
            # no ordering relationship on values (e.g. complex numbers)
            pass
        return Alea(vps)

    @staticmethod
    def fromVals(*values):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of values, so that each value occurrence is
            taken as equiprobable;
            if each value occurs exactly once, then the distribution is uniform,
            i.e. the probability of each value is equal to 1 / #values;
            if the sequence is empty, then an exception is raised
        '''
        probDict = dict()
        for value in values:
            probDict[value] = probDict.get(value,0) + 1
        return Alea.fromValFreqsDict(probDict)

    @staticmethod
    def _fromValFreqs(valueFreqs,reducing):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of (val,freq) tuples, where freq is a natural number
            so that each value is taken with the given frequency (or sum of 
            frequencies of that value if it occurs multiple times);
            if reducing is True, then the frequencies are reduced by dividing them by
            their GCD
            if the sequence is empty, then an exception is raised
        '''        
        probDict = dict()
        for (value,freq) in valueFreqs:
            probDict[value] = probDict.get(value,0) + freq
        return Alea.fromValFreqsDict(probDict,reducing)

    @staticmethod
    def fromValFreqs(*valueFreqs):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of (val,freq) tuples, where freq is a natural number
            so that each value is taken with the given frequency (or sum of 
            frequencies of that value if it occurs multiple times);
            the frequencies are reduced by dividing them by their GCD
            if the sequence is empty, then an exception is raised
        '''        
        return Alea._fromValFreqs(valueFreqs,True)

    @staticmethod
    def fromValFreqsNR(*valueFreqs):
        ''' static method, returns an Alea instance representing a distribution
            for the given sequence of (val,freq) tuples, where freq is a natural number
            so that each value is taken with the given frequency (or sum of 
            frequencies of that value if it occurs multiple times);
            if the sequence is empty, then an exception is raised
        '''
        return Alea._fromValFreqs(valueFreqs,False)

    @staticmethod
    def poisson(mean,precision):
        ''' static method, returns an Alea instance representing a Poisson probability
            distribution having the given mean; the distribution is approximated by
            the finite set of values that have probability > precision
            (i.e. low/high values with too small probabilities are dropped)
        '''
        precFactor = 0.5 / precision
        valFreqs = []
        p = exp(-mean)
        v = 0
        t = 0.
        while p > 0.0:
            valFreqs.append((v,int(0.5+p*precFactor)))
            t += p
            v += 1
            p = (p*mean) / v
        return Alea.fromValFreqs(*valFreqs)

    def asString(self,kind='/',nbDecimals=6,histoSize=100):
        ''' returns a string representation of probability distribution self;
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
        if kind not in ('/', '.', '%', '-', '/-', '.-', '%-'):
            raise Lea.Error("invalid display format '%s'"%kind)
        valueStrings = tuple(str(v) for v in self._vs)
        ps = self._ps
        vm = max(len(v) for v in valueStrings)
        linesIter = (v.rjust(vm)+' : ' for v in valueStrings)
        probRepresentation = kind[0]
        withHisto = kind[-1] == '-'
        if probRepresentation == '/':
            pStrings = tuple(str(p) for p in ps)
            pm = len(str(max(p for p in ps)))
            if self._count == 1:
                den = ''
            else:
                den = '/%d' % self._count
            linesIter = (line+pString.rjust(pm)+den for (line,pString) in zip(linesIter,pStrings))
        else:
            c = float(self._count)    
            if probRepresentation == '.':
                fmt = "%%s%%.%df" % nbDecimals
                linesIter = (fmt%(line,p/c) for (line,p) in zip(linesIter,ps))
            elif probRepresentation == '%':
                fmt = "%%s%%%d.%df %%%%" % (4+nbDecimals,nbDecimals)
                linesIter = (fmt%(line,100.*p/c) for (line,p) in zip(linesIter,ps))    
        if withHisto:
            c = float(self._count)    
            linesIter = (line+' '+int(0.5+(p/c)*histoSize)*'-' for (line,p) in zip(linesIter,ps))
        return '\n'.join(linesIter)

    def __str__(self):
        ''' returns a string representation of probability distribution self;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value  with its
            probability expressed as a rational number "n/d" or "0" or "1";
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used;
            called on evalution of "str(self)" and "repr(self)"
        '''
        return self.asString()
          
    def asFloat(self,nbDecimals=6):
        ''' returns a string representation of probability distribution self;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability expressed as decimal with given nbDecimals digits;
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used;
        '''
        return self.asString('.',nbDecimals)
        
    def asPct(self,nbDecimals=1):
        ''' returns a string representation of probability distribution self;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability expressed as percentage with given nbDecimals digits;
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used;
        '''
        return self.asString('%',nbDecimals)

    def histo(self,size=100):
        ''' returns a string representation of probability distribution self;
            it contains one line per distinct value, separated by a newline character;
            each line contains the string representation of a value with its
            probability expressed as a histogram bar made up of repeated '-',
            such that a bar length of given size represents a probability 1
            if an order relationship is defined on values, then the values are sorted by 
            increasing order; otherwise, an arbitrary order is used;
        '''
        return self.asString('-',histoSize=size)
                
    def getAleaLeavesSet(self):
        ''' returns a set containing all the Alea leaves in the tree having the root self
            in the present case of Alea instance, it returns the singleton set with self as element
        '''
        return frozenset((self,))        
     
    def _getLeaChildren(self):
        # Alea instance has no children
        return ()

    def _clone(self,cloneTable):
        return Alea(zip(self._vs,self._ps))

    def _getCount(self):
        ''' returns the total probability weight count (integer) of current Alea;
            this value depends on current binding (hence, the calculated value cannot be cached)
        '''
        if self._val is self:
            return self._count
        return 1

    def _genVPs(self):
        ''' generates tuple (v,p) where v is a value of the current probability distribution
            and p is the associated probability weight (integer > 0);
            this obeys the "binding" mechanism, so if the same variable is refered multiple times in
            a given expression, then same value will be yielded at each occurrence; 
            "Statues" algorithm: 
            before yielding a value v, this value v is bound to the current instance;
            then, if the current calculation requires to get again values on the current
            instance, then the bound value is yielded with probability 1;
            the instance is rebound to a new value at each iteration, as soon as the execution
            is resumed after the yield;
            it is unbound at the end;
            the method calls the _genVPs method implemented in Lea subclasses;
        '''
        if self._val is not self:
            # distribution already bound to a value because genVPs has been called already on self
            # it is yielded as a certain distribution (unique yield)
            yield (self._val,1)
        else:
            # distribution not yet bound to a value
            try:
                # browse all (v,p) tuples 
                for (v,p) in zip(self._vs,self._ps):
                    # bind value v: this is important if an object calls genVPs on the same instance
                    # before resuming the present generator (see above)
                    self._val = v
                    # yield the bound value v with probability weight p
                    yield (v,p)
            finally:
                # unbind value v, after all values have been bound or if an exception has been raised
                self._val = self
         
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
            the method calls the _genOneRandomMC method implemented in Lea subclasses;
        '''
        if self._val is not self:
            # distribution already bound to a value, because genOneRandomMC has been called already on self 
            # yield the bound value, in order to be consistent
            yield self._val
        else:
            try:
                # bind value v: this is important if an object calls genOneRandomMC on the same 
                # instance before resuming the present generator (see above)
                self._val = self.randomVal()
                # yield the bound value v
                yield self._val
            finally:
                # unbind value, after the random value has been bound or if an exception has been raised
                self._val = self
    
    def _p(self,val,checkValType=False):
        ''' returns the probability p/s of the given value val, as a tuple of naturals (p,s)
            where s is the sum of the probability weights of all values 
                  p is the probability weight of the given value val (from 0 to s)
            note: the ratio p/s is not reduced
            if checkValType is True, then raises an exception if some value in the
            distribution has a type different from val's
        '''
        p1 = 0
        if checkValType:
            errVal = self  # dumy value
            typeToCheck = type(val)
        # note: shall not exit the loop by a break/return (unbinding)
        for (v,p) in self._genVPs():
            if checkValType and not isinstance(v,typeToCheck):
                errVal = v
            if p1 == 0 and v == val:
                p1 = p
        if checkValType and errVal is not self:
            raise Lea.Error("found <%s> value although <%s> is expected"%(type(errVal).__name__,typeToCheck.__name__))
        return (p1,self._count)

    def cumul(self):
        ''' returns a tuple with the probability weights p that self <= value ;
            there is one element more than number of values; the first element is 0, then
            the sequence follows the order defined on values; if an order relationship is defined
            on values, then the tuples follows their increasing order; otherwise, an arbitrary
            order is used, fixed from call to call
            Note : the returned value is cached
        '''
        if self._cumul is None:
            cumulList = [0]
            pSum = 0
            for p in self._ps:
                pSum += p
                cumulList.append(pSum)
            self._cumul = tuple(cumulList)
        return self._cumul
        
    def invCumul(self):
        ''' returns a tuple with the probability weights p that self >= value ;
            there is one element more than number of values; the first element is 0, then
            the sequence follows the order defined on values; if an order relationship is defined
            on values, then the tuples follows their increasing order; otherwise, an arbitrary
            order is used, fixed from call to call
            Note : the returned value is cached
        '''
        if self._invCumul is None:
            invCumulList = []
            pSum = self._count
            for p in self._ps:
                invCumulList.append(pSum)
                pSum -= p
            invCumulList.append(0)
            self._invCumul = tuple(invCumulList)
        return self._invCumul
            
    def randomVal(self):
        ''' returns a random value among the values of self, according to their probabilities
        '''
        return next(self._randomIter)
        
    def _createRandomIter(self):
        ''' generates an infinite sequence of random values among the values of self,
            according to their probabilities
        '''
        count = self._count
        probs = self.cumul()[1:]
        vals = self._vs
        while True:
            yield vals[bisect_right(probs,randrange(count))]    
        
    def randomDraw(self,n=None,sorted=False):
        ''' if n is None, returns a tuple with all the values of the distribution,
            in a random order respecting the probabilities
            (the higher count of a value, the most likely the value will be in the
             beginning of the sequence)
            if n > 0, then only n different values will be drawn
            if sorted is True, then the returned tuple is sorted
        '''
        if n is None:
           n = len(self._vs)
        elif n < 0:
            raise Lea.Error("randomDraw method requires a positive integer")    
        if n == 0:
            return ()
        lea1 = self
        res = []
        while True:
            lea1 = lea1.getAlea()
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
    def pCumul(self,val):
        ''' returns, as an integer, the probability weight that self <= val
            note that it is not required that val is in the support of self
        '''
        return self.cumul()[bisect_right(self._vs,val)] 

    @memoize
    def pInvCumul(self,val):
        ''' returns, as an integer, the probability weight that self >= val
            note that it is not required that val is in the support of self
        '''
        return self.invCumul()[bisect_left(self._vs,val)] 

    @staticmethod
    def fastExtremum(cumulFunc,*aleaArgs):
        ''' static method, returns a new Alea instance giving the probabilities
            to have the extremum value (min or max) of each combination of the
            given Alea args;
            cumulFunc is the cumul function that determines whether max or min is
            used : respectively, Alea.pCumul or Alea.pInvCumul;
            the method uses an efficient algorithm (linear complexity), which is
            due to Nicky van Foreest; for explanations, see
            http://nicky.vanforeest.com/scheduling/cpm/stochasticMakespan.html
        '''
        if len(aleaArgs) == 1:
            return aleaArgs[0]
        if len(aleaArgs) == 2:
            (aleaArg1,aleaArg2) = aleaArgs
            valFreqsDict = defaultdict(int)
            for (v,p) in aleaArg1._genVPs():
                valFreqsDict[v] = p * cumulFunc(aleaArg2,v)
            for (v,p) in aleaArg2._genVPs():
                valFreqsDict[v] += (cumulFunc(aleaArg1,v)-aleaArg1._p(v)[0]) * p
            return Lea.fromValFreqsDict(valFreqsDict)
        return Alea.fastExtremum(cumulFunc,aleaArgs[0],Alea.fastExtremum(cumulFunc,*aleaArgs[1:]))

    # WARNING: the following methods are called without parentheses (see Lea.__getattr__)

    indicatorMethodNames = ('P','Pf','mean','var','std','mode','entropy','information')

    def P(self):
        ''' returns a ProbFraction instance representing the probability of True,
            from 0/1 to 1/1;
            raises an exception if some value in the distribution is not boolean
            (this is NOT the case with self.p(True))
            WARNING: this method is called without parentheses
        '''
        return ProbFraction(*self._p(True,checkValType=True))

    def Pf(self):
        ''' returns the probability of True, as a floating point number,
            from 0.0 to 1.0;
            raises an exception if some value in the distribution is not boolean
            (this is NOT the case with self.pmf(True))
            WARNING: this method is called without parentheses
        '''
        return float(self.P)
        
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
            value class implementation (likely, raised exception)
            WARNING: this method is called without parentheses
        '''
        res = None
        x0 = None
        for (x,p) in self._genVPs():
            if x0 is None:
                x0 = x
            elif res is None:
                res = p * (x-x0)
            else:
                res += p * (x-x0)
        if res is not None:
            try:
                x0 += res / float(self._count)
            except:
                # if the / operator is not supported with float as denominator
                # e.g. datetime.timedelta in Python 2.x 
                x0 += res / self._count    
        return x0
   
    def var(self):
        ''' returns a float number representing the variance of the probability distribution;
            requires that
            1 - the requirements of the mean() method are met,
            2 - the values can be subtracted to the mean value,
            3 - the differences between values and the mean value can be squared;
            if any of these conditions is not met, then the result depends of the
            value implementation (likely, raised exception)
            WARNING: this method is called without parentheses
        '''
        res = 0
        m = self.mean
        for (v,p) in self._genVPs():
            res += p*(v-m)**2
        return res / float(self._count)    

    def std(self):
        ''' returns a float number representing the standard deviation of the probability distribution
            requires that the requirements of the variance() method are met
            WARNING: this method is called without parentheses
        '''      
        return sqrt(self.var)
 
    def mode(self):
        ''' returns a tuple with the value(s) of the probability distribution having the highest probability 
            WARNING: this method is called without parentheses
        '''
        maxP = max(self._ps)
        return tuple(v for (v,p) in self._genVPs() if p == maxP)
            
    def entropy(self):
        ''' returns a float number representing the entropy of the probability  distribution
            WARNING: this method is called without parentheses
        '''
        res = 0
        count = float(self._count)
        for (v,p) in self._genVPs():
            if p > 0:
               p /= count
               res -= p*log(p)
        return res / LOG2
